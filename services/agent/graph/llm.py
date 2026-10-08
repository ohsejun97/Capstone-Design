"""
llm.py — 질의에서 (약물, 표적) 엔티티 추출

두 경로를 둔다:

  1. LLM 경로   — ANTHROPIC_API_KEY가 있으면 Claude로 구조화 추출.
                  한국어 약물명("글리벡"), 모호한 표현을 다룰 수 있다.
  2. 규칙 경로  — 키 없이도 동작. 사전 + 패턴 매칭.
                  CI에서 네트워크·과금 없이 그래프 전체를 테스트하기 위해 필요하다.

LLM 호출이 실패하면 조용히 규칙 경로로 떨어진다. 에이전트가 API 장애로
통째로 멈추지 않게 하기 위함이다.

**추출은 정규화하지 않는다.** 질의에 적힌 표면형("Gleevec", "BCR-ABL")을
그대로 돌려주고, 표준명 치환은 graph의 refine_* 노드가 1차 조회 실패 후에
수행한다. 정규화를 두 군데서 하면 어느 쪽이 실제로 동작했는지 알 수 없고,
재시도 분기가 영영 실행되지 않아 죽은 코드가 된다.
"""

from __future__ import annotations

import json
import logging
import os
import re

log = logging.getLogger("agent.llm")

MODEL = os.environ.get("AGENT_LLM_MODEL", "claude-sonnet-5")

_SYSTEM = """You extract exactly two entities from a drug-target binding question.

Return ONLY a JSON object, no prose:
{"drug": "<drug name>", "protein": "<protein or gene name>"}

Rules:
- Translate non-English drug names to English (비아그라 -> Viagra, 글리벡 -> Gleevec).
- Keep the name the user used; do NOT convert brand names to generic names and
  do NOT convert protein aliases to gene symbols. A downstream step does that
  only if the lookup fails.
- If an entity is genuinely absent, use null for that field."""

# 엔티티는 ASCII 식별자로 한정한다. 한국어 조사가 붙은 "EGFR의"를
# 엔티티로 잡지 않기 위해서다. 한국어 약물명은 아래 사전 탐지가 처리한다.
_ENTITY = r"[A-Za-z][A-Za-z0-9\-]*"

# 규칙 경로에서 버릴 단어들.
_STOPWORDS = {
    "does", "do", "is", "are", "the", "a", "an", "to", "with", "bind", "binds",
    "binding", "affinity", "between", "and", "or", "for", "of", "predict",
    "interaction", "interact", "inhibit", "inhibits", "inhibitor", "target",
    "protein", "drug", "kinase", "what", "how", "strongly", "strong", "vs",
}

# "A와 B", "A - B", "does A bind to B" 형태에서 두 엔티티를 직접 집어내는 패턴.
_PAIR_PATTERNS = [
    re.compile(rf"(?:does|will|can)\s+({_ENTITY})\s+bind(?:\s+to)?\s+({_ENTITY})", re.I),
    re.compile(rf"({_ENTITY})\s*(?:와|과|-{{1,2}}|–|—|vs\.?|and)\s*({_ENTITY})", re.I),
]


def extract_entities(query: str) -> tuple[str | None, str | None, str]:
    """질의 → (약물명, 표적명, 사용한 경로 라벨).

    Returns:
        (drug, protein, how). how는 "llm" 또는 "rule" — trace에 남겨
        어느 경로가 돌았는지 사후에 알 수 있게 한다.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        result = _extract_with_llm(query)
        if result:
            return result[0], result[1], "llm"
        log.info("LLM 추출 실패 — 규칙 경로로 폴백")

    drug, protein = _extract_with_rules(query)
    return drug, protein, "rule"


def _extract_with_llm(query: str) -> tuple[str | None, str | None] | None:
    try:
        import anthropic
    except ImportError:
        log.info("anthropic SDK 미설치 — 규칙 경로 사용")
        return None

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=MODEL,
            max_tokens=200,
            system=_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        text = "".join(block.text for block in response.content
                       if block.type == "text").strip()
        # 모델이 코드펜스를 붙이는 경우가 있어 JSON 본문만 잘라낸다.
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        data = json.loads(match.group())
        return data.get("drug"), data.get("protein")
    except Exception as exc:
        log.warning("LLM 엔티티 추출 실패: %s", exc)
        return None


def _find_alias(query: str, aliases) -> str | None:
    """사전 키가 질의에 등장하면 **질의에 적힌 표면형** 그대로 돌려준다.

    ASCII 키는 단어 경계를 요구한다. 그렇지 않으면 "abl"이 "BCRABL"에,
    "alk"가 "walk"에 걸린다. 한국어 키는 뒤에 조사가 붙어 \\b가 성립하지
    않으므로 단순 부분 문자열로 찾는다.
    """
    for alias in aliases:
        if alias.isascii():
            match = re.search(rf"\b{re.escape(alias)}\b", query, re.I)
        else:
            match = re.search(re.escape(alias), query, re.I)
        if match:
            return match.group(0)
    return None


def _extract_with_rules(query: str) -> tuple[str | None, str | None]:
    """사전·패턴 기반 추출.

    휴리스틱: 표적 유전자 심볼은 전부 대문자(EGFR, ABL1, BCR-ABL)이고,
    약물명은 첫 글자만 대문자인 긴 단어(Imatinib, Gleevec)인 경우가 압도적이다.
    """
    from services.agent.graph.nodes import ALIAS_TO_GENE, BRAND_TO_GENERIC

    # 1) 사전에 있는 이름이 질의에 그대로 박혀 있으면 그것이 가장 확실하다.
    #    긴 키부터 봐야 "abl"이 "bcr-abl"보다 먼저 걸리지 않는다.
    by_length = sorted(BRAND_TO_GENERIC, key=len, reverse=True)
    drug = _find_alias(query, by_length)
    protein = _find_alias(query, sorted(ALIAS_TO_GENE, key=len, reverse=True))

    # 2) 쌍 패턴
    if not (drug and protein):
        for pattern in _PAIR_PATTERNS:
            match = pattern.search(query)
            if match:
                drug = drug or match.group(1)
                protein = protein or match.group(2)
                break

    # 3) 토큰 분류 폴백
    if not (drug and protein):
        tokens = [t for t in re.findall(_ENTITY, query)
                  if t.lower() not in _STOPWORDS and len(t) > 1]
        gene_like = [t for t in tokens if t.upper() == t]
        drug_like = [t for t in tokens if t not in gene_like]
        protein = protein or (gene_like[0] if gene_like else None)
        drug = drug or (drug_like[0] if drug_like else None)

    return drug, protein
