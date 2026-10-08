"""
state.py — LangGraph 에이전트의 공유 상태 정의

ReAct 루프(smolagents)와 달리 LangGraph는 노드 간에 **명시적 상태**를 넘긴다.
어떤 정보가 어디서 채워지는지가 타입으로 드러나고, 체크포인터가 이 dict를
그대로 직렬화하므로 중단된 질의를 이어서 재개할 수 있다.

병렬 분기(fetch_structure / generate_ligand)가 동시에 쓰는 필드는
reducer(operator.add)를 붙여야 한다. 그렇지 않으면 마지막 분기의 값이
앞 분기의 값을 덮어쓴다.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

# 파이프라인이 도달한 단계 — UI 진행률 표시와 디버깅에 쓴다.
Stage = Literal[
    "parse", "resolve_drug", "resolve_protein", "scope_check",
    "enrich", "predict", "compose", "done", "failed",
]


class DTIState(TypedDict, total=False):
    """약물-표적 질의 하나가 그래프를 통과하는 동안의 전체 상태."""

    # ── 입력 ──────────────────────────────────────────────────────────────
    query: str                      # "Does Imatinib bind to BCR-ABL?"

    # ── parse_query 결과 ──────────────────────────────────────────────────
    drug_name: str                  # "Imatinib"
    protein_name: str               # "ABL1"

    # ── 이름 해석 결과 (tools/pubchem_tool.py, tools/uniprot_tool.py) ─────
    drug: dict[str, Any]            # query_name, smiles, cid, formula, ...
    protein: dict[str, Any]         # uniprot_id, gene, sequence, organism, ...

    # ── 재시도 카운터 ─────────────────────────────────────────────────────
    # 무한 루프를 막는 유일한 장치. 조건부 엣지가 이 값을 보고 포기 시점을 정한다.
    drug_attempts: int
    protein_attempts: int

    # ── 보강 단계 (병렬) ──────────────────────────────────────────────────
    structure: dict[str, Any]       # AlphaFold PDB + pLDDT
    ligand: dict[str, Any]          # RDKit 3D SDF

    # ── 예측 ──────────────────────────────────────────────────────────────
    prediction: dict[str, Any]      # pKd, interpretation, used_3di

    # ── 누적 필드 (병렬 분기가 동시에 append) ─────────────────────────────
    warnings: Annotated[list[str], operator.add]
    trace: Annotated[list[str], operator.add]   # 어떤 노드를 어떤 순서로 지났나

    # ── 종료 ──────────────────────────────────────────────────────────────
    stage: Stage
    error: str                      # 치명적 실패 사유 (있으면 answer가 에러 설명)
    answer: str                     # 사용자에게 보여줄 최종 텍스트


def new_state(query: str) -> DTIState:
    """초기 상태. reducer가 붙은 필드는 반드시 빈 리스트로 시작해야 한다."""
    return DTIState(
        query=query,
        drug_attempts=0,
        protein_attempts=0,
        warnings=[],
        trace=[],
        stage="parse",
    )
