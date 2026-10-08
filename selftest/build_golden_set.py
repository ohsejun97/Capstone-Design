"""
build_golden_set.py — 회귀 테스트용 골든셋 생성

두 계층으로 나눈다:

  model_level  DTI 모델에 직접 넣는 (SMILES, 아미노산 서열) 쌍.
               가중치·인코더·양자화 설정이 바뀌면 여기서 먼저 잡힌다.
               `data/davis_test.csv`에서 결정론적으로 추출한다.

  agent_level  (약물명, 유전자명) 쌍. PubChem/UniProt 해석 → AlphaFold →
               예측까지 전 경로를 태운다. 외부 API 계약이 조용히 바뀌면
               여기서 잡힌다. 손으로 고른다.

**라벨은 쓰지 않는다.** `data/davis_test.csv`의 `Label` 컬럼은 0~1로 정규화된
값이고, 현재 학습 스크립트(DeepPurpose 로더, pKd 5.00~10.54)와 스케일이 다르다.
정답을 안다고 주장하지 않고, **모델 자신의 출력을 기준선으로 스냅샷**해서
그로부터의 이탈(drift)을 감지한다. 이것이 서빙 회귀 테스트의 정석이기도 하다 —
"문헌값과 맞는가"가 아니라 "어제와 같은 답을 내는가"를 묻는 것.

사용:
    python -m selftest.build_golden_set                  # 기본 20 + 10쌍
    python -m selftest.build_golden_set --n-model 40
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DAVIS_CSV = ROOT / "data" / "davis_test.csv"
OUT_PATH = Path(__file__).resolve().parent / "golden_set.json"

# ── 에이전트 경로용 수동 큐레이션 ─────────────────────────────────────────────
# 선정 기준:
#   · 이름 해석의 **어려운 경우**를 일부러 섞는다 (브랜드명, 융합 유전자 표기)
#     → 재시도 분기가 실제로 동작하는지 매일 확인된다
#   · 학습 분포 밖(비키나아제) 표적을 하나 넣는다
#     → OOD 경고가 계속 붙는지 확인된다
#   · `expect`는 정답 pKd가 아니라 **정성적 기대**다. 기준선 비교와 별개로
#     "강한 결합으로 알려진 쌍이 약한 쪽보다 높게 나오는가"를 본다.
AGENT_PAIRS = [
    # id, 약물명, 표적명, 기대, 왜 골랐나
    ("ag-imatinib-abl1",   "Imatinib",     "ABL1",    "strong",
     "표준 기준 쌍 — CML 1차 치료제와 그 표적"),
    ("ag-gleevec-bcrabl",  "Gleevec",      "BCR-ABL", "strong",
     "브랜드명 + 융합 유전자 표기 — 재시도 분기 2개를 동시에 태운다"),
    ("ag-gefitinib-egfr",  "Gefitinib",    "EGFR",    "strong",
     "EGFR 저해제 — DAVIS 대표 키나아제"),
    ("ag-erlotinib-egfr",  "Erlotinib",    "EGFR",    "strong",
     "같은 표적 다른 약물 — 약물 인코더 민감도"),
    ("ag-dasatinib-abl1",  "Dasatinib",    "ABL1",    "strong",
     "2세대 ABL 저해제"),
    ("ag-sunitinib-kdr",   "Sunitinib",    "KDR",     "strong",
     "다중표적 저해제 — VEGFR2"),
    ("ag-sorafenib-braf",  "Sorafenib",    "BRAF",    "strong",
     "세린/트레오닌 키나아제 — 티로신 키나아제 편중 확인"),
    ("ag-aspirin-egfr",    "Aspirin",      "EGFR",    "weak",
     "음성 대조 — 결합할 이유가 없는 조합"),
    ("ag-caffeine-abl1",   "Caffeine",     "ABL1",    "weak",
     "음성 대조 — 작은 분자, 비저해제"),
    ("ag-atorva-hmgcr",    "Atorvastatin", "HMGCR",   "ood",
     "비키나아제 표적 — OOD 경고가 붙는지 확인"),
]

# ── 상대 순위 단언 ───────────────────────────────────────────────────────────
# 절대값은 주장하지 않지만 **순서**는 무너지면 안 된다.
# 기준선이 통째로 이동(예: 정규화 변경)해도 이 검사는 살아남는다.
ORDERING = [
    ("ord-imatinib-vs-aspirin", "ag-imatinib-abl1", "ag-aspirin-egfr",
     "알려진 저해제가 음성 대조보다 높아야 한다"),
    ("ord-gefitinib-vs-caffeine", "ag-gefitinib-egfr", "ag-caffeine-abl1",
     "알려진 저해제가 음성 대조보다 높아야 한다"),
    ("ord-dasatinib-vs-aspirin", "ag-dasatinib-abl1", "ag-aspirin-egfr",
     "알려진 저해제가 음성 대조보다 높아야 한다"),
]


def _aa_from_sa(sa_seq: str) -> str:
    """SA 토큰 서열에서 아미노산만 뽑는다.

    SaProt 포맷은 아미노산 1글자 + 구조 토큰 1글자가 번갈아 온다
    ("MaEdVcKp" 또는 구조 정보가 없으면 "M#E#V#K#"). 짝수 인덱스가 아미노산이다.
    """
    return sa_seq[0::2]


def sample_model_pairs(n: int, seed: int) -> list[dict]:
    """davis_test.csv에서 (SMILES, 서열) 쌍을 결정론적으로 뽑는다.

    단백질과 약물 모두 최대한 겹치지 않게 고른다 — 같은 단백질 20개를 뽑으면
    단백질 인코더 회귀를 한 종류밖에 못 잡는다.
    """
    if not DAVIS_CSV.exists():
        raise FileNotFoundError(
            f"{DAVIS_CSV} 없음. model_level 골든셋은 이 파일이 있어야 만들 수 있다."
        )

    with open(DAVIS_CSV) as f:
        rows = list(csv.DictReader(f))

    rng = random.Random(seed)
    rng.shuffle(rows)

    picked: list[dict] = []
    seen_proteins: set[str] = set()
    seen_drugs: set[str] = set()

    # 1차: 단백질·약물이 모두 새로운 것만
    for row in rows:
        if len(picked) >= n:
            break
        uniprot, smiles = row["uniprot_id"], row["SMILES"]
        if uniprot in seen_proteins or smiles in seen_drugs:
            continue
        seen_proteins.add(uniprot)
        seen_drugs.add(smiles)
        picked.append(_make_entry(len(picked), row))

    # 2차: 부족하면 단백질만 새로운 것으로 채운다 (DAVIS 약물은 68종뿐)
    for row in rows:
        if len(picked) >= n:
            break
        if row["uniprot_id"] in seen_proteins:
            continue
        seen_proteins.add(row["uniprot_id"])
        picked.append(_make_entry(len(picked), row))

    return picked


def _make_entry(index: int, row: dict) -> dict:
    aa_seq = _aa_from_sa(row["Target Sequence"])
    return {
        "id": f"dm-{index:03d}",
        "uniprot_id": row["uniprot_id"],
        "smiles": row["SMILES"],
        "aa_seq": aa_seq,
        "seq_len": len(aa_seq),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-model", type=int, default=20,
                        help="model_level 쌍 개수 (기본 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="샘플링 시드 — 같은 시드면 같은 골든셋")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    model_level = sample_model_pairs(args.n_model, args.seed)

    payload = {
        "version": 1,
        "generated": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "source": (
            "model_level: data/davis_test.csv에서 입력만 추출 (Label 컬럼은 "
            "0~1 정규화값이라 사용하지 않음). agent_level: 수동 큐레이션."
        ),
        "note": (
            "참조 pKd를 주장하지 않는다. 기준선은 모델 자신의 출력을 스냅샷한 "
            "selftest/baseline.json이며, 검사는 그로부터의 이탈을 본다."
        ),
        "model_level": model_level,
        "agent_level": [
            {"id": i, "drug_name": d, "protein_name": p, "expect": e, "why": w}
            for i, d, p, e, w in AGENT_PAIRS
        ],
        "ordering": [
            {"id": i, "stronger": s, "weaker": w, "reason": r}
            for i, s, w, r in ORDERING
        ],
    }

    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    n_prot = len({e["uniprot_id"] for e in model_level})
    n_drug = len({e["smiles"] for e in model_level})
    print(f"골든셋 생성: {out}")
    print(f"  model_level : {len(model_level)}쌍 (단백질 {n_prot}종, 약물 {n_drug}종)")
    print(f"  agent_level : {len(payload['agent_level'])}쌍")
    print(f"  ordering    : {len(payload['ordering'])}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
