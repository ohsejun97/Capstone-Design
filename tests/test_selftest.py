"""
test_selftest.py — SelfTest Agent 검증

SelfTest는 회귀를 잡는 장치다. 그러니 **SelfTest 자신이 회귀를 못 잡는 경우**가
가장 위험하다. 여기서는 고장을 일부러 주입하고 각 검사가 실제로 빨간불을
켜는지 확인한다:

  · 가중치가 바뀐 척 → drift가 잡는가
  · NaN이 흘러나오는 척 → range가 잡는가
  · 저해제와 음성 대조 순서가 뒤집힌 척 → ordering이 잡는가
  · OOD 경고가 사라진 척 → ood_warning이 잡는가
  · PubChem 응답 스키마가 바뀐 척 → contract가 잡는가

전부 주입된 가짜 실행 주체로 돈다 — GPU도 네트워크도 필요 없다.
"""

from __future__ import annotations

import math

import pytest

from selftest.checks import check_contracts, check_drift, check_ordering, check_range
from selftest.runner import run_selftest

# ── 최소 골든셋 ───────────────────────────────────────────────────────────────

GOLDEN = {
    "model_level": [
        {"id": "dm-000", "uniprot_id": "P00519", "smiles": "CC1", "aa_seq": "MAAAA", "seq_len": 5},
        {"id": "dm-001", "uniprot_id": "P00533", "smiles": "CC2", "aa_seq": "MBBBB", "seq_len": 5},
    ],
    "agent_level": [
        {"id": "ag-strong", "drug_name": "Imatinib", "protein_name": "ABL1", "expect": "strong"},
        {"id": "ag-weak", "drug_name": "Aspirin", "protein_name": "EGFR", "expect": "weak"},
        {"id": "ag-ood", "drug_name": "Atorvastatin", "protein_name": "HMGCR", "expect": "ood"},
    ],
    "ordering": [
        {"id": "ord-1", "stronger": "ag-strong", "weaker": "ag-weak",
         "reason": "알려진 저해제가 음성 대조보다 높아야 한다"},
    ],
}

BASELINE = {"dm-000": 8.0, "dm-001": 7.0, "ag-strong": 8.7, "ag-weak": 4.2, "ag-ood": 5.5}

MODEL_VALUES = {"CC1": 8.0, "CC2": 7.0}
AGENT_VALUES = {"Imatinib": 8.7, "Aspirin": 4.2, "Atorvastatin": 5.5}


def make_model_predictor(values=None, latency_ms=0.0):
    values = values or MODEL_VALUES

    def predict(smiles: str, aa_seq: str) -> dict:
        if smiles not in values:
            return {"error": f"unknown smiles {smiles}"}
        return {"pKd": values[smiles], "used_3di": True,
                "interpretation": "x", "smiles": smiles, "seq_length": len(aa_seq)}

    return predict


def make_agent_runner(values=None, ood_for=("Atorvastatin",)):
    values = values or AGENT_VALUES

    def run(query: str) -> dict:
        drug = next((d for d in values if d in query), None)
        if drug is None:
            return {"stage": "failed", "error": f"파싱 실패: {query}"}
        warnings = []
        if drug in ood_for:
            warnings.append("키나아제로 보이지 않습니다 ... OOD 예측입니다.")
        return {"stage": "done", "prediction": {"pKd": values[drug]},
                "warnings": warnings}

    return run


def run(**kwargs):
    params = dict(
        golden=GOLDEN,
        baseline=BASELINE,
        predict_model=make_model_predictor(),
        run_agent=make_agent_runner(),
    )
    params.update(kwargs)
    return run_selftest(**params)


def status_of(report, name):
    return next(c["status"] for c in report["checks"] if c["name"] == name)


# ══════════════════════════════════════════════════════════════════════════════
# 정상 경로
# ══════════════════════════════════════════════════════════════════════════════

def test_clean_run_passes():
    report = run()
    assert report["status"] == "PASS", report["failed_checks"]
    assert report["metrics"]["n_predictions"] == 5
    assert not report["errors"]


def test_every_check_runs():
    report = run()
    names = {c["name"] for c in report["checks"]}
    assert names == {"range", "drift", "ordering", "latency", "ood_warning"}


# ══════════════════════════════════════════════════════════════════════════════
# 고장 주입 — 각 검사가 실제로 잡는가
# ══════════════════════════════════════════════════════════════════════════════

def test_drift_is_detected():
    """가중치가 바뀐 척 — 한 쌍만 0.3 이동."""
    report = run(predict_model=make_model_predictor({"CC1": 8.3, "CC2": 7.0}))

    assert report["status"] == "FAIL"
    assert "drift" in report["failed_checks"]
    drift = next(c for c in report["checks"] if c["name"] == "drift")
    assert "dm-000" in " ".join(drift["details"])
    assert drift["metrics"]["drift_max_abs"] == pytest.approx(0.3)


def test_tiny_numeric_noise_does_not_trip_drift():
    """FP16 잡음 수준(0.01)은 통과해야 한다 — 매일 거짓 경보가 오면 아무도 안 본다."""
    report = run(predict_model=make_model_predictor({"CC1": 8.01, "CC2": 6.99}))
    assert status_of(report, "drift") == "PASS"


def test_nan_is_detected():
    report = run(predict_model=make_model_predictor({"CC1": math.nan, "CC2": 7.0}))

    assert report["status"] == "FAIL"
    assert "range" in report["failed_checks"]


def test_ordering_violation_is_detected():
    """저해제가 음성 대조보다 낮게 나오는 경우."""
    broken = {"Imatinib": 3.0, "Aspirin": 6.0, "Atorvastatin": 5.5}
    report = run(run_agent=make_agent_runner(broken))

    assert "ordering" in report["failed_checks"]
    ordering = next(c for c in report["checks"] if c["name"] == "ordering")
    assert "ord-1" in " ".join(ordering["details"])


def test_missing_ood_warning_is_detected():
    """OOD 경고가 조용히 사라지면 근거 없는 숫자를 확신에 차서 보여주게 된다."""
    report = run(run_agent=make_agent_runner(ood_for=()))

    assert "ood_warning" in report["failed_checks"]


def test_agent_failure_is_recorded_not_swallowed():
    def always_fail(query: str) -> dict:
        return {"stage": "failed", "error": "PubChem 503"}

    report = run(run_agent=always_fail)

    assert report["status"] == "FAIL"
    assert len(report["errors"]) == 3
    assert "PubChem 503" in report["errors"][0]


def test_latency_budget_breach_is_detected():
    import time

    def slow_predict(smiles: str, aa_seq: str) -> dict:
        time.sleep(0.02)
        return {"pKd": MODEL_VALUES.get(smiles, 8.0), "used_3di": True}

    report = run(predict_model=slow_predict, latency_budget_ms=5.0)
    assert "latency" in report["failed_checks"]


# ══════════════════════════════════════════════════════════════════════════════
# 외부 API 계약
# ══════════════════════════════════════════════════════════════════════════════

def test_contract_passes_on_expected_shape():
    check = check_contracts({
        "pubchem": lambda q: {"smiles": "CC1", "cid": "5291"},
        "uniprot": lambda q: {"uniprot_id": "P00533", "sequence": "MA",
                              "organism": "Homo sapiens"},
    })
    assert check.passed


def test_contract_catches_renamed_field():
    """PubChem이 'smiles'를 'canonical_smiles'로 바꾼 상황."""
    check = check_contracts({
        "pubchem": lambda q: {"canonical_smiles": "CC1", "cid": "5291"},
    })
    assert not check.passed
    assert "smiles" in check.details[0]


def test_contract_catches_exception():
    def boom(q):
        raise ConnectionError("DNS 실패")

    check = check_contracts({"uniprot": boom})
    assert not check.passed
    assert "ConnectionError" in check.details[0]


# ══════════════════════════════════════════════════════════════════════════════
# 기준선 취급
# ══════════════════════════════════════════════════════════════════════════════

def test_first_run_without_baseline_skips_drift():
    report = run(baseline={})
    assert status_of(report, "drift") == "SKIP"
    assert report["status"] == "PASS"


def test_baseline_mismatch_is_reported_not_ignored():
    """골든셋이 통째로 바뀌어 겹치는 쌍이 없으면 조용히 통과하면 안 된다."""
    check = check_drift({"new-1": 8.0}, {"old-1": 8.0})
    assert not check.passed
    assert "겹치는 쌍이 하나도 없음" in check.summary


def test_pair_missing_from_this_run_is_flagged():
    check = check_drift({"dm-000": 8.0}, {"dm-000": 8.0, "dm-001": 7.0})
    assert "dm-001" in " ".join(check.details)


# ══════════════════════════════════════════════════════════════════════════════
# 단위 검사
# ══════════════════════════════════════════════════════════════════════════════

def test_range_rejects_impossible_pkd():
    assert not check_range({"a": 99.0}).passed
    assert check_range({"a": 8.0}).passed


def test_ordering_skips_when_values_absent():
    check = check_ordering({"ag-strong": 8.0},
                           [{"id": "ord-1", "stronger": "ag-strong",
                             "weaker": "ag-missing", "reason": "x"}])
    assert check.passed          # 값이 없으면 위반으로 치지 않는다
    assert "ord-1" in " ".join(check.details)


def test_real_golden_set_is_wellformed():
    """저장소에 커밋된 골든셋이 러너가 기대하는 모양인지 확인한다."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "selftest" / "golden_set.json"
    golden = json.loads(path.read_text())

    assert golden["model_level"], "model_level이 비어 있다"
    ids = [e["id"] for e in golden["model_level"]] + \
          [e["id"] for e in golden["agent_level"]]
    assert len(ids) == len(set(ids)), "골든셋 id가 중복된다"

    valid_aa = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
    for entry in golden["model_level"]:
        assert set(entry["aa_seq"]) <= valid_aa, f"{entry['id']} 서열에 비아미노산 문자"
        assert entry["seq_len"] == len(entry["aa_seq"])

    known = set(ids)
    for rule in golden["ordering"]:
        assert rule["stronger"] in known, f"{rule['id']}: 없는 id 참조"
        assert rule["weaker"] in known, f"{rule['id']}: 없는 id 참조"
