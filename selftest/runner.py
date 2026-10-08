"""
runner.py — SelfTest Agent: 회귀·드리프트 자동 점검

n8n의 야간 크론이 이것을 부르고, 종료 코드로 알림 여부를 정한다.
사람이 직접 돌려도 된다.

동작:
    1. 골든셋(selftest/golden_set.json)을 예측한다
       · model_level — DTI 모델 직접 호출 (SMILES + 서열)
       · agent_level — LangGraph 전 경로 (약물명 → ... → pKd)
    2. 검사 5종을 돌린다 (checks.py)
    3. 리포트를 JSON + 사람이 읽는 텍스트로 낸다
    4. MLflow에 `FusionDTI-SelfTest` 실험으로 기록한다
    5. 종료 코드: 0=정상, 1=회귀 감지, 2=실행 실패

사용:
    python -m selftest.runner --record-baseline     # 최초 1회: 기준선 스냅샷
    python -m selftest.runner                       # 이후: 드리프트 점검
    python -m selftest.runner --skip-agent          # 모델 계층만 (외부 API 불필요)
    python -m selftest.runner --json-only           # n8n이 파싱하기 좋게
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from selftest.checks import (
    Check,
    check_contracts,
    check_drift,
    check_latency,
    check_ordering,
    check_range,
)

HERE = Path(__file__).resolve().parent
GOLDEN_PATH = HERE / "golden_set.json"
BASELINE_PATH = HERE / "baseline.json"
REPORT_DIR = ROOT / "results" / "selftest"

log = logging.getLogger("selftest")

EXIT_OK, EXIT_REGRESSION, EXIT_ERROR = 0, 1, 2


# ══════════════════════════════════════════════════════════════════════════════
# 예측 수집
# ══════════════════════════════════════════════════════════════════════════════

def _default_model_predictor() -> Callable[[str, str], dict]:
    from tools.dti_tool import predict_binding
    return predict_binding


def _default_agent_runner() -> Callable[[str], dict]:
    """LangGraph를 전 경로로 돌려 최종 상태를 돌려주는 함수."""
    from agent.graph import build_graph
    from agent.state import new_state

    app = build_graph()

    def run(query: str) -> dict:
        return app.invoke(new_state(query))

    return run


def collect_model_predictions(
    entries: list[dict],
    predict: Callable[[str, str], dict],
) -> tuple[dict[str, float], list[float], list[str]]:
    """model_level 쌍을 예측한다. → (id→pKd, 지연시간 목록, 오류 목록)"""
    values: dict[str, float] = {}
    latencies: list[float] = []
    errors: list[str] = []

    for entry in entries:
        started = time.perf_counter()
        result = predict(entry["smiles"], entry["aa_seq"])
        elapsed_ms = (time.perf_counter() - started) * 1000

        if not isinstance(result, dict) or "error" in result:
            reason = result.get("error", result) if isinstance(result, dict) else result
            errors.append(f"{entry['id']} ({entry['uniprot_id']}): {reason}")
            continue

        values[entry["id"]] = float(result["pKd"])
        latencies.append(elapsed_ms)

    return values, latencies, errors


def collect_agent_predictions(
    entries: list[dict],
    run_agent: Callable[[str], dict],
) -> tuple[dict[str, float], list[str], list[str]]:
    """agent_level 쌍을 전 경로로 돌린다. → (id→pKd, OOD가 붙은 id, 오류)"""
    values: dict[str, float] = {}
    ood_flagged: list[str] = []
    errors: list[str] = []

    for entry in entries:
        query = f"Does {entry['drug_name']} bind to {entry['protein_name']}?"
        state = run_agent(query)

        if state.get("stage") == "failed":
            errors.append(f"{entry['id']}: {state.get('error', '알 수 없는 실패')}")
            continue

        prediction = state.get("prediction") or {}
        if "pKd" not in prediction:
            errors.append(f"{entry['id']}: 예측값 없이 종료 (stage={state.get('stage')})")
            continue

        values[entry["id"]] = float(prediction["pKd"])
        if any("OOD" in warning for warning in state.get("warnings", [])):
            ood_flagged.append(entry["id"])

    return values, ood_flagged, errors


def check_ood_expectations(golden: list[dict], ood_flagged: list[str],
                           measured: set[str]) -> Check:
    """`expect: ood`로 표시한 쌍에 실제로 OOD 경고가 붙었는지 확인한다.

    OOD 경고는 사용자가 예측을 얼마나 믿을지 정하는 근거다. 조용히 사라지면
    비키나아제 표적에 대해 근거 없는 숫자를 확신에 차서 보여주게 된다.
    """
    expected = [e["id"] for e in golden
                if e.get("expect") == "ood" and e["id"] in measured]
    if not expected:
        return Check("ood_warning", True, "OOD 기대 쌍 없음", skipped=True)

    missing = [pair_id for pair_id in expected if pair_id not in ood_flagged]
    passed = not missing
    return Check(
        "ood_warning", passed,
        f"{len(expected) - len(missing)}/{len(expected)}건에 OOD 경고 부착됨",
        [f"{pair_id}: 비키나아제 표적인데 OOD 경고가 없다" for pair_id in missing],
        {"ood_expected": float(len(expected)), "ood_missing": float(len(missing))},
    )


# ══════════════════════════════════════════════════════════════════════════════
# 리포트
# ══════════════════════════════════════════════════════════════════════════════

def build_report(checks: list[Check], predictions: dict[str, float],
                 errors: list[str]) -> dict[str, Any]:
    failed = [c for c in checks if not c.passed and not c.skipped]
    metrics: dict[str, float] = {}
    for check in checks:
        metrics.update(check.metrics)
    metrics["n_predictions"] = float(len(predictions))
    metrics["n_errors"] = float(len(errors))

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL" if (failed or errors) else "PASS",
        "n_failed_checks": len(failed),
        "failed_checks": [c.name for c in failed],
        "errors": errors,
        "checks": [
            {"name": c.name, "status": c.status, "summary": c.summary,
             "details": c.details, "metrics": c.metrics}
            for c in checks
        ],
        "metrics": metrics,
        "predictions": predictions,
    }


def format_report(report: dict[str, Any]) -> str:
    icons = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭"}
    lines = [
        "═" * 68,
        f"  SelfTest — {report['status']}   ({report['timestamp']})",
        "═" * 68,
        "",
    ]
    for check in report["checks"]:
        lines.append(f"{icons[check['status']]} {check['name']:<12} {check['summary']}")
        lines.extend(f"      · {detail}" for detail in check["details"])

    if report["errors"]:
        lines += ["", f"실행 오류 {len(report['errors'])}건:"]
        lines += [f"      · {err}" for err in report["errors"][:10]]
        if len(report["errors"]) > 10:
            lines.append(f"      · ... 외 {len(report['errors']) - 10}건")

    lines += ["", "─" * 68,
              f"예측 {int(report['metrics']['n_predictions'])}건 · "
              f"실패한 검사 {report['n_failed_checks']}개", ""]
    return "\n".join(lines)


def log_to_mlflow(report: dict[str, Any]) -> str | None:
    """SelfTest 결과를 MLflow에 기록한다. 실패해도 검사 결과를 버리지 않는다."""
    try:
        from mlops.tracking import log_result
    except ImportError:
        return None

    payload = {
        "run_name": f"selftest-{report['timestamp'][:19]}",
        "timestamp": report["timestamp"],
        **{k: v for k, v in report["metrics"].items()},
    }
    return log_result(
        payload,
        experiment="FusionDTI-SelfTest",
        extra_tags={
            "status": report["status"],
            "failed_checks": ",".join(report["failed_checks"]) or "none",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# 오케스트레이션
# ══════════════════════════════════════════════════════════════════════════════

def run_selftest(
    golden: dict[str, Any],
    baseline: dict[str, float],
    *,
    predict_model: Callable[[str, str], dict] | None = None,
    run_agent: Callable[[str], dict] | None = None,
    contract_callers: dict[str, Callable[[str], dict]] | None = None,
    tol: float = 0.05,
    latency_budget_ms: float = 2000.0,
) -> dict[str, Any]:
    """검사 전체를 돌리고 리포트 dict를 돌려준다.

    모든 실행 주체를 주입받는다 — 테스트에서 GPU도 네트워크도 없이
    전체 흐름을 검증하기 위해서다.
    """
    predictions: dict[str, float] = {}
    errors: list[str] = []
    latencies: list[float] = []
    ood_flagged: list[str] = []

    if predict_model is not None:
        values, latencies, model_errors = collect_model_predictions(
            golden.get("model_level", []), predict_model)
        predictions.update(values)
        errors.extend(model_errors)

    if run_agent is not None:
        values, ood_flagged, agent_errors = collect_agent_predictions(
            golden.get("agent_level", []), run_agent)
        predictions.update(values)
        errors.extend(agent_errors)

    checks = [
        check_range(predictions),
        check_drift(predictions, baseline, tol=tol),
        check_ordering(predictions, golden.get("ordering", [])),
        check_latency(latencies, p95_budget_ms=latency_budget_ms),
    ]
    if run_agent is not None:
        checks.append(check_ood_expectations(
            golden.get("agent_level", []), ood_flagged, set(predictions)))
    if contract_callers:
        checks.append(check_contracts(contract_callers))

    return build_report(checks, predictions, errors)


def _load_contract_callers() -> dict[str, Callable[[str], dict]]:
    from tools.alphafold_tool import fetch_alphafold_structure
    from tools.pubchem_tool import resolve_drug_name
    from tools.uniprot_tool import resolve_protein_name

    return {
        "pubchem": resolve_drug_name,
        "uniprot": resolve_protein_name,
        "alphafold": fetch_alphafold_structure,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", default=str(GOLDEN_PATH))
    parser.add_argument("--baseline", default=str(BASELINE_PATH))
    parser.add_argument("--record-baseline", action="store_true",
                        help="이번 예측값을 새 기준선으로 저장한다 "
                             "(모델을 의도적으로 바꾼 뒤에만 쓸 것)")
    parser.add_argument("--tol", type=float, default=0.05,
                        help="쌍별 pKd 허용 이탈 (기본 0.05)")
    parser.add_argument("--latency-budget-ms", type=float, default=2000.0)
    parser.add_argument("--skip-model", action="store_true",
                        help="model_level 건너뜀")
    parser.add_argument("--skip-agent", action="store_true",
                        help="agent_level 건너뜀 (외부 API 불필요)")
    parser.add_argument("--skip-contracts", action="store_true",
                        help="외부 API 계약 검사 건너뜀")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--json-only", action="store_true",
                        help="리포트 JSON만 stdout으로 (n8n용)")
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING if args.json_only else logging.INFO,
                        format="%(message)s")

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"골든셋이 없습니다: {golden_path}\n"
              f"  python -m selftest.build_golden_set 으로 먼저 생성하세요.",
              file=sys.stderr)
        return EXIT_ERROR

    golden = json.loads(golden_path.read_text())
    baseline_path = Path(args.baseline)
    baseline = (json.loads(baseline_path.read_text()).get("predictions", {})
                if baseline_path.exists() else {})

    try:
        report = run_selftest(
            golden,
            baseline,
            predict_model=None if args.skip_model else _default_model_predictor(),
            run_agent=None if args.skip_agent else _default_agent_runner(),
            contract_callers=None if args.skip_contracts else _load_contract_callers(),
            tol=args.tol,
            latency_budget_ms=args.latency_budget_ms,
        )
    except Exception as exc:
        print(f"SelfTest 실행 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    # ── 리포트 저장 ───────────────────────────────────────────────────────────
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = report["timestamp"].replace(":", "").replace("-", "")[:15]
    (report_dir / f"selftest_{stamp}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    (report_dir / "latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    if args.json_only:
        print(json.dumps(report, ensure_ascii=False))
    else:
        print(format_report(report))
        print(f"리포트: {report_dir}/latest.json")

    if not args.no_mlflow:
        log_to_mlflow(report)

    # ── 기준선 갱신 ───────────────────────────────────────────────────────────
    if args.record_baseline:
        baseline_path.write_text(json.dumps({
            "recorded": report["timestamp"],
            "note": ("모델 자신의 출력 스냅샷. 문헌값이 아니다. "
                     "모델을 의도적으로 바꿨을 때만 갱신할 것."),
            "n": len(report["predictions"]),
            "predictions": report["predictions"],
        }, indent=2, ensure_ascii=False))
        if not args.json_only:
            print(f"기준선 기록: {baseline_path} ({len(report['predictions'])}쌍)")
        return EXIT_OK

    return EXIT_REGRESSION if report["status"] == "FAIL" else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
