"""
checks.py — SelfTest 검사 항목

각 검사는 `Check` 하나를 돌려준다. 검사끼리 독립이며, 하나가 실패해도 나머지는
계속 돈다 — 무엇이 깨졌는지 한 번에 다 보기 위해서다.

검사가 잡으려는 것:

  drift      모델 출력이 기준선에서 벗어남. 가중치 교체, 인코더 설정 변경,
             양자화 변경, 3Di 캐시 유실 등이 여기서 잡힌다.
  ordering   알려진 저해제가 음성 대조보다 낮게 나옴. 기준선이 통째로
             이동해도 살아남는 검사 — 절대값이 아니라 순서를 본다.
  range      NaN / inf / 물리적으로 불가능한 pKd. 조용한 수치 붕괴를 잡는다.
  latency    추론 지연시간 악화.
  contract   외부 API(PubChem/UniProt/AlphaFold) 응답 스키마 변경.
             이게 조용히 깨지면 예측은 계속 나오는데 값이 틀린다.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# pKd가 물리적으로 취할 수 있는 범위. BindingDB 학습셋이 2.00~13.33이었고
# (results/SaProt-650M-bindingdb-3di-chemberta), 여유를 둬서 잡는다.
PKD_MIN, PKD_MAX = 0.0, 16.0


@dataclass
class Check:
    """검사 하나의 결과."""

    name: str
    passed: bool
    summary: str
    details: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    skipped: bool = False

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        return "PASS" if self.passed else "FAIL"


# ══════════════════════════════════════════════════════════════════════════════
# 1. 드리프트 — 기준선 대비 이탈
# ══════════════════════════════════════════════════════════════════════════════

def check_drift(
    current: dict[str, float],
    baseline: dict[str, float],
    *,
    tol: float = 0.05,
    max_offenders: int = 10,
) -> Check:
    """쌍별 pKd가 기준선에서 `tol` 이상 벗어났는지 본다.

    tol 기본 0.05는 부동소수점 잡음(FP16 추론)은 흡수하되 실질적 변화는
    잡는 값이다. 모델을 의도적으로 바꿨다면 기준선을 다시 기록해야 한다
    (`--record-baseline`).
    """
    if not baseline:
        return Check("drift", True, "기준선 없음 — 첫 실행으로 간주하고 건너뜀",
                     skipped=True)

    shared = sorted(set(current) & set(baseline))
    missing = sorted(set(baseline) - set(current))
    added = sorted(set(current) - set(baseline))

    if not shared:
        return Check("drift", False,
                     "기준선과 겹치는 쌍이 하나도 없음 — 골든셋이 교체됐는가?",
                     details=[f"기준선에만 있음: {len(missing)}건",
                              f"이번에만 있음: {len(added)}건"])

    deltas = {key: current[key] - baseline[key] for key in shared}
    offenders = sorted(
        ((key, delta) for key, delta in deltas.items() if abs(delta) > tol),
        key=lambda item: -abs(item[1]),
    )

    abs_deltas = [abs(d) for d in deltas.values()]
    metrics = {
        "drift_mean_abs": sum(abs_deltas) / len(abs_deltas),
        "drift_max_abs": max(abs_deltas),
        "drift_n_compared": float(len(shared)),
        "drift_n_offenders": float(len(offenders)),
    }

    details = [
        f"{key}: {baseline[key]:.4f} → {current[key]:.4f} ({delta:+.4f})"
        for key, delta in offenders[:max_offenders]
    ]
    if len(offenders) > max_offenders:
        details.append(f"... 외 {len(offenders) - max_offenders}건")
    if missing:
        details.append(f"기준선에 있으나 이번에 측정 못 한 쌍: {len(missing)}건 "
                       f"(예: {', '.join(missing[:3])})")

    passed = not offenders
    summary = (
        f"{len(shared)}쌍 비교, 평균 이탈 {metrics['drift_mean_abs']:.4f}, "
        f"최대 {metrics['drift_max_abs']:.4f} (허용 {tol})"
    )
    if not passed:
        summary = f"{len(offenders)}쌍이 허용치 초과 — " + summary

    return Check("drift", passed, summary, details, metrics)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 상대 순위
# ══════════════════════════════════════════════════════════════════════════════

def check_ordering(
    current: dict[str, float],
    rules: list[dict[str, str]],
    *,
    margin: float = 0.0,
) -> Check:
    """"A가 B보다 높아야 한다"는 단언을 검사한다.

    기준선과 독립이다. 모델을 새로 학습해 값이 통째로 바뀌어도, 알려진
    저해제가 음성 대조보다 낮게 나오면 그건 학습이 잘못된 것이다.
    """
    if not rules:
        return Check("ordering", True, "순위 규칙 없음", skipped=True)

    violations: list[str] = []
    unavailable: list[str] = []

    for rule in rules:
        strong_key, weak_key = rule["stronger"], rule["weaker"]
        if strong_key not in current or weak_key not in current:
            unavailable.append(rule["id"])
            continue
        strong, weak = current[strong_key], current[weak_key]
        if strong <= weak + margin:
            violations.append(
                f"{rule['id']}: {strong_key}={strong:.3f} ≤ "
                f"{weak_key}={weak:.3f} — {rule['reason']}"
            )

    checked = len(rules) - len(unavailable)
    details = list(violations)
    if unavailable:
        details.append(f"예측값이 없어 검사 못 한 규칙: {', '.join(unavailable)}")

    passed = not violations
    summary = f"{checked}/{len(rules)}건 검사, 위반 {len(violations)}건"
    return Check("ordering", passed, summary, details,
                 {"ordering_violations": float(len(violations))})


# ══════════════════════════════════════════════════════════════════════════════
# 3. 값 범위 — 조용한 수치 붕괴
# ══════════════════════════════════════════════════════════════════════════════

def check_range(current: dict[str, float]) -> Check:
    """NaN / inf / 범위 밖 값을 잡는다.

    FP16 오버플로나 손상된 가중치는 예외를 던지지 않고 NaN을 흘린다.
    그러면 API는 200을 돌려주고 사용자는 이상한 숫자를 본다.
    """
    if not current:
        return Check("range", False, "예측 결과가 비어 있음")

    bad: list[str] = []
    for key, value in sorted(current.items()):
        if value is None or math.isnan(value) or math.isinf(value):
            bad.append(f"{key}: {value} (NaN/inf)")
        elif not (PKD_MIN <= value <= PKD_MAX):
            bad.append(f"{key}: {value:.3f} (허용 {PKD_MIN}~{PKD_MAX} 밖)")

    values = [v for v in current.values()
              if v is not None and not math.isnan(v) and not math.isinf(v)]
    metrics: dict[str, float] = {}
    if values:
        metrics = {
            "pred_min": min(values),
            "pred_max": max(values),
            "pred_mean": sum(values) / len(values),
        }

    passed = not bad
    summary = (f"{len(current)}개 값 검사, 이상치 {len(bad)}개"
               + (f" (범위 {metrics['pred_min']:.2f}~{metrics['pred_max']:.2f})"
                  if metrics else ""))
    return Check("range", passed, summary, bad, metrics)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 지연시간
# ══════════════════════════════════════════════════════════════════════════════

def check_latency(latencies_ms: list[float], *, p95_budget_ms: float) -> Check:
    """추론 지연시간 p95가 예산 안에 있는지 본다."""
    if not latencies_ms:
        return Check("latency", True, "측정값 없음", skipped=True)

    ordered = sorted(latencies_ms)
    # 최근접 순위법 — 표본이 적을 때 보간법보다 해석이 명확하다.
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    p95 = ordered[index]
    mean = sum(ordered) / len(ordered)

    passed = p95 <= p95_budget_ms
    return Check(
        "latency", passed,
        f"p95 {p95:.1f}ms / 예산 {p95_budget_ms:.0f}ms (평균 {mean:.1f}ms, n={len(ordered)})",
        [] if passed else [f"p95가 예산을 {p95 - p95_budget_ms:.1f}ms 초과"],
        {"latency_p95_ms": p95, "latency_mean_ms": mean},
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. 외부 API 계약
# ══════════════════════════════════════════════════════════════════════════════

# 각 도구가 반드시 채워야 하는 키. 여기 없으면 파이프라인 뒤쪽에서
# KeyError가 아니라 **조용한 오답**으로 번진다.
CONTRACTS: dict[str, tuple[str, ...]] = {
    "pubchem": ("smiles", "cid"),
    "uniprot": ("uniprot_id", "sequence", "organism"),
    "alphafold": ("pdb_path", "plddt_global"),
}

# 계약 검증용 질의 — 사라질 가능성이 없는 대표 항목으로 고른다.
CONTRACT_PROBES = {
    "pubchem": "Imatinib",
    "uniprot": "EGFR",
    "alphafold": "P00533",      # EGFR
}


def check_contracts(
    callers: dict[str, Callable[[str], dict[str, Any]]],
    *,
    probes: dict[str, str] | None = None,
) -> Check:
    """외부 API 응답이 우리가 기대하는 키를 계속 담고 있는지 확인한다.

    Args:
        callers: {"pubchem": fn, ...} — 각 fn(probe) -> dict
        probes:  질의 문자열 재정의 (기본 CONTRACT_PROBES)
    """
    if not callers:
        return Check("contract", True, "검사 대상 없음", skipped=True)

    probes = {**CONTRACT_PROBES, **(probes or {})}
    problems: list[str] = []
    ok_count = 0

    for name, call in callers.items():
        expected = CONTRACTS.get(name, ())
        try:
            result = call(probes.get(name, ""))
        except Exception as exc:
            problems.append(f"{name}: 호출 자체가 실패 — {type(exc).__name__}: {exc}")
            continue

        if not isinstance(result, dict):
            problems.append(f"{name}: dict가 아닌 {type(result).__name__} 반환")
            continue
        if "error" in result:
            problems.append(f"{name}: {result['error']}")
            continue

        missing = [key for key in expected if not result.get(key)]
        if missing:
            problems.append(
                f"{name}: 응답에 {missing} 없음 "
                f"(받은 키: {sorted(result)[:8]}) — API 스키마가 바뀌었을 수 있다"
            )
            continue

        ok_count += 1

    passed = not problems
    return Check("contract", passed,
                 f"{ok_count}/{len(callers)}개 API 계약 정상",
                 problems, {"contract_ok": float(ok_count)})
