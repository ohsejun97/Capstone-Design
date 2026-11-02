"""
loop.py — Ralph Loop: 예산 제한 자율 반복 드라이버

"Ralph"는 같은 지시를 완료될 때까지 반복시키는 방식이다. 그 자체로는 무한
루프와 구별되지 않으므로, 실제로 쓸 수 있게 만들려면 **네 가지 정지 조건**이
반드시 있어야 한다:

    1. 목표 달성   target_metric이 목표치에 도달
    2. 예산 소진   최대 반복 횟수 또는 최대 실행 시간
    3. 개선 정체   patience회 연속으로 최고 기록을 못 넘김
    4. 제안 고갈   Proposer가 더 낼 것이 없다고 선언

이 프로젝트에서의 쓰임: **하이퍼파라미터 탐색**. Phase 1h의 ChemBERTa unfreeze
층수·학습률·배치 크기를 사람이 손으로 바꿔 가며 21번 돌렸다. 그 반복을
자동화하되, MLflow에 이미 쌓인 이력을 읽어 다음 시도를 정하게 한다.

구조 — 세 부분을 갈아 끼울 수 있다:

    Proposer   다음에 무엇을 시도할지 정한다 (격자 / LLM)
    Executor   실제로 돌린다 (subprocess / 테스트용 가짜)
    Loop       정지 조건을 관리하고 상태를 디스크에 남긴다 (중단 후 재개)

상태를 파일에 남기는 이유: 학습 한 번이 수십 분이라 루프가 몇 시간 돈다.
중간에 끊기면 처음부터 다시 하는 건 말이 안 된다.

사용:
    python -m ralph.loop --dry-run                       # 무엇을 돌릴지만 출력
    python -m ralph.loop --max-iters 8 --target 0.90
    python -m ralph.loop --resume                        # 중단된 루프 이어서
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("ralph")

DEFAULT_STATE_PATH = ROOT / "results" / "ralph" / "state.json"


# ══════════════════════════════════════════════════════════════════════════════
# 자료형
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Proposal:
    """다음에 시도할 설정 하나."""

    params: dict[str, Any]
    rationale: str = ""

    def key(self) -> str:
        """중복 제안을 걸러내기 위한 정규화 키."""
        return json.dumps(self.params, sort_keys=True)

    def label(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))


@dataclass
class Outcome:
    """시도 하나의 결과."""

    ok: bool
    metrics: dict[str, float] = field(default_factory=dict)
    run_name: str = ""
    message: str = ""
    duration_sec: float = 0.0


@dataclass
class Attempt:
    """이력 한 줄 — Proposer가 이걸 읽고 다음을 정한다."""

    iteration: int
    params: dict[str, Any]
    rationale: str
    ok: bool
    score: float | None
    metrics: dict[str, float]
    message: str
    duration_sec: float
    timestamp: str


class Proposer(Protocol):
    def propose(self, history: list[Attempt]) -> Proposal | None:
        """다음 시도를 낸다. 더 낼 것이 없으면 None."""
        ...


class Executor(Protocol):
    def run(self, proposal: Proposal) -> Outcome:
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Proposer — 격자 탐색
# ══════════════════════════════════════════════════════════════════════════════

class GridProposer:
    """미리 정의한 조합을 순서대로 낸다.

    LLM 없이도 루프를 검증할 수 있어야 하므로 기본 제안자로 둔다.
    이미 시도한 조합은 건너뛴다 — 중단 후 재개 시 중복 학습을 막는다.
    """

    def __init__(self, grid: list[dict[str, Any]]):
        self.grid = grid

    def propose(self, history: list[Attempt]) -> Proposal | None:
        tried = {json.dumps(a.params, sort_keys=True) for a in history}
        for params in self.grid:
            if json.dumps(params, sort_keys=True) not in tried:
                return Proposal(params, rationale="격자 순회")
        return None


# Phase 1h에서 사람이 손으로 돌려 본 축을 그대로 격자로 옮긴 것.
# unfreeze 0은 frozen 기준선(r=0.8737), 2는 현재 최고(r=0.8923).
DEFAULT_GRID = [
    {"unfreeze": 2, "lr_cb": 1e-5, "batch_size": 32},
    {"unfreeze": 3, "lr_cb": 1e-5, "batch_size": 32},
    {"unfreeze": 2, "lr_cb": 3e-5, "batch_size": 32},
    {"unfreeze": 4, "lr_cb": 1e-5, "batch_size": 32},
    {"unfreeze": 2, "lr_cb": 1e-5, "batch_size": 64},
    {"unfreeze": 3, "lr_cb": 3e-5, "batch_size": 32},
]


# ══════════════════════════════════════════════════════════════════════════════
# Proposer — LLM (진짜 "Ralph")
# ══════════════════════════════════════════════════════════════════════════════

_LLM_SYSTEM = """You are tuning a drug-target binding affinity model.

Architecture: SaProt-650M protein encoder (frozen, FP16) + ChemBERTa drug encoder
(6 layers; the top N are unfrozen and fine-tuned) + MLP head. Trained on BindingDB
(~80K pairs). Metric: test Pearson r, higher is better. Hardware: 4GB VRAM — large
batch sizes with many unfrozen layers will OOM.

You will receive the history of attempts. Propose the NEXT configuration to try.

Return ONLY a JSON object:
{"params": {"unfreeze": <0-6>, "lr_cb": <float>, "batch_size": <int>},
 "rationale": "<one sentence>"}

Rules:
- Do not repeat a configuration already in the history.
- If the history shows the search has converged or is degrading, return
  {"params": null, "rationale": "<why you are stopping>"}.
- Prefer small, interpretable steps over large jumps."""


class LLMProposer:
    """이력을 읽고 다음 설정을 제안한다 — 매 반복 같은 프롬프트를 던지는 Ralph 방식.

    API 키가 없으면 생성 시점에 실패하지 않고, `propose`에서 fallback으로
    넘긴 Proposer에게 위임한다. 루프가 키 유무로 죽지 않게 하기 위함이다.
    """

    def __init__(self, fallback: Proposer | None = None, model: str | None = None):
        import os

        self.fallback = fallback or GridProposer(DEFAULT_GRID)
        self.model = model or os.environ.get("AGENT_LLM_MODEL", "claude-sonnet-5")
        self.enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if not self.enabled:
            log.info("ANTHROPIC_API_KEY 없음 — 격자 제안자로 폴백")

    def propose(self, history: list[Attempt]) -> Proposal | None:
        if not self.enabled:
            return self.fallback.propose(history)

        try:
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=400,
                system=_LLM_SYSTEM,
                messages=[{"role": "user", "content": self._render(history)}],
            )
            text = "".join(b.text for b in response.content if b.type == "text")
            data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except Exception as exc:
            log.warning("LLM 제안 실패 (%s) — 격자로 폴백", exc)
            return self.fallback.propose(history)

        params = data.get("params")
        if not params:
            log.info("LLM이 탐색 종료를 제안: %s", data.get("rationale", ""))
            return None

        proposal = Proposal(params, rationale=data.get("rationale", "LLM 제안"))

        # LLM은 지시해도 같은 설정을 다시 낸다. 루프가 제자리를 돌지 않도록 막는다.
        if proposal.key() in {json.dumps(a.params, sort_keys=True) for a in history}:
            log.warning("LLM이 중복 설정을 제안 — 격자로 폴백")
            return self.fallback.propose(history)

        return proposal

    @staticmethod
    def _render(history: list[Attempt]) -> str:
        if not history:
            return "No attempts yet. Propose a sensible starting configuration."
        lines = ["Attempt history (oldest first):"]
        for attempt in history:
            score = f"{attempt.score:.4f}" if attempt.score is not None else "FAILED"
            lines.append(
                f"  #{attempt.iteration} {json.dumps(attempt.params, sort_keys=True)}"
                f" → test_pearson_r={score}"
                + (f"  ({attempt.message})" if not attempt.ok else "")
            )
        best = max((a for a in history if a.score is not None),
                   key=lambda a: a.score, default=None)
        if best:
            lines.append(f"\nBest so far: {json.dumps(best.params, sort_keys=True)}"
                         f" → {best.score:.4f}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Executor — 실제 학습 실행
# ══════════════════════════════════════════════════════════════════════════════

class TrainingExecutor:
    """`scripts/train_chemberta_unfreeze.py`를 서브프로세스로 돌리고 결과를 읽는다.

    학습 스크립트를 import하지 않고 서브프로세스로 띄우는 이유: 학습이 OOM이나
    CUDA 오류로 죽어도 루프 자신은 살아남아야 하고, 반복 사이에 GPU 메모리가
    확실히 반환되어야 한다.
    """

    def __init__(self, *, python: str = sys.executable, timeout_sec: float = 7200,
                 extra_args: list[str] | None = None):
        self.python = python
        self.timeout_sec = timeout_sec
        self.extra_args = extra_args or []

    def build_command(self, proposal: Proposal) -> list[str]:
        params = proposal.params
        cmd = [self.python, "-m", "scripts.train_chemberta_unfreeze"]
        for flag in ("unfreeze", "lr_cb", "batch_size", "split", "epochs", "max_length"):
            if flag in params:
                cmd += [f"--{flag}", str(params[flag])]
        return cmd + self.extra_args

    def run(self, proposal: Proposal) -> Outcome:
        cmd = self.build_command(proposal)
        started = time.perf_counter()
        log.info("  실행: %s", " ".join(cmd))

        try:
            completed = subprocess.run(
                cmd, cwd=ROOT, capture_output=True, text=True,
                timeout=self.timeout_sec, check=False,
            )
        except subprocess.TimeoutExpired:
            return Outcome(False, message=f"타임아웃 ({self.timeout_sec}s)",
                           duration_sec=time.perf_counter() - started)

        duration = time.perf_counter() - started

        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip().splitlines()
            return Outcome(False, message=f"종료코드 {completed.returncode}: "
                                          f"{tail[-1] if tail else '출력 없음'}",
                           duration_sec=duration)

        result = self._read_result(proposal)
        if result is None:
            return Outcome(False, message="result.json을 찾지 못함", duration_sec=duration)

        return Outcome(
            True,
            metrics={k: float(v) for k, v in result.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)},
            run_name=result.get("run_name", ""),
            duration_sec=duration,
        )

    @staticmethod
    def _read_result(proposal: Proposal) -> dict | None:
        """학습 스크립트가 만든 result.json을 찾는다.

        run_name 규칙은 학습 스크립트가 정한다:
            SaProt-650M-bindingdb-3di-chemberta-unfreeze{N}-{split}
        """
        unfreeze = proposal.params.get("unfreeze")
        split = proposal.params.get("split", "cold_drug")
        path = (ROOT / "results"
                / f"SaProt-650M-bindingdb-3di-chemberta-unfreeze{unfreeze}-{split}"
                / "result.json")
        return json.loads(path.read_text()) if path.exists() else None


# ══════════════════════════════════════════════════════════════════════════════
# 루프
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LoopConfig:
    target_metric: str = "test_pearson_r"
    target_value: float | None = None       # 도달하면 즉시 종료
    maximize: bool = True
    max_iters: int = 6
    max_seconds: float | None = None
    patience: int = 3                       # 연속 미개선 허용 횟수
    min_delta: float = 0.001                # 이보다 작으면 개선으로 치지 않는다


@dataclass
class LoopResult:
    stopped_because: str
    iterations: int
    best_score: float | None
    best_params: dict[str, Any] | None
    history: list[Attempt]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stopped_because": self.stopped_because,
            "iterations": self.iterations,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "history": [asdict(a) for a in self.history],
        }


class RalphLoop:
    """예산 안에서 propose → execute → evaluate 를 반복한다."""

    def __init__(self, proposer: Proposer, executor: Executor,
                 config: LoopConfig | None = None,
                 state_path: Path | str | None = None,
                 log_to_mlflow: bool = True):
        self.proposer = proposer
        self.executor = executor
        self.config = config or LoopConfig()
        self.state_path = Path(state_path) if state_path else DEFAULT_STATE_PATH
        self.log_to_mlflow = log_to_mlflow
        self.history: list[Attempt] = []

    # ── 상태 저장·복원 ────────────────────────────────────────────────────────

    def load_state(self) -> bool:
        if not self.state_path.exists():
            return False
        data = json.loads(self.state_path.read_text())
        self.history = [Attempt(**a) for a in data.get("history", [])]
        log.info("이전 상태 복원: %d회 시도 (%s)", len(self.history), self.state_path)
        return True

    def save_state(self, stopped_because: str = "") -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({
            "updated": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "stopped_because": stopped_because,
            "history": [asdict(a) for a in self.history],
        }, indent=2, ensure_ascii=False))

    # ── 평가 ──────────────────────────────────────────────────────────────────

    def _score(self, outcome: Outcome) -> float | None:
        if not outcome.ok:
            return None
        value = outcome.metrics.get(self.config.target_metric)
        return None if value is None else float(value)

    def _best(self) -> Attempt | None:
        scored = [a for a in self.history if a.score is not None]
        if not scored:
            return None
        pick = max if self.config.maximize else min
        return pick(scored, key=lambda a: a.score)

    def _is_improvement(self, score: float) -> bool:
        best = self._best()
        if best is None or best.score is None:
            return True
        delta = score - best.score
        return (delta if self.config.maximize else -delta) > self.config.min_delta

    def _target_reached(self, score: float) -> bool:
        if self.config.target_value is None:
            return False
        return (score >= self.config.target_value if self.config.maximize
                else score <= self.config.target_value)

    # ── 본체 ──────────────────────────────────────────────────────────────────

    def run(self) -> LoopResult:
        started = time.perf_counter()
        stale = 0
        stopped = "알 수 없음"

        while True:
            # 정지 조건 — 제안을 받기 전에 먼저 확인한다.
            if len(self.history) >= self.config.max_iters:
                stopped = f"반복 예산 소진 ({self.config.max_iters}회)"
                break
            if (self.config.max_seconds is not None
                    and time.perf_counter() - started >= self.config.max_seconds):
                stopped = f"시간 예산 소진 ({self.config.max_seconds}s)"
                break
            if stale >= self.config.patience:
                stopped = f"{self.config.patience}회 연속 미개선"
                break

            proposal = self.proposer.propose(self.history)
            if proposal is None:
                stopped = "제안 고갈"
                break

            iteration = len(self.history) + 1
            log.info("[%d/%d] %s  — %s", iteration, self.config.max_iters,
                     proposal.label(), proposal.rationale)

            outcome = self.executor.run(proposal)
            score = self._score(outcome)
            improved = score is not None and self._is_improvement(score)

            self.history.append(Attempt(
                iteration=iteration,
                params=proposal.params,
                rationale=proposal.rationale,
                ok=outcome.ok,
                score=score,
                metrics=outcome.metrics,
                message=outcome.message,
                duration_sec=round(outcome.duration_sec, 1),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            self.save_state()
            self._record(proposal, outcome, score, iteration)

            if score is None:
                log.warning("   실패: %s", outcome.message)
                stale += 1
                continue

            marker = "★ 최고 기록" if improved else ""
            log.info("   %s = %.4f  %s", self.config.target_metric, score, marker)
            stale = 0 if improved else stale + 1

            if self._target_reached(score):
                stopped = f"목표 달성 ({self.config.target_metric} {score:.4f})"
                break

        best = self._best()
        self.save_state(stopped)
        log.info("종료: %s", stopped)
        if best:
            log.info("최고: %s → %.4f", Proposal(best.params).label(), best.score)

        return LoopResult(stopped, len(self.history),
                          best.score if best else None,
                          best.params if best else None,
                          self.history)

    def _record(self, proposal: Proposal, outcome: Outcome,
                score: float | None, iteration: int) -> None:
        """각 반복을 MLflow에 남긴다 — 루프가 끝난 뒤 왜 그 설정을 골랐는지 추적 가능하게."""
        if not self.log_to_mlflow:
            return
        try:
            from mlops.tracking import log_result
        except ImportError:
            return

        log_result(
            {
                "run_name": outcome.run_name or f"ralph-iter{iteration}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **proposal.params,
                **outcome.metrics,
            },
            experiment="FusionDTI-Ralph",
            extra_tags={
                "ralph_iteration": str(iteration),
                "ralph_rationale": proposal.rationale[:250],
                "ralph_ok": str(outcome.ok),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

class DryRunExecutor:
    """아무것도 실행하지 않고 명령만 보여 준다."""

    def __init__(self):
        self.inner = TrainingExecutor()
        self.seen: list[str] = []

    def run(self, proposal: Proposal) -> Outcome:
        command = " ".join(self.inner.build_command(proposal))
        self.seen.append(command)
        print(f"    $ {command}")
        return Outcome(True, metrics={"test_pearson_r": 0.0}, message="dry-run")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--target", type=float, default=None,
                        help="test_pearson_r 목표치 — 도달하면 즉시 종료")
    parser.add_argument("--proposer", choices=["grid", "llm"], default="grid",
                        help="llm은 ANTHROPIC_API_KEY 필요 (없으면 grid로 폴백)")
    parser.add_argument("--dry-run", action="store_true",
                        help="학습을 돌리지 않고 실행할 명령만 출력")
    parser.add_argument("--resume", action="store_true",
                        help="이전 상태를 이어서 진행")
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    proposer: Proposer = (LLMProposer() if args.proposer == "llm"
                          else GridProposer(DEFAULT_GRID))
    executor: Executor = DryRunExecutor() if args.dry_run else TrainingExecutor()

    # dry-run은 모든 결과가 0점이라 patience에 먼저 걸린다. 무엇을 돌릴지
    # 전부 보는 게 목적이므로 그때는 미개선 정지를 끈다.
    patience = args.max_iters if args.dry_run else args.patience

    loop = RalphLoop(
        proposer,
        executor,
        LoopConfig(target_value=args.target, max_iters=args.max_iters,
                   max_seconds=args.max_seconds, patience=patience),
        state_path=args.state,
        log_to_mlflow=not (args.no_mlflow or args.dry_run),
    )

    if args.resume:
        loop.load_state()

    result = loop.run()
    print()
    print(f"정지 사유 : {result.stopped_because}")
    print(f"시도 횟수 : {result.iterations}")
    if result.best_params:
        print(f"최고 설정 : {Proposal(result.best_params).label()}")
        print(f"최고 점수 : {result.best_score:.4f}")
    print(f"상태 파일 : {loop.state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
