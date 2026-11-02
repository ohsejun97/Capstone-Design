"""
test_ralph.py — Ralph Loop 검증

자율 루프에서 유일하게 정말 위험한 것은 **멈추지 않는 것**이다. 학습 한 번이
수십 분이므로 정지 조건이 하나라도 새면 GPU와 시간이 조용히 증발한다.
그래서 정지 조건 네 가지를 각각 독립적으로 검증한다.

그 다음으로 위험한 것은 재개 시 같은 학습을 다시 도는 것 — 이것도 검증한다.
"""

from __future__ import annotations

import json

import pytest

from ralph.loop import (
    Attempt,
    GridProposer,
    LLMProposer,
    LoopConfig,
    Outcome,
    Proposal,
    RalphLoop,
    TrainingExecutor,
)

GRID = [
    {"unfreeze": 2, "lr_cb": 1e-5},
    {"unfreeze": 3, "lr_cb": 1e-5},
    {"unfreeze": 4, "lr_cb": 1e-5},
    {"unfreeze": 5, "lr_cb": 1e-5},
]


class ScriptedExecutor:
    """제안 순서대로 정해진 점수를 돌려준다."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls: list[dict] = []

    def run(self, proposal: Proposal) -> Outcome:
        self.calls.append(proposal.params)
        if not self.scores:
            return Outcome(False, message="대본 소진")
        score = self.scores.pop(0)
        if score is None:
            return Outcome(False, message="학습 실패(주입)")
        return Outcome(True, metrics={"test_pearson_r": score},
                       run_name=f"run-{len(self.calls)}")


def make_loop(tmp_path, scores, **cfg):
    executor = ScriptedExecutor(scores)
    loop = RalphLoop(
        GridProposer(GRID),
        executor,
        LoopConfig(**{"max_iters": 10, "patience": 10, **cfg}),
        state_path=tmp_path / "state.json",
        log_to_mlflow=False,
    )
    return loop, executor


# ══════════════════════════════════════════════════════════════════════════════
# 정지 조건 4가지 — 하나씩 독립 검증
# ══════════════════════════════════════════════════════════════════════════════

def test_stops_on_target_reached(tmp_path):
    loop, executor = make_loop(tmp_path, [0.80, 0.91, 0.95], target_value=0.90)
    result = loop.run()

    assert "목표 달성" in result.stopped_because
    assert result.iterations == 2, "목표 도달 즉시 멈춰야 한다"
    assert len(executor.calls) == 2
    assert result.best_score == 0.91


def test_stops_on_iteration_budget(tmp_path):
    loop, _ = make_loop(tmp_path, [0.1, 0.2, 0.3, 0.4], max_iters=3)
    result = loop.run()

    assert "반복 예산 소진" in result.stopped_because
    assert result.iterations == 3


def test_stops_on_patience(tmp_path):
    """개선이 없으면 격자가 남아 있어도 멈춰야 한다."""
    loop, _ = make_loop(tmp_path, [0.90, 0.80, 0.81, 0.82], patience=2)
    result = loop.run()

    assert "연속 미개선" in result.stopped_because
    assert result.iterations == 3     # 첫 시도 + 미개선 2회


def test_stops_when_proposals_exhausted(tmp_path):
    loop, _ = make_loop(tmp_path, [0.1, 0.2, 0.3, 0.4], max_iters=99)
    result = loop.run()

    assert result.stopped_because == "제안 고갈"
    assert result.iterations == len(GRID)


def test_stops_on_time_budget(tmp_path):
    import time

    class SlowExecutor:
        def run(self, proposal):
            time.sleep(0.05)
            return Outcome(True, metrics={"test_pearson_r": 0.5})

    loop = RalphLoop(GridProposer(GRID * 10), SlowExecutor(),
                     LoopConfig(max_iters=999, patience=999, max_seconds=0.1),
                     state_path=tmp_path / "s.json", log_to_mlflow=False)
    result = loop.run()

    assert "시간 예산" in result.stopped_because
    assert result.iterations < 10


def test_failures_alone_cannot_loop_forever(tmp_path):
    """학습이 계속 죽어도 patience가 루프를 끝내야 한다."""
    loop, _ = make_loop(tmp_path, [None] * 20, patience=3, max_iters=99)
    result = loop.run()

    assert result.iterations == 3
    assert result.best_score is None
    assert all(a.score is None for a in result.history)


# ══════════════════════════════════════════════════════════════════════════════
# 상태 저장 · 재개
# ══════════════════════════════════════════════════════════════════════════════

def test_state_is_written_every_iteration(tmp_path):
    """중간에 죽어도 이미 끝난 학습은 보존되어야 한다."""
    state_path = tmp_path / "state.json"

    class CrashingExecutor:
        def __init__(self):
            self.n = 0

        def run(self, proposal):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("CUDA 오류(주입)")
            return Outcome(True, metrics={"test_pearson_r": 0.5})

    loop = RalphLoop(GridProposer(GRID), CrashingExecutor(),
                     LoopConfig(max_iters=5, patience=5),
                     state_path=state_path, log_to_mlflow=False)

    with pytest.raises(RuntimeError):
        loop.run()

    saved = json.loads(state_path.read_text())
    assert len(saved["history"]) == 1, "1회차 결과가 디스크에 남아야 한다"


def test_resume_does_not_repeat_finished_work(tmp_path):
    state_path = tmp_path / "state.json"

    first, exec1 = make_loop(tmp_path, [0.50, 0.60], max_iters=2)
    first.state_path = state_path
    first.run()
    assert len(exec1.calls) == 2

    second, exec2 = make_loop(tmp_path, [0.70, 0.80], max_iters=4)
    second.state_path = state_path
    assert second.load_state()
    second.run()

    # 2회차 루프는 남은 격자 2개만 돌아야 한다.
    assert len(exec2.calls) == 2
    already = [a["unfreeze"] for a in exec1.calls]
    assert all(call["unfreeze"] not in already for call in exec2.calls)


def test_resume_keeps_best_from_before(tmp_path):
    state_path = tmp_path / "state.json"
    first, _ = make_loop(tmp_path, [0.95, 0.10], max_iters=2)
    first.state_path = state_path
    first.run()

    second, _ = make_loop(tmp_path, [0.20, 0.30], max_iters=4)
    second.state_path = state_path
    second.load_state()
    result = second.run()

    assert result.best_score == 0.95, "재개 후에도 이전 최고 기록을 기억해야 한다"


# ══════════════════════════════════════════════════════════════════════════════
# 제안자
# ══════════════════════════════════════════════════════════════════════════════

def test_grid_proposer_skips_tried_configs():
    proposer = GridProposer(GRID)
    history = [Attempt(1, GRID[0], "", True, 0.5, {}, "", 0.0, "t")]

    proposal = proposer.propose(history)
    assert proposal.params == GRID[1]


def test_grid_proposer_returns_none_when_done():
    proposer = GridProposer(GRID)
    history = [Attempt(i, p, "", True, 0.5, {}, "", 0.0, "t")
               for i, p in enumerate(GRID)]
    assert proposer.propose(history) is None


def test_llm_proposer_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    proposer = LLMProposer(fallback=GridProposer(GRID))

    proposal = proposer.propose([])
    assert proposal.params == GRID[0]
    assert not proposer.enabled


def test_llm_duplicate_proposal_is_rejected(monkeypatch):
    """LLM이 이미 시도한 설정을 다시 내면 루프가 제자리를 돈다 — 막아야 한다."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    proposer = LLMProposer(fallback=GridProposer(GRID))

    # LLM 호출부를 가로채 항상 GRID[0]을 제안하게 만든다.
    monkeypatch.setattr(
        proposer, "_render", lambda history: "x")

    class FakeBlock:
        type = "text"
        text = json.dumps({"params": GRID[0], "rationale": "중복"})

    class FakeResponse:
        def __init__(self):
            self.content = [FakeBlock()]

    class FakeMessages:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        messages = FakeMessages()

    import sys
    import types
    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = lambda *a, **k: FakeClient()
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    history = [Attempt(1, GRID[0], "", True, 0.5, {}, "", 0.0, "t")]
    proposal = proposer.propose(history)

    assert proposal.params == GRID[1], "중복 제안을 그대로 받아들이면 안 된다"


def test_llm_can_declare_convergence(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    proposer = LLMProposer()

    class FakeBlock:
        type = "text"
        text = json.dumps({"params": None, "rationale": "수렴"})

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                return type("R", (), {"content": [FakeBlock()]})()

    import sys
    import types
    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda *a, **k: FakeClient()
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    assert proposer.propose([]) is None


# ══════════════════════════════════════════════════════════════════════════════
# 실행기 — 명령 조립
# ══════════════════════════════════════════════════════════════════════════════

def test_training_command_maps_params_to_flags():
    executor = TrainingExecutor(python="python3")
    cmd = executor.build_command(Proposal({"unfreeze": 3, "lr_cb": 3e-5,
                                           "batch_size": 64, "split": "random"}))

    assert cmd[:3] == ["python3", "-m", "scripts.train_chemberta_unfreeze"]
    assert "--unfreeze" in cmd and cmd[cmd.index("--unfreeze") + 1] == "3"
    assert "--split" in cmd and cmd[cmd.index("--split") + 1] == "random"


def test_training_command_ignores_unknown_params():
    """제안자가 모르는 키를 내도 CLI가 깨지면 안 된다."""
    executor = TrainingExecutor()
    cmd = executor.build_command(Proposal({"unfreeze": 2, "made_up_knob": 7}))

    assert "--made_up_knob" not in cmd
