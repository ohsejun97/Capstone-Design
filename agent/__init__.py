"""Phase 3 — LangGraph 기반 에이전트 오케스트레이션."""

from agent.state import DTIState, new_state

__all__ = ["DTIState", "build_graph", "new_state", "run_query"]


def __getattr__(name: str):
    # graph 모듈은 langgraph를 import한다. agent.state만 쓰는 곳(예: 테스트)에서
    # 불필요한 의존성을 끌어오지 않도록 지연 로딩한다.
    if name in ("build_graph", "run_query"):
        from agent import graph
        return getattr(graph, name)
    raise AttributeError(name)
