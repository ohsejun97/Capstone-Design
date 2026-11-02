"""
graph.py — Phase 3 에이전트 오케스트레이션 (LangGraph)

왜 ReAct 루프가 아니라 상태 그래프인가
───────────────────────────────────────
DTI 질의의 제어 흐름은 사실 정해져 있다: 이름 해석 → 범위 점검 → 구조 보강
→ 예측 → 답변. LLM이 매 스텝 도구를 고르게 두면(ReAct) 비용과 비결정성만
늘어나고, 틀린 순서로 도구를 부르거나 무한 루프에 빠질 수 있다.

반면 "실패하면 어떻게 할 것인가"는 실제로 분기가 필요하다:
  · 브랜드명이라 PubChem이 못 찾음      → 성분명으로 보정 후 1회 재시도
  · 융합 유전자라 UniProt 항목이 없음   → 별칭 사전으로 치환 후 재시도
  · 키나아제가 아님                     → 차단하지 않고 OOD 경고를 달아 계속
  · AlphaFold 구조 없음                 → 3Di 없이 예측하고 정확도 하락을 고지

그래서 LLM은 맨 앞 엔티티 추출에만 쓰고, 나머지는 명시적 그래프로 둔다.
부수 효과로 체크포인터가 붙는다 — AlphaFold PDB 다운로드처럼 수십 초
걸리는 단계에서 끊겨도 같은 thread_id로 이어서 재개할 수 있다.

그래프 모양
───────────
    START → parse
              │ (실패)─────────────────────────────────┐
              ▼                                        │
         resolve_drug ──retry──→ refine_drug_name ──┐  │
              │ ok        ▲                         │  │
              │           └─────────────────────────┘  │
              │ give_up → mark_drug_failure ───────────┤
              ▼                                        │
       resolve_protein ──retry──→ refine_protein_name ─┤
              │ ok                                     │
              │ give_up → mark_protein_failure ────────┤
              ▼                                        │
         scope_check                                   │
              │                                        │
       ┌──────┴──────┐  (병렬 fan-out)                 │
       ▼             ▼                                 │
 fetch_structure  generate_ligand                      │
       └──────┬──────┘  (fan-in)                       │
              ▼                                        │
           predict ─────────────────────────────────→ compose → END

사용:
    from services.agent.graph.graph import build_graph, run_query

    print(run_query("Does Imatinib bind to BCR-ABL?"))

    # 체크포인트를 써서 중단된 질의 재개
    app = build_graph(checkpoint="sqlite:///agent_checkpoints.db")
    config = {"configurable": {"thread_id": "user-42"}}
    app.invoke(new_state(q), config)
"""

from __future__ import annotations

import logging
from typing import Any

from services.agent.graph import nodes
from services.agent.graph.state import DTIState, new_state

log = logging.getLogger("agent.graph")


def build_graph(checkpoint: str | None = None):
    """컴파일된 LangGraph 앱을 만든다.

    Args:
        checkpoint: None이면 체크포인트 없음(단발 실행).
                    "memory"면 인메모리(프로세스 수명).
                    "sqlite:///path.db"면 디스크에 저장 — 재개 가능.
    """
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(DTIState)

    builder.add_node("parse", nodes.parse_query)
    builder.add_node("resolve_drug", nodes.resolve_drug)
    builder.add_node("refine_drug_name", nodes.refine_drug_name)
    builder.add_node("mark_drug_failure", nodes.mark_drug_failure)
    builder.add_node("resolve_protein", nodes.resolve_protein)
    builder.add_node("refine_protein_name", nodes.refine_protein_name)
    builder.add_node("mark_protein_failure", nodes.mark_protein_failure)
    builder.add_node("scope_check", nodes.scope_check)
    builder.add_node("fetch_structure", nodes.fetch_structure)
    builder.add_node("generate_ligand", nodes.generate_ligand)
    builder.add_node("predict", nodes.predict)
    builder.add_node("compose", nodes.compose_answer)

    builder.add_edge(START, "parse")

    # 파싱 실패는 도구를 하나도 부르지 않고 바로 답변 구성으로 간다.
    builder.add_conditional_edges(
        "parse",
        lambda s: "failed" if s.get("error") else "ok",
        {"ok": "resolve_drug", "failed": "compose"},
    )

    # ── 약물 해석: 보정 후 최대 1회 재시도 (사이클) ──────────────────────
    builder.add_conditional_edges(
        "resolve_drug",
        nodes.route_after_drug,
        {
            "ok": "resolve_protein",
            "retry": "refine_drug_name",
            "give_up": "mark_drug_failure",
        },
    )
    builder.add_edge("refine_drug_name", "resolve_drug")
    builder.add_edge("mark_drug_failure", "compose")

    # ── 표적 해석: 동일 구조 ──────────────────────────────────────────────
    builder.add_conditional_edges(
        "resolve_protein",
        nodes.route_after_protein,
        {
            "ok": "scope_check",
            "retry": "refine_protein_name",
            "give_up": "mark_protein_failure",
        },
    )
    builder.add_edge("refine_protein_name", "resolve_protein")
    builder.add_edge("mark_protein_failure", "compose")

    # ── 보강: 두 도구는 서로 독립이므로 병렬로 돈다 ───────────────────────
    # AlphaFold 다운로드(네트워크)와 RDKit 형태 생성(CPU)이 겹쳐 실행된다.
    builder.add_edge("scope_check", "fetch_structure")
    builder.add_edge("scope_check", "generate_ligand")
    builder.add_edge("fetch_structure", "predict")
    builder.add_edge("generate_ligand", "predict")   # 둘 다 끝나야 predict 진입

    builder.add_edge("predict", "compose")
    builder.add_edge("compose", END)

    return builder.compile(checkpointer=_make_checkpointer(checkpoint))


def _make_checkpointer(spec: str | None):
    if spec is None:
        return None
    if spec == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    if spec.startswith("sqlite"):
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SQLite 체크포인터에는 langgraph-checkpoint-sqlite가 필요합니다: "
                "pip install langgraph-checkpoint-sqlite"
            ) from exc
        path = spec.replace("sqlite:///", "").replace("sqlite://", "")
        return SqliteSaver.from_conn_string(path).__enter__()
    raise ValueError(f"알 수 없는 체크포인트 지정: {spec}")


def run_query(
    query: str,
    *,
    checkpoint: str | None = None,
    thread_id: str = "default",
    verbose: bool = False,
) -> str:
    """질의 하나를 끝까지 돌려 답변 문자열을 돌려준다."""
    app = build_graph(checkpoint=checkpoint)
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    final = app.invoke(new_state(query), config if checkpoint else None)

    if verbose:
        print("── 실행 경로 " + "─" * 50)
        for step in final.get("trace", []):
            print(f"   {step}")
        print("─" * 62)

    return final.get("answer", "(답변 없음)")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="FusionDTI LangGraph 에이전트")
    parser.add_argument("query", nargs="*", help="자연어 질의")
    parser.add_argument("--verbose", "-v", action="store_true", help="실행 경로 출력")
    parser.add_argument("--checkpoint", default=None,
                        help="memory | sqlite:///agent_checkpoints.db")
    parser.add_argument("--thread-id", default="cli")
    parser.add_argument("--draw", action="store_true",
                        help="그래프 구조를 Mermaid로 출력하고 종료")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.draw:
        print(build_graph().get_graph().draw_mermaid())
        return 0

    query = " ".join(args.query) or "Does Imatinib bind to BCR-ABL?"
    print(run_query(query, checkpoint=args.checkpoint,
                    thread_id=args.thread_id, verbose=args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
