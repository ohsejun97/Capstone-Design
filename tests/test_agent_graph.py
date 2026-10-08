"""
test_agent_graph.py — LangGraph 제어 흐름 검증

여기서 확인하는 것은 모델 성능이 아니라 **그래프가 분기를 제대로 타는가**다:
  · 정상 경로가 5개 도구를 모두 부르는가
  · PubChem 실패 시 이름을 보정해 재시도하는가, 그리고 멈추는가
  · 키나아제가 아닌 표적에 OOD 경고가 붙는가 (차단이 아니라 경고)
  · AlphaFold 실패가 예측을 막지 않는가
  · 병렬 분기 두 개가 남긴 경고가 서로 덮어쓰지 않는가 (reducer 동작)
"""

from __future__ import annotations

import pytest

from agent.state import new_state
from tests.conftest import HMGCR, PREDICTION_OK


def run(query: str, **kwargs):
    from agent.graph import build_graph
    return build_graph().invoke(new_state(query), **kwargs)


# ── 정상 경로 ─────────────────────────────────────────────────────────────────

def test_happy_path_calls_all_five_tools(tools):
    final = run("Does Imatinib bind to ABL1?")

    assert final["stage"] == "done"
    assert set(tools.names()) == {"pubchem", "uniprot", "alphafold", "rdkit", "dti"}
    assert "8.7412" in final["answer"]
    assert "P00519" in final["answer"]


def test_dti_tool_receives_resolved_smiles_and_sequence(tools):
    run("Does Imatinib bind to ABL1?")

    smiles, sequence = next(args for name, args in tools.calls if name == "dti")
    assert smiles.startswith("Cc1ccc")
    assert len(sequence) == 1130


def test_trace_records_every_node(tools):
    final = run("Does Imatinib bind to ABL1?")
    joined = " | ".join(final["trace"])

    for expected in ("parse:rule", "resolve_drug[1]", "resolve_protein[1]",
                     "scope_check", "fetch_structure", "generate_ligand", "predict"):
        assert expected in joined, f"{expected} 누락: {joined}"


# ── 재시도 사이클 ─────────────────────────────────────────────────────────────

def test_brand_name_retry_succeeds(tools):
    """PubChem이 브랜드명을 거부하면 성분명으로 바꿔 한 번 더 시도한다.

    응답 리스트를 하나만 넣으면 소진 후 기본값(IMATINIB)으로 떨어진다 —
    즉 1차 실패, 2차 성공 시나리오가 된다.
    """
    tools.responses["pubchem"] = [
        {"error": "Drug 'Gleevec' not found in PubChem.", "query_name": "Gleevec"}
    ]

    final = run("Does Gleevec bind to ABL1?")

    assert tools.names().count("pubchem") == 2, "재시도가 일어나지 않았다"
    assert final["stage"] == "done"
    assert any("Gleevec" in w and "Imatinib" in w for w in final["warnings"])


def test_retry_is_bounded_then_gives_up(tools):
    """계속 실패해도 무한 루프에 빠지지 않고 실패 답변으로 끝난다."""
    tools.responses["pubchem"] = {"error": "not found", "query_name": "Zzzz"}

    final = run("Does Zzzzqqq bind to ABL1?")

    assert final["stage"] == "failed"
    assert tools.names().count("pubchem") <= 2
    # 약물 해석이 끝내 실패했으면 DTI 예측은 호출되지 않아야 한다.
    assert "dti" not in tools.names()
    assert "❌" in final["answer"]


def test_protein_alias_retry(tools):
    """UniProt에 없는 융합 표기(BCR-ABL)를 ABL1로 바꿔 재시도한다."""
    tools.responses["uniprot"] = [
        {"error": "Protein 'BCR-ABL' not found in UniProt.", "query_name": "BCR-ABL"}
    ]

    final = run("Does Imatinib bind to BCR-ABL?")

    assert tools.names().count("uniprot") == 2
    assert final["stage"] == "done"


# ── 범위 점검 (차단이 아니라 경고) ────────────────────────────────────────────

def test_non_kinase_target_warns_but_still_predicts(tools):
    tools.responses["uniprot"] = HMGCR

    final = run("Does Atorvastatin bind to HMGCR?")

    assert final["stage"] == "done", "OOD는 차단이 아니라 경고여야 한다"
    assert "dti" in tools.names()
    assert any("OOD" in w for w in final["warnings"])


def test_non_human_target_warns(tools):
    tools.responses["uniprot"] = dict(HMGCR, organism="Mus musculus",
                                      name="Tyrosine-protein kinase")

    final = run("Does Imatinib bind to ABL1?")

    assert any("인간 단백질이 아닙니다" in w for w in final["warnings"])


# ── 부분 실패 내성 ────────────────────────────────────────────────────────────

def test_alphafold_failure_does_not_block_prediction(tools):
    tools.responses["alphafold"] = {"error": "UniProt ID not found in AlphaFold DB.",
                                    "uniprot_id": "P00519"}

    final = run("Does Imatinib bind to ABL1?")

    assert final["stage"] == "done"
    assert "dti" in tools.names()
    assert any("AlphaFold" in w for w in final["warnings"])


def test_missing_3di_is_reported(tools):
    tools.responses["dti"] = dict(PREDICTION_OK, used_3di=False)

    final = run("Does Imatinib bind to ABL1?")

    assert any("placeholder" in w for w in final["warnings"])
    assert "미사용" in final["answer"]


def test_parallel_branches_accumulate_warnings(tools):
    """fan-out 두 분기가 같은 슈퍼스텝에서 남긴 경고가 둘 다 살아남아야 한다.

    warnings에 operator.add reducer가 없으면 한쪽이 다른 쪽을 덮어쓴다.
    """
    tools.responses["alphafold"] = {"error": "not found", "uniprot_id": "P00519"}
    tools.responses["uniprot"] = HMGCR          # scope_check 경고
    tools.responses["dti"] = dict(PREDICTION_OK, used_3di=False)   # predict 경고

    final = run("Does Atorvastatin bind to HMGCR?")

    assert len(final["warnings"]) >= 3, final["warnings"]
    assert any("OOD" in w for w in final["warnings"])
    assert any("AlphaFold" in w for w in final["warnings"])
    assert any("placeholder" in w for w in final["warnings"])


def test_dti_failure_short_circuits_to_error_answer(tools):
    tools.responses["dti"] = {"error": "CUDA out of memory", "smiles": "x"}

    final = run("Does Imatinib bind to ABL1?")

    assert final["stage"] == "failed"
    assert "CUDA out of memory" in final["answer"]


# ── 입력 처리 ─────────────────────────────────────────────────────────────────

def test_unparseable_query_calls_no_tools(tools):
    final = run("안녕하세요")

    assert final["stage"] == "failed"
    assert tools.calls == [], "파싱 실패 시 도구를 부르면 안 된다"


@pytest.mark.parametrize("query,drug,protein", [
    ("Does Imatinib bind to ABL1?", "Imatinib", "ABL1"),
    ("Imatinib과 EGFR의 결합력", "Imatinib", "EGFR"),
    ("글리벡 ABL1 결합 예측", "글리벡", "ABL1"),
    ("Gleevec - HER2", "Gleevec", "HER2"),
    ("Does Imatinib bind to BCR-ABL?", "Imatinib", "BCR-ABL"),
])
def test_rule_based_entity_extraction(query, drug, protein):
    from agent.llm import _extract_with_rules

    got_drug, got_protein = _extract_with_rules(query)
    # 추출은 표면형을 그대로 돌려준다 — 표준명 치환은 refine_* 노드의 몫이다.
    assert got_drug == drug
    assert got_protein == protein


# ── 체크포인트 ────────────────────────────────────────────────────────────────

def test_checkpointer_persists_state(tools):
    """같은 thread_id로 재호출하면 이전 상태가 남아 있어야 한다."""
    from agent.graph import build_graph

    app = build_graph(checkpoint="memory")
    config = {"configurable": {"thread_id": "t1"}}
    app.invoke(new_state("Does Imatinib bind to ABL1?"), config)

    snapshot = app.get_state(config)
    assert snapshot.values["stage"] == "done"
    assert snapshot.values["prediction"]["pKd"] == 8.7412
