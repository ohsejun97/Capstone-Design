"""
conftest.py — 에이전트 테스트용 가짜 도구 주입

tools/ 하위 모듈은 torch·rdkit·네트워크를 요구한다. 그래프의 제어 흐름
(재시도 / OOD 경고 / 병렬 fan-in / 실패 단락)을 검증하는 데는 그게 필요 없고,
CI에서 GPU와 외부 API를 쓸 수도 없다.

agent.nodes는 도구를 **함수 안에서 지연 import** 하므로, 노드가 처음 실행되는
시점에 sys.modules에 가짜 모듈이 꽂혀 있으면 그것을 집는다. 그래서 여기서
sys.modules를 통째로 대체한다.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class ToolRecorder:
    """가짜 도구의 호출 기록 + 응답 대본을 담는다."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        # name -> 응답 또는 응답 리스트(호출 순서대로 소비)
        self.responses: dict[str, object] = {}

    def record(self, tool: str, *args):
        self.calls.append((tool, args))

    def reply(self, tool: str, default: dict) -> dict:
        spec = self.responses.get(tool)
        if spec is None:
            return default
        if isinstance(spec, list):
            return spec.pop(0) if spec else default
        return spec

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


IMATINIB = {
    "query_name": "Imatinib", "name": "Imatinib", "cid": "5291",
    "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "formula": "C29H31N7O", "mol_weight": "493.6", "cached": True,
}

ABL1 = {
    "query_name": "ABL1", "uniprot_id": "P00519", "gene": "ABL1",
    "name": "Tyrosine-protein kinase ABL1", "organism": "Homo sapiens",
    "seq_length": "1130", "sequence": "M" + "A" * 1129, "cached": True,
}

# 키나아제가 아닌 표적 — scope_check의 OOD 경고를 밟게 한다.
HMGCR = {
    "query_name": "HMGCR", "uniprot_id": "P04035", "gene": "HMGCR",
    "name": "3-hydroxy-3-methylglutaryl-coenzyme A reductase",
    "organism": "Homo sapiens", "seq_length": "888",
    "sequence": "M" + "G" * 887, "cached": True,
}

STRUCTURE_OK = {
    "uniprot_id": "P00519", "entry_id": "AF-P00519-F1", "gene": "ABL1",
    "plddt_global": "88.5", "pdb_path": "/tmp/P00519.pdb", "cached": True,
}

LIGAND_OK = {"smiles": IMATINIB["smiles"], "sdf_path": "/tmp/imatinib.sdf",
             "status": "cached"}

PREDICTION_OK = {
    "smiles": IMATINIB["smiles"], "seq_length": 1130, "pKd": 8.7412,
    "interpretation": "Strong binding (pKd 7–9, Kd 1–100 nM)", "used_3di": True,
}


@pytest.fixture
def tools(monkeypatch) -> ToolRecorder:
    """5개 도구 모듈을 전부 가짜로 교체한다."""
    rec = ToolRecorder()

    def make(module_name: str, func_name: str, tool_key: str, default: dict):
        module = types.ModuleType(module_name)

        def fake(*args, **kwargs):
            rec.record(tool_key, *args)
            return rec.reply(tool_key, default)

        setattr(module, func_name, fake)
        monkeypatch.setitem(sys.modules, module_name, module)

    make("tools.pubchem_tool", "resolve_drug_name", "pubchem", IMATINIB)
    make("tools.uniprot_tool", "resolve_protein_name", "uniprot", ABL1)
    make("tools.alphafold_tool", "fetch_alphafold_structure", "alphafold", STRUCTURE_OK)
    make("tools.rdkit_tool", "generate_3d_ligand", "rdkit", LIGAND_OK)
    make("tools.dti_tool", "predict_binding", "dti", PREDICTION_OK)

    # LLM 경로가 켜지면 테스트가 네트워크를 타므로 규칙 경로로 고정한다.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return rec
