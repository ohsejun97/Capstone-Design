"""
nodes.py — LangGraph 노드 구현

각 노드는 `DTIState -> dict` 순수 함수다. 반환한 dict의 키만 상태에 병합된다
(전체 상태를 되돌려줄 필요 없음). 노드는 예외를 던지지 않고 `error` 또는
`warnings`에 기록한다 — 네트워크 실패 하나로 그래프 전체가 죽으면
체크포인트에서 재개할 수 없기 때문이다.

Tool 매핑 (docs/PHASE2_AGENT_TOOLS.md):
    Tool 1 DTI 예측      → tools.dti_tool.predict_binding
    Tool 2 단백질 구조   → tools.alphafold_tool.fetch_alphafold_structure
    Tool 3 리간드 3D     → tools.rdkit_tool.generate_3d_ligand
    Tool 4 약물명 해석   → tools.pubchem_tool.resolve_drug_name
    Tool 5 단백질명 해석 → tools.uniprot_tool.resolve_protein_name
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.state import DTIState

log = logging.getLogger("agent")

MAX_DRUG_ATTEMPTS = 2
MAX_PROTEIN_ATTEMPTS = 2

# 브랜드명 → 성분명. PubChem은 성분명에서 훨씬 안정적으로 맞춘다.
# 1차 조회가 실패했을 때만 쓰는 재시도 힌트다.
BRAND_TO_GENERIC = {
    "gleevec": "Imatinib", "글리벡": "Imatinib",
    "viagra": "Sildenafil", "비아그라": "Sildenafil",
    "tarceva": "Erlotinib", "iressa": "Gefitinib",
    "sutent": "Sunitinib", "nexavar": "Sorafenib",
    "sprycel": "Dasatinib", "tasigna": "Nilotinib",
    "lipitor": "Atorvastatin", "tylenol": "Acetaminophen",
}

# 흔한 표적 별칭 → UniProt가 아는 유전자 심볼.
# "BCR-ABL"은 융합 유전자라 UniProt 항목이 없다 → 키나아제 도메인을 가진 ABL1로.
ALIAS_TO_GENE = {
    "bcr-abl": "ABL1", "bcr abl": "ABL1", "abl": "ABL1",
    "her2": "ERBB2", "her1": "EGFR", "c-kit": "KIT",
    "vegfr2": "KDR", "pdgfr": "PDGFRA", "alk": "ALK",
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. 질의 파싱
# ══════════════════════════════════════════════════════════════════════════════

def parse_query(state: DTIState) -> dict[str, Any]:
    """자연어 질의에서 약물명과 표적 단백질명을 뽑는다.

    LLM이 붙어 있으면 LLM으로, 없으면 규칙 기반으로 처리한다.
    규칙 기반 경로가 있어야 API 키 없이 CI에서 그래프 전체를 테스트할 수 있다.
    """
    query = state.get("query", "").strip()
    if not query:
        return {"error": "빈 질의입니다.", "stage": "failed", "trace": ["parse:empty"]}

    from agent.llm import extract_entities

    drug, protein, how = extract_entities(query)
    if not drug or not protein:
        return {
            "error": (
                f"질의에서 약물과 표적을 모두 찾지 못했습니다 "
                f"(약물={drug or '?'}, 표적={protein or '?'}). "
                f"'Imatinib과 ABL1의 결합력' 형태로 다시 입력해 주세요."
            ),
            "stage": "failed",
            "trace": [f"parse:{how}:incomplete"],
        }

    return {
        "drug_name": drug,
        "protein_name": protein,
        "stage": "resolve_drug",
        "trace": [f"parse:{how} → drug={drug}, protein={protein}"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. 약물명 해석 (Tool 4) + 재시도
# ══════════════════════════════════════════════════════════════════════════════

def resolve_drug(state: DTIState) -> dict[str, Any]:
    """PubChem으로 약물명 → SMILES."""
    from tools.pubchem_tool import resolve_drug_name

    name = state["drug_name"]
    attempts = state.get("drug_attempts", 0) + 1
    result = resolve_drug_name(name)

    if "error" in result:
        return {
            "drug": result,
            "drug_attempts": attempts,
            "trace": [f"resolve_drug[{attempts}]: '{name}' 실패 — {result['error'][:60]}"],
        }

    return {
        "drug": result,
        "drug_attempts": attempts,
        "stage": "resolve_protein",
        "trace": [f"resolve_drug[{attempts}]: '{name}' → CID {result.get('cid')}"],
    }


def refine_drug_name(state: DTIState) -> dict[str, Any]:
    """1차 실패 후 약물명을 고쳐 본다.

    실패 원인의 대부분은 브랜드명(Gleevec)이나 용량 표기("아스피린 100mg")다.
    성분명 치환 → 숫자/단위 제거 순으로 시도한다.
    """
    name = state["drug_name"]
    key = name.strip().lower()

    if key in BRAND_TO_GENERIC:
        fixed = BRAND_TO_GENERIC[key]
        reason = "브랜드명 → 성분명"
    else:
        fixed = re.sub(r"\s*\d+\s*(mg|mcg|g|ml|정|캡슐)\b", "", name, flags=re.I).strip()
        reason = "용량 표기 제거"

    if not fixed or fixed.lower() == key:
        # 고칠 여지가 없다 — 카운터만 올려서 조건부 엣지가 포기하게 한다.
        return {
            "drug_attempts": MAX_DRUG_ATTEMPTS,
            "trace": [f"refine_drug: '{name}' 보정 불가"],
        }

    return {
        "drug_name": fixed,
        "warnings": [f"약물명을 '{name}' → '{fixed}'로 해석했습니다 ({reason})."],
        "trace": [f"refine_drug: '{name}' → '{fixed}' ({reason})"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. 단백질명 해석 (Tool 5) + 재시도
# ══════════════════════════════════════════════════════════════════════════════

def resolve_protein(state: DTIState) -> dict[str, Any]:
    """UniProt로 유전자/단백질명 → accession + 아미노산 서열."""
    from tools.uniprot_tool import resolve_protein_name

    name = state["protein_name"]
    attempts = state.get("protein_attempts", 0) + 1
    result = resolve_protein_name(name)

    if "error" in result:
        return {
            "protein": result,
            "protein_attempts": attempts,
            "trace": [f"resolve_protein[{attempts}]: '{name}' 실패 — {result['error'][:60]}"],
        }

    return {
        "protein": result,
        "protein_attempts": attempts,
        "stage": "scope_check",
        "trace": [f"resolve_protein[{attempts}]: '{name}' → {result.get('uniprot_id')}"],
    }


def refine_protein_name(state: DTIState) -> dict[str, Any]:
    """별칭 사전으로 표적명을 UniProt 유전자 심볼로 바꾼다."""
    name = state["protein_name"]
    key = name.strip().lower()

    fixed = ALIAS_TO_GENE.get(key)
    if fixed is None and "-" in key:
        # "BCR-ABL" 같은 융합 표기 → 뒷부분을 단독 유전자로 재시도
        tail = key.rsplit("-", 1)[-1]
        fixed = ALIAS_TO_GENE.get(tail, tail.upper() if tail.isalnum() else None)

    if not fixed or fixed.lower() == key:
        return {
            "protein_attempts": MAX_PROTEIN_ATTEMPTS,
            "trace": [f"refine_protein: '{name}' 보정 불가"],
        }

    return {
        "protein_name": fixed,
        "warnings": [f"표적명을 '{name}' → '{fixed}'로 해석했습니다."],
        "trace": [f"refine_protein: '{name}' → '{fixed}'"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. 적용 범위 점검
# ══════════════════════════════════════════════════════════════════════════════

def scope_check(state: DTIState) -> dict[str, Any]:
    """학습 분포를 벗어난 질의에 경고를 붙인다 (차단하지는 않는다).

    DTI Head는 BindingDB → DAVIS/KIBA로 전이 학습됐고, DAVIS/KIBA는
    인간 키나아제 전용이다. GPCR·프로테아제·바이러스 단백질 예측은
    분포 밖(OOD)이므로 수치를 그대로 믿으면 안 된다.
    README 'Scope & Limitations' 참조.
    """
    protein = state.get("protein", {})
    description = f"{protein.get('name', '')} {protein.get('gene', '')}".lower()
    organism = protein.get("organism", "")

    warnings: list[str] = []
    if "kinase" not in description:
        warnings.append(
            f"'{protein.get('gene', '?')}'는 키나아제로 보이지 않습니다. "
            f"학습 데이터(DAVIS/KIBA)는 인간 키나아제 전용이라 "
            f"이 예측은 분포 밖(OOD)이며 신뢰도가 낮습니다."
        )
    if organism and "Homo sapiens" not in organism:
        warnings.append(f"표적이 인간 단백질이 아닙니다 (organism={organism}). OOD 예측입니다.")

    seq_len = len(protein.get("sequence", ""))
    if seq_len > 2000:
        warnings.append(
            f"서열이 {seq_len} aa로 깁니다. SaProt 입력이 잘려 "
            f"결합 부위가 누락될 수 있습니다."
        )

    return {
        "warnings": warnings,
        "stage": "enrich",
        "trace": [f"scope_check: OOD 경고 {len(warnings)}건"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. 보강 — 병렬 분기 (둘 다 실패해도 예측은 진행된다)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_structure(state: DTIState) -> dict[str, Any]:
    """AlphaFold DB에서 3D 구조를 받는다 (Tool 2).

    3Di 토큰의 원천이다. 실패하면 DTI Tool이 '#' placeholder로 떨어지는데,
    Phase 1c 실험상 그때 DAVIS Pearson r이 약 0.023 낮아진다. 치명적이진 않다.
    """
    from tools.alphafold_tool import fetch_alphafold_structure

    uniprot_id = state.get("protein", {}).get("uniprot_id")
    if not uniprot_id:
        return {"trace": ["fetch_structure: UniProt ID 없음 — 건너뜀"]}

    result = fetch_alphafold_structure(uniprot_id)
    if "error" in result:
        return {
            "structure": result,
            "warnings": [
                (f"AlphaFold 구조를 받지 못했습니다 ({result['error'][:60]}). "
                 f"3Di 없이 예측하므로 정확도가 다소 낮습니다."),
            ],
            "trace": [f"fetch_structure: {uniprot_id} 실패"],
        }

    plddt = result.get("plddt_global", "?")
    warnings = []
    try:
        if float(plddt) < 70:
            warnings.append(
                f"AlphaFold 구조 신뢰도가 낮습니다 (pLDDT {plddt}). "
                f"3Di 토큰이 부정확할 수 있습니다."
            )
    except (TypeError, ValueError):
        pass

    return {
        "structure": result,
        "warnings": warnings,
        "trace": [f"fetch_structure: {uniprot_id} pLDDT={plddt}"],
    }


def generate_ligand(state: DTIState) -> dict[str, Any]:
    """RDKit으로 3D 리간드 형태를 만든다 (Tool 3). 시각화용 — 예측에는 쓰지 않는다."""
    from tools.rdkit_tool import generate_3d_ligand

    drug = state.get("drug", {})
    smiles = drug.get("smiles")
    if not smiles:
        return {"trace": ["generate_ligand: SMILES 없음 — 건너뜀"]}

    result = generate_3d_ligand(smiles, name=drug.get("name"))
    if "error" in result:
        return {
            "ligand": result,
            "trace": [f"generate_ligand: 실패 — {result['error'][:60]}"],
        }
    return {"ligand": result, "trace": ["generate_ligand: 3D SDF 생성"]}


# ══════════════════════════════════════════════════════════════════════════════
# 6. 예측 (Tool 1)
# ══════════════════════════════════════════════════════════════════════════════

def predict(state: DTIState) -> dict[str, Any]:
    """SaProt + ft-ChemBERTa + MLP Head로 pKd를 예측한다."""
    from tools.dti_tool import predict_binding

    smiles = state.get("drug", {}).get("smiles", "")
    sequence = state.get("protein", {}).get("sequence", "")

    result = predict_binding(smiles, sequence)
    if "error" in result:
        return {
            "prediction": result,
            "error": f"DTI 예측 실패: {result['error']}",
            "stage": "failed",
            "trace": [f"predict: 실패 — {result['error'][:60]}"],
        }

    warnings = []
    if not result.get("used_3di"):
        warnings.append(
            "3Di 캐시에 없는 단백질이라 placeholder 토큰으로 예측했습니다 "
            "(DAVIS 기준 Pearson r 약 0.023 하락에 해당)."
        )

    return {
        "prediction": result,
        "warnings": warnings,
        "stage": "compose",
        "trace": [f"predict: pKd={result.get('pKd')}"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. 답변 구성
# ══════════════════════════════════════════════════════════════════════════════

def compose_answer(state: DTIState) -> dict[str, Any]:
    """최종 사용자 답변을 만든다. 실패 경로도 여기로 모인다."""
    if state.get("error"):
        lines = [f"❌ {state['error']}"]
        if state.get("warnings"):
            lines += ["", "참고:"] + [f"  • {w}" for w in state["warnings"]]
        return {"answer": "\n".join(lines), "stage": "failed"}

    drug = state.get("drug", {})
    protein = state.get("protein", {})
    pred = state.get("prediction", {})

    lines = [
        f"**{drug.get('name', '?')} — {protein.get('gene', '?')} 결합 예측**",
        "",
        f"  예측 pKd : {pred.get('pKd')}  ({pred.get('interpretation')})",
        f"  약물     : {drug.get('name')} (CID {drug.get('cid')}, {drug.get('formula')})",
        (f"  표적     : {protein.get('name')} "
         f"[{protein.get('uniprot_id')}, {protein.get('seq_length')} aa]"),
        f"  3Di 구조 : {'사용' if pred.get('used_3di') else '미사용 (placeholder)'}",
    ]

    structure = state.get("structure", {})
    if structure.get("pdb_path"):
        lines.append(f"  단백질 3D: {structure['pdb_path']} (pLDDT {structure.get('plddt_global')})")
    ligand = state.get("ligand", {})
    if ligand.get("sdf_path"):
        lines.append(f"  리간드 3D: {ligand['sdf_path']}")

    if state.get("warnings"):
        lines += ["", "⚠️  주의:"] + [f"  • {w}" for w in state["warnings"]]

    return {"answer": "\n".join(lines), "stage": "done"}


# ══════════════════════════════════════════════════════════════════════════════
# 조건부 엣지 라우터
# ══════════════════════════════════════════════════════════════════════════════

def route_after_drug(state: DTIState) -> str:
    """약물 해석 결과에 따라 진행 / 재시도 / 포기."""
    if "error" not in state.get("drug", {}):
        return "ok"
    if state.get("drug_attempts", 0) < MAX_DRUG_ATTEMPTS:
        return "retry"
    return "give_up"


def route_after_protein(state: DTIState) -> str:
    if "error" not in state.get("protein", {}):
        return "ok"
    if state.get("protein_attempts", 0) < MAX_PROTEIN_ATTEMPTS:
        return "retry"
    return "give_up"


def mark_drug_failure(state: DTIState) -> dict[str, Any]:
    err = state.get("drug", {}).get("error", "알 수 없는 오류")
    return {
        "error": f"약물 '{state.get('drug_name')}'을(를) PubChem에서 찾지 못했습니다. {err}",
        "stage": "failed",
        "trace": ["give_up:drug"],
    }


def mark_protein_failure(state: DTIState) -> dict[str, Any]:
    err = state.get("protein", {}).get("error", "알 수 없는 오류")
    return {
        "error": f"표적 '{state.get('protein_name')}'을(를) UniProt에서 찾지 못했습니다. {err}",
        "stage": "failed",
        "trace": ["give_up:protein"],
    }
