"""
api.py — FusionDTI FastAPI Backend

REST API for Drug-Target Interaction prediction.

Endpoints:
    GET  /health                  — Health check + model status
    GET  /resolve/drug/{name}     — Drug name → SMILES via PubChem
    GET  /resolve/protein/{name}  — Protein name → sequence via UniProt
    POST /predict                 — Full DTI prediction pipeline

Run (프로젝트 루트에서):
    uvicorn web_pipeline.api:app --reload --port 8000
"""

import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent   # Capstone-Design/
sys.path.insert(0, str(ROOT))

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic FusionDTI API",
    description=(
        "Drug-Target Interaction prediction using "
        "SaProt-650M (protein encoder) + ft-ChemBERTa (drug encoder) + MLP Head. "
        "Trained on BindingDB ~80K pairs. Pearson r = 0.892 on test set."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_models_loaded = False
_load_start = None


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global _models_loaded, _load_start
    _load_start = time.time()
    print("[API] Pre-loading DTI models...")
    from tools.dti_tool import _load_models
    _load_models()
    _models_loaded = True
    print(f"[API] Models ready in {time.time() - _load_start:.1f}s")


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    drug_name: Optional[str] = None
    smiles: Optional[str] = None
    protein_name: Optional[str] = None
    sequence: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "drug_name": "Imatinib",
                    "protein_name": "ABL1",
                },
                {
                    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                    "protein_name": "PTGS1",
                },
            ]
        }
    }


class DrugInfo(BaseModel):
    query_name: str
    name: str
    cid: Optional[str] = None
    smiles: str
    formula: Optional[str] = None
    mol_weight: Optional[str] = None
    iupac_name: Optional[str] = None
    pubchem_url: Optional[str] = None
    cached: bool = False


class ProteinInfo(BaseModel):
    query_name: str
    uniprot_id: Optional[str] = None
    gene: Optional[str] = None
    name: Optional[str] = None
    organism: Optional[str] = None
    seq_length: str
    sequence: str
    reviewed: bool = False
    cached: bool = False


class PredictionResult(BaseModel):
    pKd: float
    kd_nM: Optional[float] = None
    interpretation: str
    smiles: str
    seq_length: int
    used_3di: bool


class PredictResponse(BaseModel):
    drug: dict
    protein: dict
    prediction: PredictionResult
    elapsed_seconds: float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "models_loaded": _models_loaded,
        "model_description": "SaProt-650M (FP16) + ft-ChemBERTa + MLP Head",
        "trained_on": "BindingDB ~80K pairs",
        "pearson_r": 0.8923,
    }


@app.get("/resolve/drug/{name}", tags=["Resolution"], response_model=DrugInfo)
async def resolve_drug(name: str):
    """Resolve a drug name to SMILES and properties via PubChem."""
    from tools.pubchem_tool import resolve_drug_name
    result = resolve_drug_name(name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/resolve/protein/{name}", tags=["Resolution"], response_model=ProteinInfo)
async def resolve_protein(
    name: str,
    organism_id: int = Query(default=9606, description="NCBI taxonomy ID (9606 = Homo sapiens)"),
):
    """Resolve a gene/protein name to UniProt accession and amino acid sequence."""
    from tools.uniprot_tool import resolve_protein_name
    result = resolve_protein_name(name, organism_id=organism_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/predict", tags=["Prediction"])
async def predict(req: PredictRequest):
    """
    Predict drug-target binding affinity (pKd).

    Provide either `drug_name` (resolved via PubChem) or `smiles` directly,
    and either `protein_name` (resolved via UniProt) or `sequence` directly.
    """
    t0 = time.time()

    # ── Resolve drug ──────────────────────────────────────────────────────────
    smiles = req.smiles
    drug_meta: dict = {}

    if smiles:
        drug_meta = {"name": "Custom SMILES", "smiles": smiles, "query_name": "custom"}
    elif req.drug_name:
        from tools.pubchem_tool import resolve_drug_name
        drug_meta = resolve_drug_name(req.drug_name)
        if "error" in drug_meta:
            raise HTTPException(status_code=400, detail=f"Drug resolution failed: {drug_meta['error']}")
        smiles = drug_meta["smiles"]
    else:
        raise HTTPException(status_code=422, detail="Provide either 'drug_name' or 'smiles'.")

    # ── Resolve protein ───────────────────────────────────────────────────────
    aa_seq = req.sequence
    prot_meta: dict = {}

    if aa_seq:
        aa_seq = aa_seq.upper().strip()
        prot_meta = {
            "gene": "Custom",
            "uniprot_id": "N/A",
            "query_name": "custom",
            "seq_length": str(len(aa_seq)),
            "sequence": aa_seq,
        }
    elif req.protein_name:
        from tools.uniprot_tool import resolve_protein_name
        prot_meta = resolve_protein_name(req.protein_name)
        if "error" in prot_meta:
            raise HTTPException(status_code=400, detail=f"Protein resolution failed: {prot_meta['error']}")
        aa_seq = prot_meta["sequence"]
    else:
        raise HTTPException(status_code=422, detail="Provide either 'protein_name' or 'sequence'.")

    # ── Predict ───────────────────────────────────────────────────────────────
    from tools.dti_tool import predict_binding
    result = predict_binding(smiles, aa_seq)

    if "error" in result:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {result['error']}")

    # pKd → Kd (nM)
    pkd = result["pKd"]
    kd_nM = round(10 ** (9 - pkd), 4)

    return {
        "drug": {k: v for k, v in drug_meta.items() if k != "sequence"},
        "protein": {k: v for k, v in prot_meta.items() if k != "sequence"},
        "prediction": {
            "pKd": pkd,
            "kd_nM": kd_nM,
            "interpretation": result["interpretation"],
            "smiles": smiles,
            "seq_length": len(aa_seq),
            "used_3di": result.get("used_3di", False),
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Automation endpoints
#
# n8n 워크플로가 부르는 자리다. n8n 이미지에는 Python도 torch도 없으므로
# executeCommand로 스크립트를 돌릴 수 없다 — 모델과 코드를 가진 이 컨테이너가
# HTTP로 노출하는 것이 맞다. n8n은 "언제 부를지"와 "어디로 알릴지"만 맡는다.
# ══════════════════════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    query: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "Does Imatinib bind to BCR-ABL?"}]
        }
    }


@app.post("/agent/ask", tags=["Agent"])
async def agent_ask(req: AskRequest):
    """자연어 질의를 LangGraph 에이전트에 그대로 태운다.

    /predict 와의 차이: 이쪽은 이름 해석 실패 시 재시도, OOD 경고, 구조 보강까지
    전 경로를 돈다. 실행 경로(trace)도 함께 돌려주므로 디버깅에 쓸 수 있다.
    """
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query가 비어 있습니다.")

    from services.agent.graph.graph import build_graph
    from services.agent.graph.state import new_state

    final = build_graph().invoke(new_state(req.query))

    return {
        "ok": final.get("stage") == "done",
        "stage": final.get("stage"),
        "answer": final.get("answer", ""),
        "prediction": final.get("prediction", {}),
        "drug": {k: v for k, v in final.get("drug", {}).items() if k != "sequence"},
        "protein": {k: v for k, v in final.get("protein", {}).items()
                    if k != "sequence"},
        "warnings": final.get("warnings", []),
        "trace": final.get("trace", []),
    }


@app.post("/selftest", tags=["System"])
async def run_selftest_endpoint(
    skip_agent: bool = Query(default=False,
                             description="에이전트 계층을 건너뛴다 (외부 API 미사용)"),
    skip_contracts: bool = Query(default=True,
                                 description="외부 API 계약 검사를 건너뛴다"),
    tol: float = Query(default=0.05, description="쌍별 pKd 허용 이탈"),
):
    """골든셋 회귀·드리프트 점검을 실행하고 리포트를 돌려준다.

    HTTP 상태는 항상 200이다 — 회귀는 '서버 오류'가 아니라 '보고할 사실'이고,
    n8n이 본문의 status를 보고 알림 여부를 정한다. 실행 자체가 실패하면 500.
    """
    import json as _json
    from pathlib import Path as _Path

    golden_path = ROOT / "selftest" / "golden_set.json"
    if not golden_path.exists():
        raise HTTPException(
            status_code=503,
            detail="골든셋이 없습니다. python -m selftest.build_golden_set 을 먼저 실행하세요.",
        )

    from selftest.runner import (
        _default_agent_runner,
        _default_model_predictor,
        _load_contract_callers,
        run_selftest,
    )

    golden = _json.loads(golden_path.read_text())
    baseline_path = ROOT / "selftest" / "baseline.json"
    baseline = (_json.loads(baseline_path.read_text()).get("predictions", {})
                if baseline_path.exists() else {})

    try:
        report = run_selftest(
            golden,
            baseline,
            predict_model=_default_model_predictor(),
            run_agent=None if skip_agent else _default_agent_runner(),
            contract_callers=None if skip_contracts else _load_contract_callers(),
            tol=tol,
        )
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(status_code=500,
                            detail=f"SelfTest 실행 실패: {exc}") from exc

    # 리포트를 파일로도 남긴다 — 사람이 나중에 들여다볼 수 있도록.
    report_dir = _Path(ROOT / "results" / "selftest")
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "latest.json").write_text(
            _json.dumps(report, indent=2, ensure_ascii=False))
    except OSError:
        pass        # 읽기 전용 마운트여도 리포트 반환은 계속한다

    try:
        from selftest.runner import log_to_mlflow
        log_to_mlflow(report)
    except Exception:                                           # noqa: BLE001
        pass

    return report

# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, workers=1)
