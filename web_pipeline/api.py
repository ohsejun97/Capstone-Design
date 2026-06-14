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


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, workers=1)
