"""
dti_tool.py
===========
DTI Prediction Tool — Agent Tool #1

Given a SMILES string and an amino acid sequence, predicts the binding
affinity (pKd) using SaProt-650M (NF4 4-bit) + trained DTI MLP head.

Model: SaProt-650M-4bit + DTIHead trained on DAVIS (Pearson r=0.7914)
       checkpoint: results/SaProt-650M-4bit/dti_head.pt

The SaProt model is loaded once and reused (singleton pattern).

Usage (standalone test):
  python tools/dti_tool.py
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("pip install rdkit")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
CKPT_PATH  = ROOT / "results" / "SaProt-650M-4bit" / "dti_head.pt"
SAPROT_ID  = "westlake-repl/SaProt_650M_AF2"
PROT_DIM   = 1280
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Singleton model state ─────────────────────────────────────────────────────
_saprot    = None
_tokenizer = None
_head      = None


# ══════════════════════════════════════════════════════════════════════════════
# Model architecture (mirrors train_dti_saprot.py)
# ══════════════════════════════════════════════════════════════════════════════
class DTIHead(nn.Module):
    def __init__(self, prot_dim, drug_dim=2048, hidden=512):
        super().__init__()
        self.prot_enc = nn.Sequential(
            nn.Linear(prot_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 256), nn.GELU(),
        )
        self.drug_enc = nn.Sequential(
            nn.Linear(drug_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, 256), nn.GELU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 64),  nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, prot_emb, drug_fp):
        return self.regressor(
            torch.cat([self.prot_enc(prot_emb), self.drug_enc(drug_fp)], dim=-1)
        ).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════
def aa_to_sa(seq: str) -> str:
    """AA sequence → SA token string ('M#E#T#...')"""
    return "".join(aa + "#" for aa in seq)


def smiles_to_fp(smiles: str):
    """SMILES → 2048-bit Morgan fingerprint (radius=2)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.array(list(fp), dtype=np.float32)


def _load_models():
    """Load SaProt + DTIHead once (singleton)."""
    global _saprot, _tokenizer, _head

    if _saprot is not None:
        return  # already loaded

    print("  [DTI Tool] Loading SaProt-650M-4bit ...")
    from transformers import EsmModel, EsmTokenizer, BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    _tokenizer = EsmTokenizer.from_pretrained(SAPROT_ID)
    _saprot    = EsmModel.from_pretrained(
        SAPROT_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
        add_pooling_layer=False,
    )
    _saprot.eval()
    for p in _saprot.parameters():
        p.requires_grad_(False)

    print("  [DTI Tool] Loading DTI head ...")
    _head = DTIHead(PROT_DIM).to(DEVICE)
    _head.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
    _head.eval()
    print("  [DTI Tool] Models ready.\n")


def _encode_protein(aa_seq: str) -> torch.Tensor:
    """AA sequence → mean-pooled SaProt embedding [1, PROT_DIM]"""
    sa_seq = aa_to_sa(aa_seq)
    inputs = _tokenizer(
        sa_seq, return_tensors="pt",
        truncation=True, max_length=1024, padding=False
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out    = _saprot(**inputs)
        hidden = out.last_hidden_state[0, 1:-1, :].float()  # skip CLS, EOS
        emb    = hidden.mean(0).unsqueeze(0)                 # [1, D]
    return emb


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════
def predict_binding(smiles: str, aa_seq: str) -> dict:
    """
    Predict drug-target binding affinity (pKd).

    Args:
        smiles: Drug SMILES string
        aa_seq: Protein amino acid sequence (one-letter code)

    Returns:
        dict with keys:
          pKd, interpretation, smiles, seq_length
        or dict with key 'error' on failure
    """
    _load_models()

    # ── Drug fingerprint ──────────────────────────────────────────
    fp = smiles_to_fp(smiles)
    if fp is None:
        return {"error": f"Invalid SMILES: '{smiles}'", "smiles": smiles}
    drug_fp = torch.tensor(fp).unsqueeze(0).to(DEVICE)   # [1, 2048]

    # ── Protein embedding ─────────────────────────────────────────
    if not aa_seq or len(aa_seq) < 5:
        return {"error": "Amino acid sequence too short.", "smiles": smiles}
    prot_emb = _encode_protein(aa_seq)                    # [1, 1280]

    # ── Predict ───────────────────────────────────────────────────
    with torch.no_grad():
        pKd = _head(prot_emb, drug_fp).item()

    interpretation = _interpret_pkd(pKd)

    return {
        "smiles":         smiles,
        "seq_length":     len(aa_seq),
        "pKd":            round(pKd, 4),
        "interpretation": interpretation,
    }


def _interpret_pkd(pkd: float) -> str:
    """Qualitative interpretation of pKd value."""
    if pkd >= 9.0:
        return "Very strong binding (pKd ≥ 9.0, Kd ≤ 1 nM)"
    elif pkd >= 7.0:
        return "Strong binding (pKd 7–9, Kd 1–100 nM)"
    elif pkd >= 5.0:
        return "Moderate binding (pKd 5–7, Kd 0.1–10 µM)"
    else:
        return "Weak / no significant binding (pKd < 5)"


def format_result(r: dict) -> str:
    """Human-readable summary for Agent output."""
    if "error" in r:
        return f"[DTI Tool] Error: {r['error']}"
    return (
        f"[DTI Tool]\n"
        f"  pKd          : {r['pKd']}\n"
        f"  Interpretation: {r['interpretation']}\n"
        f"  Drug SMILES  : {r['smiles'][:60]}{'...' if len(r['smiles']) > 60 else ''}\n"
        f"  Protein length: {r['seq_length']} aa"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Standalone test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Imatinib SMILES + ABL1 sequence (first 100 aa for quick test)
    TEST_SMILES = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
    TEST_SEQ    = (
        "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGD"
        "NTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQR"
        "SISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHK"
    )

    print("=" * 55)
    print("  DTI Tool — Standalone Test")
    print("  Drug : Imatinib")
    print("  Target: ABL1 (partial sequence)")
    print("=" * 55)
    result = predict_binding(TEST_SMILES, TEST_SEQ)
    print(format_result(result))
