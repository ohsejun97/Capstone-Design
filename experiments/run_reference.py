"""
run_reference.py
================
SaProt-650M + SPRINT 아키텍처를 이용한 DAVIS DTI Reference Score 생성
(panspecies-dti 체크포인트 구조와 1:1 매칭)

아키텍처 (체크포인트 분석 결과):
    단백질: SA 시퀀스 → SaProt-650M (EsmModel) → 4-head attention pooling
            → LN + LeakyReLU + Linear(1280→1024) × 3 + LN → L2 normalize
    약물:   SMILES → Morgan FP (2048-bit)
            → Linear(2048→1260→1024→1024) + BN + LeakyReLU → L2 normalize
    융합:   코사인 유사도 (NoSigmoid)

V1 실패 원인:
    ① AutoModelForSequenceClassification: 랜덤 헤드 사용
    ② tokenizer(protein, SMILES): SMILES를 단백질 토크나이저에 입력
    ③ DTI 파인튜닝 없는 순수 PLM 사용

환경변수:
    SPRINT_CHECKPOINT   가중치 파일 경로 (기본: ./sprint_weights.ckpt)
    HF_TOKEN            Hugging Face 토큰 (gated model 접근 시)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

# ── 선택적 의존성 ─────────────────────────────────────────────────────────────
try:
    from transformers import EsmModel, EsmTokenizer
except ImportError:
    sys.exit("❌ pip install transformers")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")   # RDKit deprecation 경고 억제
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    sys.exit("❌ pip install rdkit")

try:
    from scipy.stats import pearsonr
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
torch.set_num_threads(32)

DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAPROT_MODEL      = "westlake-repl/SaProt_650M_AF2"
MAX_PROTEIN_LEN   = 1024
PROTEIN_DIM       = 1280   # SaProt-650M hidden dim
DRUG_DIM          = 2048   # Morgan FP bits
LATENT_DIM        = 1024   # SPRINT latent dim (체크포인트 확인)
NUM_HEADS_AGG     = 4      # 어텐션 풀링 헤드 수 (hyper_parameters 확인)

HF_TOKEN          = os.environ.get("HF_TOKEN", None)
SPRINT_CHECKPOINT = os.environ.get("SPRINT_CHECKPOINT", "sprint_weights.ckpt")

print("=" * 60)
print("  Agentic FusionDTI — Reference Score Pipeline v2")
print("=" * 60)
print(f"  Device     : {DEVICE}")
print(f"  CUDA       : {torch.cuda.is_available()}")
print(f"  Checkpoint : {SPRINT_CHECKPOINT}")
print("=" * 60, "\n")

# ══════════════════════════════════════════════════════════════════════════════
# [1] 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
print("[1] 데이터 로드...")
df = pd.read_csv("davis_test.csv", index_col=0)
print(f"    {len(df)} 샘플\n")

# ══════════════════════════════════════════════════════════════════════════════
# [2] SaProt-650M 로드 (EsmModel — 순수 인코더, LM head 없음)
# ══════════════════════════════════════════════════════════════════════════════
print("[2] SaProt-650M 로드...")
_hf = {"token": HF_TOKEN} if HF_TOKEN else {}
try:
    tokenizer = EsmTokenizer.from_pretrained(SAPROT_MODEL, **_hf)
    saprot    = EsmModel.from_pretrained(SAPROT_MODEL, low_cpu_mem_usage=True, **_hf).to(DEVICE)
    saprot.eval()
    print(f"    ✅ 로드 완료 ({sum(p.numel() for p in saprot.parameters())/1e6:.0f}M params)\n")
except Exception as e:
    print(f"    ❌ 실패: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# [3] SPRINT 아키텍처 정의 (체크포인트 키/shape와 1:1 매칭)
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttentionPool(nn.Module):
    """
    panspecies-dti AggregateAggregation
    - CLS 토큰이 단백질 시퀀스를 query → 고정 크기 벡터로 압축
    - keys: target_projector.0.proj.*
    """
    def __init__(self, dim: int = 1280, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.q         = nn.Linear(dim, dim, bias=False)
        self.k         = nn.Linear(dim, dim, bias=False)
        self.v         = nn.Linear(dim, dim, bias=False)
        self.proj      = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        cls = self.cls_token.expand(B, -1, -1)          # [B, 1, D]
        q   = self.q(cls).reshape(B, 1, H, Dh).transpose(1, 2)   # [B,H,1,Dh]
        k   = self.k(x).reshape(B, N, H, Dh).transpose(1, 2)     # [B,H,N,Dh]
        v   = self.v(x).reshape(B, N, H, Dh).transpose(1, 2)     # [B,H,N,Dh]

        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)  # [B,H,1,N]
        out  = (attn @ v).transpose(1, 2).reshape(B, 1, D).squeeze(1)       # [B,D]
        return self.proj(out)


class TargetProjector(nn.Module):
    """
    단백질 인코더 헤드
    keys: target_projector.0.proj.*, target_projector.0.res.*
    activation: LeakyReLU (hyper_parameters 확인)
    """
    def __init__(self):
        super().__init__()
        act = nn.LeakyReLU()

        self.proj = MultiHeadAttentionPool(PROTEIN_DIM, NUM_HEADS_AGG)
        self.res  = nn.Sequential(
            nn.LayerNorm(PROTEIN_DIM),   # .0
            nn.LeakyReLU(),              # .1  (파라미터 없어 state_dict 미등록)
            nn.Linear(PROTEIN_DIM, LATENT_DIM),  # .2
            nn.LeakyReLU(),              # .3
            nn.LayerNorm(LATENT_DIM),    # .4
            nn.LeakyReLU(),              # .5
            nn.Linear(LATENT_DIM, LATENT_DIM),   # .6
            nn.LeakyReLU(),              # .7
            nn.LayerNorm(LATENT_DIM),    # .8
            nn.LeakyReLU(),              # .9
            nn.Linear(LATENT_DIM, LATENT_DIM),   # .10
            nn.LeakyReLU(),              # .11
            nn.LayerNorm(LATENT_DIM),    # .12
        )

    def forward(self, token_embs: torch.Tensor) -> torch.Tensor:
        pooled = self.proj(token_embs)       # [B, 1280]
        return F.normalize(self.res(pooled), dim=-1)  # [B, 1024]


class DrugProjector(nn.Module):
    """
    약물 인코더 헤드
    keys: drug_projector.linear1~4, drug_projector.bn1~3
    activation: LeakyReLU
    """
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(DRUG_DIM, 1260)
        self.bn1     = nn.BatchNorm1d(1260)
        self.linear2 = nn.Linear(1260, LATENT_DIM)
        self.bn2     = nn.BatchNorm1d(LATENT_DIM)
        self.linear3 = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.bn3     = nn.BatchNorm1d(LATENT_DIM)
        self.linear4 = nn.Linear(LATENT_DIM, LATENT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn1(self.linear1(x)))
        x = F.leaky_relu(self.bn2(self.linear2(x)))
        x = F.leaky_relu(self.bn3(self.linear3(x)))
        return F.normalize(self.linear4(x), dim=-1)  # [B, 1024]

# ══════════════════════════════════════════════════════════════════════════════
# [4] 가중치 로드
# ══════════════════════════════════════════════════════════════════════════════
target_proj = TargetProjector().to(DEVICE)
drug_proj   = DrugProjector().to(DEVICE)

print("[3] SPRINT 가중치 로드...")
if os.path.exists(SPRINT_CHECKPOINT):
    ckpt  = torch.load(SPRINT_CHECKPOINT, map_location=DEVICE, weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    # ── 단백질 투영 헤드 (target_projector.0.* → target_proj.*) ──
    t_state = {
        k.replace("target_projector.0.", ""): v
        for k, v in state.items()
        if k.startswith("target_projector.0.")
    }
    missing_t, unexpected_t = target_proj.load_state_dict(t_state, strict=False)
    print(f"    target_proj | matched: {len(t_state) - len(missing_t)}/{len(t_state)}")
    if missing_t:
        print(f"    missing keys: {missing_t}")

    # ── 약물 투영 헤드 (drug_projector.* → drug_proj.*) ──
    d_state = {
        k.replace("drug_projector.", ""): v
        for k, v in state.items()
        if k.startswith("drug_projector.")
    }
    missing_d, unexpected_d = drug_proj.load_state_dict(d_state, strict=False)
    print(f"    drug_proj   | matched: {len(d_state) - len(missing_d)}/{len(d_state)}")
    if missing_d:
        print(f"    missing keys: {missing_d}")

    print("    ✅ 가중치 로드 완료\n")
else:
    print(f"    ❌ {SPRINT_CHECKPOINT} 없음 → zero-shot (Pearson r 낮음)\n")

target_proj.eval()
drug_proj.eval()

# ══════════════════════════════════════════════════════════════════════════════
# [5] 인코딩 함수
# ══════════════════════════════════════════════════════════════════════════════
def encode_protein(sa_seq: str) -> torch.Tensor:
    """
    SA 시퀀스 → SaProt 토큰별 임베딩 [seq_len, 1280]
    CLS(0)·EOS(-1) 제외 후 반환 (어텐션 풀링은 target_proj에서 처리)
    """
    inputs = tokenizer(
        sa_seq,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROTEIN_LEN,
        padding=False,
    ).to(DEVICE)
    with torch.no_grad():
        out = saprot(**inputs)
    # last_hidden_state: [1, seq_len, 1280] → [1, seq_len-2, 1280]
    return out.last_hidden_state[:, 1:-1, :]   # CLS/EOS 제외


def encode_drug(smiles: str):
    """SMILES → Morgan FP [1, 2048]"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=DRUG_DIM)
    return torch.tensor(list(fp), dtype=torch.float32).unsqueeze(0).to(DEVICE)


def score_pair(prot_tokens: torch.Tensor, drug_fp: torch.Tensor) -> float:
    """코사인 유사도 스코어 (범위: -1 ~ 1)"""
    with torch.no_grad():
        p = target_proj(prot_tokens)   # [1, 1024]
        d = drug_proj(drug_fp)         # [1, 1024]
        return (p * d).sum().item()    # cosine sim (L2 정규화 완료)

# ══════════════════════════════════════════════════════════════════════════════
# [6] 추론 루프
# ══════════════════════════════════════════════════════════════════════════════
print(f"[4] 추론 시작 ({len(df)} 샘플, device={DEVICE})...\n")
results   = []
n_errors  = 0

with torch.no_grad():
    for i, row in tqdm(df.iterrows(), total=len(df), desc="DTI Inference"):
        try:
            prot_tokens = encode_protein(row["Target Sequence"])
            drug_fp     = encode_drug(row["SMILES"])
            if drug_fp is None:
                n_errors += 1
                continue

            score = score_pair(prot_tokens, drug_fp)
            results.append({
                "smiles":          row["SMILES"],
                "reference_score": score,
                "label":           row["Label"],
            })
        except KeyboardInterrupt:
            print("\n⚠️  중단됨 — 현재까지 결과 저장...")
            break
        except Exception:
            n_errors += 1
            continue

# ══════════════════════════════════════════════════════════════════════════════
# [7] 결과 저장 및 Pearson r 계산
# ══════════════════════════════════════════════════════════════════════════════
df_out = pd.DataFrame(results)
df_out.to_csv("reference_scores_osj.csv", index=False)

print(f"\n[5] 저장 완료 → reference_scores_osj.csv")
print(f"    성공: {len(results)}개 | 오류: {n_errors}개")

if SCIPY_OK:
    valid = df_out["label"].notna()
    if valid.sum() > 10:
        r, p_val = pearsonr(df_out.loc[valid, "reference_score"],
                            df_out.loc[valid, "label"])
        print(f"\n{'='*40}")
        print(f"  Pearson R : {r:+.4f}")
        print(f"  p-value   : {p_val:.2e}")
        print(f"  샘플 수   : {int(valid.sum())}")
        if r >= 0.8:
            print("  ✅ 목표 달성! (r ≥ 0.8)")
        elif r >= 0.5:
            print("  △  중간 상관 — 추가 파인튜닝 권장")
        else:
            print("  ❌ 낮은 상관")
        print("=" * 40)

print("\n[완료]")
