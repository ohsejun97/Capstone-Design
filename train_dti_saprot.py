"""
train_dti_saprot.py
====================
SaProt + DTI 회귀 헤드를 DAVIS 연속 pKd 데이터로 학습 및 평가

실험 구성:
  기준 모델:   SaProt-650M (frozen) + DTI MLP 헤드  → Pearson r 측정
  경량 모델:   SaProt-35M  (frozen) + DTI MLP 헤드  → Pearson r 측정
  양자화 모델: SaProt-650M 4-bit   + DTI MLP 헤드  → Pearson r 측정

핵심 전략:
  SaProt 인코더는 완전히 frozen (기울기 없음)
  훈련 대상: DTI 헤드(소형 MLP)만 → GTX 1650 4GB에서 10~15분 내 완료 가능

사용법:
  python train_dti_saprot.py --encoder 650M              # 기준 모델
  python train_dti_saprot.py --encoder 35M               # 경량 모델
  python train_dti_saprot.py --encoder 650M --quant 4bit # 양자화 모델
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import EsmModel, EsmTokenizer

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("❌ pip install rdkit")

try:
    from DeepPurpose.dataset import load_process_DAVIS
except ImportError:
    sys.exit("❌ pip install DeepPurpose")

# ══════════════════════════════════════════════════════════════════════════════
# 인자 파싱
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="SaProt DTI Trainer")
parser.add_argument("--encoder", default="650M", choices=["650M", "35M"],
                    help="SaProt 인코더 크기")
parser.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"],
                    help="양자화 수준 (none: FP32/FP16)")
parser.add_argument("--epochs",     type=int,   default=50)
parser.add_argument("--batch_size", type=int,   default=128)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--patience",   type=int,   default=10,
                    help="Early stopping patience")
parser.add_argument("--seed",       type=int,   default=42)
args = parser.parse_args()

# ── 재현성 ────────────────────────────────────────────────────────────────────
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_num_threads(32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 모델 ID 매핑 ──────────────────────────────────────────────────────────────
SAPROT_IDS = {
    "650M": "westlake-repl/SaProt_650M_AF2",
    "35M":  "westlake-repl/SaProt_35M_AF2",
}
SAPROT_DIMS = {"650M": 1280, "35M": 480}  # ESM2 hidden size

run_name = f"SaProt-{args.encoder}"
if args.quant != "none":
    run_name += f"-{args.quant}"

print("=" * 60)
print(f"  DTI Training — {run_name}")
print(f"  실행: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Device: {DEVICE} | Encoder: {args.encoder} | Quant: {args.quant}")
print("=" * 60, "\n")

# ══════════════════════════════════════════════════════════════════════════════
# [1] DAVIS 데이터 로드 (연속 pKd)
# ══════════════════════════════════════════════════════════════════════════════
print("[1] DAVIS 연속 pKd 데이터 로드...")
X_drugs, X_targets, y = load_process_DAVIS(
    path="./data", binary=False, convert_to_log=True
)
y = np.array(y, dtype=np.float32)
print(f"    전체: {len(y):,}쌍  |  pKd: {y.min():.2f} ~ {y.max():.2f}")

# Train/Val/Test 분할 (70/10/20, seed=42)
rng   = np.random.default_rng(args.seed)
idx   = rng.permutation(len(y))
n_tr  = int(len(y) * 0.70)
n_val = int(len(y) * 0.10)
tr_idx  = idx[:n_tr]
val_idx = idx[n_tr:n_tr + n_val]
te_idx  = idx[n_tr + n_val:]
print(f"    Train: {len(tr_idx):,}  Val: {len(val_idx):,}  Test: {len(te_idx):,}\n")

# ══════════════════════════════════════════════════════════════════════════════
# [2] SaProt 로드 (선택적 양자화)
# ══════════════════════════════════════════════════════════════════════════════
print(f"[2] SaProt-{args.encoder} 로드 (quant={args.quant})...")
model_id  = SAPROT_IDS[args.encoder]
prot_dim  = SAPROT_DIMS[args.encoder]
tokenizer = EsmTokenizer.from_pretrained(model_id)

if args.quant == "4bit":
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    saprot = EsmModel.from_pretrained(
        model_id, quantization_config=bnb_cfg, device_map="auto",
        low_cpu_mem_usage=True, add_pooling_layer=False,
    )
elif args.quant == "8bit":
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    saprot = EsmModel.from_pretrained(
        model_id, quantization_config=bnb_cfg, device_map="auto",
        low_cpu_mem_usage=True, add_pooling_layer=False,
    )
else:
    saprot = EsmModel.from_pretrained(
        model_id, low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).to(DEVICE)

saprot.eval()
for p in saprot.parameters():
    p.requires_grad_(False)

n_params = sum(p.numel() for p in saprot.parameters()) / 1e6
print(f"    ✅ {n_params:.0f}M params, frozen\n")

# ══════════════════════════════════════════════════════════════════════════════
# [3] 단백질 임베딩 사전 계산 (unique 379개, 캐시)
#     AA 시퀀스 → SA 포맷 ("P#F#W#...") → SaProt → mean pool → [prot_dim]
# ══════════════════════════════════════════════════════════════════════════════
def aa_to_sa(seq: str) -> str:
    """표준 AA 시퀀스 → SaProt SA 포맷 (구조 없으면 '#' 대체)"""
    return "".join(aa + "#" for aa in seq)

cache_path = Path(f"./cache/prot_embs_{args.encoder}_{args.quant}.pt")
cache_path.parent.mkdir(exist_ok=True)

unique_targets = list(dict.fromkeys(X_targets))   # 순서 유지 고유 목록
tgt2idx = {t: i for i, t in enumerate(unique_targets)}

if cache_path.exists():
    print(f"[3] 단백질 임베딩 캐시 로드: {cache_path}")
    prot_embs = torch.load(cache_path, weights_only=True)
else:
    print(f"[3] 단백질 임베딩 사전 계산 ({len(unique_targets)}개)...")
    t0 = time.time()
    prot_embs = torch.zeros(len(unique_targets), prot_dim, dtype=torch.float32)

    with torch.no_grad():
        for i, seq in enumerate(unique_targets):
            sa_seq  = aa_to_sa(seq)
            inputs  = tokenizer(
                sa_seq, return_tensors="pt",
                truncation=True, max_length=1024, padding=False,
            )
            inputs  = {k: v.to(DEVICE) for k, v in inputs.items()}
            out     = saprot(**inputs)
            # CLS/EOS 제외 mean pool → [prot_dim]
            hidden  = out.last_hidden_state[0, 1:-1, :].float()
            prot_embs[i] = hidden.mean(0).cpu()

            if (i + 1) % 50 == 0 or i == len(unique_targets) - 1:
                elapsed = time.time() - t0
                eta     = elapsed / (i + 1) * (len(unique_targets) - i - 1)
                print(f"    {i+1}/{len(unique_targets)}  "
                      f"({elapsed:.0f}s 경과, ETA {eta:.0f}s)")

    torch.save(prot_embs, cache_path)
    print(f"    ✅ 캐시 저장: {cache_path}\n")

# 각 샘플의 단백질 인덱스
tgt_indices = np.array([tgt2idx[t] for t in X_targets])

# ══════════════════════════════════════════════════════════════════════════════
# [4] 약물 Morgan Fingerprint 계산
# ══════════════════════════════════════════════════════════════════════════════
print("[4] 약물 Morgan FP 계산 (2048-bit)...")

def smiles_to_fp(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.array(list(fp), dtype=np.float32)

unique_drugs  = list(dict.fromkeys(X_drugs))
drug2idx      = {d: i for i, d in enumerate(unique_drugs)}
drug_fps      = np.zeros((len(unique_drugs), 2048), dtype=np.float32)
n_invalid     = 0

for i, smi in enumerate(unique_drugs):
    fp = smiles_to_fp(smi)
    if fp is not None:
        drug_fps[i] = fp
    else:
        n_invalid += 1

drug_fps     = torch.tensor(drug_fps)
drug_indices = np.array([drug2idx[d] for d in X_drugs])
print(f"    ✅ {len(unique_drugs)}개 약물 | 유효하지 않은 SMILES: {n_invalid}개\n")

# ══════════════════════════════════════════════════════════════════════════════
# [5] DTI 데이터셋 & 데이터로더
# ══════════════════════════════════════════════════════════════════════════════
class DTIDataset(Dataset):
    def __init__(self, indices):
        self.prot_idx = tgt_indices[indices]
        self.drug_idx = drug_indices[indices]
        self.labels   = y[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (
            prot_embs[self.prot_idx[i]],          # [prot_dim]
            drug_fps[self.drug_idx[i]],             # [2048]
            torch.tensor(self.labels[i], dtype=torch.float32),
        )

train_loader = DataLoader(DTIDataset(tr_idx),  batch_size=args.batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(DTIDataset(val_idx), batch_size=256,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(DTIDataset(te_idx),  batch_size=256,
                          shuffle=False, num_workers=2, pin_memory=True)

# ══════════════════════════════════════════════════════════════════════════════
# [6] DTI 헤드 정의
#     단백질 임베딩 [prot_dim] + 약물 FP [2048] → pKd 회귀값
# ══════════════════════════════════════════════════════════════════════════════
class DTIHead(nn.Module):
    def __init__(self, prot_dim: int, drug_dim: int = 2048, hidden: int = 512):
        super().__init__()
        self.prot_enc = nn.Sequential(
            nn.Linear(prot_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 256),
            nn.GELU(),
        )
        self.drug_enc = nn.Sequential(
            nn.Linear(drug_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, 256),
            nn.GELU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, prot_emb, drug_fp):
        p   = self.prot_enc(prot_emb)                      # [B, 256]
        d   = self.drug_enc(drug_fp)                       # [B, 256]
        out = self.regressor(torch.cat([p, d], dim=-1))    # [B, 1]
        return out.squeeze(-1)                             # [B]

head = DTIHead(prot_dim).to(DEVICE)
n_head_params = sum(p.numel() for p in head.parameters()) / 1e6
print(f"[5] DTI 헤드: {n_head_params:.2f}M params (훈련 대상)\n")

# ══════════════════════════════════════════════════════════════════════════════
# [7] 훈련
# ══════════════════════════════════════════════════════════════════════════════
optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-5
)
criterion = nn.HuberLoss(delta=1.0)   # MSE보다 이상치 강건

best_val_r   = -1.0
best_state   = None
patience_cnt = 0
history      = []

print(f"[6] 훈련 시작 (epochs={args.epochs}, batch={args.batch_size}, lr={args.lr})")
print(f"    {'Epoch':>5} | {'Train Loss':>10} | {'Val r':>7} | {'Best':>7}")
print("    " + "-" * 42)

t_train_start = time.time()

for epoch in range(1, args.epochs + 1):
    # ── train ──
    head.train()
    train_loss = 0.0
    for prot, drug, label in train_loader:
        prot, drug, label = prot.to(DEVICE), drug.to(DEVICE), label.to(DEVICE)
        pred  = head(prot, drug)
        loss  = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * len(label)
    train_loss /= len(tr_idx)
    scheduler.step()

    # ── validate ──
    head.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for prot, drug, label in val_loader:
            prot, drug = prot.to(DEVICE), drug.to(DEVICE)
            pred = head(prot, drug).cpu().numpy()
            val_preds.extend(pred)
            val_labels.extend(label.numpy())
    val_r, _ = pearsonr(val_preds, val_labels)

    is_best = val_r > best_val_r
    if is_best:
        best_val_r = val_r
        best_state = {k: v.clone() for k, v in head.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1

    history.append({"epoch": epoch, "train_loss": train_loss, "val_r": val_r})

    marker = " ★" if is_best else ""
    print(f"    {epoch:>5} | {train_loss:>10.4f} | {val_r:>7.4f} | "
          f"{best_val_r:>7.4f}{marker}")

    if patience_cnt >= args.patience:
        print(f"\n    Early stopping (patience={args.patience})")
        break

t_train_end = time.time()
train_time  = t_train_end - t_train_start

# ══════════════════════════════════════════════════════════════════════════════
# [8] 테스트 평가 (best checkpoint)
# ══════════════════════════════════════════════════════════════════════════════
head.load_state_dict(best_state)
head.eval()

test_preds, test_labels = [], []
with torch.no_grad():
    for prot, drug, label in test_loader:
        prot, drug = prot.to(DEVICE), drug.to(DEVICE)
        pred = head(prot, drug).cpu().numpy()
        test_preds.extend(pred)
        test_labels.extend(label.numpy())

test_r, test_p = pearsonr(test_preds, test_labels)

# ══════════════════════════════════════════════════════════════════════════════
# [9] 결과 저장
# ══════════════════════════════════════════════════════════════════════════════
out_dir = Path("./results") / run_name
out_dir.mkdir(parents=True, exist_ok=True)

# 예측값 저장
pd.DataFrame({
    "y_pred": test_preds,
    "y_true": test_labels,
}).to_csv(out_dir / "test_predictions.csv", index=False)

# 학습 곡선 저장
pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

# 모델 저장
torch.save(best_state, out_dir / "dti_head.pt")

# 결과 요약 JSON
result = {
    "run_name":       run_name,
    "encoder":        args.encoder,
    "quant":          args.quant,
    "prot_dim":       prot_dim,
    "timestamp":      datetime.now().isoformat(),
    "test_pearson_r": float(test_r),
    "test_p_value":   float(test_p),
    "best_val_r":     float(best_val_r),
    "epochs_trained": len(history),
    "train_time_sec": round(train_time, 1),
    "n_test":         len(te_idx),
    "n_train":        len(tr_idx),
}
with open(out_dir / "result.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n{'='*50}")
print(f"  모델:      {run_name}")
print(f"  Pearson R: {test_r:+.4f}  (p={test_p:.2e})")
print(f"  Val best:  {best_val_r:.4f}")
print(f"  학습 시간: {train_time:.0f}초")
if test_r >= 0.8:
    print("  ✅ 목표 달성! (r ≥ 0.8)")
elif test_r >= 0.6:
    print("  △  양호 (r ≥ 0.6)")
else:
    print("  ❌ 성능 미달")
print(f"  결과 저장: {out_dir}/")
print("=" * 50)
print("\n[완료]")
