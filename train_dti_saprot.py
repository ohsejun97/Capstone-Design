"""
train_dti_saprot.py  (v2 — LoRA 지원)
======================================
SaProt + DTI 회귀 헤드를 DAVIS 연속 pKd 데이터로 학습 및 평가

모드:
  frozen  : SaProt 완전 고정, 임베딩 캐시 사용 → DTI 헤드만 학습 (빠름, ~1분)
  LoRA    : SaProt 어텐션에 rank-16 어댑터 삽입 → SaProt + 헤드 함께 학습 (느림, 수 시간)

사용법:
  python train_dti_saprot.py --encoder 650M                       # frozen 기준
  python train_dti_saprot.py --encoder 35M                        # frozen 경량
  python train_dti_saprot.py --encoder 650M --quant 4bit          # frozen 4bit
  python train_dti_saprot.py --encoder 650M --lora                # LoRA 기준
  python train_dti_saprot.py --encoder 35M  --lora                # LoRA 경량  ← 핵심 실험
  python train_dti_saprot.py --encoder 650M --quant 4bit --lora   # LoRA 4bit
"""

import os, sys, time, json, argparse
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
    import DeepPurpose.dataset as dp_dataset
except ImportError:
    sys.exit("❌ pip install DeepPurpose")

# ══════════════════════════════════════════════════════════════════════════════
# 인자 파싱
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="SaProt DTI Trainer")
parser.add_argument("--dataset",    default="davis", choices=["davis", "kiba"],
                    help="Training dataset (default: davis)")
parser.add_argument("--encoder",    default="650M", choices=["650M", "35M"])
parser.add_argument("--quant",      default="none", choices=["none", "8bit", "4bit"])
parser.add_argument("--lora",       action="store_true")
parser.add_argument("--lora_r",     type=int,   default=16)
parser.add_argument("--lora_alpha", type=int,   default=32)
parser.add_argument("--epochs",     type=int,   default=50)
parser.add_argument("--batch_size", type=int,   default=0,
                    help="0=auto (frozen:128, LoRA 650M:8, LoRA 35M:32)")
parser.add_argument("--lr",         type=float, default=0.0,
                    help="0=auto (frozen:1e-3, LoRA:5e-5)")
parser.add_argument("--patience",   type=int,   default=10)
parser.add_argument("--seed",       type=int,   default=42)
args = parser.parse_args()

# 자동 기본값
if args.batch_size == 0:
    if args.lora:
        args.batch_size = 8 if args.encoder == "650M" else 32
    else:
        args.batch_size = 128
if args.lr == 0.0:
    args.lr = 5e-5 if args.lora else 1e-3

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_num_threads(32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAPROT_IDS  = {"650M": "westlake-repl/SaProt_650M_AF2",
               "35M":  "westlake-repl/SaProt_35M_AF2"}
SAPROT_DIMS = {"650M": 1280, "35M": 480}

run_name = f"SaProt-{args.encoder}"
if args.quant != "none": run_name += f"-{args.quant}"
if args.lora:            run_name += "-lora"
run_name += f"-{args.dataset}"

print("=" * 60)
print(f"  DTI Training — {run_name}")
print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Device: {DEVICE} | Dataset: {args.dataset.upper()} | "
      f"Encoder: {args.encoder} | Quant: {args.quant}")
print(f"  batch={args.batch_size} | lr={args.lr}")
print("=" * 60, "\n")

# ══════════════════════════════════════════════════════════════════════════════
# [1] 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
if args.dataset == "davis":
    print("[1] Loading DAVIS (continuous pKd)...")
    X_drugs, X_targets, y = dp_dataset.load_process_DAVIS(
        path="./data", binary=False, convert_to_log=True
    )
else:
    print("[1] Loading KIBA (KIBA score)...")
    X_drugs, X_targets, y = dp_dataset.load_process_KIBA(
        path="./data", binary=False
    )

y = np.array(y, dtype=np.float32)
print(f"    Total: {len(y):,} pairs  |  target: {y.min():.2f} ~ {y.max():.2f}")

rng    = np.random.default_rng(args.seed)
idx    = rng.permutation(len(y))
n_tr   = int(len(y) * 0.70)
n_val  = int(len(y) * 0.10)
tr_idx  = idx[:n_tr]
val_idx = idx[n_tr:n_tr + n_val]
te_idx  = idx[n_tr + n_val:]
print(f"    Train: {len(tr_idx):,}  Val: {len(val_idx):,}  Test: {len(te_idx):,}\n")

# ══════════════════════════════════════════════════════════════════════════════
# [2] SaProt 로드
# ══════════════════════════════════════════════════════════════════════════════
print(f"[2] SaProt-{args.encoder} 로드 (quant={args.quant}, lora={args.lora})...")
model_id  = SAPROT_IDS[args.encoder]
prot_dim  = SAPROT_DIMS[args.encoder]
tokenizer = EsmTokenizer.from_pretrained(model_id)

if args.quant == "4bit":
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    saprot = EsmModel.from_pretrained(
        model_id, quantization_config=bnb_cfg,
        device_map="auto", low_cpu_mem_usage=True, add_pooling_layer=False,
    )
elif args.quant == "8bit":
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    saprot = EsmModel.from_pretrained(
        model_id, quantization_config=bnb_cfg,
        device_map="auto", low_cpu_mem_usage=True, add_pooling_layer=False,
    )
else:
    saprot = EsmModel.from_pretrained(
        model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16,
    ).to(DEVICE)

# ── LoRA 적용 ─────────────────────────────────────────────────────────────────
if args.lora:
    from peft import LoraConfig, get_peft_model, TaskType
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05,
        bias="none",
        # FEATURE_EXTRACTION: 분류 헤드 없는 인코더 전용
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    saprot = get_peft_model(saprot, lora_cfg)
    saprot.print_trainable_parameters()
    saprot.train()
    # 650M에서 VRAM 절약을 위한 gradient checkpointing
    if args.encoder == "650M" and args.quant == "none":
        saprot.enable_input_require_grads()
        saprot.base_model.model.gradient_checkpointing_enable()
    saprot = saprot.to(DEVICE)
else:
    saprot.eval()
    for p in saprot.parameters():
        p.requires_grad_(False)

n_params     = sum(p.numel() for p in saprot.parameters()) / 1e6
n_trainable  = sum(p.numel() for p in saprot.parameters() if p.requires_grad) / 1e6
print(f"    ✅ {n_params:.0f}M params total | {n_trainable:.2f}M trainable\n")

# ══════════════════════════════════════════════════════════════════════════════
# [3] 단백질 처리 — frozen: 임베딩 캐시 / LoRA: 토큰 사전 계산
# ══════════════════════════════════════════════════════════════════════════════
def aa_to_sa(seq: str) -> str:
    return "".join(aa + "#" for aa in seq)

unique_targets = list(dict.fromkeys(X_targets))
tgt2idx        = {t: i for i, t in enumerate(unique_targets)}
tgt_indices    = np.array([tgt2idx[t] for t in X_targets])

if not args.lora:
    # ── frozen 모드: 임베딩 캐시 ─────────────────────────────────────────────
    cache_path = Path(f"./cache/prot_embs_{args.encoder}_{args.quant}.pt")
    cache_path.parent.mkdir(exist_ok=True)

    if cache_path.exists():
        print(f"[3] 단백질 임베딩 캐시 로드: {cache_path}")
        prot_embs = torch.load(cache_path, weights_only=True)
    else:
        print(f"[3] 단백질 임베딩 사전 계산 ({len(unique_targets)}개)...")
        t0 = time.time()
        prot_embs = torch.zeros(len(unique_targets), prot_dim, dtype=torch.float32)
        with torch.no_grad():
            for i, seq in enumerate(unique_targets):
                inputs = tokenizer(aa_to_sa(seq), return_tensors="pt",
                                   truncation=True, max_length=1024, padding=False)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                out    = saprot(**inputs)
                hidden = out.last_hidden_state[0, 1:-1, :].float()
                prot_embs[i] = hidden.mean(0).cpu()
                if (i + 1) % 50 == 0 or i == len(unique_targets) - 1:
                    elapsed = time.time() - t0
                    eta     = elapsed / (i + 1) * (len(unique_targets) - i - 1)
                    print(f"    {i+1}/{len(unique_targets)}  "
                          f"({elapsed:.0f}s 경과, ETA {eta:.0f}s)")
        torch.save(prot_embs, cache_path)
        print(f"    ✅ 캐시 저장: {cache_path}\n")

else:
    # ── LoRA 모드: 토큰 사전 계산 (실제 임베딩은 학습 중 계산) ─────────────
    print(f"[3] 단백질 토큰 사전 계산 ({len(unique_targets)}개)...")
    all_input_ids  = []
    all_attn_masks = []
    MAX_LEN = 512   # 650M + LoRA VRAM 제약
    for seq in unique_targets:
        enc = tokenizer(aa_to_sa(seq), return_tensors="pt",
                        truncation=True, max_length=MAX_LEN, padding=False)
        all_input_ids.append(enc["input_ids"][0])
        all_attn_masks.append(enc["attention_mask"][0])
    print(f"    ✅ 토큰화 완료 (max_len={MAX_LEN})\n")

# ══════════════════════════════════════════════════════════════════════════════
# [4] 약물 Morgan Fingerprint
# ══════════════════════════════════════════════════════════════════════════════
print("[4] 약물 Morgan FP 계산 (2048-bit)...")

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.array(list(fp), dtype=np.float32)

unique_drugs  = list(dict.fromkeys(X_drugs))
drug2idx      = {d: i for i, d in enumerate(unique_drugs)}
drug_fps      = np.zeros((len(unique_drugs), 2048), dtype=np.float32)
n_invalid     = 0
for i, smi in enumerate(unique_drugs):
    fp = smiles_to_fp(smi)
    if fp is not None: drug_fps[i] = fp
    else: n_invalid += 1
drug_fps     = torch.tensor(drug_fps)
drug_indices = np.array([drug2idx[d] for d in X_drugs])
print(f"    ✅ {len(unique_drugs)}개 약물 | 유효하지 않은 SMILES: {n_invalid}개\n")

# ══════════════════════════════════════════════════════════════════════════════
# [5] 데이터셋 & 데이터로더
# ══════════════════════════════════════════════════════════════════════════════
class FrozenDataset(Dataset):
    def __init__(self, indices):
        self.prot_idx = tgt_indices[indices]
        self.drug_idx = drug_indices[indices]
        self.labels   = y[indices]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (prot_embs[self.prot_idx[i]],
                drug_fps[self.drug_idx[i]],
                torch.tensor(self.labels[i], dtype=torch.float32))

class LoRADataset(Dataset):
    def __init__(self, indices):
        self.prot_idx = tgt_indices[indices]
        self.drug_idx = drug_indices[indices]
        self.labels   = y[indices]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (self.prot_idx[i],          # int → collate 시 토큰 로드
                drug_fps[self.drug_idx[i]],
                torch.tensor(self.labels[i], dtype=torch.float32))

def lora_collate(batch):
    prot_idxs, drug_fps_b, labels = zip(*batch)
    ids   = [all_input_ids[j]  for j in prot_idxs]
    masks = [all_attn_masks[j] for j in prot_idxs]
    ids_pad   = torch.nn.utils.rnn.pad_sequence(ids,   batch_first=True, padding_value=tokenizer.pad_token_id)
    masks_pad = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    return ids_pad, masks_pad, torch.stack(drug_fps_b), torch.tensor(labels)

if args.lora:
    DS = LoRADataset
    train_loader = DataLoader(DS(tr_idx),  batch_size=args.batch_size,
                              shuffle=True,  collate_fn=lora_collate, num_workers=0)
    val_loader   = DataLoader(DS(val_idx), batch_size=args.batch_size,
                              shuffle=False, collate_fn=lora_collate, num_workers=0)
    test_loader  = DataLoader(DS(te_idx),  batch_size=args.batch_size,
                              shuffle=False, collate_fn=lora_collate, num_workers=0)
else:
    train_loader = DataLoader(FrozenDataset(tr_idx),  batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(FrozenDataset(val_idx), batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(FrozenDataset(te_idx),  batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)

# ══════════════════════════════════════════════════════════════════════════════
# [6] DTI 헤드
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

head = DTIHead(prot_dim).to(DEVICE)
n_head = sum(p.numel() for p in head.parameters()) / 1e6
print(f"[5] DTI 헤드: {n_head:.2f}M params\n")

# ══════════════════════════════════════════════════════════════════════════════
# [7] 옵티마이저 — LoRA: SaProt 어댑터 + 헤드 / frozen: 헤드만
# ══════════════════════════════════════════════════════════════════════════════
if args.lora:
    lora_params = [p for p in saprot.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr},
         {"params": head.parameters(), "lr": args.lr * 10}],  # 헤드는 10배 lr
        weight_decay=1e-4,
    )
else:
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-6
)
criterion = nn.HuberLoss(delta=1.0)

# ══════════════════════════════════════════════════════════════════════════════
# [8] 학습 루프 공통 헬퍼
# ══════════════════════════════════════════════════════════════════════════════
def get_prot_emb(input_ids, attention_mask):
    """LoRA 모드: SaProt forward → mean pool"""
    out    = saprot(input_ids=input_ids, attention_mask=attention_mask)
    hidden = out.last_hidden_state.float()    # [B, L, D]
    # CLS(0), EOS(-1) 제외 mean pool
    mask   = attention_mask[:, 1:-1].unsqueeze(-1).float()
    emb    = (hidden[:, 1:-1, :] * mask).sum(1) / mask.sum(1).clamp(min=1)
    return emb    # [B, D]

# ══════════════════════════════════════════════════════════════════════════════
# [9] 훈련
# ══════════════════════════════════════════════════════════════════════════════
best_val_r, best_head_state, patience_cnt = -1.0, None, 0
best_lora_state = None
history = []

print(f"[6] 훈련 시작 (epochs={args.epochs}, batch={args.batch_size}, lr={args.lr})")
print(f"    {'Epoch':>5} | {'Train Loss':>10} | {'Val r':>7} | {'Best':>7}")
print("    " + "-" * 42)

t_start = time.time()

for epoch in range(1, args.epochs + 1):
    # ── train ──────────────────────────────────────────────────────────────
    head.train()
    if args.lora: saprot.train()
    train_loss = 0.0

    for batch in train_loader:
        if args.lora:
            ids, masks, drug, label = batch
            ids, masks = ids.to(DEVICE), masks.to(DEVICE)
            drug, label = drug.to(DEVICE), label.to(DEVICE)
            prot = get_prot_emb(ids, masks)
        else:
            prot, drug, label = batch
            prot, drug, label = prot.to(DEVICE), drug.to(DEVICE), label.to(DEVICE)

        pred  = head(prot, drug)
        loss  = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(head.parameters()) + (list(saprot.parameters()) if args.lora else []),
            1.0
        )
        optimizer.step()
        train_loss += loss.item() * len(label)

    train_loss /= len(tr_idx)
    scheduler.step()

    # ── validate ───────────────────────────────────────────────────────────
    head.eval()
    if args.lora: saprot.eval()
    val_preds, val_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            if args.lora:
                ids, masks, drug, label = batch
                ids, masks = ids.to(DEVICE), masks.to(DEVICE)
                drug = drug.to(DEVICE)
                prot = get_prot_emb(ids, masks)
            else:
                prot, drug, label = batch
                prot, drug = prot.to(DEVICE), drug.to(DEVICE)
            pred = head(prot, drug).cpu().numpy()
            val_preds.extend(pred)
            val_labels.extend(label.numpy())

    val_r, _ = pearsonr(val_preds, val_labels)
    is_best   = val_r > best_val_r

    if is_best:
        best_val_r    = val_r
        best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
        if args.lora:
            best_lora_state = {k: v.clone() for k, v in saprot.state_dict().items()
                               if "lora" in k}
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

train_time = time.time() - t_start

# ══════════════════════════════════════════════════════════════════════════════
# [10] 테스트
# ══════════════════════════════════════════════════════════════════════════════
head.load_state_dict(best_head_state)
head.eval()
if args.lora:
    # best LoRA 가중치 복원
    cur = saprot.state_dict()
    cur.update(best_lora_state)
    saprot.load_state_dict(cur)
    saprot.eval()

test_preds, test_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        if args.lora:
            ids, masks, drug, label = batch
            ids, masks = ids.to(DEVICE), masks.to(DEVICE)
            drug = drug.to(DEVICE)
            prot = get_prot_emb(ids, masks)
        else:
            prot, drug, label = batch
            prot, drug = prot.to(DEVICE), drug.to(DEVICE)
        pred = head(prot, drug).cpu().numpy()
        test_preds.extend(pred)
        test_labels.extend(label.numpy())

test_r, test_p = pearsonr(test_preds, test_labels)

# ══════════════════════════════════════════════════════════════════════════════
# [11] 결과 저장
# ══════════════════════════════════════════════════════════════════════════════
out_dir = Path("./results") / run_name
out_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame({"y_pred": test_preds, "y_true": test_labels}).to_csv(
    out_dir / "test_predictions.csv", index=False)
pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
torch.save(best_head_state, out_dir / "dti_head.pt")
if args.lora:
    torch.save(best_lora_state, out_dir / "lora_adapter.pt")

result = {
    "run_name":       run_name,
    "dataset":        args.dataset,
    "encoder":        args.encoder,
    "quant":          args.quant,
    "lora":           args.lora,
    "lora_r":         args.lora_r if args.lora else None,
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
if   test_r >= 0.9: print("  ✅✅ 목표 달성! (r ≥ 0.9)")
elif test_r >= 0.8: print("  ✅  Phase 1 완료 (r ≥ 0.8)")
elif test_r >= 0.6: print("  △   양호 (r ≥ 0.6)")
else:               print("  ❌  성능 미달")
print(f"  결과 저장: {out_dir}/")
print("=" * 50)
print("\n[완료]")
