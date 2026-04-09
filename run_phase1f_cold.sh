#!/bin/bash
# run_phase1f_cold.sh
# BindingDB cold-split 실험 + DAVIS/KIBA cross-eval + git push
# nohup bash run_phase1f_cold.sh > logs/phase1f_cold.log 2>&1 &

set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "========================================"
echo "  Phase 1f Cold-Split 실험 시작: $(date)"
echo "========================================"

# ── 1. cold_drug split ────────────────────────────────────────────────────────
echo ""
echo "[1/3] ChemBERTa — cold_drug split"
conda run -n bioinfo python train_dti_saprot.py \
    --dataset bindingdb --encoder 650M --use_3di \
    --drug_encoder chemberta --split cold_drug \
    2>&1 | tee logs/bindingdb_chemberta_cold_drug.log
echo "  완료: $(date)"

# ── 2. cold_protein split ─────────────────────────────────────────────────────
echo ""
echo "[2/3] ChemBERTa — cold_protein split"
conda run -n bioinfo python train_dti_saprot.py \
    --dataset bindingdb --encoder 650M --use_3di \
    --drug_encoder chemberta --split cold_protein \
    2>&1 | tee logs/bindingdb_chemberta_cold_protein.log
echo "  완료: $(date)"

# ── 3. Cross-dataset evaluation (cold_drug 모델 → DAVIS, KIBA) ───────────────
echo ""
echo "[3/3] Cross-eval: cold_drug 모델 → DAVIS / KIBA"
conda run -n bioinfo python scripts/cross_eval.py \
    --model_dir results/SaProt-650M-bindingdb-3di-chemberta-cold_drug \
    --eval_datasets davis kiba \
    2>&1 | tee logs/cross_eval_cold_drug.log
echo "  완료: $(date)"

# ── 결과 파싱 ─────────────────────────────────────────────────────────────────
echo ""
echo "[요약] 결과 파싱..."
conda run -n bioinfo python - <<'PYEOF'
import json
from pathlib import Path

configs = [
    ("random",       "results/SaProt-650M-bindingdb-3di-chemberta/result.json"),
    ("cold_drug",    "results/SaProt-650M-bindingdb-3di-chemberta-cold_drug/result.json"),
    ("cold_protein", "results/SaProt-650M-bindingdb-3di-chemberta-cold_protein/result.json"),
]
cross_files = [
    ("DAVIS (cross)", "results/SaProt-650M-bindingdb-3di-chemberta-cold_drug/cross_eval_davis.json", "cross_pearson_r"),
    ("KIBA  (cross)", "results/SaProt-650M-bindingdb-3di-chemberta-cold_drug/cross_eval_kiba.json",  "cross_pearson_r"),
]

print(f"\n{'Split':<18} {'BindingDB r':>12} {'RMSE':>8} {'CI':>8}")
print("-" * 52)
for label, path in configs:
    p = Path(path)
    if p.exists():
        d = json.load(open(p))
        print(f"{label:<18} {d['test_pearson_r']:>12.4f} {d['test_rmse']:>8.4f} {d['test_ci']:>8.4f}")
    else:
        print(f"{label:<18} {'N/A':>12}")

print()
for label, path, key in cross_files:
    p = Path(path)
    if p.exists():
        d = json.load(open(p))
        print(f"{label:<18} r={d[key]:.4f}  RMSE={d['cross_rmse']:.4f}  CI={d['cross_ci']:.4f}")
    else:
        print(f"{label:<18} N/A")
PYEOF

# ── git push ─────────────────────────────────────────────────────────────────
echo ""
echo "[git] 결과 push..."
git add results/SaProt-650M-bindingdb-3di-chemberta-cold_drug/ \
        results/SaProt-650M-bindingdb-3di-chemberta-cold_protein/ \
        scripts/cross_eval.py run_phase1f_cold.sh
git commit -m "feat: Phase 1f cold-split 실험 + DAVIS/KIBA cross-eval 완료

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push origin main
echo "  git push 완료"

echo ""
echo "========================================"
echo "  전체 완료: $(date)"
echo "========================================"
