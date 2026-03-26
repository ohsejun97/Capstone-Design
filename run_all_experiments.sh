#!/bin/bash
# ================================================================
# run_all_experiments.sh
# Run all SaProt DTI frozen experiments (4 variants)
#
# Usage:
#   bash run_all_experiments.sh
# ================================================================
set -e

PYTHON=/home/ohsejun/miniconda3/envs/bioinfo/bin/python3
LOG_DIR=/home/ohsejun/Capstone_Design/logs
SCRIPT=/home/ohsejun/Capstone_Design/train_dti_saprot.py

mkdir -p "$LOG_DIR"

echo "========================================================"
echo "  SaProt DTI — Quantization Comparison Experiments"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

echo ""
echo "[A] SaProt-650M (FP16 baseline)..."
$PYTHON -u "$SCRIPT" --encoder 650M 2>&1 | tee "$LOG_DIR/train_650M.log"

echo ""
echo "[B] SaProt-35M (lightweight backbone)..."
$PYTHON -u "$SCRIPT" --encoder 35M 2>&1 | tee "$LOG_DIR/train_35M.log"

echo ""
echo "[C] SaProt-650M-8bit (INT8 quantization)..."
$PYTHON -u "$SCRIPT" --encoder 650M --quant 8bit 2>&1 | tee "$LOG_DIR/train_650M_8bit.log"

echo ""
echo "[D] SaProt-650M-4bit (NF4 quantization)..."
$PYTHON -u "$SCRIPT" --encoder 650M --quant 4bit 2>&1 | tee "$LOG_DIR/train_650M_4bit.log"

echo ""
echo "========================================================"
echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""
echo "  Results:"
for dir in /home/ohsejun/Capstone_Design/results/SaProt-*/; do
    if [ -f "$dir/result.json" ]; then
        $PYTHON -c "
import json
d = json.load(open('$dir/result.json'))
print(f\"    {d['run_name']:30s}  r = {d['test_pearson_r']:.4f}\")
" 2>/dev/null
    fi
done
echo ""
echo "  Logs:    $LOG_DIR/"
echo "  Results: /home/ohsejun/Capstone_Design/results/"
