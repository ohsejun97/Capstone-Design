#!/bin/bash
# ================================================================
# run_all_experiments.sh
# SaProt DTI 전체 실험 자동 실행 (frozen + LoRA)
#
# 실행 모드:
#   bash run_all_experiments.sh          # frozen 3종 + LoRA 3종
#   bash run_all_experiments.sh frozen   # frozen 3종만
#   bash run_all_experiments.sh lora     # LoRA 3종만
# ================================================================
set -e

PYTHON=/home/ohsejun/miniconda3/envs/bioinfo/bin/python3
LOG_DIR=/home/ohsejun/Capstone_Design/logs
SCRIPT=/home/ohsejun/Capstone_Design/train_dti_saprot.py

mkdir -p "$LOG_DIR"

MODE=${1:-"all"}   # all / frozen / lora

echo "========================================================"
echo "  Agentic FusionDTI — 전체 실험 자동 실행 (mode=$MODE)"
echo "  시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# ── Frozen 실험 ───────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "frozen" ]]; then
    echo ""
    echo "[A] SaProt-650M frozen..."
    $PYTHON -u "$SCRIPT" --encoder 650M 2>&1 | tee "$LOG_DIR/train_650M.log"

    echo ""
    echo "[B] SaProt-35M frozen..."
    $PYTHON -u "$SCRIPT" --encoder 35M  2>&1 | tee "$LOG_DIR/train_35M.log"

    echo ""
    echo "[C] SaProt-650M-4bit frozen..."
    $PYTHON -u "$SCRIPT" --encoder 650M --quant 4bit 2>&1 | tee "$LOG_DIR/train_650M_4bit.log"
fi

# ── LoRA 실험 ─────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "lora" ]]; then
    echo ""
    echo "[D] SaProt-650M + LoRA (기준 모델, ~4~6h)..."
    $PYTHON -u "$SCRIPT" --encoder 650M --lora 2>&1 | tee "$LOG_DIR/train_650M_lora.log"

    echo ""
    echo "[E] SaProt-35M + LoRA (핵심 실험, ~30~60min)..."
    $PYTHON -u "$SCRIPT" --encoder 35M  --lora 2>&1 | tee "$LOG_DIR/train_35M_lora.log"

    echo ""
    echo "[F] SaProt-650M-4bit + LoRA (~2~3h)..."
    $PYTHON -u "$SCRIPT" --encoder 650M --quant 4bit --lora 2>&1 | tee "$LOG_DIR/train_650M_4bit_lora.log"
fi

# ── 결과 요약 ─────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""
echo "  결과 요약:"
for dir in /home/ohsejun/Capstone_Design/results/SaProt-*/; do
    if [ -f "$dir/result.json" ]; then
        $PYTHON -c "
import json
d = json.load(open('$dir/result.json'))
lora = '(LoRA)' if d.get('lora') else '(frozen)'
print(f\"    {d['run_name']:30s} {lora}  r = {d['test_pearson_r']:.4f}\")
" 2>/dev/null
    fi
done
echo ""
echo "  로그:    $LOG_DIR/"
echo "  결과:    /home/ohsejun/Capstone_Design/results/"
