#!/bin/bash
# ================================================================
# run_all_experiments.sh
# 세 가지 SaProt DTI 모델을 순차적으로 학습 후 결과 요약
# ================================================================
set -e

PYTHON=/home/ohsejun/miniconda3/envs/bioinfo/bin/python3
LOG_DIR=/home/ohsejun/Capstone_Design/logs
SCRIPT=/home/ohsejun/Capstone_Design/train_dti_saprot.py

mkdir -p "$LOG_DIR"

echo "========================================================"
echo "  Agentic FusionDTI — 전체 실험 자동 실행"
echo "  시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# ── 실험 A: SaProt-650M (기준) ────────────────────────────────
echo ""
echo "[A] SaProt-650M 기준 모델 학습 시작..."
$PYTHON -u "$SCRIPT" --encoder 650M 2>&1 | tee "$LOG_DIR/train_650M.log"
echo "[A] 완료: $(date '+%H:%M:%S')"

# ── 실험 B: SaProt-35M (경량) ─────────────────────────────────
echo ""
echo "[B] SaProt-35M 경량 모델 학습 시작..."
$PYTHON -u "$SCRIPT" --encoder 35M 2>&1 | tee "$LOG_DIR/train_35M.log"
echo "[B] 완료: $(date '+%H:%M:%S')"

# ── 실험 C: SaProt-650M 4bit (양자화) ───────────────────────
echo ""
echo "[C] SaProt-650M-4bit 양자화 모델 학습 시작..."
$PYTHON -u "$SCRIPT" --encoder 650M --quant 4bit 2>&1 | tee "$LOG_DIR/train_650M_4bit.log"
echo "[C] 완료: $(date '+%H:%M:%S')"

# ── 결과 요약 ─────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  전체 실험 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""
echo "  결과 요약:"
for dir in /home/ohsejun/Capstone_Design/results/SaProt-*/; do
    if [ -f "$dir/result.json" ]; then
        name=$(python3 -c "import json; d=json.load(open('$dir/result.json')); print(d['run_name'])" 2>/dev/null)
        r=$(python3 -c "import json; d=json.load(open('$dir/result.json')); print(f\"{d['test_pearson_r']:.4f}\")" 2>/dev/null)
        echo "    $name : Pearson r = $r"
    fi
done
echo ""
echo "  로그 파일: $LOG_DIR/"
echo "  결과 파일: /home/ohsejun/Capstone_Design/results/"
