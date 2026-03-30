#!/bin/bash
# ================================================================
# run_3di_experiments.sh
# Phase 3 — FoldSeek 3Di token experiments
#
# 순서:
#   [0] 3Di 토큰 캐시 빌드 (DAVIS + KIBA)
#   [A~D] DAVIS 3Di — 4개 모델 학습
#   [E~H] KIBA  3Di — 4개 모델 교차검증
#   [결과] placeholder vs 3Di 비교 출력
#
# Usage:
#   bash run_3di_experiments.sh
#   bash run_3di_experiments.sh --skip-cache   # 캐시 이미 있을 때
# ================================================================
set -e

PYTHON=/home/ohsejun/miniconda3/envs/bioinfo/bin/python3
LOG_DIR=/home/ohsejun/Capstone_Design/logs
SCRIPT=/home/ohsejun/Capstone_Design/train_dti_saprot.py

export PATH="/home/ohsejun/tools/foldseek/bin:$PATH"

mkdir -p "$LOG_DIR"

SKIP_CACHE=false
for arg in "$@"; do
    [ "$arg" = "--skip-cache" ] && SKIP_CACHE=true
done

echo "========================================================"
echo "  Phase 3 — FoldSeek 3Di Experiments"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# ── [0] 3Di 캐시 빌드 ──────────────────────────────────────────
if [ "$SKIP_CACHE" = false ]; then
    echo ""
    echo "[0-A] DAVIS 3Di 캐시 빌드..."
    $PYTHON -u /home/ohsejun/Capstone_Design/scripts/build_3di_cache.py \
        --dataset davis --resume 2>&1 | tee "$LOG_DIR/build_3di_davis.log"

    echo ""
    echo "[0-B] KIBA 3Di 캐시 빌드..."
    $PYTHON -u /home/ohsejun/Capstone_Design/scripts/build_3di_cache.py \
        --dataset kiba --resume 2>&1 | tee "$LOG_DIR/build_3di_kiba.log"
else
    echo "[0] 캐시 빌드 스킵 (--skip-cache)"
fi

# ── [A~D] DAVIS 3Di 학습 ───────────────────────────────────────
echo ""
echo "[A] SaProt-650M + 3Di (DAVIS)..."
$PYTHON -u "$SCRIPT" --encoder 650M --dataset davis --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_davis_3di.log"

echo ""
echo "[B] SaProt-35M + 3Di (DAVIS)..."
$PYTHON -u "$SCRIPT" --encoder 35M --dataset davis --use_3di \
    2>&1 | tee "$LOG_DIR/train_35M_davis_3di.log"

echo ""
echo "[C] SaProt-650M-8bit + 3Di (DAVIS)..."
$PYTHON -u "$SCRIPT" --encoder 650M --quant 8bit --dataset davis --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_8bit_davis_3di.log"

echo ""
echo "[D] SaProt-650M-4bit + 3Di (DAVIS)..."
$PYTHON -u "$SCRIPT" --encoder 650M --quant 4bit --dataset davis --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_4bit_davis_3di.log"

# ── [E~H] KIBA 3Di 교차검증 ────────────────────────────────────
echo ""
echo "[E] SaProt-650M + 3Di (KIBA)..."
$PYTHON -u "$SCRIPT" --encoder 650M --dataset kiba --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_kiba_3di.log"

echo ""
echo "[F] SaProt-35M + 3Di (KIBA)..."
$PYTHON -u "$SCRIPT" --encoder 35M --dataset kiba --use_3di \
    2>&1 | tee "$LOG_DIR/train_35M_kiba_3di.log"

echo ""
echo "[G] SaProt-650M-8bit + 3Di (KIBA)..."
$PYTHON -u "$SCRIPT" --encoder 650M --quant 8bit --dataset kiba --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_8bit_kiba_3di.log"

echo ""
echo "[H] SaProt-650M-4bit + 3Di (KIBA)..."
$PYTHON -u "$SCRIPT" --encoder 650M --quant 4bit --dataset kiba --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_4bit_kiba_3di.log"

# ── 결과 요약 ──────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""
echo "  === DAVIS: Placeholder vs 3Di ==="
$PYTHON -c "
import json
from pathlib import Path

pairs = [
    ('SaProt-650M',      'SaProt-650M-davis-3di'),
    ('SaProt-35M',       'SaProt-35M-davis-3di'),
    ('SaProt-650M-8bit', 'SaProt-650M-8bit-davis-3di'),
    ('SaProt-650M-4bit', 'SaProt-650M-4bit-davis-3di'),
]
base = Path('/home/ohsejun/Capstone_Design/results')
print(f\"  {'Model':<20} {'Placeholder':>12} {'3Di':>8} {'Delta':>8}\")
print('  ' + '-'*50)
for base_name, di_name in pairs:
    b = base / base_name / 'result.json'
    d = base / di_name   / 'result.json'
    rb = json.load(open(b))['test_pearson_r'] if b.exists() else None
    rd = json.load(open(d))['test_pearson_r'] if d.exists() else None
    label = base_name.replace('SaProt-', '')
    if rb and rd:
        delta = rd - rb
        sign  = '+' if delta >= 0 else ''
        print(f\"  {label:<20} {rb:>12.4f} {rd:>8.4f} {sign}{delta:>7.4f}\")
    elif rd:
        print(f\"  {label:<20} {'N/A':>12} {rd:>8.4f}\")
" 2>/dev/null

echo ""
echo "  === KIBA: Placeholder vs 3Di ==="
$PYTHON -c "
import json
from pathlib import Path

pairs = [
    ('SaProt-35M-kiba',       'SaProt-35M-kiba-3di'),
    ('SaProt-650M-8bit-kiba', 'SaProt-650M-8bit-kiba-3di'),
    ('SaProt-650M-4bit-kiba', 'SaProt-650M-4bit-kiba-3di'),
    (None,                    'SaProt-650M-kiba-3di'),
]
base = Path('/home/ohsejun/Capstone_Design/results')
print(f\"  {'Model':<20} {'Placeholder':>12} {'3Di':>8} {'Delta':>8}\")
print('  ' + '-'*50)
for base_name, di_name in pairs:
    d = base / di_name / 'result.json'
    rd = json.load(open(d))['test_pearson_r'] if d.exists() else None
    if base_name:
        b = base / base_name / 'result.json'
        rb = json.load(open(b))['test_pearson_r'] if b.exists() else None
    else:
        rb = None
    label = di_name.replace('SaProt-', '').replace('-kiba-3di', '')
    if rb and rd:
        delta = rd - rb
        sign  = '+' if delta >= 0 else ''
        print(f\"  {label:<20} {rb:>12.4f} {rd:>8.4f} {sign}{delta:>7.4f}\")
    elif rd:
        print(f\"  {label:<20} {'N/A':>12} {rd:>8.4f}\")
" 2>/dev/null

echo ""
echo "  Logs:    $LOG_DIR/"
echo "  Results: /home/ohsejun/Capstone_Design/results/"
