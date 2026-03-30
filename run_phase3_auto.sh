#!/bin/bash
# ================================================================
# run_phase3_auto.sh
# DAVIS 캐시 빌드 완료 후 전체 Phase 3 자동 실행
#
# 순서:
#   [0] DAVIS 캐시 빌드 완료 대기 (PID 인자로 전달)
#   [1] KIBA 3Di 캐시 빌드
#   [2] DAVIS 4개 모델 학습 (--use_3di)
#   [3] KIBA 4개 모델 교차검증 (--use_3di)
#   [4] 결과 시각화
# ================================================================

export PATH="/home/ohsejun/tools/foldseek/bin:$PATH"

PYTHON=/home/ohsejun/miniconda3/envs/bioinfo/bin/python3
LOG_DIR=/home/ohsejun/Capstone_Design/logs
SCRIPT=/home/ohsejun/Capstone_Design/train_dti_saprot.py
DAVIS_PID=${1:-3066}

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_DIR/phase3_auto.log"; }

log "=========================================="
log "  Phase 3 자동 실행 시작"
log "  DAVIS 캐시 PID: $DAVIS_PID"
log "=========================================="

# ── [0] DAVIS 캐시 완료 대기 ───────────────────────────────────
log "[0] DAVIS 캐시 빌드 완료 대기 (PID $DAVIS_PID)..."
while kill -0 "$DAVIS_PID" 2>/dev/null; do
    sleep 30
done
log "[0] DAVIS 캐시 빌드 완료"

# ── [1] KIBA 3Di 캐시 빌드 ────────────────────────────────────
log "[1] KIBA 3Di 캐시 빌드 시작..."
$PYTHON -u /home/ohsejun/Capstone_Design/scripts/build_3di_cache.py \
    --dataset kiba --resume 2>&1 | tee "$LOG_DIR/build_3di_kiba.log"
log "[1] KIBA 캐시 빌드 완료"

# ── [2] DAVIS 4개 모델 학습 ────────────────────────────────────
log "[2-A] SaProt-650M + 3Di (DAVIS)..."
$PYTHON -u "$SCRIPT" --encoder 650M --dataset davis --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_davis_3di.log"

log "[2-B] SaProt-35M + 3Di (DAVIS)..."
$PYTHON -u "$SCRIPT" --encoder 35M --dataset davis --use_3di \
    2>&1 | tee "$LOG_DIR/train_35M_davis_3di.log"

log "[2-C] SaProt-650M-8bit + 3Di (DAVIS)..."
$PYTHON -u "$SCRIPT" --encoder 650M --quant 8bit --dataset davis --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_8bit_davis_3di.log"

log "[2-D] SaProt-650M-4bit + 3Di (DAVIS)..."
$PYTHON -u "$SCRIPT" --encoder 650M --quant 4bit --dataset davis --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_4bit_davis_3di.log"

log "[2] DAVIS 학습 완료"

# ── [3] KIBA 4개 모델 교차검증 ────────────────────────────────
log "[3-E] SaProt-650M + 3Di (KIBA)..."
$PYTHON -u "$SCRIPT" --encoder 650M --dataset kiba --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_kiba_3di.log"

log "[3-F] SaProt-35M + 3Di (KIBA)..."
$PYTHON -u "$SCRIPT" --encoder 35M --dataset kiba --use_3di \
    2>&1 | tee "$LOG_DIR/train_35M_kiba_3di.log"

log "[3-G] SaProt-650M-8bit + 3Di (KIBA)..."
$PYTHON -u "$SCRIPT" --encoder 650M --quant 8bit --dataset kiba --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_8bit_kiba_3di.log"

log "[3-H] SaProt-650M-4bit + 3Di (KIBA)..."
$PYTHON -u "$SCRIPT" --encoder 650M --quant 4bit --dataset kiba --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_4bit_kiba_3di.log"

log "[3] KIBA 교차검증 완료"

# ── [4] 결과 시각화 ────────────────────────────────────────────
log "[4] 결과 시각화..."
$PYTHON -u /home/ohsejun/Capstone_Design/experiments/visualize_results.py \
    2>&1 | tee "$LOG_DIR/visualize_3di.log"
$PYTHON -u /home/ohsejun/Capstone_Design/experiments/evaluate_results.py \
    2>&1 | tee "$LOG_DIR/evaluate_3di.log"

# ── 최종 요약 ──────────────────────────────────────────────────
log "=========================================="
log "  Phase 3 전체 완료"
log "=========================================="
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
for bn, dn in pairs:
    b = base / bn / 'result.json'
    d = base / dn / 'result.json'
    rb = json.load(open(b))['test_pearson_r'] if b.exists() else None
    rd = json.load(open(d))['test_pearson_r'] if d.exists() else None
    label = bn.replace('SaProt-', '')
    if rb and rd:
        delta = rd - rb
        sign = '+' if delta >= 0 else ''
        print(f\"  {label:<20} {rb:>12.4f} {rd:>8.4f} {sign}{delta:>7.4f}\")
" 2>/dev/null | tee -a "$LOG_DIR/phase3_auto.log"
