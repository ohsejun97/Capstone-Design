#!/bin/bash
# ================================================================
# resume_phase3.sh
# Phase 3 이어서 실행 (WSL 꺼도 안 죽도록 nohup 사용)
#
# Usage:
#   bash resume_phase3.sh
#   tail -f logs/phase3_resume.log   # 진행 상황 확인
# ================================================================

PYTHON=/home/ohsejun/miniconda3/envs/bioinfo/bin/python3
SCRIPT_DIR=/home/ohsejun/Capstone_Design
LOG_DIR=$SCRIPT_DIR/logs
LOG=$LOG_DIR/phase3_resume.log

export PATH="/home/ohsejun/tools/foldseek/bin:$PATH"

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

# 이미 실행 중인지 확인
if pgrep -f "build_3di_cache.py\|train_dti_saprot.py" > /dev/null 2>&1; then
    echo "이미 Phase 3 프로세스가 실행 중입니다:"
    pgrep -af "build_3di_cache.py\|train_dti_saprot.py"
    exit 1
fi

log "=========================================="
log "  Phase 3 재개 (resume)"
log "=========================================="

# ── [0-A] DAVIS 3Di 캐시 이어서 빌드 ─────────────────────────
log "[0-A] DAVIS 3Di 캐시 이어서 빌드..."
$PYTHON -u "$SCRIPT_DIR/scripts/build_3di_cache.py" \
    --dataset davis --resume \
    2>&1 | tee -a "$LOG_DIR/build_3di_davis.log"
log "[0-A] DAVIS 캐시 완료"

# ── [0-B] KIBA 3Di 캐시 빌드 ──────────────────────────────────
log "[0-B] KIBA 3Di 캐시 빌드..."
$PYTHON -u "$SCRIPT_DIR/scripts/build_3di_cache.py" \
    --dataset kiba --resume \
    2>&1 | tee -a "$LOG_DIR/build_3di_kiba.log"
log "[0-B] KIBA 캐시 완료"

# ── [A~D] DAVIS 3Di 학습 ───────────────────────────────────────
log "[A] SaProt-650M + 3Di (DAVIS)..."
$PYTHON -u "$SCRIPT_DIR/train_dti_saprot.py" \
    --encoder 650M --dataset davis --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_davis_3di.log"

log "[B] SaProt-35M + 3Di (DAVIS)..."
$PYTHON -u "$SCRIPT_DIR/train_dti_saprot.py" \
    --encoder 35M --dataset davis --use_3di \
    2>&1 | tee "$LOG_DIR/train_35M_davis_3di.log"

log "[C] SaProt-650M-8bit + 3Di (DAVIS)..."
$PYTHON -u "$SCRIPT_DIR/train_dti_saprot.py" \
    --encoder 650M --quant 8bit --dataset davis --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_8bit_davis_3di.log"

log "[D] SaProt-650M-4bit + 3Di (DAVIS)..."
$PYTHON -u "$SCRIPT_DIR/train_dti_saprot.py" \
    --encoder 650M --quant 4bit --dataset davis --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_4bit_davis_3di.log"

# ── [E~H] KIBA 3Di 교차검증 ────────────────────────────────────
log "[E] SaProt-650M + 3Di (KIBA)..."
$PYTHON -u "$SCRIPT_DIR/train_dti_saprot.py" \
    --encoder 650M --dataset kiba --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_kiba_3di.log"

log "[F] SaProt-35M + 3Di (KIBA)..."
$PYTHON -u "$SCRIPT_DIR/train_dti_saprot.py" \
    --encoder 35M --dataset kiba --use_3di \
    2>&1 | tee "$LOG_DIR/train_35M_kiba_3di.log"

log "[G] SaProt-650M-8bit + 3Di (KIBA)..."
$PYTHON -u "$SCRIPT_DIR/train_dti_saprot.py" \
    --encoder 650M --quant 8bit --dataset kiba --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_8bit_kiba_3di.log"

log "[H] SaProt-650M-4bit + 3Di (KIBA)..."
$PYTHON -u "$SCRIPT_DIR/train_dti_saprot.py" \
    --encoder 650M --quant 4bit --dataset kiba --use_3di \
    2>&1 | tee "$LOG_DIR/train_650M_4bit_kiba_3di.log"

log "=========================================="
log "  Phase 3 전체 완료"
log "=========================================="
