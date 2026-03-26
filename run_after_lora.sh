#!/bin/bash
# ================================================================
# run_after_lora.sh
# 35M LoRA 완료 후 자동 실행:
#   1. 650M-4bit frozen 재실행 (add_pooling_layer=False 패치 적용)
#   2. 결과 요약 출력
# ================================================================
PYTHON=/home/ohsejun/miniconda3/envs/bioinfo/bin/python3
LOG_DIR=/home/ohsejun/Capstone_Design/logs
SCRIPT=/home/ohsejun/Capstone_Design/train_dti_saprot.py

echo "========================================================"
echo "  35M LoRA 완료. 후속 실험 시작"
echo "  시각: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

echo ""
echo "[C] SaProt-650M-4bit frozen 재실행..."
$PYTHON -u "$SCRIPT" --encoder 650M --quant 4bit 2>&1 | tee "$LOG_DIR/train_650M_4bit.log"
echo "[C] 완료: $(date '+%H:%M:%S')"

echo ""
echo "========================================================"
echo "  전체 실험 결과 요약"
echo "========================================================"
for dir in /home/ohsejun/Capstone_Design/results/SaProt-*/; do
    if [ -f "$dir/result.json" ]; then
        $PYTHON -c "
import json
d = json.load(open('$dir/result.json'))
lora = '(LoRA)  ' if d.get('lora') else '(frozen)'
print(f\"  {d['run_name']:35s} {lora}  Pearson r = {d['test_pearson_r']:.4f}\")
" 2>/dev/null
    fi
done
echo ""
echo "  로그:  $LOG_DIR/"
echo "  결과:  /home/ohsejun/Capstone_Design/results/"
echo "========================================================"
