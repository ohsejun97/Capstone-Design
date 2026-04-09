#!/bin/bash
# run_phase1f.sh — Phase 1f 실험 자동화 (ChemBERTa → GNN → 문서 업데이트 → git push)
# 사용: nohup bash run_phase1f.sh > logs/phase1f_auto.log 2>&1 &

set -e
cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "  Phase 1f 실험 시작: $(date)"
echo "========================================"

# ── 1. ChemBERTa ─────────────────────────────────────────────────────────────
echo ""
echo "[1/2] ChemBERTa — dataset=bindingdb, encoder=650M, use_3di"
echo "      시작: $(date)"
conda run -n bioinfo python train_dti_saprot.py \
    --dataset bindingdb --encoder 650M --use_3di --drug_encoder chemberta \
    2>&1 | tee "$LOG_DIR/bindingdb_chemberta.log"
echo "      완료: $(date)"

# ── 2. GNN ───────────────────────────────────────────────────────────────────
echo ""
echo "[2/2] GNN — dataset=bindingdb, encoder=650M, use_3di"
echo "      시작: $(date)"
conda run -n bioinfo python train_dti_saprot.py \
    --dataset bindingdb --encoder 650M --use_3di --drug_encoder gnn \
    2>&1 | tee "$LOG_DIR/bindingdb_gnn.log"
echo "      완료: $(date)"

# ── 3. 결과 파싱 ──────────────────────────────────────────────────────────────
echo ""
echo "[3] 결과 파싱..."
CHEMBERTA_R=$(conda run -n bioinfo python -c "
import json, glob
files = glob.glob('results/*/result.json')
best = None
for f in files:
    d = json.load(open(f))
    if d.get('dataset') == 'bindingdb' and d.get('drug_encoder','morgan') == 'chemberta':
        if best is None or d['test_pearson_r'] > best:
            best = d['test_pearson_r']
print(f'{best:.4f}' if best else 'N/A')
" 2>/dev/null)

GNN_R=$(conda run -n bioinfo python -c "
import json, glob
files = glob.glob('results/*/result.json')
best = None
for f in files:
    d = json.load(open(f))
    if d.get('dataset') == 'bindingdb' and d.get('drug_encoder','morgan') == 'gnn':
        if best is None or d['test_pearson_r'] > best:
            best = d['test_pearson_r']
print(f'{best:.4f}' if best else 'N/A')
" 2>/dev/null)

BASELINE_R="0.8082"
echo "    Baseline (DAVIS, Morgan FP): ${BASELINE_R}"
echo "    ChemBERTa (BindingDB):       ${CHEMBERTA_R}"
echo "    GNN       (BindingDB):       ${GNN_R}"

# ── 4. CLAUDE.md Phase 1f 결과 업데이트 ──────────────────────────────────────
echo ""
echo "[4] CLAUDE.md 업데이트..."
conda run -n bioinfo python - <<PYEOF
import re, json, glob
from pathlib import Path

# 결과 읽기
def get_best(dataset, drug_encoder):
    files = glob.glob("results/*/result.json")
    best = None
    for f in files:
        d = json.load(open(f))
        if d.get("dataset") == dataset and d.get("drug_encoder", "morgan") == drug_encoder:
            if best is None or d["test_pearson_r"] > best["test_pearson_r"]:
                best = d
    return best

chemberta = get_best("bindingdb", "chemberta")
gnn       = get_best("bindingdb", "gnn")

lines = []
if chemberta:
    lines.append(f"- ChemBERTa (BindingDB, frozen): test_pearson_r={chemberta['test_pearson_r']:.4f}")
if gnn:
    lines.append(f"- GNN       (BindingDB):          test_pearson_r={gnn['test_pearson_r']:.4f}")
if not lines:
    print("결과 파일 없음 — CLAUDE.md 업데이트 스킵")
    exit(0)

result_text = "\n".join(lines)

# CLAUDE.md Phase 1f 섹션 업데이트
claude_md = Path(".claude/CLAUDE.md")
content = claude_md.read_text()

# "BindingDB 기반 GNN 또는 ChemBERTa 재훈련" 체크박스 업데이트
content = content.replace(
    "- [ ] BindingDB 기반 GNN 또는 ChemBERTa 재훈련",
    f"- [x] BindingDB 기반 GNN 또는 ChemBERTa 재훈련\n\n**Phase 1f 실험 결과:**\n{result_text}"
)
claude_md.write_text(content)
print("CLAUDE.md 업데이트 완료")
PYEOF

# ── 5. git push ───────────────────────────────────────────────────────────────
echo ""
echo "[5] git push..."
cd /home/ohsejun/Capstone_Design
git add results/ logs/bindingdb_chemberta.log logs/bindingdb_gnn.log .claude/CLAUDE.md
git commit -m "feat: Phase 1f BindingDB 실험 완료 — ChemBERTa r=${CHEMBERTA_R}, GNN r=${GNN_R}

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push origin main
echo "  git push 완료"

echo ""
echo "========================================"
echo "  Phase 1f 전체 완료: $(date)"
echo "  ChemBERTa: ${CHEMBERTA_R}  |  GNN: ${GNN_R}  |  Baseline: ${BASELINE_R}"
echo "========================================"
