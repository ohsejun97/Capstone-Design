"""
visualize_results.py
====================
SaProt DTI 실험 결과 시각화
- 학습 곡선 (val Pearson r per epoch)
- 예측값 vs 실제값 산점도
- 모델 비교 바 차트
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_DIR     = Path(__file__).parent.parent / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 색상 팔레트 ────────────────────────────────────────────────
COLORS = {
    "SaProt-650M":      "#4C72B0",
    "SaProt-35M":       "#DD8452",
    "SaProt-650M-4bit": "#55A868",
    "SaProt-650M-8bit": "#C44E52",
}

# ── 결과 로드 ──────────────────────────────────────────────────
runs = {}
for d in sorted(RESULTS_DIR.iterdir()):
    if not d.is_dir(): continue
    rj = d / "result.json"
    th = d / "training_history.csv"
    tp = d / "test_predictions.csv"
    if not (rj.exists() and th.exists() and tp.exists()): continue
    name = json.load(open(rj))["run_name"]
    runs[name] = {
        "result":  json.load(open(rj)),
        "history": pd.read_csv(th),
        "preds":   pd.read_csv(tp),
    }

if not runs:
    print("❌ results/ 디렉터리에 결과가 없습니다.")
    exit(1)

print(f"로드된 실험: {list(runs.keys())}")

# ══════════════════════════════════════════════════════════════
# Figure 1 — 학습 곡선 (val Pearson r)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

for name, data in runs.items():
    hist = data["history"]
    color = COLORS.get(name, "#888888")
    ax.plot(hist["epoch"], hist["val_r"], label=name, color=color, linewidth=1.8)

ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="목표 (r=0.8)")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Validation Pearson r", fontsize=12)
ax.set_title("SaProt DTI — 학습 곡선", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_training_curves.png", dpi=150)
plt.close()
print("✅ 01_training_curves.png 저장")

# ══════════════════════════════════════════════════════════════
# Figure 2 — 예측값 vs 실제값 산점도 (모델별)
# ══════════════════════════════════════════════════════════════
n = len(runs)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
if n == 1: axes = [axes]

for ax, (name, data) in zip(axes, runs.items()):
    preds  = data["preds"]
    y_pred = preds["y_pred"].values
    y_true = preds["y_true"].values
    r, _   = pearsonr(y_pred, y_true)
    color  = COLORS.get(name, "#888888")

    ax.scatter(y_true, y_pred, alpha=0.15, s=6, color=color)
    lims = [min(y_true.min(), y_pred.min()) - 0.2,
            max(y_true.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True pKd", fontsize=11)
    ax.set_ylabel("Predicted pKd", fontsize=11)
    ax.set_title(f"{name}\nr = {r:.4f}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

fig.suptitle("SaProt DTI — 예측 vs 실제 (Test Set)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "02_scatter_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ 02_scatter_plots.png 저장")

# ══════════════════════════════════════════════════════════════
# Figure 3 — 모델 비교 바 차트 (Test Pearson r)
# ══════════════════════════════════════════════════════════════
names  = list(runs.keys())
test_r = [runs[n]["result"]["test_pearson_r"] for n in names]
val_r  = [runs[n]["result"]["best_val_r"]     for n in names]
colors = [COLORS.get(n, "#888888") for n in names]

x = np.arange(len(names))
w = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - w/2, val_r,  w, label="Val Pearson r",  color=colors, alpha=0.6)
bars2 = ax.bar(x + w/2, test_r, w, label="Test Pearson r", color=colors, alpha=1.0)

ax.axhline(0.8, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="목표 (r=0.8)")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
ax.set_ylabel("Pearson r", fontsize=12)
ax.set_title("SaProt DTI — 모델별 성능 비교", fontsize=14, fontweight="bold")
ax.set_ylim(0.7, 0.87)
ax.legend(fontsize=10)

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_model_comparison.png", dpi=150)
plt.close()
print("✅ 03_model_comparison.png 저장")

# ══════════════════════════════════════════════════════════════
# 결과 요약 출력
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print(f"  {'모델':<25} {'Test r':>8}  {'Val r':>8}  {'학습시간':>8}")
print("  " + "-"*50)
for name, data in runs.items():
    r  = data["result"]
    print(f"  {name:<25} {r['test_pearson_r']:>8.4f}  {r['best_val_r']:>8.4f}  {r['train_time_sec']:>6.0f}s")
print("="*55)
print(f"\n그래프 저장 위치: {OUT_DIR}/")
