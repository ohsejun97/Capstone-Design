"""
visualize_results.py
====================
Visualize SaProt DTI experiment results.

Outputs (saved to outputs/figures/):
  01_training_curves.png  - Validation Pearson r per epoch
  02_scatter_plots.png    - Predicted vs True pKd scatter plots
  03_model_comparison.png - Test/Val Pearson r bar chart per model

Usage:
  python experiments/visualize_results.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_DIR     = Path(__file__).parent.parent / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "SaProt-650M":      "#4C72B0",
    "SaProt-35M":       "#DD8452",
    "SaProt-650M-4bit": "#55A868",
    "SaProt-650M-8bit": "#C44E52",
}

# ── Load results ───────────────────────────────────────────────
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
    print("No results found in results/")
    exit(1)

print(f"Loaded: {list(runs.keys())}")

# ══════════════════════════════════════════════════════════════
# Figure 1 — Training curves (val Pearson r per epoch)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

for name, data in runs.items():
    hist  = data["history"]
    color = COLORS.get(name, "#888888")
    ax.plot(hist["epoch"], hist["val_r"], label=name, color=color, linewidth=1.8)

ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Target (r=0.8)")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Validation Pearson r", fontsize=12)
ax.set_title("SaProt DTI — Training Curves", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_training_curves.png", dpi=150)
plt.close()
print("Saved: 01_training_curves.png")

# ══════════════════════════════════════════════════════════════
# Figure 2 — Predicted vs True scatter plots
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

fig.suptitle("SaProt DTI — Predicted vs True pKd (Test Set)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "02_scatter_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_scatter_plots.png")

# ══════════════════════════════════════════════════════════════
# Figure 3 — Model comparison bar chart (Test/Val Pearson r)
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

ax.axhline(0.8, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Target (r=0.8)")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
ax.set_ylabel("Pearson r", fontsize=12)
ax.set_title("SaProt DTI — Model Comparison", fontsize=14, fontweight="bold")
ax.set_ylim(0.7, 0.87)
ax.legend(fontsize=10)

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_model_comparison.png", dpi=150)
plt.close()
print("Saved: 03_model_comparison.png")

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print(f"  {'Model':<25} {'Test r':>8}  {'Val r':>8}  {'Time':>8}")
print("  " + "-" * 50)
for name, data in runs.items():
    r = data["result"]
    print(f"  {name:<25} {r['test_pearson_r']:>8.4f}  {r['best_val_r']:>8.4f}  {r['train_time_sec']:>6.0f}s")
print("=" * 55)
print(f"\nFigures saved to: {OUT_DIR}/")
