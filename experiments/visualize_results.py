"""
visualize_results.py
====================
Visualize SaProt DTI experiment results.

Figures saved to outputs/figures/:
  01_training_curves.png     - Val Pearson r per epoch (DAVIS models)
  02_scatter_plots.png       - Predicted vs True pKd scatter (DAVIS models)
  03_model_comparison.png    - Test/Val Pearson r bar chart (DAVIS models)
  05_davis_vs_kiba.png       - DAVIS vs KIBA cross-dataset comparison

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
    "SaProt-650M":           "#4C72B0",
    "SaProt-35M":            "#DD8452",
    "SaProt-650M-4bit":      "#55A868",
    "SaProt-650M-8bit":      "#C44E52",
    "SaProt-650M-4bit-kiba": "#8172B3",
}

# ── Load all results ───────────────────────────────────────────
all_runs = {}
for d in sorted(RESULTS_DIR.iterdir()):
    if not d.is_dir(): continue
    rj = d / "result.json"
    th = d / "training_history.csv"
    tp = d / "test_predictions.csv"
    if not (rj.exists() and th.exists() and tp.exists()): continue
    meta = json.load(open(rj))
    name = meta["run_name"]
    all_runs[name] = {
        "result":  meta,
        "history": pd.read_csv(th),
        "preds":   pd.read_csv(tp),
        "dataset": meta.get("dataset", "davis"),
    }

davis_runs = {k: v for k, v in all_runs.items() if v["dataset"] == "davis"}
kiba_runs  = {k: v for k, v in all_runs.items() if v["dataset"] == "kiba"}

print(f"DAVIS models: {list(davis_runs.keys())}")
print(f"KIBA  models: {list(kiba_runs.keys())}")

# ══════════════════════════════════════════════════════════════
# Figure 1 — Training curves (DAVIS)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
for name, data in davis_runs.items():
    hist  = data["history"]
    color = COLORS.get(name, "#888888")
    ax.plot(hist["epoch"], hist["val_r"], label=name, color=color, linewidth=1.8)
ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Target (r=0.8)")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Validation Pearson r", fontsize=12)
ax.set_title("SaProt DTI — Training Curves (DAVIS)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_training_curves.png", dpi=150)
plt.close()
print("Saved: 01_training_curves.png")

# ══════════════════════════════════════════════════════════════
# Figure 2 — Predicted vs True scatter (DAVIS)
# ══════════════════════════════════════════════════════════════
n = len(davis_runs)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
if n == 1: axes = [axes]
for ax, (name, data) in zip(axes, davis_runs.items()):
    y_pred = data["preds"]["y_pred"].values
    y_true = data["preds"]["y_true"].values
    r, _   = pearsonr(y_pred, y_true)
    color  = COLORS.get(name, "#888888")
    ax.scatter(y_true, y_pred, alpha=0.15, s=6, color=color)
    lims = [min(y_true.min(), y_pred.min()) - 0.2,
            max(y_true.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True pKd", fontsize=11)
    ax.set_ylabel("Predicted pKd", fontsize=11)
    ax.set_title(f"{name}\nr = {r:.4f}", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
fig.suptitle("SaProt DTI — Predicted vs True pKd (DAVIS Test Set)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "02_scatter_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_scatter_plots.png")

# ══════════════════════════════════════════════════════════════
# Figure 3 — Model comparison bar chart (DAVIS)
# ══════════════════════════════════════════════════════════════
names  = list(davis_runs.keys())
test_r = [davis_runs[n]["result"]["test_pearson_r"] for n in names]
val_r  = [davis_runs[n]["result"]["best_val_r"]     for n in names]
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
ax.set_title("SaProt DTI — Model Comparison (DAVIS)", fontsize=14, fontweight="bold")
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
# Figure 5 — DAVIS vs KIBA cross-dataset comparison
# ══════════════════════════════════════════════════════════════
if kiba_runs:
    # Collect metrics for cross-dataset comparison (4-bit model only)
    davis_4bit = all_runs.get("SaProt-650M-4bit")
    kiba_4bit  = all_runs.get("SaProt-650M-4bit-kiba")

    if davis_4bit and kiba_4bit:
        datasets = ["DAVIS", "KIBA"]
        test_rs  = [davis_4bit["result"]["test_pearson_r"],
                    kiba_4bit["result"]["test_pearson_r"]]
        val_rs   = [davis_4bit["result"]["best_val_r"],
                    kiba_4bit["result"]["best_val_r"]]
        ds_colors = ["#55A868", "#8172B3"]

        x = np.arange(2)
        w = 0.35
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Left: Val r comparison
        ax = axes[0]
        bars = ax.bar(x, val_rs, color=ds_colors, alpha=0.85, edgecolor="white")
        ax.axhline(0.8, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Target (r=0.8)")
        ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=12)
        ax.set_ylabel("Val Pearson r", fontsize=12)
        ax.set_title("Validation Pearson r\n(SaProt-650M-4bit)", fontsize=12, fontweight="bold")
        ax.set_ylim(0.75, 0.85)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, val_rs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        # Right: Test r comparison
        ax = axes[1]
        bars = ax.bar(x, test_rs, color=ds_colors, alpha=0.85, edgecolor="white")
        ax.axhline(0.8, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Target (r=0.8)")
        ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=12)
        ax.set_ylabel("Test Pearson r", fontsize=12)
        ax.set_title("Test Pearson r\n(SaProt-650M-4bit)", fontsize=12, fontweight="bold")
        ax.set_ylim(0.75, 0.85)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, test_rs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        fig.suptitle("Cross-Dataset Generalization: DAVIS vs KIBA",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "05_davis_vs_kiba.png", dpi=150)
        plt.close()
        print("Saved: 05_davis_vs_kiba.png")

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  {'Model':<30} {'Dataset':>6}  {'Test r':>7}  {'Val r':>7}")
print("  " + "-" * 55)
for name, data in all_runs.items():
    r  = data["result"]
    ds = r.get("dataset", "davis").upper()
    print(f"  {name:<30} {ds:>6}  {r['test_pearson_r']:>7.4f}  {r['best_val_r']:>7.4f}")
print("=" * 60)
print(f"\nFigures saved to: {OUT_DIR}/")
