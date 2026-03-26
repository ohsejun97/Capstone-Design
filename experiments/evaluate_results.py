"""
evaluate_results.py
===================
Compute evaluation metrics for all models in results/.

Metrics:
  Pearson r  - Linear correlation between predicted and true values
  RMSE       - Root Mean Squared Error (pKd / KIBA score units)
  MAE        - Mean Absolute Error
  CI         - Concordance Index: probability of correctly ranking a pair

Outputs:
  outputs/evaluation_metrics.csv
  outputs/figures/04_metrics_comparison.png  (DAVIS models)
  outputs/figures/06_kiba_metrics.png        (KIBA model, if available)

Usage:
  python experiments/evaluate_results.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
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

def concordance_index(y_true, y_pred, sample=3000, seed=42):
    """0.5=random, 1.0=perfect ranking."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true) > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_true), sample, replace=False)
        y_true, y_pred = y_true[idx], y_pred[idx]
    concordant = total = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] == y_true[j]:
                continue
            total += 1
            if (y_true[i] > y_true[j]) == (y_pred[i] > y_pred[j]):
                concordant += 1
    return concordant / total if total > 0 else 0.0

# ── Collect metrics ────────────────────────────────────────────
rows = []
for d in sorted(RESULTS_DIR.iterdir()):
    if not d.is_dir(): continue
    rj = d / "result.json"
    tp = d / "test_predictions.csv"
    if not (rj.exists() and tp.exists()): continue

    meta   = json.load(open(rj))
    df     = pd.read_csv(tp)
    y_pred = df["y_pred"].values
    y_true = df["y_true"].values

    r, _  = pearsonr(y_pred, y_true)
    rmse  = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
    mae   = float(np.abs(y_pred - y_true).mean())
    print(f"  Computing CI: {meta['run_name']} ...", end=" ", flush=True)
    ci = concordance_index(y_true, y_pred)
    print(f"done (CI={ci:.4f})")

    rows.append({
        "Model":     meta["run_name"],
        "Dataset":   meta.get("dataset", "davis").upper(),
        "Quant":     meta.get("quant", "none"),
        "Pearson r": round(r,    4),
        "RMSE":      round(rmse, 4),
        "MAE":       round(mae,  4),
        "CI":        round(ci,   4),
    })

if not rows:
    print("No results found in results/")
    exit(1)

df_result = pd.DataFrame(rows)
davis_df  = df_result[df_result["Dataset"] == "DAVIS"]
kiba_df   = df_result[df_result["Dataset"] == "KIBA"]

# ── Print ──────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  SaProt DTI — Full Evaluation Metrics")
print("=" * 75)
print(df_result.to_string(index=False))
print("=" * 75)
print("""
Metric guide:
  Pearson r : higher is better. Linear correlation (trend).
  RMSE      : lower is better. Average error in target units.
  MAE       : lower is better. Median-robust error.
  CI        : higher is better. 0.5=random, 1.0=perfect ranking.
""")

# ── Save CSV ───────────────────────────────────────────────────
csv_path = Path(__file__).parent.parent / "outputs" / "evaluation_metrics.csv"
csv_path.parent.mkdir(exist_ok=True)
df_result.to_csv(csv_path, index=False)
print(f"CSV saved: {csv_path}")

# ══════════════════════════════════════════════════════════════
# Figure 4 — DAVIS multi-metric comparison (2x2)
# ══════════════════════════════════════════════════════════════
metrics   = ["Pearson r", "RMSE", "MAE", "CI"]
better    = ["higher",    "lower", "lower", "higher"]
ref_lines = {
    "Pearson r": (0.8,  "Target (r=0.8)"),
    "CI":        (0.85, "Target (CI=0.85)"),
}

names  = davis_df["Model"].tolist()
colors = [COLORS.get(n, "#888888") for n in names]
x      = np.arange(len(names))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, metric, direction in zip(axes, metrics, better):
    values = davis_df[metric].values
    bars   = ax.bar(x, values, color=colors, alpha=0.85, edgecolor="white")
    if metric in ref_lines:
        yval, label = ref_lines[metric]
        ax.axhline(yval, color="red", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=label)
        ax.legend(fontsize=9)
    ymin = min(values) * 0.97
    ymax = max(values) * 1.05
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_title(f"{metric}  ({direction} is better)", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ymax - ymin) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("SaProt DTI — Full Metrics Comparison (DAVIS)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_metrics_comparison.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/04_metrics_comparison.png")

# ══════════════════════════════════════════════════════════════
# Figure 6 — DAVIS vs KIBA metrics comparison (4-bit model)
# ══════════════════════════════════════════════════════════════
if not kiba_df.empty:
    davis_row = davis_df[davis_df["Model"] == "SaProt-650M-4bit"]
    kiba_row  = kiba_df[kiba_df["Model"]  == "SaProt-650M-4bit-kiba"]

    if not davis_row.empty and not kiba_row.empty:
        ds_labels = ["DAVIS", "KIBA"]
        ds_colors = ["#55A868", "#8172B3"]
        x = np.arange(2)

        fig, axes = plt.subplots(1, 4, figsize=(14, 5))
        for ax, metric, direction in zip(axes, metrics, better):
            vals = [float(davis_row[metric].iloc[0]),
                    float(kiba_row[metric].iloc[0])]
            bars = ax.bar(x, vals, color=ds_colors, alpha=0.85, edgecolor="white")
            ymin = min(vals) * 0.97
            ymax = max(vals) * 1.05
            ax.set_ylim(ymin, ymax)
            ax.set_xticks(x)
            ax.set_xticklabels(ds_labels, fontsize=11)
            ax.set_title(f"{metric}\n({direction} is better)", fontsize=11, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (ymax - ymin) * 0.01,
                        f"{val:.4f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")

        fig.suptitle("Cross-Dataset Generalization: DAVIS vs KIBA (SaProt-650M-4bit)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "06_kiba_metrics.png", dpi=150)
        plt.close()
        print(f"Saved: {OUT_DIR}/06_kiba_metrics.png")
