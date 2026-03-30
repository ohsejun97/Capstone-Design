"""
visualize_results.py
====================
Visualize SaProt DTI experiment results.

Figures saved to outputs/figures/:
  --- Baseline (placeholder '#') ---
  01_training_curves.png        - Val Pearson r per epoch (DAVIS models)
  02_scatter_plots.png          - Predicted vs True pKd scatter (DAVIS models)
  03_model_comparison.png       - Test/Val Pearson r bar chart (DAVIS models)
  05_kiba_model_comparison.png  - KIBA multi-model comparison
  06_cross_dataset.png          - DAVIS vs KIBA cross-dataset (baseline)

  --- Phase 3: FoldSeek 3Di ---
  08_3di_davis_comparison.png   - DAVIS: placeholder vs 3Di per model
  09_3di_kiba_comparison.png    - KIBA:  placeholder vs 3Di per model
  10_3di_cross_dataset.png      - 3Di cross-dataset (DAVIS-3Di vs KIBA-3Di)

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

# ── Color palette ──────────────────────────────────────────────
# Baseline (placeholder)
COLORS = {
    "SaProt-650M":            "#4C72B0",
    "SaProt-35M":             "#DD8452",
    "SaProt-650M-4bit":       "#55A868",
    "SaProt-650M-8bit":       "#C44E52",
    "SaProt-650M-kiba":       "#4C72B0",
    "SaProt-35M-kiba":        "#DD8452",
    "SaProt-650M-4bit-kiba":  "#55A868",
    "SaProt-650M-8bit-kiba":  "#C44E52",
    # 3Di variants (darker shades of same model colors)
    "SaProt-650M-davis-3di":      "#1A3F7A",
    "SaProt-35M-davis-3di":       "#A04010",
    "SaProt-650M-4bit-davis-3di": "#1D6B33",
    "SaProt-650M-8bit-davis-3di": "#7A1012",
    "SaProt-650M-kiba-3di":       "#1A3F7A",
    "SaProt-35M-kiba-3di":        "#A04010",
    "SaProt-650M-4bit-kiba-3di":  "#1D6B33",
    "SaProt-650M-8bit-kiba-3di":  "#7A1012",
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
        "use_3di": meta.get("use_3di", False),
    }

# 분류: dataset × 3di 여부
davis_base = {k: v for k, v in all_runs.items()
              if v["dataset"] == "davis" and not v["use_3di"]}
davis_3di  = {k: v for k, v in all_runs.items()
              if v["dataset"] == "davis" and v["use_3di"]}
kiba_base  = {k: v for k, v in all_runs.items()
              if v["dataset"] == "kiba"  and not v["use_3di"]}
kiba_3di   = {k: v for k, v in all_runs.items()
              if v["dataset"] == "kiba"  and v["use_3di"]}

print(f"DAVIS base : {list(davis_base.keys())}")
print(f"DAVIS 3Di  : {list(davis_3di.keys())}")
print(f"KIBA  base : {list(kiba_base.keys())}")
print(f"KIBA  3Di  : {list(kiba_3di.keys())}")


# ══════════════════════════════════════════════════════════════
# Figure 1 — Training curves (DAVIS baseline)
# ══════════════════════════════════════════════════════════════
if davis_base:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in davis_base.items():
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
# Figure 2 — Predicted vs True scatter (DAVIS baseline)
# ══════════════════════════════════════════════════════════════
if davis_base:
    n = len(davis_base)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1: axes = [axes]
    for ax, (name, data) in zip(axes, davis_base.items()):
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
# Figure 3 — Model comparison bar chart (DAVIS baseline)
# ══════════════════════════════════════════════════════════════
if davis_base:
    names  = list(davis_base.keys())
    test_r = [davis_base[n]["result"]["test_pearson_r"] for n in names]
    val_r  = [davis_base[n]["result"]["best_val_r"]     for n in names]
    colors = [COLORS.get(n, "#888888") for n in names]
    x = np.arange(len(names))
    w = 0.35

    all_vals = test_r + val_r
    ymin = min(all_vals) * 0.97
    ymax = max(all_vals) * 1.03

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, val_r,  w, label="Val Pearson r",  color=colors, alpha=0.6)
    bars2 = ax.bar(x + w/2, test_r, w, label="Test Pearson r", color=colors, alpha=1.0)
    ax.axhline(0.8, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Target (r=0.8)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_title("SaProt DTI — Model Comparison (DAVIS)", fontsize=14, fontweight="bold")
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=10)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (ymax - ymin) * 0.01,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_model_comparison.png", dpi=150)
    plt.close()
    print("Saved: 03_model_comparison.png")

# ══════════════════════════════════════════════════════════════
# Figure 5 — KIBA baseline comparison
# ══════════════════════════════════════════════════════════════
if kiba_base:
    kiba_names  = list(kiba_base.keys())
    kiba_test_r = [kiba_base[n]["result"]["test_pearson_r"] for n in kiba_names]
    kiba_val_r  = [kiba_base[n]["result"]["best_val_r"]     for n in kiba_names]
    kiba_colors = [COLORS.get(n, "#888888") for n in kiba_names]

    all_vals = kiba_test_r + kiba_val_r
    ymin = min(all_vals) * 0.97
    ymax = max(all_vals) * 1.03

    x = np.arange(len(kiba_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, kiba_val_r,  w, label="Val Pearson r",  color=kiba_colors, alpha=0.6)
    bars2 = ax.bar(x + w/2, kiba_test_r, w, label="Test Pearson r", color=kiba_colors, alpha=1.0)
    ax.axhline(0.8, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Target (r=0.8)")
    ax.set_xticks(x)
    ax.set_xticklabels(kiba_names, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_title("SaProt DTI — Model Comparison (KIBA)", fontsize=14, fontweight="bold")
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=10)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (ymax - ymin) * 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_kiba_model_comparison.png", dpi=150)
    plt.close()
    print("Saved: 05_kiba_model_comparison.png")

# ══════════════════════════════════════════════════════════════
# Figure 6 — Cross-dataset: DAVIS vs KIBA (baseline)
# ══════════════════════════════════════════════════════════════
if kiba_base:
    pairs = [
        ("SaProt-35M",        "SaProt-35M-kiba"),
        ("SaProt-650M-8bit",  "SaProt-650M-8bit-kiba"),
        ("SaProt-650M-4bit",  "SaProt-650M-4bit-kiba"),
    ]
    valid_pairs = [(d, k) for d, k in pairs if d in all_runs and k in all_runs]

    if valid_pairs:
        labels     = [d.replace("SaProt-", "") for d, _ in valid_pairs]
        davis_test = [all_runs[d]["result"]["test_pearson_r"] for d, _ in valid_pairs]
        kiba_test  = [all_runs[k]["result"]["test_pearson_r"] for _, k in valid_pairs]

        all_vals = davis_test + kiba_test
        ymin = min(all_vals) * 0.97
        ymax = max(all_vals) * 1.03

        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        bars1 = ax.bar(x - w/2, davis_test, w, label="DAVIS", color="#4878D0", alpha=0.9)
        bars2 = ax.bar(x + w/2, kiba_test,  w, label="KIBA",  color="#EE854A", alpha=0.9)
        ax.axhline(0.8, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Target (r=0.8)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Test Pearson r", fontsize=12)
        ax.set_title("Cross-Dataset Generalization: DAVIS vs KIBA (Baseline)", fontsize=14, fontweight="bold")
        ax.set_ylim(ymin, ymax)
        ax.legend(fontsize=11)
        for bar in list(bars1) + list(bars2):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (ymax - ymin) * 0.005,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "06_cross_dataset.png", dpi=150)
        plt.close()
        print("Saved: 06_cross_dataset.png")

# ══════════════════════════════════════════════════════════════
# Figure 8 — DAVIS: Placeholder vs 3Di comparison
# ══════════════════════════════════════════════════════════════
if davis_3di:
    pairs_davis = [
        ("SaProt-650M",      "SaProt-650M-davis-3di",      "650M"),
        ("SaProt-35M",       "SaProt-35M-davis-3di",        "35M"),
        ("SaProt-650M-8bit", "SaProt-650M-8bit-davis-3di",  "650M-8bit"),
        ("SaProt-650M-4bit", "SaProt-650M-4bit-davis-3di",  "650M-4bit"),
    ]
    valid = [(b, d, lbl) for b, d, lbl in pairs_davis if d in all_runs]

    if valid:
        labels    = [lbl for _, _, lbl in valid]
        base_vals = [all_runs[b]["result"]["test_pearson_r"] if b in all_runs else None
                     for b, _, _ in valid]
        di_vals   = [all_runs[d]["result"]["test_pearson_r"] for _, d, _ in valid]

        x = np.arange(len(labels))
        w = 0.35

        all_vals = [v for v in base_vals + di_vals if v is not None]
        ymin = min(all_vals) * 0.97
        ymax = max(all_vals) * 1.03

        fig, ax = plt.subplots(figsize=(9, 5))

        # Baseline bars (lighter)
        base_colors = ["#4C72B0", "#DD8452", "#C44E52", "#55A868"]
        di_colors   = ["#1A3F7A", "#A04010", "#7A1012", "#1D6B33"]

        for i, (bv, dv, bc, dc) in enumerate(zip(base_vals, di_vals, base_colors, di_colors)):
            if bv is not None:
                b1 = ax.bar(i - w/2, bv, w, color=bc, alpha=0.6,
                            label="Placeholder '#'" if i == 0 else "")
            b2 = ax.bar(i + w/2, dv, w, color=dc, alpha=0.95,
                        label="FoldSeek 3Di" if i == 0 else "")

            # Delta annotation
            if bv is not None:
                delta = dv - bv
                sign  = "+" if delta >= 0 else ""
                color = "green" if delta >= 0 else "red"
                ax.text(i, max(bv, dv) + (ymax - ymin) * 0.02,
                        f"{sign}{delta:.4f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color=color)

        ax.axhline(0.8, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Target (r=0.8)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Test Pearson r", fontsize=12)
        ax.set_title("DAVIS — Placeholder '#' vs FoldSeek 3Di Tokens", fontsize=14, fontweight="bold")
        ax.set_ylim(ymin, ymax)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "08_3di_davis_comparison.png", dpi=150)
        plt.close()
        print("Saved: 08_3di_davis_comparison.png")

# ══════════════════════════════════════════════════════════════
# Figure 9 — KIBA: Placeholder vs 3Di comparison
# ══════════════════════════════════════════════════════════════
if kiba_3di:
    pairs_kiba = [
        ("SaProt-650M-kiba",      "SaProt-650M-kiba-3di",      "650M"),
        ("SaProt-35M-kiba",       "SaProt-35M-kiba-3di",        "35M"),
        ("SaProt-650M-8bit-kiba", "SaProt-650M-8bit-kiba-3di",  "650M-8bit"),
        ("SaProt-650M-4bit-kiba", "SaProt-650M-4bit-kiba-3di",  "650M-4bit"),
    ]
    valid = [(b, d, lbl) for b, d, lbl in pairs_kiba if d in all_runs]

    if valid:
        labels    = [lbl for _, _, lbl in valid]
        base_vals = [all_runs[b]["result"]["test_pearson_r"] if b in all_runs else None
                     for b, _, _ in valid]
        di_vals   = [all_runs[d]["result"]["test_pearson_r"] for _, d, _ in valid]

        x = np.arange(len(labels))
        w = 0.35

        all_vals = [v for v in base_vals + di_vals if v is not None]
        ymin = min(all_vals) * 0.97
        ymax = max(all_vals) * 1.03

        base_colors = ["#4C72B0", "#DD8452", "#C44E52", "#55A868"]
        di_colors   = ["#1A3F7A", "#A04010", "#7A1012", "#1D6B33"]

        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (bv, dv, bc, dc) in enumerate(zip(base_vals, di_vals, base_colors, di_colors)):
            if bv is not None:
                ax.bar(i - w/2, bv, w, color=bc, alpha=0.6,
                       label="Placeholder '#'" if i == 0 else "")
            ax.bar(i + w/2, dv, w, color=dc, alpha=0.95,
                   label="FoldSeek 3Di" if i == 0 else "")

            if bv is not None:
                delta = dv - bv
                sign  = "+" if delta >= 0 else ""
                color = "green" if delta >= 0 else "red"
                ax.text(i, max(bv, dv) + (ymax - ymin) * 0.02,
                        f"{sign}{delta:.4f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color=color)

        ax.axhline(0.8, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Target (r=0.8)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Test Pearson r", fontsize=12)
        ax.set_title("KIBA — Placeholder '#' vs FoldSeek 3Di Tokens", fontsize=14, fontweight="bold")
        ax.set_ylim(ymin, ymax)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "09_3di_kiba_comparison.png", dpi=150)
        plt.close()
        print("Saved: 09_3di_kiba_comparison.png")

# ══════════════════════════════════════════════════════════════
# Figure 10 — 3Di cross-dataset: DAVIS-3Di vs KIBA-3Di
# ══════════════════════════════════════════════════════════════
if davis_3di and kiba_3di:
    pairs_3di = [
        ("SaProt-35M-davis-3di",       "SaProt-35M-kiba-3di",        "35M"),
        ("SaProt-650M-8bit-davis-3di", "SaProt-650M-8bit-kiba-3di",  "650M-8bit"),
        ("SaProt-650M-4bit-davis-3di", "SaProt-650M-4bit-kiba-3di",  "650M-4bit"),
        ("SaProt-650M-davis-3di",      "SaProt-650M-kiba-3di",       "650M"),
    ]
    valid = [(d, k, lbl) for d, k, lbl in pairs_3di if d in all_runs and k in all_runs]

    if valid:
        labels     = [lbl for _, _, lbl in valid]
        davis_vals = [all_runs[d]["result"]["test_pearson_r"] for d, _, _ in valid]
        kiba_vals  = [all_runs[k]["result"]["test_pearson_r"] for _, k, _ in valid]

        all_vals = davis_vals + kiba_vals
        ymin = min(all_vals) * 0.97
        ymax = max(all_vals) * 1.03

        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        bars1 = ax.bar(x - w/2, davis_vals, w, label="DAVIS (3Di)", color="#1A3F7A", alpha=0.9)
        bars2 = ax.bar(x + w/2, kiba_vals,  w, label="KIBA (3Di)",  color="#A04010", alpha=0.9)
        ax.axhline(0.8, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Target (r=0.8)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Test Pearson r", fontsize=12)
        ax.set_title("Cross-Dataset Generalization: DAVIS-3Di vs KIBA-3Di", fontsize=14, fontweight="bold")
        ax.set_ylim(ymin, ymax)
        ax.legend(fontsize=11)
        for bar in list(bars1) + list(bars2):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (ymax - ymin) * 0.005,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "10_3di_cross_dataset.png", dpi=150)
        plt.close()
        print("Saved: 10_3di_cross_dataset.png")

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  {'Model':<35} {'Dataset':>6}  {'3Di':>5}  {'Test r':>7}  {'Val r':>7}")
print("  " + "-" * 60)
for name, data in all_runs.items():
    r  = data["result"]
    ds = r.get("dataset", "davis").upper()
    di = "Yes" if r.get("use_3di") else "No"
    print(f"  {name:<35} {ds:>6}  {di:>5}  "
          f"{r['test_pearson_r']:>7.4f}  {r['best_val_r']:>7.4f}")
print("=" * 65)
print(f"\nFigures saved to: {OUT_DIR}/")
