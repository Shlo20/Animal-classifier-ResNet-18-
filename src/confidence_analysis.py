import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import beta

# ----------------------------
# Paths
# ----------------------------
PRED_PATH = "../reports/predictions.csv"
OUT_DIR = "../reports/confidence_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def clamp01(x, eps=1e-6):
    """Avoid exactly 0 or 1 which can break Beta fitting."""
    return np.clip(x, eps, 1 - eps)

def fit_beta_mle(conf_values):
    """
    Fit Beta(alpha, beta) via MLE using scipy.
    Force loc=0, scale=1.
    Returns (alpha, beta).
    """
    x = clamp01(np.asarray(conf_values, dtype=float))
    a, b, loc, scale = beta.fit(x, floc=0, fscale=1)
    return float(a), float(b)

def compute_ece(conf, correct, n_bins=10):
    """
    Expected Calibration Error.
    conf: probabilities in [0,1]
    correct: 1 if correct else 0
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_acc = []
    bin_conf = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # include right edge on last bin
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        count = mask.sum()
        if count == 0:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_counts.append(0)
            continue

        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()

        bin_acc.append(acc)
        bin_conf.append(avg_conf)
        bin_counts.append(int(count))

        ece += (count / len(conf)) * abs(acc - avg_conf)

    return float(ece), bins, np.array(bin_acc), np.array(bin_conf), np.array(bin_counts)

def plot_reliability(bin_acc, bin_conf, bins, out_path):
    """
    Reliability diagram: accuracy vs confidence per bin + y=x reference.
    """
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1])  # perfect calibration line

    # x = avg confidence, y = accuracy
    mask = ~np.isnan(bin_acc) & ~np.isnan(bin_conf)
    plt.scatter(bin_conf[mask], bin_acc[mask])

    plt.xlabel("Average confidence in bin")
    plt.ylabel("Accuracy in bin")
    plt.title("Reliability Diagram (Calibration)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ----------------------------
# Load predictions
# ----------------------------
df = pd.read_csv(PRED_PATH)
print(f"Loaded: {PRED_PATH}")
print(f"Columns: {list(df.columns)}")
print(f"Total predictions: {len(df)}")

# pick confidence column automatically
if "confidence" in df.columns:
    conf_col = "confidence"
elif "top1_confidence" in df.columns:
    conf_col = "top1_confidence"
else:
    raise ValueError("No confidence column found. Expected 'confidence' or 'top1_confidence'.")

print(f"Using confidence column: {conf_col}")

# sanity
required = ["correct"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing required column '{c}' in predictions.csv")

conf_all = clamp01(df[conf_col].values)
correct_all = df["correct"].astype(int).values

# split
correct_df = df[df["correct"] == 1]
incorrect_df = df[df["correct"] == 0]

print(f"Correct predictions: {len(correct_df)}")
print(f"Incorrect predictions: {len(incorrect_df)}")

# ----------------------------
# Basic statistics
# ----------------------------
print("\nConfidence stats (Correct):")
print(correct_df[conf_col].describe())

print("\nConfidence stats (Incorrect):")
print(incorrect_df[conf_col].describe())

# ----------------------------
# Histogram (Correct vs Incorrect)
# ----------------------------
plt.figure(figsize=(8, 5))
plt.hist(clamp01(correct_df[conf_col].values), bins=30, alpha=0.7, label="Correct")
plt.hist(clamp01(incorrect_df[conf_col].values), bins=30, alpha=0.7, label="Incorrect")
plt.xlabel("Prediction Confidence")
plt.ylabel("Count")
plt.title("Confidence Distribution: Correct vs Incorrect")
plt.legend()
plt.tight_layout()

hist_path = os.path.join(OUT_DIR, "confidence_histogram.png")
plt.savefig(hist_path)
plt.close()
print(f"\nSaved histogram: {hist_path}")

# ----------------------------
# Beta distribution fit
# ----------------------------
beta_rows = []

# All
a_all, b_all = fit_beta_mle(conf_all)
beta_rows.append({"group": "all", "alpha": a_all, "beta": b_all})

# Correct
a_c, b_c = fit_beta_mle(correct_df[conf_col].values)
beta_rows.append({"group": "correct", "alpha": a_c, "beta": b_c})

# Incorrect
if len(incorrect_df) > 0:
    a_i, b_i = fit_beta_mle(incorrect_df[conf_col].values)
    beta_rows.append({"group": "incorrect", "alpha": a_i, "beta": b_i})
else:
    beta_rows.append({"group": "incorrect", "alpha": np.nan, "beta": np.nan})

beta_df = pd.DataFrame(beta_rows)
beta_path = os.path.join(OUT_DIR, "beta_fit.csv")
beta_df.to_csv(beta_path, index=False)
print(f"Saved Beta fit params: {beta_path}")
print(beta_df)

# Optional plot: Beta pdf overlay for correct vs incorrect
xs = np.linspace(0.001, 0.999, 400)
plt.figure(figsize=(8, 5))
plt.plot(xs, beta.pdf(xs, a_c, b_c), label="Beta fit (Correct)")
if len(incorrect_df) > 0:
    plt.plot(xs, beta.pdf(xs, a_i, b_i), label="Beta fit (Incorrect)")
plt.xlabel("Confidence")
plt.ylabel("Density")
plt.title("Beta Distribution Fits")
plt.legend()
plt.tight_layout()
beta_plot_path = os.path.join(OUT_DIR, "beta_fit_overlay.png")
plt.savefig(beta_plot_path)
plt.close()
print(f"Saved Beta overlay plot: {beta_plot_path}")

# ----------------------------
# ECE + Reliability Diagram
# ----------------------------
ece, bins, bin_acc, bin_conf, bin_counts = compute_ece(conf_all, correct_all, n_bins=10)
print(f"\nECE (10 bins): {ece:.6f}")

ece_df = pd.DataFrame({
    "bin_left": bins[:-1],
    "bin_right": bins[1:],
    "count": bin_counts,
    "avg_confidence": bin_conf,
    "accuracy": bin_acc
})
ece_table_path = os.path.join(OUT_DIR, "ece_bins.csv")
ece_df.to_csv(ece_table_path, index=False)
print(f"Saved ECE bin table: {ece_table_path}")

reliability_path = os.path.join(OUT_DIR, "reliability_diagram.png")
plot_reliability(bin_acc, bin_conf, bins, reliability_path)
print(f"Saved reliability diagram: {reliability_path}")

# ----------------------------
# Summary CSV 
# ----------------------------
summary = pd.DataFrame({
    "group": ["correct", "incorrect", "all"],
    "mean_confidence": [
        clamp01(correct_df[conf_col].values).mean() if len(correct_df) else np.nan,
        clamp01(incorrect_df[conf_col].values).mean() if len(incorrect_df) else np.nan,
        conf_all.mean()
    ],
    "median_confidence": [
        np.median(clamp01(correct_df[conf_col].values)) if len(correct_df) else np.nan,
        np.median(clamp01(incorrect_df[conf_col].values)) if len(incorrect_df) else np.nan,
        np.median(conf_all)
    ],
    "count": [len(correct_df), len(incorrect_df), len(df)]
})
summary_path = os.path.join(OUT_DIR, "confidence_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"\nSaved summary: {summary_path}")

print("\nDone.")
