"""
analysis_observations.py

Performs exploratory analysis on dataset/processed_acc_gyr/observations.csv.
Writes plots to dataset/processed_acc_gyr/analysis_plots/ and summary to dataset/processed_acc_gyr/analysis_summary.json

This script is written to be memory-friendly (uses chunked reading) and to produce
useful diagnostics for modeling decisions.

Run from the repo root:
    python analysis_observations.py

"""

import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
CSV_PATH = os.path.join("dataset", "processed_acc_gyr", "observations.csv")
OUT_DIR = os.path.join("dataset", "processed_acc_gyr", "analysis_plots")
ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
ensure_dir(OUT_DIR)

# Columns expected
cols = [
    "time",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "activity_id",
    "subject_id",
]

# Chunked pass: counts, sums, sumsqs for online mean/std, min/max
chunksize = 200000

# accumulators
count = 0
sum_cols = None
sumsq_cols = None
min_cols = None
max_cols = None

activity_counts = defaultdict(int)
subject_counts = defaultdict(int)

# sampling statistics per (subject, activity) -> list of dt means
sampling_stats = defaultdict(
    lambda: {"n": 0, "dt_sum": 0.0, "dt_sq_sum": 0.0, "dt_min": np.inf, "dt_max": 0.0}
)

# We'll sample one (subject,activity) pair for detailed plotting (the one with most samples)
sample_candidate = None

print("Starting chunked read of:", CSV_PATH)
for chunk in pd.read_csv(CSV_PATH, usecols=cols, chunksize=chunksize):
    # ensure correct dtypes
    chunk = chunk.dropna()
    numeric = chunk[["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]].astype(
        float
    )

    # initialize accumulators
    if sum_cols is None:
        sum_cols = np.zeros(numeric.shape[1], dtype=float)
        sumsq_cols = np.zeros(numeric.shape[1], dtype=float)
        min_cols = np.full(numeric.shape[1], np.inf, dtype=float)
        max_cols = np.full(numeric.shape[1], -np.inf, dtype=float)

    n = len(numeric)
    count += n
    arr = numeric.values
    sum_cols += arr.sum(axis=0)
    sumsq_cols += (arr**2).sum(axis=0)
    min_cols = np.minimum(min_cols, arr.min(axis=0))
    max_cols = np.maximum(max_cols, arr.max(axis=0))

    # activity and subject counts
    for a, c in chunk["activity_id"].value_counts().items():
        activity_counts[int(a)] += int(c)
    for s, c in chunk["subject_id"].value_counts().items():
        subject_counts[str(s)] += int(c)

    # sampling stats grouped by subject+activity (requires sorting by time per group)
    for (s, a), g in chunk.groupby(["subject_id", "activity_id"]):
        times = np.sort(g["time"].values.astype(float))
        if len(times) < 2:
            continue
        diffs = np.diff(times)
        dsum = float(diffs.sum())
        d2 = float((diffs**2).sum())
        st = sampling_stats[(str(s), int(a))]
        st["n"] += len(diffs)
        st["dt_sum"] += dsum
        st["dt_sq_sum"] += d2
        st["dt_min"] = min(st["dt_min"], float(diffs.min()))
        st["dt_max"] = max(st["dt_max"], float(diffs.max()))

    # pick candidate for detailed plotting: most frequent subject/activity encountered so far
    # update sample_candidate to the group with max count
    if sample_candidate is None:
        # pick top activity and subject in this chunk
        top = (
            chunk.groupby(["subject_id", "activity_id"])
            .size()
            .reset_index(name="cnt")
            .sort_values("cnt", ascending=False)
        )
        if len(top) > 0:
            sample_candidate = (
                str(top.iloc[0]["subject_id"]),
                int(top.iloc[0]["activity_id"]),
            )

print("Finished chunked pass. Total samples:", count)

# compute global mean/std
means = sum_cols / count
stds = np.sqrt((sumsq_cols / count) - (means**2))

axis_names = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
summary = {"n_samples": int(count)}
for i, name in enumerate(axis_names):
    summary[name] = {
        "mean": float(means[i]),
        "std": float(stds[i]),
        "min": float(min_cols[i]),
        "max": float(max_cols[i]),
    }

# activity and subject counts
summary["activity_counts"] = {int(k): int(v) for k, v in activity_counts.items()}
summary["subject_counts"] = {str(k): int(v) for k, v in subject_counts.items()}

# sampling summary: compute mean dt per group and global
group_sampling = {}
all_dts = []
for k, v in sampling_stats.items():
    if v["n"] == 0:
        continue
    mean_dt = v["dt_sum"] / v["n"]
    var = (v["dt_sq_sum"] / v["n"]) - (mean_dt**2)
    std_dt = np.sqrt(max(var, 0.0))
    group_sampling[k] = {
        "n_intervals": int(v["n"]),
        "mean_dt": float(mean_dt),
        "std_dt": float(std_dt),
        "min_dt": float(v["dt_min"]),
        "max_dt": float(v["dt_max"]),
    }
    all_dts.append(mean_dt)

if len(all_dts) > 0:
    summary["sampling_global_mean_dt"] = float(np.mean(all_dts))
    summary["sampling_global_median_dt"] = float(np.median(all_dts))
    summary["sampling_global_std_mean_dt"] = float(np.std(all_dts))
else:
    summary["sampling_global_mean_dt"] = None

# write summary JSON
summary_path = os.path.join(os.path.dirname(CSV_PATH), "analysis_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print("Wrote summary to", summary_path)

# Now load selected sample group for detailed visualizations
if sample_candidate is None:
    print("No suitable sample candidate found for plotting - dataset may be empty.")
else:
    s_subj, s_act = sample_candidate
    print("Selected sample candidate for time-series plotting:", sample_candidate)
    # load filtered rows for that subject/activity
    df_sample = pd.read_csv(CSV_PATH, usecols=cols)
    df_sample = df_sample[
        (df_sample["subject_id"].astype(str) == str(s_subj))
        & (df_sample["activity_id"] == int(s_act))
    ]
    df_sample = df_sample.sort_values("time")

    if len(df_sample) == 0:
        print("Sample candidate yielded no rows after full load.")
    else:
        t0 = df_sample["time"].values.astype(float)
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(t0 - t0[0], df_sample[["acc_x", "acc_y", "acc_z"]].values)
        ax[0].set_title(f"Accelerometer - subject={s_subj}, activity={s_act}")
        ax[0].legend(["acc_x", "acc_y", "acc_z"])
        ax[0].set_ylabel("m/s^2 (or device units)")

        ax[1].plot(t0 - t0[0], df_sample[["gyr_x", "gyr_y", "gyr_z"]].values)
        ax[1].set_title(f"Gyroscope - subject={s_subj}, activity={s_act}")
        ax[1].legend(["gyr_x", "gyr_y", "gyr_z"])
        ax[1].set_ylabel("rad/s (or device units)")
        ax[1].set_xlabel("time (s)")
        plt.tight_layout()
        p = os.path.join(OUT_DIR, f"timeseries_subj_{s_subj}_act_{s_act}.png")
        fig.savefig(p)
        plt.close(fig)
        print("Saved time-series plot to", p)

# Histograms and violin plots per activity: load a sample frame (for memory, load a fraction)
# We'll load the full CSV but only keep necessary columns; this may still be large but reasonable for saving plots.
print("Loading reduced CSV for activity-wise plots (may take a moment)...")
df = pd.read_csv(CSV_PATH, usecols=cols)

# histograms per axis
for axis in axis_names:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[axis].dropna(), bins=200, kde=True)
    plt.title(f"Histogram: {axis}")
    plt.xlabel(axis)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"hist_{axis}.png")
    plt.savefig(p)
    plt.close()
    print("Saved", p)

# violin plots per activity (to show distribution differences)
plt.figure(figsize=(12, 6))
sns.violinplot(x="activity_id", y="acc_x", data=df.sample(frac=0.1, random_state=1))
plt.title("acc_x distribution per activity (sampled 10%)")
plt.tight_layout()
p = os.path.join(OUT_DIR, "violin_acc_x_by_activity.png")
plt.savefig(p)
plt.close()
print("Saved", p)

# correlation heatmap across the 6 sensor channels
corr = df[axis_names].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation between sensor channels")
plt.tight_layout()
p = os.path.join(OUT_DIR, "corr_heatmap.png")
plt.savefig(p)
plt.close()
print("Saved", p)

# sampling interval histogram (global)
# compute diffs on a per-subject full load basis for accuracy
all_diffs = []
for (s, a), meta in group_sampling.items():
    all_diffs.append(meta["mean_dt"])
if len(all_diffs) > 0:
    plt.figure(figsize=(8, 4))
    sns.histplot(all_diffs, bins=100)
    plt.title("Distribution of mean sampling intervals (per subject+activity)")
    plt.xlabel("mean dt (s)")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "sampling_intervals_hist.png")
    plt.savefig(p)
    plt.close()
    print("Saved", p)

# activity counts bar
plt.figure(figsize=(10, 4))
items = sorted(summary["activity_counts"].items(), key=lambda x: x[0])
acts, counts = zip(*items) if len(items) > 0 else ([], [])
sns.barplot(x=list(acts), y=list(counts))
plt.title("Activity counts (total samples)")
plt.xlabel("activity_id")
plt.ylabel("samples")
plt.tight_layout()
p = os.path.join(OUT_DIR, "activity_counts.png")
plt.savefig(p)
plt.close()
print("Saved", p)

# Save a short recommendations text
rec_path = os.path.join(os.path.dirname(CSV_PATH), "model_recommendations.txt")
with open(rec_path, "w") as f:
    f.write("Data analysis summary and model guidance\n")
    f.write("------------------------------------\n")
    f.write(f"Total samples: {summary['n_samples']}\n")
    f.write("Sensor channels: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z\n")
    if summary["sampling_global_mean_dt"] is not None:
        sr = 1.0 / summary["sampling_global_mean_dt"]
        f.write(
            f"Estimated global sampling rate (approx): {sr:.2f} Hz (median dt = {summary['sampling_global_median_dt']:.4f} s)\n"
        )
    f.write("\nSuggested modeling guidelines:\n")
    f.write(
        "- Use windowed inputs (the repo already uses 4s windows). This is sensible for activity recognition.\n"
    )
    f.write(
        "- If sampling is stable around 50-100Hz, preserve temporal resolution; otherwise resample to a fixed rate.\n"
    )
    f.write(
        "- Models: 1D CNNs over windows (good baseline), hybrid CNN+RNN for temporal modeling, or Transformers if you have substantial data and compute.\n"
    )
    f.write(
        "- Use per-axis normalization (z-score) per-subject for robustness to device placement differences.\n"
    )
    f.write(
        "- Evaluate with LOSO/subject-wise splits to measure generalization across people.\n"
    )
    f.write(
        "- Augmentations: jitter, scaling, small time-warping, axis dropout, and noise injection.\n"
    )
print("Wrote recommendations to", rec_path)

print("\nAll analysis plots saved to:", OUT_DIR)
print("Summary JSON:", summary_path)
print("Done.")