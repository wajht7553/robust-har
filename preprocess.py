import os
import json
import numpy as np
import pandas as pd
from scipy import interpolate


INPUT_DIR = "dataset/raw_acc_gyr"  # folder created after extraction
OUTPUT_DIR = "dataset/processed_acc_gyr"  # final ML-ready dataset
TARGET_RATE = 50  # uniform sampling frequency
WINDOW_SIZE_SEC = 5.0  # window length
STRIDE_SEC = 2.0  # window stride
MIN_SAMPLES = TARGET_RATE * WINDOW_SIZE_SEC

ACTIVITIES = [
    "walking",
    "running",
    "sitting",
    "standing",
    "lying",
    "climbingup",
    "climbingdown",
    "jumping",
]

activity_to_label = {act: i for i, act in enumerate(ACTIVITIES)}

# ================================================================
# UTILITY FUNCTIONS
# ================================================================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_csv(path):
    """
    Loads CSV with columns:
    id, attr_time, attr_x, attr_y, attr_z
    Handles millisecond timestamps automatically.
    """
    df = pd.read_csv(path)

    t = df["attr_time"].values.astype(float)

    # detect ms-scale timestamps
    if np.median(np.diff(t)) > 1.0:
        t = t / 1000.0  # convert ms → s

    data = df[["attr_x", "attr_y", "attr_z"]].values.astype(float)
    return t, data


def synchronize_streams(acc_t, acc_v, gyr_t, gyr_v, rate=100):
    """
    Interpolates acc and gyr onto a common uniform timeline.
    Returns:
        t_uniform, acc_uniform, gyr_uniform
    """

    # intersect time range
    t_start = max(acc_t[0], gyr_t[0])
    t_end = min(acc_t[-1], gyr_t[-1])

    if t_end - t_start < 2.5:
        # too short
        return None, None, None

    # build uniform timeline
    t_uniform = np.arange(t_start, t_end, 1.0 / rate)

    # define helpers
    def interp_columns(t_src, v_src, t_new):
        return np.vstack(
            [
                interpolate.interp1d(
                    t_src,
                    v_src[:, i],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )(t_new)
                for i in range(v_src.shape[1])
            ]
        ).T

    acc_u = interp_columns(acc_t, acc_v, t_uniform)
    gyr_u = interp_columns(gyr_t, gyr_v, t_uniform)

    return t_uniform, acc_u, gyr_u


def make_windows(signal_array, rate, win_sec=2.0, stride_sec=1.0):
    """
    Splits signal_array (N,6) into windows.
    Returns array: (num_windows, win_len, 6)
    """
    win_len = int(win_sec * rate)
    stride = int(stride_sec * rate)

    windows = []
    for start in range(0, len(signal_array) - win_len + 1, stride):
        w = signal_array[start : start + win_len]
        windows.append(w)

    if len(windows) == 0:
        return np.zeros((0, win_len, 6))

    return np.stack(windows)


# ================================================================
# MAIN PROCESSING PIPELINE
# ================================================================


def process_participant(proband):
    """
    Loads all activities for a single participant,
    returns X, y arrays.
    """
    X_list, y_list, raw_list = [], [], []

    p_dir = os.path.join(INPUT_DIR, proband)
    acc_dir = os.path.join(p_dir, "acc")
    gyr_dir = os.path.join(p_dir, "gyr")

    if not os.path.isdir(acc_dir) or not os.path.isdir(gyr_dir):
        print(f"Skipping {proband}: missing acc/ or gyr/")
        return None, None

    for activity in ACTIVITIES:
        # find matching file names for acc & gyr
        acc_file = None
        gyr_file = None

        # find accelerometer file
        for f in os.listdir(acc_dir):
            if activity in f.lower() and f.endswith("_forearm.csv"):
                acc_file = os.path.join(acc_dir, f)
                break

        # find gyroscope file
        for f in os.listdir(gyr_dir):
            if activity in f.lower() and f.endswith("_forearm.csv"):
                gyr_file = os.path.join(gyr_dir, f)
                break

        if acc_file is None or gyr_file is None:
            continue

        print(f"   Processing {proband} | {activity}")

        # load CSVs
        acc_t, acc_v = load_csv(acc_file)
        gyr_t, gyr_v = load_csv(gyr_file)

        # synchronize
        t_u, acc_u, gyr_u = synchronize_streams(acc_t, acc_v, gyr_t, gyr_v, TARGET_RATE)
        if t_u is None:
            print(f"      (Too short, skipping)")
            continue

        # combine acc+gyr → (N, 6)
        imu = np.hstack([acc_u, gyr_u])

        # store per-sample observations (time, 6-d imu, activity_id, subject_id)
        try:
            subj_id = int(proband.replace("proband", ""))
        except Exception:
            subj_id = proband
        n = imu.shape[0]
        if n > 0:
            times_col = t_u.reshape(-1, 1)
            labels_col = np.full((n, 1), activity_to_label[activity], dtype=int)
            subj_col = np.full((n, 1), subj_id, dtype=object)
            rec = np.hstack([times_col, imu, labels_col, subj_col])
            raw_list.append(rec)

        # create windows
        windows = make_windows(imu, TARGET_RATE, WINDOW_SIZE_SEC, STRIDE_SEC)
        labels = np.array([activity_to_label[activity]] * len(windows), dtype=int)

        if len(windows) == 0:
            continue

        X_list.append(windows)
        y_list.append(labels)

    if len(X_list) == 0:
        return None, None

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    raw_arr = np.vstack(raw_list)

    return X, y, raw_arr


def build_dataset():
    ensure_dir(OUTPUT_DIR)

    all_X = []
    all_y = []
    all_raw = []
    participants = sorted(
        os.listdir(INPUT_DIR), key=lambda x: int(x.replace("proband", ""))
    )

    # store per-subject sizes (useful for LOSO)
    subject_index = {}

    print("Participants:", participants)

    sample_offset = 0
    for proband in participants:
        print(f"\n========== {proband} ==========")
        Xp, yp, rawp = process_participant(proband)
        if Xp is None and (rawp is None or len(rawp) == 0):
            print(f"No data for {proband}, skipping.")
            continue
        # only add to subject_index if there are windowed labels (y)
        if yp is not None:
            # Use cumulative sample counts (running offset), not number of subjects
            subject_index[proband] = (sample_offset, sample_offset + len(yp))
            all_X.append(Xp)
            all_y.append(yp)
            sample_offset += len(yp)
        # always collect raw per-sample observations if present
        if rawp is not None and len(rawp) > 0:
            all_raw.append(rawp)

    # Combine all participants windows (if any)
    if len(all_X) > 0:
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        # save windowed outputs as before
        np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
        np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    else:
        X = np.zeros((0,))
        y = np.zeros((0,))

    # save subject_index (unchanged)
    with open(os.path.join(OUTPUT_DIR, "subject_index.json"), "w") as f:
        json.dump(subject_index, f, indent=4)

    # Combine and save raw observations CSV (time, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, activity_id, subject_id)
    if len(all_raw) > 0:
        raw_all = np.vstack(all_raw)
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
        df_raw = pd.DataFrame(raw_all, columns=cols)
        # ensure integer types where appropriate
        df_raw["activity_id"] = df_raw["activity_id"].astype(int)
        # subject_id may be int or string; try cast to int when possible
        try:
            df_raw["subject_id"] = df_raw["subject_id"].astype(int)
        except Exception:
            pass
        csv_path = os.path.join(OUTPUT_DIR, "observations.csv")
        df_raw.to_csv(csv_path, index=False)
        print(f"Saved per-sample observations CSV to: {csv_path}")

    print("\nDataset created!")
    print("X.shape =", X.shape)
    print("y.shape =", y.shape)
    print("Subject index mapping saved.")
    print(f"Saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    build_dataset()
