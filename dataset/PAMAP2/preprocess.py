import os
import json
import numpy as np
import pandas as pd

INPUT_DIR = "raw_acc_gyr"
OUTPUT_DIR = "processed_acc_gyr"

# 100Hz as per original PAMAP2 layout
TARGET_RATE = 100
WINDOW_SIZE_SEC = 5.0
STRIDE_SEC = 2.0

ACTIVITIES = [
    'walking', 'running', 'sitting', 'standing', 'lying', 
    'climbingup', 'climbingdown', 'jumping'
]
activity_to_label = {act: i for i, act in enumerate(ACTIVITIES)}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_csv(path):
    df = pd.read_csv(path)
    df_numeric = df.select_dtypes(include=["number"])
    df[df_numeric.columns] = df_numeric.interpolate(method='linear').ffill().bfill()
    t = df["attr_time"].values.astype(float)
    # PAMAP2 natively handles time as seconds so no adjustment is needed
    data = df[["attr_x", "attr_y", "attr_z"]].values.astype(float)
    return t, data

def synchronize_streams(acc_t, acc_v, gyr_t, gyr_v):
    # Perfect parallel match truncation logic
    t_start = max(acc_t[0], gyr_t[0])
    t_end = min(acc_t[-1], gyr_t[-1])

    if t_end - t_start < 2.5:
        return None, None, None

    acc_mask = (acc_t >= t_start) & (acc_t <= t_end)
    gyr_mask = (gyr_t >= t_start) & (gyr_t <= t_end)

    acc_t_crop = acc_t[acc_mask]
    acc_v_crop = acc_v[acc_mask]
    gyr_t_crop = gyr_t[gyr_mask]
    gyr_v_crop = gyr_v[gyr_mask]

    n = min(len(acc_t_crop), len(gyr_t_crop))
    if n == 0:
        return None, None, None

    return acc_t_crop[:n], acc_v_crop[:n], gyr_v_crop[:n]

def make_windows(signal_array, rate, win_sec, stride_sec):
    win_len = int(win_sec * rate)
    stride = int(stride_sec * rate)
    windows = []
    for start in range(0, len(signal_array) - win_len + 1, stride):
        w = signal_array[start : start + win_len]
        windows.append(w)
    if not windows:
        return np.zeros((0, win_len, 6))
    return np.stack(windows)

def process_participant(proband):
    X_list, y_list, raw_list = [], [], []
    
    p_dir = os.path.join(INPUT_DIR, proband)
    acc_dir = os.path.join(p_dir, "acc")
    gyr_dir = os.path.join(p_dir, "gyr")

    if not os.path.isdir(acc_dir) or not os.path.isdir(gyr_dir):
        return None, None, None

    for activity in ACTIVITIES:
        acc_file, gyr_file = None, None

        for f in os.listdir(acc_dir):
            if activity in f.lower() and f.endswith("_forearm.csv"):
                acc_file = os.path.join(acc_dir, f)
                break
        for f in os.listdir(gyr_dir):
            if activity in f.lower() and f.endswith("_forearm.csv"):
                gyr_file = os.path.join(gyr_dir, f)
                break

        if not acc_file or not gyr_file:
            continue

        print(f"   Processing {proband} | {activity}")

        acc_t, acc_v = load_csv(acc_file)
        gyr_t, gyr_v = load_csv(gyr_file)

        t_u, acc_u, gyr_u = synchronize_streams(acc_t, acc_v, gyr_t, gyr_v)
        if t_u is None:
            continue

        imu = np.hstack([acc_u, gyr_u])

        try:
            subj_id = int(''.join(filter(str.isdigit, proband)))
        except:
            subj_id = proband

        n_samples = imu.shape[0]
        if n_samples > 0:
            times_col = t_u.reshape(-1, 1)
            labels_col = np.full((n_samples, 1), activity_to_label[activity], dtype=int)
            subj_col = np.full((n_samples, 1), subj_id, dtype=object)
            rec = np.hstack([times_col, imu, labels_col, subj_col])
            raw_list.append(rec)

        windows = make_windows(imu, TARGET_RATE, WINDOW_SIZE_SEC, STRIDE_SEC)
        if len(windows) > 0:
            labels = np.full((len(windows),), activity_to_label[activity], dtype=int)
            X_list.append(windows)
            y_list.append(labels)

    if not X_list:
        return None, None, None

    return np.vstack(X_list), np.concatenate(y_list), np.vstack(raw_list)

def build_dataset():
    ensure_dir(OUTPUT_DIR)
    
    all_X, all_y, all_raw = [], [], []
    # Sorted generically
    participants = sorted(os.listdir(INPUT_DIR))
    subject_index = {}

    sample_offset = 0
    for proband in participants:
        print(f"\n========== {proband} ==========")
        Xp, yp, rawp = process_participant(proband)
        if Xp is None and (rawp is None or len(rawp) == 0):
            print(f"No data for {proband}, skipping.")
            continue
            
        if yp is not None:
            subject_index[proband] = (sample_offset, sample_offset + len(yp))
            all_X.append(Xp)
            all_y.append(yp)
            sample_offset += len(yp)
            
        if rawp is not None and len(rawp) > 0:
            all_raw.append(rawp)

    if len(all_X) > 0:
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
        np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    else:
        X = np.zeros((0,))
        y = np.zeros((0,))

    with open(os.path.join(OUTPUT_DIR, "subject_index.json"), "w") as f:
        json.dump(subject_index, f, indent=4)

    if len(all_raw) > 0:
        raw_all = np.vstack(all_raw)
        cols = ["time", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "activity_id", "subject_id"]
        df_raw = pd.DataFrame(raw_all, columns=cols)
        df_raw["activity_id"] = df_raw["activity_id"].astype(int)
        csv_path = os.path.join(OUTPUT_DIR, "observations.csv")
        df_raw.to_csv(csv_path, index=False)
        print(f"Saved observations CSV to: {csv_path}")

    print(f"\nDataset extracted! X.shape={X.shape}, y.shape={y.shape}")

if __name__ == "__main__":
    build_dataset()
