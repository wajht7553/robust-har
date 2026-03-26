import os
import glob
import pandas as pd

RAW_DIR = "raw"
OUT_DIR = "raw_acc_gyr"

# Overlapping activities modeled from user feedback
ACTIVITY_MAP = {
    1: 'lying',
    2: 'sitting',
    3: 'standing',
    4: 'walking',
    5: 'running',
    12: 'climbingup',
    13: 'climbingdown',
    24: 'jumping'
}

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def process_all():
    ensure_dir(OUT_DIR)
    # The pamap files are .dat in text-based column layout
    files = glob.glob(os.path.join(RAW_DIR, "*.dat"))
    
    print(f"Found {len(files)} .dat files to process.")
    for file_path in files:
        basename = os.path.basename(file_path)
        subject_id = basename.split('.')[0]
        
        print(f"Processing {subject_id} ...")
        
        acc_dir = os.path.join(OUT_DIR, subject_id, "acc")
        gyr_dir = os.path.join(OUT_DIR, subject_id, "gyr")
        ensure_dir(acc_dir)
        ensure_dir(gyr_dir)
        
        # Pull wrist IMU mappings natively from the documentation
        cols_to_use = [0, 1, 4, 5, 6, 10, 11, 12]
        col_names = ["attr_time", "activity", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
        
        # Read the file via space delimiter
        df = pd.read_csv(file_path, sep=r"\s+", usecols=cols_to_use, names=col_names, header=None)
        
        # Fill missing values natively per column
        df = df.interpolate(method='linear').bfill().ffill()
        
        # Act mapping
        df_valid = df[df["activity"].isin(ACTIVITY_MAP.keys())].copy()
        
        for act_code, group in df_valid.groupby("activity"):
            act_name = ACTIVITY_MAP[int(act_code)]
            
            # Output matching the downstream ingestion formats
            acc_csv_path = os.path.join(acc_dir, f"{act_name}_acc_forearm.csv")
            acc_df = group[["activity", "attr_time", "acc_x", "acc_y", "acc_z"]].copy()
            acc_df.rename(columns={"activity": "id", "acc_x": "attr_x", "acc_y": "attr_y", "acc_z": "attr_z"}, inplace=True)
            acc_df["id"] = subject_id
            acc_df.to_csv(acc_csv_path, index=False)
            
            gyr_csv_path = os.path.join(gyr_dir, f"{act_name}_gyr_forearm.csv")
            gyr_df = group[["activity", "attr_time", "gyr_x", "gyr_y", "gyr_z"]].copy()
            gyr_df.rename(columns={"activity": "id", "gyr_x": "attr_x", "gyr_y": "attr_y", "gyr_z": "attr_z"}, inplace=True)
            gyr_df["id"] = subject_id
            gyr_df.to_csv(gyr_csv_path, index=False)

    print("Extraction complete!")

if __name__ == "__main__":
    process_all()
