import os
import glob
import pandas as pd

# Directories
RAW_DIR = "raw/watch"
OUT_DIR = "raw_acc_gyr"

# Full activity map based on activity_key.txt
ACTIVITY_MAP = {
    'A': 'walking',
    'B': 'jogging',
    'C': 'stairs',
    'D': 'sitting',
    'E': 'standing',
    'F': 'typing',
    'G': 'teeth',
    'H': 'soup',
    'I': 'chips',
    'J': 'pasta',
    'K': 'drinking',
    'L': 'sandwich',
    'M': 'kicking',
    'O': 'catch',
    'P': 'dribbling',
    'Q': 'writing',
    'R': 'clapping',
    'S': 'folding'
}

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def process_sensor_files(sensor_name):
    """
    sensor_name is 'accel' or 'gyro'
    maps to output folder 'acc' or 'gyr'
    """
    assert sensor_name in ['accel', 'gyro']
    out_sensor_dir = "acc" if sensor_name == "accel" else "gyr"
    sensor_prefix = "acc" if sensor_name == "accel" else "gyr"
    
    sensor_folder = os.path.join(RAW_DIR, sensor_name)
    files = glob.glob(os.path.join(sensor_folder, "*.txt"))
    
    print(f"Found {len(files)} {sensor_name} files to process.")
    for file_path in files:
        # data_1600_accel_watch.txt
        basename = os.path.basename(file_path)
        parts = basename.split('_')
        subject_id = parts[1]
        
        print(f"Processing {sensor_name} for subject {subject_id}")
        
        # Output directory: raw_acc_gyr/1600/acc/
        subject_out_dir = os.path.join(OUT_DIR, subject_id, out_sensor_dir)
        ensure_dir(subject_out_dir)
        
        # Read the file
        # Format: Subject-id, Activity Label, Timestamp, x, y, z;
        data = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Remove trailing ; if present
                if line.endswith(';'):
                    line = line[:-1]
                
                parts = line.split(',')
                if len(parts) != 6:
                    continue
                
                sub_id, act_label, timestamp, x, y, z = parts
                data.append({
                    "id": sub_id,
                    "attr_time": timestamp,
                    "activity": act_label,
                    "attr_x": x,
                    "attr_y": y,
                    "attr_z": z
                })
        
        if not data:
            continue
            
        df = pd.DataFrame(data)
        
        # Group by activity and save
        for act_code, group in df.groupby("activity"):
            act_name = ACTIVITY_MAP.get(act_code, act_code)
            
            # File name mapping e.g. walking_acc_forearm.csv
            csv_name = f"{act_name}_{sensor_prefix}_forearm.csv"
            csv_path = os.path.join(subject_out_dir, csv_name)
            
            # Select ordered columns
            out_df = group[["id", "attr_time", "attr_x", "attr_y", "attr_z"]]
            out_df.to_csv(csv_path, index=False)

def process_all():
    ensure_dir(OUT_DIR)
    process_sensor_files('accel')
    process_sensor_files('gyro')
    print(f"Conversion complete. Data is in {OUT_DIR}/")

if __name__ == "__main__":
    process_all()
