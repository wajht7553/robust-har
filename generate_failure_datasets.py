import os
import numpy as np
import torch
import json
from tqdm import tqdm
from src.data.transforms import *

INPUT_DIR = "dataset/processed_acc_gyr"
OUTPUT_BASE = "dataset"

def load_base_dataset():
    X_path = os.path.join(INPUT_DIR, "X.npy")
    y_path = os.path.join(INPUT_DIR, "y.npy")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Base dataset not found in {INPUT_DIR}")
        
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y

def save_dataset(X, y, name):
    out_dir = os.path.join(OUTPUT_BASE, name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    print(f"Saved {name}: {X.shape}")

def apply_transform(X, transform_fn):
    X_new = []
    print(f"Applying transform: {type(transform_fn).__name__}")
    for i in tqdm(range(len(X))):
        tensor = torch.from_numpy(X[i]).float()
        aug = transform_fn(tensor)
        X_new.append(aug.numpy())
    return np.stack(X_new)

def main():
    print("Loading base dataset...")
    X, y = load_base_dataset()
    
    # 1. Bias/Drift (Gyro Z)
    print("\nGenerating: drift_gyro_z")
    # Drift rate: 0.01 rad/s per sample? No, per second usually.
    # If 50Hz, 0.01 per second = 0.01/50 per sample.
    # Let's assume drift_rate in transform is per sample for simplicity or adjust.
    # Transform implementation: drift = t * self.drift_rate.
    # If we want 0.1 rad drift over 4 seconds (window), rate = 0.1/200 = 0.0005
    t_drift = DriftTransform(axis_idx=5, drift_rate=0.001, p=1.0)
    X_drift = apply_transform(X, t_drift)
    save_dataset(X_drift, y, "drift_gyro_z")
    
    # 2. Packet Loss/Dropout (0.5s - 2s)
    print("\nGenerating: packet_loss")
    t_loss = PacketLossTransform(sampling_rate=50, min_duration=0.5, max_duration=2.0, p=1.0)
    X_loss = apply_transform(X, t_loss)
    save_dataset(X_loss, y, "packet_loss")
    
    # 3. Saturation (Clip at 15.0 - assuming m/s^2 or rad/s, typical acc range +-20)
    # RealWorld HAR might be m/s^2. 15 is reasonable (approx 1.5g).
    print("\nGenerating: saturation")
    t_sat = SaturationTransform(threshold=15.0, p=1.0)
    X_sat = apply_transform(X, t_sat)
    save_dataset(X_sat, y, "saturation")
    
    # 4. Noise Injection (Accel, 0.2g approx 2.0 m/s^2)
    print("\nGenerating: noise_acc")
    t_noise = NoiseInjectionTransform(noise_std=2.0, channels=[0,1,2], p=1.0)
    X_noise = apply_transform(X, t_noise)
    save_dataset(X_noise, y, "noise_acc")

    print("\nAll datasets generated successfully.")

if __name__ == "__main__":
    main()
