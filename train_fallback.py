
import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_loso import LOSOExperiment, get_mobilevit_config, get_deepconvlstm_config
from utils.dataset_loader import LOSOSplitter

class FallbackLOSOExperiment(LOSOExperiment):
    """
    Specialized experiment for Fallback (Accel-only) models.
    It filters the input data to keep only accelerometer channels (0-2)
    BEFORE creating dataloaders.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def train_subject_fold(self, subject, X_train, y_train, X_test, y_test, **kwargs):
        # Filter to keep only first 3 channels (Accel x,y,z)
        # Assuming original data is (N, 200, 6) -> (N, 200, 3)
        print("Filtering data to keep only Accelerometer channels (0-2)...")
        X_train = X_train[:, :, :3]
        X_test = X_test[:, :, :3]
        
        return super().train_subject_fold(subject, X_train, y_train, X_test, y_test, **kwargs)

def main():
    parser = argparse.ArgumentParser(description="Train Fallback (Accel-only) models")
    parser.add_argument("--model", type=str, default="mobilevit", choices=["deepconvlstm", "mobilevit"], help="Model to train")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs per fold")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="dataset/processed_acc_gyr", help="Path to processed data")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to save results")

    args = parser.parse_args()

    # Config for Accel-only
    window_size = 200
    nb_channels = 3 # Only Accel
    nb_classes = 8

    if args.model == "deepconvlstm":
        config = get_deepconvlstm_config(window_size, nb_channels, nb_classes)
    else:
        config = get_mobilevit_config(window_size, nb_channels, nb_classes)

    # Create experiment
    # We append "_fallback" to the model name for clarity in results
    experiment = FallbackLOSOExperiment(
        model_name=f"{args.model}_fallback",
        config=config,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )

    experiment.run_loso(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

if __name__ == "__main__":
    main()
