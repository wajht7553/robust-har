
##################################################
# Main training script for LOSO validation
# Trains DeepConvLSTM and MobileViT models
##################################################

import os
import sys
import json
import torch
import argparse
import numpy as np


from torch import nn
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_loader import LOSOSplitter, create_dataloaders, SensorFailureTransform
from utils.training_utils import Trainer, compute_metrics, save_model

# Import models
from models.MobileViT import MobileViT
from models.DeepConvLSTM import DeepConvLSTM


class LOSOExperiment:
    """Manager for Leave-One-Subject-Out experiments"""

    def __init__(
        self,
        model_name,
        config,
        data_dir="dataset/processed_acc_gyr",
        results_dir="results",
        device=None,
        modality_dropout=0.0,
    ):
        """
        Args:
            model_name: 'deepconvlstm' or 'mobilevit'
            config: model configuration dict
            data_dir: path to processed data
            results_dir: path to save results
            device: torch device (auto-detect if None)
            modality_dropout: probability of dropping gyro during training
        """
        self.model_name = model_name
        self.config = config
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.modality_dropout = modality_dropout

        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(results_dir, f"{model_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Initialize data splitter
        self.splitter = LOSOSplitter(data_dir)

        # Storage for results
        self.results = {
            "model_name": model_name,
            "config": config,
            "device": str(self.device),
            "timestamp": timestamp,
            "modality_dropout": modality_dropout,
            "subjects": {},
            "aggregate_metrics": {},
        }

        print(f"\n{'='*80}")
        print(f"LOSO Experiment: {model_name}")
        print(f"Device: {self.device}")
        print(f"Modality Dropout (Gyro): {self.modality_dropout}")
        print(f"Results directory: {self.experiment_dir}")
        print(f"{'='*80}\n")

    def create_model(self):
        """Create a fresh model instance"""
        if self.model_name.lower() == "deepconvlstm":
            return DeepConvLSTM(self.config)
        elif self.model_name.lower() == "mobilevit":
            return MobileViT(self.config)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def train_subject_fold(
        self,
        subject,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=50,
        batch_size=32,
        lr=1e-3,
    ):
        """
        Train and evaluate for one subject fold

        Args:
            subject: test subject name
            X_train, y_train: training data
            X_test, y_test: test data
            epochs: number of training epochs
            batch_size: batch size
            lr: learning rate

        Returns:
            dict with results for this fold
        """
        print(f"\n{'-'*80}")
        print(f"Training fold: Test Subject = {subject}")
        print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
        print(f"{'-'*80}\n")

        # Define transforms
        train_transform = None
        if self.modality_dropout > 0:
            train_transform = SensorFailureTransform(p_dropout_gyro=self.modality_dropout)
            print(f"Applying Modality Dropout (p={self.modality_dropout}) to training data")

        # Create dataloaders
        train_loader, test_loader, norm_stats = create_dataloaders(
            X_train, y_train, X_test, y_test, 
            batch_size=batch_size, 
            num_workers=0,
            train_transform=train_transform
        )

        # Create model
        model = self.create_model()
        print(f"Model created: {model.number_of_parameters():,} parameters")

        # Prepare checkpoint path for best model during training
        checkpoint_path = os.path.join(self.experiment_dir, f"best_model_{subject}.pt")

        # Create optimizer and trainer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(
            model, self.device, criterion, optimizer, 
            early_stopping_patience=10, 
            checkpoint_path=checkpoint_path
        )

        # Train
        print(f"\nTraining for up to {epochs} epochs (early stopping enabled, patience=10)...")
        history = trainer.train(train_loader, test_loader, epochs, verbose=True)

        # Final evaluation on test set
        print(f"\nFinal evaluation on test subject {subject}...")
        test_loss, test_acc, y_pred, y_true = trainer.validate(test_loader)

        # Compute detailed metrics
        metrics = compute_metrics(y_true, y_pred, num_classes=self.config["nb_classes"])

        print(f"\nTest Results for {subject}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        # Report early stopping info
        if history.get('early_stopped', False):
            print(f"  Early stopped at epoch {history['total_epochs']} (best: epoch {history['best_epoch']})")

        # Save final model checkpoint (best model is already saved during training)
        model_path = os.path.join(self.experiment_dir, f"model_{subject}.pt")
        save_model(model, model_path)

        # Prepare results
        fold_results = {
            "subject": subject,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "history": history,
            "test_metrics": metrics,
            "model_path": model_path,
            "normalization_stats": {
                "mean": norm_stats[0].tolist() if norm_stats[0] is not None else None,
                "std": norm_stats[1].tolist() if norm_stats[1] is not None else None,
            },
        }

        return fold_results

    def run_loso(self, epochs=50, batch_size=32, lr=1e-3):
        """
        Run complete LOSO cross-validation

        Args:
            epochs: number of epochs per fold
            batch_size: batch size
            lr: learning rate
        """
        all_test_accs = []
        all_test_f1s = []

        # Iterate through LOSO splits
        for (
            subject,
            X_train,
            y_train,
            X_test,
            y_test,
        ) in self.splitter.get_loso_splits():
            # Train and evaluate this fold
            fold_results = self.train_subject_fold(
                subject,
                X_train,
                y_train,
                X_test,
                y_test,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )

            # Store results
            self.results["subjects"][subject] = fold_results

            # Track aggregate metrics
            all_test_accs.append(fold_results["test_metrics"]["accuracy"])
            all_test_f1s.append(fold_results["test_metrics"]["f1_macro"])

            # Save intermediate results
            self.save_results()

        # Compute aggregate statistics
        self.results["aggregate_metrics"] = {
            "mean_accuracy": float(np.mean(all_test_accs)),
            "std_accuracy": float(np.std(all_test_accs)),
            "mean_f1_macro": float(np.mean(all_test_f1s)),
            "std_f1_macro": float(np.std(all_test_f1s)),
            "all_accuracies": all_test_accs,
            "all_f1_macros": all_test_f1s,
        }

        # Print summary
        print(f"\n{'='*80}")
        print(f"LOSO Cross-Validation Results for {self.model_name}")
        print(f"{'='*80}")
        print(
            f"Mean Accuracy: {self.results['aggregate_metrics']['mean_accuracy']:.4f} "
            f"± {self.results['aggregate_metrics']['std_accuracy']:.4f}"
        )
        print(
            f"Mean F1 (macro): {self.results['aggregate_metrics']['mean_f1_macro']:.4f} "
            f"± {self.results['aggregate_metrics']['std_f1_macro']:.4f}"
        )
        print(f"{'='*80}\n")

        # Save final results
        self.save_results()

        return self.results

    def save_results(self):
        """Save results to JSON file"""
        results_path = os.path.join(self.experiment_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {results_path}")


def get_deepconvlstm_config(window_size=200, nb_channels=6, nb_classes=8):
    """Get default config for DeepConvLSTM"""
    return {
        "window_size": window_size,
        "nb_channels": nb_channels,
        "nb_classes": nb_classes,
        "nb_filters": 64,
        "filter_width": 11,
        "nb_units_lstm": 128,
        "nb_layers_lstm": 2,
        "nb_conv_blocks": 4,
        "conv_block_type": "normal",
        "dilation": 1,
        "drop_prob": 0.5,
        "pooling": False,
        "batch_norm": True,
        "reduce_layer": False,
        "reduce_layer_output": 8,
        "pool_type": "max",
        "pool_kernel_width": 2,
        "no_lstm": False,
        "weights_init": "xavier_normal",
        "seed": 42,
    }


def get_mobilevit_config(window_size=200, nb_channels=6, nb_classes=8):
    """Get default config for MobileViT"""
    return {
        "window_size": window_size,
        "nb_channels": nb_channels,
        "nb_classes": nb_classes,
        "dims": [32, 64, 96, 128],
        "num_transformer_layers": [2, 4],
        "patch_size": 2,
        "num_heads": 4,
        "dropout": 0.1,
    }


def main():
    parser = argparse.ArgumentParser(description="Train models with LOSO validation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["deepconvlstm", "mobilevit", "both"],
        help="Model to train",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs per fold"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/processed_acc_gyr",
        help="Path to processed data",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results", help="Path to save results"
    )
    parser.add_argument(
        "--modality_dropout",
        type=float,
        default=0.0,
        help="Probability of dropping gyro during training",
    )

    args = parser.parse_args()

    # Determine window size and channels from data
    # For this dataset: 4 sec windows at 50 Hz = 200 timesteps, 6 channels (acc+gyr)
    window_size = 200
    nb_channels = 6
    nb_classes = 8  # number of activities

    models_to_train = []
    if args.model == "both":
        models_to_train = ["deepconvlstm", "mobilevit"]
    else:
        models_to_train = [args.model]

    # Train each model
    for model_name in models_to_train:
        # Get model config
        if model_name == "deepconvlstm":
            config = get_deepconvlstm_config(window_size, nb_channels, nb_classes)
        else:
            config = get_mobilevit_config(window_size, nb_channels, nb_classes)

        # Create experiment and run LOSO
        experiment = LOSOExperiment(
            model_name=model_name,
            config=config,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            modality_dropout=args.modality_dropout,
        )

        experiment.run_loso(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
