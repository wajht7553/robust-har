import os
import argparse
import yaml
import torch
import numpy as np
from datetime import datetime
from torch import nn

from src.data.splitter import LOSOSplitter, create_dataloaders
from src.data.transforms import SensorFailureTransform
from src.models.factory import create_model
from src.training.trainer import Trainer
from src.utils.metrics import compute_metrics
from src.utils.common import save_model, save_json


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class LOSOExperiment:
    def __init__(self, model_name, model_config, train_config, device=None):
        self.model_name = model_name
        self.model_config = model_config
        self.train_config = train_config
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Setup results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            train_config["results_dir"], f"{model_name}_{timestamp}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Save configs
        save_json(model_config, os.path.join(self.experiment_dir, "model_config.json"))
        save_json(train_config, os.path.join(self.experiment_dir, "train_config.json"))

        self.splitter = LOSOSplitter(train_config["data_dir"])
        self.results = {
            "model_name": model_name,
            "timestamp": timestamp,
            "subjects": {},
            "aggregate_metrics": {},
        }

        print(f"Experiment initialized: {self.experiment_dir}")

    def train_fold(self, subject, X_train, y_train, X_test, y_test):
        print(f"\nTraining fold: Test Subject = {subject}")

        # Apply modality dropout if configured
        train_transform = None
        if self.train_config.get("modality_dropout", 0.0) > 0:
            train_transform = SensorFailureTransform(
                p_dropout_gyro=self.train_config["modality_dropout"]
            )
            print(
                f"Applying Modality Dropout (p={self.train_config['modality_dropout']})"
            )

        # Filter input channels if configured
        input_channels = self.train_config.get("input_channels", None)
        if input_channels:
            X_train = X_train[:, :, input_channels]
            X_test = X_test[:, :, input_channels]
            # Update model config nb_channels to match
            self.model_config["nb_channels"] = len(input_channels)

        train_loader, test_loader, norm_stats = create_dataloaders(
            X_train,
            y_train,
            X_test,
            y_test,
            batch_size=self.train_config["batch_size"],
            train_transform=train_transform,
        )

        model = create_model(self.model_name, self.model_config)

        checkpoint_path = os.path.join(self.experiment_dir, f"best_model_{subject}.pt")
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_config["lr"])

        trainer = Trainer(
            model,
            self.device,
            optimizer=optimizer,
            early_stopping_patience=10,
            checkpoint_path=checkpoint_path,
        )

        history = trainer.train(train_loader, test_loader, self.train_config["epochs"])

        # Final evaluation
        _, _, y_pred, y_true = trainer.validate(test_loader)
        metrics = compute_metrics(y_true, y_pred)

        print(
            f"Subject {subject} Results: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}"
        )

        # Save final model
        save_model(model, os.path.join(self.experiment_dir, f"model_{subject}.pt"))

        return {
            "subject": subject,
            "history": history,
            "test_metrics": metrics,
            "normalization_stats": {
                "mean": norm_stats[0].tolist() if norm_stats[0] is not None else None,
                "std": norm_stats[1].tolist() if norm_stats[1] is not None else None,
            },
        }

    def run(self):
        all_accs = []
        all_f1s = []

        for (
            subject,
            X_train,
            y_train,
            X_test,
            y_test,
        ) in self.splitter.get_loso_splits():
            fold_results = self.train_fold(subject, X_train, y_train, X_test, y_test)
            self.results["subjects"][subject] = fold_results
            all_accs.append(fold_results["test_metrics"]["accuracy"])
            all_f1s.append(fold_results["test_metrics"]["f1_macro"])

            save_json(self.results, os.path.join(self.experiment_dir, "results.json"))

        self.results["aggregate_metrics"] = {
            "mean_accuracy": float(np.mean(all_accs)),
            "std_accuracy": float(np.std(all_accs)),
            "mean_f1_macro": float(np.mean(all_f1s)),
            "std_f1_macro": float(np.std(all_f1s)),
        }

        save_json(self.results, os.path.join(self.experiment_dir, "results.json"))
        print("\nLOSO Experiment Completed.")
        print(
            f"Mean Accuracy: {self.results['aggregate_metrics']['mean_accuracy']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (mobilevit, deepconvlstm, mamba)",
    )
    parser.add_argument("--config", type=str, help="Path to model config override")
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config",
    )
    args = parser.parse_args()

    # Load configs
    train_config = load_config(args.train_config)

    # Load default model config if not provided
    if args.config:
        model_config = load_config(args.config)
    else:
        model_config_path = f"configs/model/{args.model}.yaml"
        if os.path.exists(model_config_path):
            model_config = load_config(model_config_path)
        else:
            raise ValueError(
                f"No default config found for {args.model} at {model_config_path}"
            )

    experiment = LOSOExperiment(args.model, model_config, train_config)
    experiment.run()


if __name__ == "__main__":
    main()
