import os
import argparse
import json
import torch
import numpy as np
from src.models.factory import create_model
from src.data.splitter import LOSOSplitter, create_dataloaders
from src.data.transforms import (
    MissingModalityTransform,
    NoiseInjectionTransform,
)
from src.utils.metrics import compute_metrics
from src.utils.common import load_model
from src.utils.plotting import plot_robustness_curve, plot_confusion_matrix


def evaluate_with_transform(model, test_loader, device, transform_name):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    return metrics, all_targets, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir", type=str, required=True, help="Path to experiment directory"
    )
    parser.add_argument(
        "--subject", type=str, default="proband1", help="Subject to evaluate"
    )
    parser.add_argument("--data_dir", type=str, default="dataset/processed_acc_gyr")
    parser.add_argument(
        "--output_dir", type=str, default="money_plots", help="Directory to save plots"
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = os.path.join(args.experiment_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load experiment info
    with open(os.path.join(args.experiment_dir, "results.json"), "r") as f:
        results = json.load(f)

    with open(os.path.join(args.experiment_dir, "model_config.json"), "r") as f:
        model_config = json.load(f)

    model_name = results["model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    splitter = LOSOSplitter(args.data_dir)
    X_train, y_train, X_test, y_test = splitter.get_train_test_split(args.subject)

    # Handle channel mismatch if necessary
    if model_config["nb_channels"] < X_test.shape[2]:
        print(
            f"Model expects {model_config['nb_channels']} channels, data has {X_test.shape[2]}. Truncating..."
        )
        X_train = X_train[:, :, : model_config["nb_channels"]]
        X_test = X_test[:, :, : model_config["nb_channels"]]

    # Load model
    model = create_model(model_name, model_config)
    model_path = os.path.join(args.experiment_dir, f"model_{args.subject}.pt")

    if os.path.exists(model_path):
        load_model(model, model_path, device)
        print(f"Loaded model from {model_path}")
    else:
        print(
            f"WARNING: Model file not found at {model_path}. Using random weights for verification."
        )

    # --- 1. Robustness Curve (Noise Injection) ---
    print("\nGenerating Robustness Curve...")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    robustness_results = []

    for std in noise_levels:
        if std == 0.0:
            transform = None
        else:
            # Using NoiseInjectionTransform directly for cleaner control
            # Assuming channels 0,1,2 are Accel. Adjust if needed.
            transform = NoiseInjectionTransform(
                noise_std=std, channels=[0, 1, 2], p=1.0
            )

        _, test_loader, _ = create_dataloaders(
            X_train, y_train, X_test, y_test, batch_size=32, test_transform=transform
        )

        metrics, _, _ = evaluate_with_transform(
            model, test_loader, device, f"Noise std={std}"
        )
        print(f"Noise std={std}: Accuracy={metrics['accuracy']:.4f}")

        robustness_results.append(
            {
                "noise_std": std,
                "accuracy": metrics["accuracy"] * 100,  # Convert to percentage
            }
        )

    plot_robustness_curve(
        robustness_results,
        os.path.join(output_dir, "robustness_curve.png"),
        model_name=model_name,
    )

    # --- 2. Graceful Failure (Modality Dropout) ---
    print("\nGenerating Graceful Failure Matrix...")
    # Drop Gyro (channels 3,4,5 usually)
    transform = MissingModalityTransform(modality="gyro", p=1.0)

    _, test_loader, _ = create_dataloaders(
        X_train, y_train, X_test, y_test, batch_size=32, test_transform=transform
    )

    metrics, y_true, y_pred = evaluate_with_transform(
        model, test_loader, device, "Missing Gyro"
    )
    print(f"Missing Gyro: Accuracy={metrics['accuracy']:.4f}")

    # Get class names (assuming standard HAR classes, but ideally should be loaded)
    # For now, using generic names or trying to infer.
    # Since I don't have the class map handy in code, I'll use generic "Class 0", "Class 1"...
    # OR better, I'll try to find where class names are stored.
    # Looking at previous file listings, maybe in `dataset` dir?
    # For now, I'll use placeholders and user can update.
    classes = [f"Class {i}" for i in range(len(metrics["confusion_matrix"]))]

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        classes,
        os.path.join(output_dir, "confusion_matrix_missing_gyro.png"),
        title="Confusion Matrix (Missing Gyro)",
    )

    # Save raw results
    all_results = {"robustness_curve": robustness_results, "graceful_failure": metrics}
    with open(os.path.join(output_dir, "money_plots_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll plots and results saved to {output_dir}")


if __name__ == "__main__":
    main()
