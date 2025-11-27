import os
import argparse
import json
import yaml
import torch
import numpy as np
import glob
from omegaconf import OmegaConf
from src.models.factory import create_model
from src.data.splitter import LOSOSplitter, create_dataloaders
from src.data.transforms import (
    MissingModalityTransform,
    NoiseInjectionTransform,
)
from src.utils.metrics import compute_metrics
from src.utils.common import load_model
from src.utils.plotting import plot_robustness_curve, plot_confusion_matrix, plot_tsne
from sklearn.metrics import confusion_matrix


def load_config(experiment_dir, config_name):
    """
    Load configuration file (YAML or JSON) with backward compatibility.

    Args:
        experiment_dir: Path to experiment directory
        config_name: Base name of config (e.g., 'model_config' or 'train_config')

    Returns:
        Config dictionary
    """
    # Try YAML first (new format)
    yaml_path = os.path.join(experiment_dir, f"{config_name}.yaml")
    if os.path.exists(yaml_path):
        return OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)

    # Fall back to JSON (old format)
    json_path = os.path.join(experiment_dir, f"{config_name}.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)

    raise FileNotFoundError(
        f"Config file not found: neither {yaml_path} nor {json_path} exists"
    )


def evaluate_with_transform(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_features = []

    # Hook to capture features (penultimate layer)
    # This is a bit hacky and depends on model structure.
    # For now, we'll try to get features if the model has a 'features' method or similar.
    # If not, we might just use the output logits as "features" for t-SNE (less ideal but works).

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_features.extend(
                outputs.cpu().numpy()
            )  # Using logits as features for now

    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    return metrics, np.array(all_targets), np.array(all_preds), np.array(all_features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir", type=str, required=True, help="Path to experiment directory"
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

    # Load model config (YAML or JSON)
    model_config = load_config(args.experiment_dir, "model_config")

    model_name = results["model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find all trained models
    model_files = glob.glob(os.path.join(args.experiment_dir, "best_model_*.pt"))
    subjects = sorted(
        [
            os.path.basename(f).replace("best_model_", "").replace(".pt", "")
            for f in model_files
        ],
        key=lambda x: int(x.replace("proband", "")),
    )

    if not subjects:
        print("No model files found! Cannot proceed.")
        return

    print(f"Found {len(subjects)} subjects: {subjects}")

    # Load data splitter
    splitter = LOSOSplitter(args.data_dir)

    # Storage for aggregation
    robustness_data = {}  # subject -> list of {noise, acc}
    aggregated_preds_missing_gyro = []
    aggregated_targets_missing_gyro = []
    subject_accuracies = {}  # subject -> clean accuracy (for representative selection)

    # --- Loop over all subjects ---
    for subject in subjects:
        print(f"\nProcessing Subject: {subject}")

        # Load Data
        X_train, y_train, X_test, y_test = splitter.get_train_test_split(subject)

        # Handle channel mismatch
        # Support both old (nb_channels/input_dim) and new (channels) config keys
        try:
            if "channels" in model_config:
                expected_channels = model_config["channels"]
            elif "nb_channels" in model_config:
                expected_channels = model_config["nb_channels"]
            elif "input_dim" in model_config:
                expected_channels = model_config["input_dim"]
            else:
                expected_channels = X_test.shape[2]  # No truncation

            if expected_channels < X_test.shape[2]:
                X_test = X_test[:, :, :expected_channels]
        except Exception as e:
            print(f"Warning: Could not determine channel count, using full data: {e}")

        # Load Model
        model = create_model(model_name, model_config)
        model_path = os.path.join(args.experiment_dir, f"best_model_{subject}.pt")
        load_model(model, model_path, device)
        model.to(device)

        # 1. Robustness Curve (Noise Injection)
        robustness_data[subject] = []
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        for std in noise_levels:
            if std == 0.0:
                transform = None
            else:
                transform = NoiseInjectionTransform(
                    noise_std=std, channels=[0, 1, 2], p=1.0
                )

            _, test_loader, _ = create_dataloaders(
                X_train,
                y_train,
                X_test,
                y_test,
                batch_size=32,
                test_transform=transform,
            )

            metrics, _, _, _ = evaluate_with_transform(model, test_loader, device)

            robustness_data[subject].append(
                {"noise_std": std, "accuracy": metrics["accuracy"] * 100}
            )

            if std == 0.0:
                subject_accuracies[subject] = metrics["accuracy"] * 100

        # 2. Graceful Failure (Missing Gyro)
        transform = MissingModalityTransform(modality="gyro", p=1.0)
        _, test_loader, _ = create_dataloaders(
            X_train, y_train, X_test, y_test, batch_size=32, test_transform=transform
        )
        _, targets, preds, _ = evaluate_with_transform(model, test_loader, device)

        aggregated_targets_missing_gyro.extend(targets)
        aggregated_preds_missing_gyro.extend(preds)

    # --- Generate Plots ---

    # 1. Robustness Curve (Mean +/- Std)
    print("\nGenerating Robustness Curve...")
    plot_robustness_curve(
        robustness_data,
        os.path.join(output_dir, "robustness_curve_aggregated.png"),
        model_name=model_name,
    )

    # 2. Graceful Failure Matrix (Aggregated)
    print("\nGenerating Aggregated Confusion Matrix (Missing Gyro)...")
    classes = ["Walk", "Run", "Sit", "Stand", "Lie", "ClimbUp", "ClimbDn", "Jump"]

    cm = confusion_matrix(
        aggregated_targets_missing_gyro, aggregated_preds_missing_gyro
    )
    plot_confusion_matrix(
        cm,
        classes,
        os.path.join(output_dir, "confusion_matrix_aggregated_missing_gyro.png"),
        title="Aggregated Confusion Matrix (Missing Gyro)",
    )

    # 3. t-SNE (Representative Subject)
    print("\nGenerating t-SNE for Representative Subject...")
    mean_acc = np.mean(list(subject_accuracies.values()))
    print(f"Mean Accuracy: {mean_acc:.2f}%")

    # Find subject closest to mean
    representative_subject = min(
        subject_accuracies, key=lambda k: abs(subject_accuracies[k] - mean_acc)
    )
    print(
        f"Representative Subject: {representative_subject} (Acc: {subject_accuracies[representative_subject]:.2f}%)"
    )

    # Reload representative subject data/model
    X_train, y_train, X_test, y_test = splitter.get_train_test_split(
        representative_subject
    )

    # Handle channel mismatch
    try:
        if "channels" in model_config:
            expected_channels = model_config["channels"]
        elif "nb_channels" in model_config:
            expected_channels = model_config["nb_channels"]
        elif "input_dim" in model_config:
            expected_channels = model_config["input_dim"]
        else:
            expected_channels = X_test.shape[2]  # No truncation

        if expected_channels < X_test.shape[2]:
            X_test = X_test[:, :, :expected_channels]
    except Exception as e:
        print(f"Warning: Could not determine channel count, using full data: {e}")

    model = create_model(model_name, model_config)
    model_path = os.path.join(
        args.experiment_dir, f"best_model_{representative_subject}.pt"
    )
    load_model(model, model_path, device)
    model.to(device)

    _, test_loader, _ = create_dataloaders(
        X_train, y_train, X_test, y_test, batch_size=32, test_transform=None
    )

    _, targets, _, features = evaluate_with_transform(model, test_loader, device)

    plot_tsne(
        features,
        targets,
        classes,
        os.path.join(output_dir, f"tsne_representative_{representative_subject}.png"),
    )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
