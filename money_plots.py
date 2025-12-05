import os
import argparse
import json
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from src.models.factory import create_model
from src.data.splitter import LOSOSplitter, create_dataloaders
from src.data.transforms import (
    MissingModalityTransform,
    NoiseInjectionTransform,
)
from src.utils.metrics import compute_metrics
from src.utils.common import load_model
from sklearn.metrics import confusion_matrix


# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)


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
    """Evaluate model on test loader and return metrics, predictions, and features."""
    model.eval()
    all_preds = []
    all_targets = []
    all_features = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_features.extend(outputs.cpu().numpy())

    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    return metrics, np.array(all_targets), np.array(all_preds), np.array(all_features)


def plot_aggregated_metrics_bar(metrics_dict, save_path, model_name="Model"):
    """
    Plot aggregated accuracy and F1-macro across test conditions as grouped bar chart.

    Args:
        metrics_dict: Dict with keys as conditions, values as dict with 'accuracy' and 'f1_macro' lists
        save_path: Path to save the figure
        model_name: Name of the model for title
    """
    conditions = list(metrics_dict.keys())

    # Compute mean and std for each condition
    acc_means = [np.mean(metrics_dict[c]["accuracy"]) * 100 for c in conditions]
    acc_stds = [np.std(metrics_dict[c]["accuracy"]) * 100 for c in conditions]
    f1_means = [np.mean(metrics_dict[c]["f1_macro"]) * 100 for c in conditions]
    f1_stds = [np.std(metrics_dict[c]["f1_macro"]) * 100 for c in conditions]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(conditions))
    width = 0.35

    # Color palette
    colors = sns.color_palette("Set2", 2)

    # Plot bars
    bars1 = ax.bar(
        x - width / 2,
        acc_means,
        width,
        yerr=acc_stds,
        label="Accuracy",
        color=colors[0],
        capsize=3,
        edgecolor="black",
        linewidth=0.8,
        error_kw={"linewidth": 0.9},
    )
    bars2 = ax.bar(
        x + width / 2,
        f1_means,
        width,
        yerr=f1_stds,
        label="F1-Macro",
        color=colors[1],
        capsize=3,
        edgecolor="black",
        linewidth=0.8,
        error_kw={"linewidth": 0.9},
    )

    # Annotate bars with values
    def annotate_bars(bars, means, stds):
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.annotate(
                f"{mean:.1f}±{std:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(3, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    annotate_bars(bars1, acc_means, acc_stds)
    annotate_bars(bars2, f1_means, f1_stds)

    # Formatting
    ax.set_xlabel("Test Condition", fontweight="bold")
    ax.set_ylabel("Performance (%)", fontweight="bold")
    ax.set_title(
        f'{model_name}: Performance Under Different Test Conditions\n(LOSO, N={len(metrics_dict[conditions[0]]["accuracy"])} subjects)',
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, 110)

    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Remove top and right spines
    sns.despine()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved aggregated metrics bar chart to {save_path}")


def plot_robustness_curve(robustness_data, save_path, model_name="Model"):
    """
    Plot publication-quality robustness curve with mean ± std shading.

    Args:
        robustness_data: Dict with subject -> list of {noise_std, accuracy}
        save_path: Path to save figure
        model_name: Model name for title
    """
    # Extract noise levels and accuracies
    noise_levels = [entry["noise_std"] for entry in list(robustness_data.values())[0]]

    # Build matrix: subjects x noise_levels
    acc_matrix = []
    for subject, data in robustness_data.items():
        acc_matrix.append([entry["accuracy"] for entry in data])
    acc_matrix = np.array(acc_matrix)

    means = np.mean(acc_matrix, axis=0)
    stds = np.std(acc_matrix, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    color = sns.color_palette("Set2")[0]

    ax.plot(
        noise_levels,
        means,
        "o-",
        color=color,
        linewidth=2.5,
        markersize=8,
        markeredgecolor="black",
        markeredgewidth=1,
        label=model_name,
    )
    ax.fill_between(noise_levels, means - stds, means + stds, alpha=0.5, color=color)

    # Annotate points
    for i, (x, y, s) in enumerate(zip(noise_levels, means, stds)):
        ax.annotate(
            f"{y:.1f}",
            xy=(x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Noise Standard Deviation (σ)", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title(
        f"{model_name}: Robustness to Accelerometer Noise\n(LOSO, N={len(robustness_data)} subjects)",
        fontweight="bold",
    )
    ax.set_xlim(-0.02, max(noise_levels) + 0.02)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", framealpha=0.9)

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    sns.despine()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved robustness curve to {save_path}")


def plot_confusion_matrix(
    cm, classes, save_path, title="Confusion Matrix", normalize=True
):
    """
    Plot publication-quality confusion matrix.

    Args:
        cm: Confusion matrix array
        classes: List of class names
        save_path: Path to save figure
        title: Plot title
        normalize: Whether to normalize the matrix
    """
    if normalize:
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized * 100
        fmt = ".1f"
        vmin, vmax = 0, 100
    else:
        cm_display = cm
        fmt = "d"
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = sns.color_palette("Blues", as_cmap=True)

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        square=True,
        xticklabels=classes,
        yticklabels=classes,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Percentage (%)" if normalize else "Count"},
    )

    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)

    # Rotate x labels
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix to {save_path}")


def plot_tsne(
    features, labels, classes, save_path, title="t-SNE Visualization"
):
    """
    Plot publication-quality t-SNE visualization.

    Args:
        features: Feature array (N, D)
        labels: Label array (N,)
        classes: List of class names
        save_path: Path to save figure
        title: Plot title
    """
    from sklearn.manifold import TSNE

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use a distinct color palette
    palette = sns.color_palette("husl", len(classes))


    for i, class_name in enumerate(classes):
        mask = labels == i
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[palette[i]],
            label=class_name,
            s=50,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_xlabel("t-SNE Dimension 1", fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, ncol=2)

    sns.despine()

    # plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved t-SNE plot to {save_path}")


def plot_per_subject_metrics(subject_metrics, output_path, model_name="Model"):
    """
    Plot per-subject accuracy and F1-macro as a grouped bar chart.

    Args:
        subject_metrics: Dict with subject -> {'accuracy': float, 'f1_macro': float}
        output_path: Path to save the figure
        model_name: Name of the model for title
    """
    subjects = list(subject_metrics.keys())
    accuracies = [subject_metrics[s]["accuracy"] * 100 for s in subjects]
    f1_scores = [subject_metrics[s]["f1_macro"] * 100 for s in subjects]

    # Compute mean for reference line
    mean_acc = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(subjects))
    width = 0.35

    colors = sns.color_palette("Set2", 2)

    bars1 = ax.bar(
        x - width / 2,
        accuracies,
        width,
        label=f"Accuracy (μ={mean_acc:.1f}%)",
        color=colors[0],
        edgecolor="black",
        linewidth=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        f1_scores,
        width,
        label=f"F1-Macro (μ={mean_f1:.1f}%)",
        color=colors[1],
        edgecolor="black",
        linewidth=0.8,
    )

    # Add mean reference lines
    ax.axhline(y=mean_acc, color=colors[0], linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axhline(y=mean_f1, color=colors[1], linestyle="--", linewidth=1.5, alpha=0.7)

    # Formatting
    ax.set_xlabel("Subject", fontweight="bold")
    ax.set_ylabel("Performance (%)", fontweight="bold")
    ax.set_title(
        f"{model_name}: Per-Subject Performance (LOSO Cross-Validation)",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("proband", "S") for s in subjects], rotation=45, ha="right"
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 105)

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    sns.despine()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved per-subject metrics plot to {output_path}")


def plot_per_subject_conditions(
    per_subject_condition_metrics, output_path, model_name="Model"
):
    """
    Plot per-subject accuracy across different test conditions.

    Args:
        per_subject_condition_metrics: Dict with subject -> {condition -> {'accuracy': float, 'f1_macro': float}}
        output_path: Path to save the figure
        model_name: Name of the model for title
    """
    subjects = list(per_subject_condition_metrics.keys())
    conditions = list(per_subject_condition_metrics[subjects[0]].keys())

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(subjects))
    width = 0.25
    n_conditions = len(conditions)

    colors = sns.color_palette("Set2", n_conditions)

    for i, condition in enumerate(conditions):
        accuracies = [
            per_subject_condition_metrics[s][condition]["accuracy"] * 100
            for s in subjects
        ]
        offset = (i - n_conditions / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            accuracies,
            width,
            label=condition,
            color=colors[i],
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xlabel("Subject", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title(
        f"{model_name}: Per-Subject Accuracy Under Different Conditions",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("proband", "S") for s in subjects], rotation=45, ha="right"
    )
    ax.legend(loc="lower right", framealpha=0.9, ncol=len(conditions))
    ax.set_ylim(0, 105)

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    sns.despine()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved per-subject conditions plot to {output_path}")


def save_results_to_json(results_dict, save_path):
    """Save results dictionary to JSON file."""

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_results = convert_to_serializable(results_dict)

    with open(save_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Saved results to {save_path}")


def load_results_from_json(load_path):
    """Load results dictionary from JSON file."""
    with open(load_path, "r") as f:
        return json.load(f)


def compute_all_metrics(args, model_name, model_config, subjects, splitter, device):
    """
    Compute all metrics for all subjects under all conditions.

    Returns:
        Dictionary containing all computed metrics and data for plotting.
    """
    # Storage for aggregation
    robustness_data = {}
    condition_metrics = {
        "Clean": {"accuracy": [], "f1_macro": []},
        f"Noisy (σ={args.noise_std:.1f})": {"accuracy": [], "f1_macro": []},
        "Gyro Dropout": {"accuracy": [], "f1_macro": []},
    }
    condition_keys = list(condition_metrics.keys())

    # Per-subject metrics storage
    per_subject_metrics = {}  # subject -> {accuracy, f1_macro} for clean condition
    per_subject_condition_metrics = {}  # subject -> {condition -> {accuracy, f1_macro}}

    # Confusion matrix aggregation (store predictions and targets)
    aggregated_preds = {k: [] for k in condition_keys}
    aggregated_targets = {k: [] for k in condition_keys}

    # --- Loop over all subjects ---
    for subject in subjects:
        print(f"\nProcessing Subject: {subject}")

        # Load Data
        X_train, y_train, X_test, y_test = splitter.get_train_test_split(subject)

        # Handle channel mismatch
        try:
            if "channels" in model_config:
                expected_channels = model_config["channels"]
            elif "nb_channels" in model_config:
                expected_channels = model_config["nb_channels"]
            elif "input_dim" in model_config:
                expected_channels = model_config["input_dim"]
            else:
                expected_channels = X_test.shape[2]

            if expected_channels < X_test.shape[2]:
                X_test = X_test[:, :, :expected_channels]
                X_train = X_train[:, :, :expected_channels]
        except Exception as e:
            print(f"Warning: Could not determine channel count, using full data: {e}")

        # Load Model
        model = create_model(model_name, model_config)
        model_path = os.path.join(args.experiment_dir, f"best_model_{subject}.pt")
        load_model(model, model_path, device)
        model.to(device)

        per_subject_condition_metrics[subject] = {}

        # === Evaluate under three conditions ===

        # 1. Clean data
        _, test_loader, _ = create_dataloaders(
            X_train, y_train, X_test, y_test, batch_size=32, test_transform=None
        )
        metrics, targets, preds, _ = evaluate_with_transform(model, test_loader, device)
        condition_metrics[condition_keys[0]]["accuracy"].append(metrics["accuracy"])
        condition_metrics[condition_keys[0]]["f1_macro"].append(metrics["f1_macro"])
        aggregated_targets[condition_keys[0]].extend(targets.tolist())
        aggregated_preds[condition_keys[0]].extend(preds.tolist())
        per_subject_metrics[subject] = {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
        }
        per_subject_condition_metrics[subject][condition_keys[0]] = {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
        }

        # 2. Noisy data
        transform = NoiseInjectionTransform(
            noise_std=args.noise_std, channels=[0, 1, 2], p=1.0
        )
        _, test_loader, _ = create_dataloaders(
            X_train, y_train, X_test, y_test, batch_size=32, test_transform=transform
        )
        metrics, targets, preds, _ = evaluate_with_transform(model, test_loader, device)
        condition_metrics[condition_keys[1]]["accuracy"].append(metrics["accuracy"])
        condition_metrics[condition_keys[1]]["f1_macro"].append(metrics["f1_macro"])
        aggregated_targets[condition_keys[1]].extend(targets.tolist())
        aggregated_preds[condition_keys[1]].extend(preds.tolist())
        per_subject_condition_metrics[subject][condition_keys[1]] = {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
        }

        # 3. Gyro dropout
        transform = MissingModalityTransform(modality="gyro", p=1.0)
        _, test_loader, _ = create_dataloaders(
            X_train, y_train, X_test, y_test, batch_size=32, test_transform=transform
        )
        metrics, targets, preds, _ = evaluate_with_transform(model, test_loader, device)
        condition_metrics[condition_keys[2]]["accuracy"].append(metrics["accuracy"])
        condition_metrics[condition_keys[2]]["f1_macro"].append(metrics["f1_macro"])
        aggregated_targets[condition_keys[2]].extend(targets.tolist())
        aggregated_preds[condition_keys[2]].extend(preds.tolist())
        per_subject_condition_metrics[subject][condition_keys[2]] = {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
        }

        # === Robustness Curve (multiple noise levels) ===
        robustness_data[subject] = []
        noise_levels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

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

            metrics_noise, _, _, _ = evaluate_with_transform(model, test_loader, device)
            robustness_data[subject].append(
                {"noise_std": std, "accuracy": metrics_noise["accuracy"] * 100}
            )

    # Compile all results
    results = {
        "model_name": model_name,
        "noise_std": args.noise_std,
        "condition_metrics": condition_metrics,
        "per_subject_metrics": per_subject_metrics,
        "per_subject_condition_metrics": per_subject_condition_metrics,
        "robustness_data": robustness_data,
        "aggregated_targets": aggregated_targets,
        "aggregated_preds": aggregated_preds,
        "subjects": subjects,
    }

    return results


def generate_plots(results, output_dir, args):
    """Generate all plots from results dictionary."""
    model_name = results["model_name"]
    condition_metrics = results["condition_metrics"]
    per_subject_metrics = results["per_subject_metrics"]
    per_subject_condition_metrics = results["per_subject_condition_metrics"]
    robustness_data = results["robustness_data"]
    aggregated_targets = results["aggregated_targets"]
    aggregated_preds = results["aggregated_preds"]

    classes = ["Walk", "Run", "Sit", "Stand", "Lie", "ClimbUp", "ClimbDn", "Jump"]
    condition_keys = list(condition_metrics.keys())

    # 1. Aggregated Metrics Bar Chart (Accuracy & F1 for all conditions)
    print("\nGenerating Aggregated Metrics Bar Chart...")
    plot_aggregated_metrics_bar(
        condition_metrics,
        os.path.join(output_dir, "aggregated_metrics_bar.png"),
        model_name=model_name,
    )

    # 2. Robustness Curve (Mean +/- Std)
    print("\nGenerating Robustness Curve...")
    plot_robustness_curve(
        robustness_data,
        os.path.join(output_dir, "robustness_curve.png"),
        model_name=model_name,
    )

    # 3. Confusion Matrices for each condition
    print("\nGenerating Confusion Matrices...")
    for condition in condition_keys:
        cm = confusion_matrix(
            aggregated_targets[condition], aggregated_preds[condition]
        )
        safe_name = (
            condition.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("=", "")
            .replace(".", "")
        )
        plot_confusion_matrix(
            cm,
            classes,
            os.path.join(output_dir, f"confusion_matrix_{safe_name}.png"),
            title=f"Confusion Matrix: {condition}",
            normalize=True,
        )

    # 4. Per-Subject Metrics (Clean condition)
    print("\nGenerating Per-Subject Metrics Plot...")
    plot_per_subject_metrics(
        per_subject_metrics,
        os.path.join(output_dir, "per_subject_metrics.png"),
        model_name=model_name,
    )

    # 5. Per-Subject Metrics Across Conditions
    print("\nGenerating Per-Subject Conditions Plot...")
    plot_per_subject_conditions(
        per_subject_condition_metrics,
        os.path.join(output_dir, "per_subject_conditions.png"),
        model_name=model_name,
    )

    # 6. t-SNE (Representative Subject)
    print("\nGenerating t-SNE for Representative Subject (clean and gyro dropout)...")
    subject_accuracies = {
        s: per_subject_metrics[s]["accuracy"] * 100 for s in per_subject_metrics
    }
    mean_acc = np.mean(list(subject_accuracies.values()))
    print(f"Mean Accuracy: {mean_acc:.2f}%")

    # Find subject closest to mean
    representative_subject = min(
        subject_accuracies, key=lambda k: abs(subject_accuracies[k] - mean_acc)
    )
    print(
        f"Representative Subject: {representative_subject} (Acc: {subject_accuracies[representative_subject]:.2f}%)"
    )

    # Reload representative subject data/model for t-SNE
    splitter = LOSOSplitter(args.data_dir)
    X_train, y_train, X_test, y_test = splitter.get_train_test_split(
        representative_subject
    )

    # Load model config
    model_config = load_config(args.experiment_dir, "model_config")

    # Handle channel mismatch
    try:
        if "channels" in model_config:
            expected_channels = model_config["channels"]
        elif "nb_channels" in model_config:
            expected_channels = model_config["nb_channels"]
        elif "input_dim" in model_config:
            expected_channels = model_config["input_dim"]
        else:
            expected_channels = X_test.shape[2]

        if expected_channels < X_test.shape[2]:
            X_test = X_test[:, :, :expected_channels]
    except Exception as e:
        print(f"Warning: Could not determine channel count, using full data: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(results["model_name"], model_config)
    model_path = os.path.join(
        args.experiment_dir, f"best_model_{representative_subject}.pt"
    )
    load_model(model, model_path, device)
    model.to(device)

    # Clean data t-SNE
    _, test_loader, _ = create_dataloaders(
        X_train, y_train, X_test, y_test, batch_size=32, test_transform=None
    )

    _, targets_clean, _, features_clean = evaluate_with_transform(
        model, test_loader, device
    )

    plot_tsne(
        features_clean,
        targets_clean,
        classes,
        os.path.join(output_dir, f"tsne_{representative_subject}_clean.png"),
        title=f"t-SNE Feature Visualization (Clean, {representative_subject})",
    )

    # Gyro dropout t-SNE
    transform_gyro = MissingModalityTransform(modality="gyro", p=1.0)
    _, test_loader, _ = create_dataloaders(
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size=32,
        test_transform=transform_gyro,
    )

    _, targets_gyro, _, features_gyro = evaluate_with_transform(
        model, test_loader, device
    )

    plot_tsne(
        features_gyro,
        targets_gyro,
        classes,
        os.path.join(output_dir, f"tsne_{representative_subject}_gyro_dropout.png"),
        title=f"t-SNE Feature Visualization (Gyro Dropout, {representative_subject})",
    )

    # --- Print Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    for condition in condition_keys:
        acc_mean = np.mean(condition_metrics[condition]["accuracy"]) * 100
        acc_std = np.std(condition_metrics[condition]["accuracy"]) * 100
        f1_mean = np.mean(condition_metrics[condition]["f1_macro"]) * 100
        f1_std = np.std(condition_metrics[condition]["f1_macro"]) * 100
        print(f"\n{condition}:")
        print(f"  Accuracy: {acc_mean:.2f} ± {acc_std:.2f}%")
        print(f"  F1-Macro: {f1_mean:.2f} ± {f1_std:.2f}%")
    print("=" * 60)

    print(f"\nAll plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots for HAR experiments"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to experiment directory",
    )
    parser.add_argument("--data_dir", type=str, default="dataset/processed_acc_gyr")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="money_plots/corrected",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=2.0,
        help="Noise std for noisy condition",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation of metrics even if cached",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Only generate plots from cached results (skip computation)",
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = os.path.join(args.experiment_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Path for cached results
    results_cache_path = os.path.join(output_dir, "evaluation_results.json")

    # Check if cached results exist
    if os.path.exists(results_cache_path) and not args.force_recompute:
        print(f"Found cached results at {results_cache_path}")
        print("Loading cached results... (use --force_recompute to recompute)")
        results = load_results_from_json(results_cache_path)
    elif args.plot_only:
        if os.path.exists(results_cache_path):
            print("Loading cached results for plotting...")
            results = load_results_from_json(results_cache_path)
        else:
            print(f"ERROR: No cached results found at {results_cache_path}")
            print("Run without --plot_only first to compute metrics.")
            return
    else:
        print("Computing metrics for all subjects...")

        # Load experiment info
        with open(os.path.join(args.experiment_dir, "results.json"), "r") as f:
            exp_results = json.load(f)

        # Load model config (YAML or JSON)
        model_config = load_config(args.experiment_dir, "model_config")

        model_name = exp_results["model_name"]
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

        # Compute all metrics
        results = compute_all_metrics(
            args, model_name, model_config, subjects, splitter, device
        )

        # Save results to JSON
        save_results_to_json(results, results_cache_path)

    # Generate all plots
    generate_plots(results, output_dir, args)


if __name__ == "__main__":
    main()
