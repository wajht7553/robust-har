##################################################
# Visualize LOSO training results
##################################################

import os
import json
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_results(results_path):
    """Load results JSON file"""
    with open(results_path, "r") as f:
        return json.load(f)


def plot_loso_results(results, output_dir=None):
    """
    Create comprehensive visualization of LOSO results

    Args:
        results: results dictionary from JSON
        output_dir: directory to save plots (if None, show instead)
    """
    model_name = results["model_name"]
    subjects = results["subjects"]

    # Extract data for plotting
    subject_names = []
    accuracies = []
    f1_scores = []

    for subj_name in sorted(
        subjects.keys(), key=lambda x: int(x.replace("proband", ""))
    ):
        subject_names.append(subj_name.replace("proband", "S"))
        accuracies.append(subjects[subj_name]["test_metrics"]["accuracy"])
        f1_scores.append(subjects[subj_name]["test_metrics"]["f1_macro"])

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Per-subject accuracy and F1
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(subject_names))
    width = 0.35
    ax1.bar(x - width / 2, accuracies, width, label="Accuracy", alpha=0.8)
    ax1.bar(x + width / 2, f1_scores, width, label="F1-Macro", alpha=0.8)
    ax1.set_xlabel("Test Subject")
    ax1.set_ylabel("Score")
    ax1.set_title(f"{model_name.upper()}: Per-Subject Performance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(subject_names, rotation=45)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim([0, 1])

    # Add mean lines
    mean_acc = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)
    ax1.axhline(
        mean_acc,
        color="C0",
        linestyle="--",
        alpha=0.5,
        label=f"Mean Acc: {mean_acc:.3f}",
    )
    ax1.axhline(
        mean_f1, color="C1", linestyle="--", alpha=0.5, label=f"Mean F1: {mean_f1:.3f}"
    )

    # 2. Training curves for first subject (example)
    ax2 = plt.subplot(2, 3, 2)
    first_subject = sorted(
        subjects.keys(), key=lambda x: int(x.replace("proband", ""))
    )[14]
    history = subjects[first_subject]["history"]
    epochs = range(1, len(history["train_losses"]) + 1)
    ax2.plot(epochs, history["train_losses"], label="Train Loss", alpha=0.8)
    ax2.plot(epochs, history["val_losses"], label="Val Loss", alpha=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"Training Curves (Test Subject: {first_subject})")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Accuracy curves for first subject
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, history["train_accs"], label="Train Acc", alpha=0.8)
    ax3.plot(epochs, history["val_accs"], label="Val Acc", alpha=0.8)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy")
    ax3.set_title(f"Accuracy Curves (Test Subject: {first_subject})")
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 1])

    # 4. Confusion matrix (aggregate across all subjects)
    ax4 = plt.subplot(2, 3, 4)
    # Sum confusion matrices from all subjects
    # Sum confusion matrices from all subjects, padding smaller matrices with zeros
    cms = []
    max_n = 0
    for subj_name in subjects:
        cm = np.array(subjects[subj_name]["test_metrics"]["confusion_matrix"])
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError(
                f"Confusion matrix for {subj_name} is not square: {cm.shape}"
            )
        cms.append(cm)
        if cm.shape[0] > max_n:
            max_n = cm.shape[0]

    cm_total = np.zeros((max_n, max_n), dtype=float)
    for cm in cms:
        n = cm.shape[0]
        if n < max_n:
            padded = np.zeros((max_n, max_n), dtype=cm.dtype)
            padded[:n, :n] = cm
            cm_total += padded
        else:
            cm_total += cm

    # cm_norm = cm_total.astype("float")
    # Normalize by row
    cm_norm = cm_total.astype("float") / cm_total.sum(axis=1, keepdims=True)

    activity_labels = [
        "Walk",
        "Run",
        "Sit",
        "Stand",
        "Lie",
        "ClimbUp",
        "ClimbDn",
        "Jump",
    ]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        square=True,
        xticklabels=activity_labels,
        yticklabels=activity_labels,
        ax=ax4,
    )
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("True")
    ax4.set_title("Normalized Confusion Matrix (All Subjects)")

    # 5. Per-class F1 scores (averaged across subjects)
    ax5 = plt.subplot(2, 3, 5)
    # Collect per-class F1 scores and pad to a common length with NaN so arrays are homogeneous
    per_class_f1_list = []
    for subj_name in subjects:
        per_class_f1_list.append(
            np.asarray(subjects[subj_name]["test_metrics"]["f1_per_class"], dtype=float)
        )

    # Determine the maximum number of classes across subjects
    max_n = max((arr.size for arr in per_class_f1_list), default=0)

    # Create a padded array filled with NaN and copy values
    padded = np.full((len(per_class_f1_list), max_n), np.nan, dtype=float)
    for i, arr in enumerate(per_class_f1_list):
        padded[i, : arr.size] = arr

    # Use nanmean/nanstd to ignore padding NaNs in statistics
    mean_per_class_f1 = np.nanmean(padded, axis=0)
    std_per_class_f1 = np.nanstd(padded, axis=0)

    # Adjust activity labels to match the number of classes (extend with generic names if needed)
    if max_n <= len(activity_labels):
        labels = activity_labels[:max_n]
    else:
        labels = activity_labels + [
            f"Class{j}" for j in range(len(activity_labels), max_n)
        ]

    x = np.arange(len(labels))
    ax5.bar(x, mean_per_class_f1, yerr=std_per_class_f1, capsize=5, alpha=0.8)
    ax5.set_xlabel("Activity")
    ax5.set_ylabel("F1 Score")
    ax5.set_title("Per-Class F1 Scores (Mean ± Std)")
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels, rotation=45, ha="right")
    ax5.grid(axis="y", alpha=0.3)
    ax5.set_ylim([0, 1])

    # 6. Distribution of accuracies
    ax6 = plt.subplot(2, 3, 6)
    ax6.boxplot([accuracies, f1_scores], tick_labels=["Accuracy", "F1-Macro"])
    ax6.set_ylabel("Score")
    ax6.set_title("Score Distribution Across Subjects")
    ax6.grid(axis="y", alpha=0.3)
    ax6.set_ylim([0, 1])

    # Add aggregate metrics as text
    agg = results["aggregate_metrics"]
    textstr = f"Mean Accuracy: {agg['mean_accuracy']:.4f} ± {agg['std_accuracy']:.4f}\n"
    textstr += f"Mean F1-Macro: {agg['mean_f1_macro']:.4f} ± {agg['std_f1_macro']:.4f}"
    ax6.text(
        0.5,
        0.05,
        textstr,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f"{model_name}_loso_results.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Plot saved to {output_dir}/{model_name}_loso_results.png")
    else:
        plt.show()

    plt.close()


def print_summary(results):
    """Print text summary of results"""
    print("\n" + "=" * 80)
    print(f"LOSO Results Summary: {results['model_name'].upper()}")
    print("=" * 80)

    agg = results["aggregate_metrics"]
    print("\nAggregate Performance:")
    print(f"  Mean Accuracy:  {agg['mean_accuracy']:.4f} ± {agg['std_accuracy']:.4f}")
    print(f"  Mean F1-Macro:  {agg['mean_f1_macro']:.4f} ± {agg['std_f1_macro']:.4f}")

    print("\nPer-Subject Results:")
    print(
        f"  {'Subject':<12} {'Accuracy':<12} {'F1-Macro':<12} {'Train Size':<12} {'Test Size':<12}"
    )
    print(f"  {'-'*60}")

    subjects = results["subjects"]
    for subj_name in sorted(
        subjects.keys(), key=lambda x: int(x.replace("proband", ""))
    ):
        subj = subjects[subj_name]
        acc = subj["test_metrics"]["accuracy"]
        f1 = subj["test_metrics"]["f1_macro"]
        train_size = subj["train_size"]
        test_size = subj["test_size"]
        print(
            f"  {subj_name:<12} {acc:<12.4f} {f1:<12.4f} {train_size:<12} {test_size:<12}"
        )

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize LOSO training results")
    parser.add_argument("results_path", type=str, help="Path to results.json file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (if not specified, plots are shown)",
    )
    parser.add_argument(
        "--no_plot", action="store_true", help="Skip plotting, only print summary"
    )

    args = parser.parse_args()

    # Load results
    results = load_results(args.results_path)

    # Print summary
    print_summary(results)

    # Plot results
    if not args.no_plot:
        output_dir = args.output_dir
        if output_dir is None:
            # Save to same directory as results.json
            output_dir = os.path.dirname(args.results_path)
        plot_loso_results(results, output_dir)


if __name__ == "__main__":
    main()
