import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_robustness_curve(results, output_path, model_name="Model"):
    """
    Plots the Robustness Curve (Accuracy vs. Noise Intensity).

    Args:
        results: List of dicts with 'noise_std' and 'accuracy'.
        output_path: Path to save the plot.
        model_name: Name of the model for the legend.
    """
    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    sns.lineplot(
        data=df,
        x="noise_std",
        y="accuracy",
        marker="o",
        label=model_name,
        linewidth=2.5,
    )

    plt.title("Robustness Curve: Accuracy vs. Noise Intensity", fontsize=16)
    plt.xlabel("Noise Intensity (Standard Deviation)", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.ylim(0, 100)  # Assuming accuracy is 0-100 or close to it
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Robustness curve saved to {output_path}")


def plot_confusion_matrix(cm, classes, output_path, title="Confusion Matrix"):
    """
    Plots the Confusion Matrix.

    Args:
        cm: Confusion matrix (numpy array or list of lists).
        classes: List of class names.
        output_path: Path to save the plot.
        title: Title of the plot.
    """
    cm = np.array(cm)
    # Normalize
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.set_style("white")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")
