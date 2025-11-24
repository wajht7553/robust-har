import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE


def plot_robustness_curve(results_per_subject, output_path, model_name="Model"):
    """
    Plots the Robustness Curve (Accuracy vs. Noise Intensity) with Mean +/- Std.

    Args:
        results_per_subject: Dictionary where key is subject_id and value is list of dicts
                             {'noise_std': float, 'accuracy': float}.
        output_path: Path to save the plot.
        model_name: Name of the model for the legend.
    """
    # Aggregate data
    all_data = []
    for subject, res_list in results_per_subject.items():
        for item in res_list:
            all_data.append(item)

    df = pd.DataFrame(all_data)

    # Calculate Mean and Std per noise level
    summary = df.groupby("noise_std")["accuracy"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Plot Mean Line
    plt.plot(
        summary["noise_std"],
        summary["mean"],
        marker="o",
        label=f"{model_name} (Mean)",
        linewidth=2.5,
        color="blue",
    )

    # Plot Shaded Region (Mean +/- Std)
    plt.fill_between(
        summary["noise_std"],
        summary["mean"] - summary["std"],
        summary["mean"] + summary["std"],
        color="blue",
        alpha=0.2,
        label="Standard Deviation",
    )

    plt.title("Robustness Curve: Accuracy vs. Noise Intensity", fontsize=16)
    plt.xlabel("Noise Intensity (Standard Deviation)", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.ylim(0, 100)
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


def plot_tsne(features, labels, classes, output_path, title="t-SNE Visualization"):
    """
    Plots t-SNE embeddings.

    Args:
        features: Numpy array of features (N, D).
        labels: Numpy array of labels (N,).
        classes: List of class names.
        output_path: Path to save the plot.
        title: Title of the plot.
    """
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")

    # Create a scatter plot
    scatter = plt.scatter(
        embeddings[:, 0], embeddings[:, 1], c=labels, cmap="tab10", alpha=0.7, s=10
    )

    # Create legend
    handles, _ = scatter.legend_elements()
    plt.legend(handles, classes, title="Classes", fontsize=10, loc="best")

    plt.title(title, fontsize=16)
    plt.axis("off")  # Remove axes for cleaner look
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"t-SNE plot saved to {output_path}")
