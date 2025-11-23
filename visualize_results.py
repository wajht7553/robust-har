import argparse
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Define class names based on the dataset (MobiAct/HAR)
CLASS_NAMES = ['Walk', 'Run', 'Sit', 'Stand', 'Lie', 'ClimbUp', 'ClimbDn', 'Jump']

def load_results(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_aggregate_metrics(results, output_dir):
    """Plots aggregate metrics (Accuracy, F1-Macro) for different conditions."""
    metrics = results.get("aggregate_metrics", {})
    if not metrics:
        print("No aggregate metrics found.")
        return

    conditions = list(metrics.keys())
    
    # Create DataFrame for plotting
    data = []
    for c in conditions:
        data.append({"Condition": c, "Metric": "Accuracy", "Value": metrics[c]["mean_accuracy"]})
        data.append({"Condition": c, "Metric": "F1-Macro", "Value": metrics[c]["mean_f1_macro"]})
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Condition", y="Value", hue="Metric")
    plt.title("Aggregate Metrics by Condition")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregate_metrics.png"))
    plt.close()
    print(f"Saved aggregate_metrics.png to {output_dir}")

def plot_per_subject_performance(results, output_dir, ax=None):
    """Plots per-subject Accuracy and F1-Macro."""
    subjects = results.get("subjects", {})
    if not subjects:
        print("No subject data found.")
        return

    data = []
    for subj_id, subj_data in subjects.items():
        # Assuming 'clean' condition for performance plot
        metrics = subj_data.get("metrics", {}).get("clean", {})
        if metrics:
            data.append({
                "Subject": subj_id,
                "Metric": "Accuracy",
                "Score": metrics.get("accuracy", 0)
            })
            data.append({
                "Subject": subj_id,
                "Metric": "F1-Macro",
                "Score": metrics.get("f1_macro", 0)
            })
    
    if not data:
        return

    df = pd.DataFrame(data)
    
    # Sort by subject ID if possible
    try:
        df["SubjectNum"] = df["Subject"].str.extract(r'(\d+)').astype(int)
        df = df.sort_values("SubjectNum")
    except:
        pass

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        save = True
    else:
        save = False

    sns.barplot(data=df, x="Subject", y="Score", hue="Metric", ax=ax)
    ax.set_title("Per-Subject Performance")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add mean line
    mean_acc = df[df["Metric"]=="Accuracy"]["Score"].mean()
    ax.axhline(mean_acc, color='blue', linestyle='--', alpha=0.5, label=f'Mean Acc: {mean_acc:.2f}')

    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "per_subject_performance.png"))
        plt.close()
        print(f"Saved per_subject_performance.png to {output_dir}")

def plot_training_curves(results, output_dir, ax_loss=None, ax_acc=None):
    """Plots training curves for a sample subject."""
    subjects = results.get("subjects", {})
    if not subjects:
        return

    # Pick the last subject as sample or one with good data
    subj_id = list(subjects.keys())[-1]
    history = subjects[subj_id].get("history", {})
    
    if not history:
        return

    epochs = range(1, len(history["train_losses"]) + 1)
    
    # Plot Loss
    if ax_loss is not None:
        ax_loss.plot(epochs, history["train_losses"], label="Train Loss")
        ax_loss.plot(epochs, history["val_losses"], label="Val Loss")
        ax_loss.set_title(f"Training Curves (Subject: {subj_id})")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
    elif ax_acc is None: # Standalone mode
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, history["train_losses"], label="Train Loss")
        plt.plot(epochs, history["val_losses"], label="Val Loss")
        plt.title(f"Training Curves (Subject: {subj_id})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"training_curves_{subj_id}.png"))
        plt.close()
        print(f"Saved training_curves_{subj_id}.png to {output_dir}")

    # Plot Accuracy
    if ax_acc is not None:
        ax_acc.plot(epochs, history["train_accs"], label="Train Acc")
        ax_acc.plot(epochs, history["val_accs"], label="Val Acc")
        ax_acc.set_title(f"Accuracy Curves (Subject: {subj_id})")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.0)
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)
    elif ax_loss is None: # Standalone mode
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, history["train_accs"], label="Train Acc")
        plt.plot(epochs, history["val_accs"], label="Val Acc")
        plt.title(f"Accuracy Curves (Subject: {subj_id})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"accuracy_curves_{subj_id}.png"))
        plt.close()
        print(f"Saved accuracy_curves_{subj_id}.png to {output_dir}")

def plot_confusion_matrices(results, output_dir, ax=None):
    """Plots aggregated confusion matrix for 'clean' condition."""
    subjects = results.get("subjects", {})
    if not subjects:
        return

    # Aggregate confusion matrices
    total_cm = None
    
    for subj_data in subjects.values():
        cm = subj_data.get("metrics", {}).get("clean", {}).get("confusion_matrix")
        if cm:
            cm = np.array(cm)
            if total_cm is None:
                total_cm = cm
            else:
                if total_cm.shape == cm.shape:
                    total_cm += cm

    if total_cm is None:
        return

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        save = True
    else:
        save = False

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("Normalized Confusion Matrix (All Subjects)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis='x', rotation=45)

    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix_aggregated.png"))
        plt.close()
        print(f"Saved confusion_matrix_aggregated.png to {output_dir}")

def plot_per_class_metrics(results, output_dir, ax=None):
    """Plots per-class F1 scores (Mean ± Std) across subjects."""
    subjects = results.get("subjects", {})
    if not subjects:
        return

    # Collect per-class F1 scores
    class_scores = {i: [] for i in range(len(CLASS_NAMES))}
    
    for subj_data in subjects.values():
        f1_per_class = subj_data.get("metrics", {}).get("clean", {}).get("f1_per_class")
        if f1_per_class:
            for i, score in enumerate(f1_per_class):
                if i < len(CLASS_NAMES):
                    class_scores[i].append(score)

    # Prepare data for plotting
    means = []
    stds = []
    classes = []
    
    for i in range(len(CLASS_NAMES)):
        scores = class_scores[i]
        if scores:
            means.append(np.mean(scores))
            stds.append(np.std(scores))
            classes.append(CLASS_NAMES[i])
        else:
            means.append(0)
            stds.append(0)
            classes.append(CLASS_NAMES[i])

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        save = True
    else:
        save = False

    ax.bar(classes, means, yerr=stds, capsize=5, alpha=0.8)
    ax.set_title("Per-Class F1 Scores (Mean ± Std)")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "per_class_f1_aggregated.png"))
        plt.close()
        print(f"Saved per_class_f1_aggregated.png to {output_dir}")

def plot_score_distribution(results, output_dir, ax=None):
    """Plots distribution of Accuracy and F1-Macro across subjects."""
    subjects = results.get("subjects", {})
    if not subjects:
        return

    data = []
    for subj_data in subjects.values():
        metrics = subj_data.get("metrics", {}).get("clean", {})
        if metrics:
            data.append({"Metric": "Accuracy", "Score": metrics.get("accuracy", 0)})
            data.append({"Metric": "F1-Macro", "Score": metrics.get("f1_macro", 0)})
            
    df = pd.DataFrame(data)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        save = True
    else:
        save = False

    sns.boxplot(data=df, x="Metric", y="Score", width=0.5, ax=ax)
    
    # Add text box with mean/std
    acc_scores = df[df["Metric"]=="Accuracy"]["Score"]
    f1_scores = df[df["Metric"]=="F1-Macro"]["Score"]
    
    text_str = f"Mean Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}\n"
    text_str += f"Mean F1-Macro: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.05, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    ax.set_title("Score Distribution Across Subjects")
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "score_distribution.png"))
        plt.close()
        print(f"Saved score_distribution.png to {output_dir}")

def create_summary_plot(results, output_dir):
    """Creates a combined summary plot with all metrics."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_per_subject_performance(results, output_dir, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_training_curves(results, output_dir, ax_loss=ax2, ax_acc=None)
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_training_curves(results, output_dir, ax_loss=None, ax_acc=ax3)

    ax4 = fig.add_subplot(gs[1, 0])
    plot_confusion_matrices(results, output_dir, ax=ax4)

    ax5 = fig.add_subplot(gs[1, 1])
    plot_per_class_metrics(results, output_dir, ax=ax5)

    ax6 = fig.add_subplot(gs[1, 2])
    plot_score_distribution(results, output_dir, ax=ax6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_plot.png"))
    plt.close()
    print(f"Saved summary_plot.png to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize results from results.json")
    parser.add_argument("json_path", help="Path to results.json file")
    parser.add_argument("--output_dir", help="Directory to save plots", default=None)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_path):
        print(f"Error: File {args.json_path} not found.")
        return

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.json_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = load_results(args.json_path)
    
    # Generate individual plots
    plot_aggregate_metrics(results, args.output_dir)
    plot_per_subject_performance(results, args.output_dir)
    plot_training_curves(results, args.output_dir)
    plot_confusion_matrices(results, args.output_dir)
    plot_per_class_metrics(results, args.output_dir)
    plot_score_distribution(results, args.output_dir)
    
    # Generate summary plot
    create_summary_plot(results, args.output_dir)

if __name__ == "__main__":
    main()
