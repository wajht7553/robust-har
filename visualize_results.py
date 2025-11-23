import argparse
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
    accuracies = [metrics[c]["mean_accuracy"] for c in conditions]
    f1_scores = [metrics[c]["mean_f1_macro"] for c in conditions]
    
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

def plot_per_class_metrics(results, output_dir):
    """Plots per-class F1 scores for different conditions."""
    # We use the first seed's results for per-class metrics as aggregate doesn't have per-class breakdown usually
    # Or we can check if aggregate has it. In the provided JSON, aggregate only has mean_accuracy and mean_f1_macro.
    # So we look at per_seed_metrics.
    
    per_seed = results.get("per_seed_metrics", {})
    if not per_seed:
        print("No per-seed metrics found.")
        return
    
    # Use the first seed found
    first_seed = list(per_seed.keys())[0]
    seed_data = per_seed[first_seed]
    
    conditions = list(seed_data.keys())
    
    # Collect data
    data = []
    for cond in conditions:
        f1_per_class = seed_data[cond].get("f1_per_class", [])
        for i, score in enumerate(f1_per_class):
            data.append({"Condition": cond, "Class": f"Class {i}", "F1-Score": score})
            
    if not data:
        print("No per-class data found.")
        return

    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Class", y="F1-Score", hue="Condition")
    plt.title(f"Per-Class F1 Scores (Seed {first_seed})")
    plt.ylim(0, 1.0)
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_f1.png"))
    plt.close()
    print(f"Saved per_class_f1.png to {output_dir}")

def plot_confusion_matrices(results, output_dir):
    """Plots confusion matrices for each condition (from the first seed)."""
    per_seed = results.get("per_seed_metrics", {})
    if not per_seed:
        return

    first_seed = list(per_seed.keys())[0]
    seed_data = per_seed[first_seed]
    
    for cond, metrics in seed_data.items():
        cm = metrics.get("confusion_matrix")
        if cm:
            cm = np.array(cm)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {cond} (Seed {first_seed})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"confusion_matrix_{cond}.png"))
            plt.close()
            print(f"Saved confusion_matrix_{cond}.png to {output_dir}")

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
    
    plot_aggregate_metrics(results, args.output_dir)
    plot_per_class_metrics(results, args.output_dir)
    plot_confusion_matrices(results, args.output_dir)

if __name__ == "__main__":
    main()
