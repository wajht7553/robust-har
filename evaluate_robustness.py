
import os
import sys
import json
import torch
import argparse
import numpy as np
from torch import nn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_loader import LOSOSplitter, create_dataloaders, SensorFailureTransform
from utils.training_utils import Trainer, compute_metrics
from models.MobileViT import MobileViT
from models.DeepConvLSTM import DeepConvLSTM

def load_model(model_path, model_name, config, device):
    if model_name.lower() == "deepconvlstm":
        model = DeepConvLSTM(config)
    elif model_name.lower() == "mobilevit":
        model = MobileViT(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_scenario(model, test_loader, device, scenario_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters()) # Dummy optimizer
    trainer = Trainer(model, device, criterion, optimizer)
    
    print(f"Evaluating scenario: {scenario_name}")
    test_loss, test_acc, y_pred, y_true = trainer.validate(test_loader)
    metrics = compute_metrics(y_true, y_pred, num_classes=8) # Hardcoded 8 classes for now
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate model robustness")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to experiment directory containing results.json")
    parser.add_argument("--data_dir", type=str, default="dataset/processed_acc_gyr", help="Path to processed data")
    parser.add_argument("--subject", type=str, default="proband1", help="Subject to use for evaluation (must match a trained model fold)")
    
    args = parser.parse_args()
    
    # Load experiment results
    results_path = os.path.join(args.experiment_dir, "results.json")
    with open(results_path, "r") as f:
        results = json.load(f)
        
    model_name = results["model_name"]
    config = results["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data for the specific subject
    splitter = LOSOSplitter(args.data_dir)
    # We need to recreate the split for this subject
    # Note: In LOSO, the subject is the TEST set.
    X_train, y_train, X_test, y_test = splitter.get_train_test_split(args.subject)
    
    # Load the model for this subject fold
    # The training script saves models as model_{subject}.pt
    model_path = os.path.join(args.experiment_dir, f"model_{args.subject}.pt")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
        
    model = load_model(model_path, model_name, config, device)
    
    # Define scenarios
    scenarios = {
        "Clean": None,
        "Missing Gyro": SensorFailureTransform(p_dropout_gyro=1.0),
        "Noisy Accel (std=0.1)": SensorFailureTransform(p_noise=1.0, noise_std=0.1),
        "Noisy Accel (std=0.2)": SensorFailureTransform(p_noise=1.0, noise_std=0.2),
        "Random Channel Drop (p=0.2)": SensorFailureTransform(p_channel_drop=0.2),
    }
    
    scenario_results = {}
    
    print(f"\n{'='*80}")
    print(f"Robustness Evaluation for {model_name} (Subject: {args.subject})")
    print(f"{'='*80}\n")
    
    for name, transform in scenarios.items():
        # Create dataloader with the specific transform
        # We only care about the test loader here
        # We pass the SAME normalization stats as training (which we need to re-compute or load)
        # Ideally we should load them from results.json if saved, but re-computing from X_train is fine/safer
        
        # Re-compute mean/std from training data
        # (In a real deployment, these would be saved constants)
        _, test_loader, _ = create_dataloaders(
            X_train, y_train, X_test, y_test, 
            batch_size=32, 
            num_workers=0,
            test_transform=transform
        )
        
        metrics = evaluate_scenario(model, test_loader, device, name)
        scenario_results[name] = metrics
        
        print(f"  {name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

    # Save robustness report
    report_path = os.path.join(args.experiment_dir, f"robustness_report_{args.subject}.json")
    with open(report_path, "w") as f:
        json.dump(scenario_results, f, indent=2)
    print(f"\nRobustness report saved to: {report_path}")

if __name__ == "__main__":
    main()
