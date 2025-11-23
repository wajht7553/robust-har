import os
import argparse
import yaml
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

from src.data.splitter import LOSOSplitter
from src.data.dataset import HARDataset
from src.data.transforms import MixedDistributionTransform, MissingModalityTransform, NoiseInjectionTransform
from src.models.factory import create_model
from src.training.trainer import Trainer
from src.utils.metrics import compute_metrics
from src.utils.common import save_model, save_json

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class RobustLOSOExperiment:
    def __init__(self, model_name, model_config, train_config, device=None):
        self.model_name = model_name
        self.model_config = model_config
        self.train_config = train_config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            train_config["results_dir"], f"{model_name}_robust_{timestamp}"
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

        print(f"Robust Experiment initialized: {self.experiment_dir}")

    def train_fold(self, test_subject, val_subject, X_train, y_train, X_val, y_val, X_test, y_test):
        print(f"\nTraining fold: Test={test_subject}, Val={val_subject}")

        # 1. Prepare Data
        # Train: Mixed Distribution
        train_transform = MixedDistributionTransform()
        
        # Create Datasets
        train_dataset = HARDataset(X_train, y_train, normalize=True, transform=train_transform)
        mean, std = train_dataset.get_stats()
        
        # Validation: Clean (for early stopping)
        val_dataset = HARDataset(X_val, y_val, normalize=True, mean=mean, std=std)
        
        # Test: 3 Variants
        # A. Clean
        test_clean_dataset = HARDataset(X_test, y_test, normalize=True, mean=mean, std=std)
        
        # B. Noisy (Noise Injection)
        # Using NoiseInjectionTransform as representative of "Noisy"
        test_noisy_dataset = HARDataset(X_test, y_test, normalize=True, mean=mean, std=std, 
                                        transform=NoiseInjectionTransform(p=1.0))
        
        # C. Dropout (Missing Gyro)
        test_dropout_dataset = HARDataset(X_test, y_test, normalize=True, mean=mean, std=std, 
                                          transform=MissingModalityTransform(modality='gyro', p=1.0))

        # Loaders
        train_loader = DataLoader(train_dataset, batch_size=self.train_config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.train_config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
        test_clean_loader = DataLoader(test_clean_dataset, batch_size=self.train_config["batch_size"], shuffle=False)
        test_noisy_loader = DataLoader(test_noisy_dataset, batch_size=self.train_config["batch_size"], shuffle=False)
        test_dropout_loader = DataLoader(test_dropout_dataset, batch_size=self.train_config["batch_size"], shuffle=False)

        # 2. Train Model
        model = create_model(self.model_name, self.model_config)
        checkpoint_path = os.path.join(self.experiment_dir, f"best_model_{test_subject}.pt")
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_config["lr"])

        trainer = Trainer(
            model,
            self.device,
            optimizer=optimizer,
            early_stopping_patience=10,
            checkpoint_path=checkpoint_path,
        )

        history = trainer.train(train_loader, val_loader, self.train_config["epochs"])

        # 3. Evaluate on Test Variants
        # Load best model
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(self.device)
        model.eval()
        
        def evaluate_loader(loader):
            _, _, y_pred, y_true = trainer.validate(loader)
            return compute_metrics(y_true, y_pred)

        metrics_clean = evaluate_loader(test_clean_loader)
        metrics_noisy = evaluate_loader(test_noisy_loader)
        metrics_dropout = evaluate_loader(test_dropout_loader)

        print(f"Subject {test_subject} Results:")
        print(f"  Clean:   Acc={metrics_clean['accuracy']:.4f}, F1={metrics_clean['f1_macro']:.4f}")
        print(f"  Noisy:   Acc={metrics_noisy['accuracy']:.4f}, F1={metrics_noisy['f1_macro']:.4f}")
        print(f"  Dropout: Acc={metrics_dropout['accuracy']:.4f}, F1={metrics_dropout['f1_macro']:.4f}")

        return {
            "subject": test_subject,
            "val_subject": val_subject,
            "history": history,
            "metrics": {
                "clean": metrics_clean,
                "noisy": metrics_noisy,
                "dropout": metrics_dropout
            }
        }

    def run(self, limit_folds=None):
        all_metrics = {"clean": [], "noisy": [], "dropout": []}
        
        for i, (test_subj, val_subj, X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(self.splitter.get_robust_loso_splits()):
            if limit_folds is not None and i >= limit_folds:
                break
                
            fold_results = self.train_fold(test_subj, val_subj, X_train, y_train, X_val, y_val, X_test, y_test)
            self.results["subjects"][test_subj] = fold_results
            
            for key in all_metrics:
                all_metrics[key].append(fold_results["metrics"][key])
                
            save_json(self.results, os.path.join(self.experiment_dir, "results.json"))

        # Aggregate
        self.results["aggregate_metrics"] = {}
        for key in all_metrics:
            accs = [m["accuracy"] for m in all_metrics[key]]
            f1s = [m["f1_macro"] for m in all_metrics[key]]
            self.results["aggregate_metrics"][key] = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_f1_macro": float(np.mean(f1s)),
                "std_f1_macro": float(np.std(f1s))
            }

        save_json(self.results, os.path.join(self.experiment_dir, "results.json"))
        print("\nRobust LOSO Experiment Completed.")
        for key, metrics in self.results["aggregate_metrics"].items():
            print(f"{key.capitalize()}: Mean Acc = {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--config", type=str, help="Path to model config override")
    parser.add_argument("--train_config", type=str, default="configs/train_robust.yaml", help="Path to training config")
    parser.add_argument("--limit_folds", type=int, default=None, help="Limit number of folds for debugging")
    args = parser.parse_args()

    train_config = load_config(args.train_config)
    
    if args.config:
        model_config = load_config(args.config)
    else:
        model_config_path = f"configs/model/{args.model}.yaml"
        if os.path.exists(model_config_path):
            model_config = load_config(model_config_path)
        else:
            raise ValueError(f"No default config found for {args.model}")

    experiment = RobustLOSOExperiment(args.model, model_config, train_config)
    experiment.run(limit_folds=args.limit_folds)

if __name__ == "__main__":
    main()
