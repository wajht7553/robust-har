"""Evaluation utilities for robust LOSO experiments"""

import torch
from src.training.trainer import Trainer
from src.utils.metrics import compute_metrics


class RobustEvaluator:
    """Evaluates models on multiple test scenarios"""

    def __init__(self, model, trainer: Trainer, device):
        """
        Initialize evaluator.

        Args:
            model: PyTorch model
            trainer: Trainer instance
            device: Torch device
        """
        self.model = model
        self.trainer = trainer
        self.device = device

    def evaluate_all_scenarios(
        self,
        test_clean_loader,
        test_noisy_loader,
        test_dropout_loader,
        checkpoint_path: str,
    ):
        """
        Evaluate model on all test scenarios.

        Args:
            test_clean_loader: DataLoader for clean test data
            test_noisy_loader: DataLoader for noisy test data
            test_dropout_loader: DataLoader for dropout test data
            checkpoint_path: Path to model checkpoint

        Returns:
            dict: Metrics for each scenario
        """
        # Load best model
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        def evaluate_loader(loader):
            _, _, y_pred, y_true = self.trainer.validate(loader)
            return compute_metrics(y_true, y_pred)

        metrics_clean = evaluate_loader(test_clean_loader)
        metrics_noisy = evaluate_loader(test_noisy_loader)
        metrics_dropout = evaluate_loader(test_dropout_loader)

        return {
            "clean": metrics_clean,
            "noisy": metrics_noisy,
            "dropout": metrics_dropout,
        }

    def print_results(self, test_subject: str, metrics: dict):
        """Print evaluation results for a subject"""
        print(f"Subject {test_subject} Results:")
        print(
            f"  Clean:   Acc={metrics['clean']['accuracy']:.4f}, F1={metrics['clean']['f1_macro']:.4f}"
        )
        print(
            f"  Noisy:   Acc={metrics['noisy']['accuracy']:.4f}, F1={metrics['noisy']['f1_macro']:.4f}"
        )
        print(
            f"  Dropout: Acc={metrics['dropout']['accuracy']:.4f}, F1={metrics['dropout']['f1_macro']:.4f}"
        )
