"""Evaluation utilities for robust LOSO experiments"""

import torch
from src.utils.metrics import compute_metrics


class Evaluator:
    """Evaluates models on multiple test scenarios"""

    def __init__(self, model, trainer, device):
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

    def evaluate(
        self,
        test_loaders,
        checkpoint_path,
    ):
        """
        Evaluate model on all test scenarios.

        Args:
            test_loaders: Dictionary of DataLoaders for test scenarios
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

        metrics = {}
        for scenario, loader in test_loaders.items():
            metrics[scenario] = evaluate_loader(loader)

        return metrics

    def print_results(self, test_subject: str, metrics: dict):
        """Print evaluation results for a subject"""
        print(f"Subject {test_subject} Results:")
        for scenario, res in metrics.items():
            print(
                f"  {scenario.capitalize()}:   Acc={res['accuracy']:.4f}, F1={res['f1_macro']:.4f}"
            )
