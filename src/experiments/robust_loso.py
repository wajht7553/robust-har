"""Robust LOSO cross-validation experiment"""

import torch
import numpy as np
from typing import Optional, Union, Dict, Any
from omegaconf import DictConfig

from src.data.splitter import LOSOSplitter
from src.models.factory import create_model
from src.training.trainer import Trainer
from .experiment_manager import ExperimentManager
from .data_preparation import RobustDataPreparator
from .evaluator import RobustEvaluator


class RobustLOSOExperiment:
    """Robust Leave-One-Subject-Out cross-validation experiment"""

    def __init__(
        self,
        model_name: str,
        model_config: Union[Dict[str, Any], DictConfig],
        train_config: Union[Dict[str, Any], DictConfig],
        device: Optional[torch.device] = None,
        resume_dir: Optional[str] = None,
    ):
        """
        Initialize robust LOSO experiment.

        Args:
            model_name: Name of the model
            model_config: Model configuration dictionary
            train_config: Training configuration dictionary
            device: Torch device (default: auto-detect)
            resume_dir: Optional path to resume from existing experiment
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Setup experiment management
        self.exp_manager = ExperimentManager(
            model_name, model_config, train_config, resume_dir
        )

        # Setup data splitter
        self.splitter = LOSOSplitter(train_config["data_dir"])

        # Setup data preparator
        self.data_preparator = RobustDataPreparator(
            batch_size=train_config["batch_size"],
            num_workers=0,
        )

        self.model_name = self.exp_manager.model_name
        self.model_config = self.exp_manager.model_config
        self.train_config = self.exp_manager.train_config

    @property
    def experiment_dir(self):
        """Get experiment directory"""
        return self.exp_manager.experiment_dir

    @property
    def results(self):
        """Get results dictionary"""
        return self.exp_manager.results

    def train_fold(
        self,
        test_subject: str,
        val_subject: str,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ):
        """
        Train and evaluate a single fold.

        Args:
            test_subject: Test subject identifier
            val_subject: Validation subject identifier
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data

        Returns:
            dict: Fold results including metrics and history
        """
        print(f"\nTraining fold: Test={test_subject}, Val={val_subject}")

        # Prepare data loaders
        (
            train_loader,
            val_loader,
            test_clean_loader,
            test_noisy_loader,
            test_dropout_loader,
            _,
        ) = self.data_preparator.prepare_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test
        )

        # Create and train model
        model = create_model(self.model_name, self.model_config)
        checkpoint_path = self.exp_manager.get_checkpoint_path(test_subject)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.train_config["lr"],
            weight_decay=self.train_config.get("weight_decay", 0.0),
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.5, patience=5)

        # Cosine Annealing Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.train_config["epochs"], eta_min=1e-6
        )

        trainer = Trainer(
            model,
            self.device,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping_patience=self.train_config.get("patience", 10),
            checkpoint_path=checkpoint_path,
            aux_weight=self.train_config.get("aux_weight", 0.4),
        )

        history = trainer.train(train_loader, val_loader, self.train_config["epochs"])

        # Evaluate on all test scenarios
        evaluator = RobustEvaluator(model, trainer, self.device)
        metrics = evaluator.evaluate_all_scenarios(
            test_clean_loader,
            test_noisy_loader,
            test_dropout_loader,
            checkpoint_path,
        )

        evaluator.print_results(test_subject, metrics)

        return {
            "subject": test_subject,
            "val_subject": val_subject,
            "history": history,
            "metrics": metrics,
        }

    def run(self, limit_folds: Optional[int] = None):
        """
        Run the experiment, skipping completed folds if resuming.
        Each fold trains an independent model with fresh, randomly initialized weights.

        Args:
            limit_folds: Limit number of folds to process (for debugging)
        """
        completed_subjects = self.exp_manager.get_completed_subjects()
        all_metrics = {"clean": [], "noisy": [], "dropout": []}

        # Collect metrics from already completed folds for aggregation
        for subject in completed_subjects:
            subject_result = self.results["subjects"].get(subject, {})
            metrics = subject_result.get("metrics", {})
            for key in all_metrics:
                if key in metrics:
                    all_metrics[key].append(metrics[key])

        print("\nStarting experiment run...")
        if completed_subjects:
            print(
                f"Skipping {len(completed_subjects)} already completed folds: {sorted(completed_subjects)}"
            )

        fold_count = 0
        for i, (
            test_subj,
            val_subj,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        ) in enumerate(self.splitter.get_robust_loso_splits()):
            # Skip if already completed
            if test_subj in completed_subjects:
                print(f"Skipping fold {i + 1}: Test={test_subj} (already completed)")
                continue

            if limit_folds is not None and fold_count >= limit_folds:
                break

            fold_count += 1
            print(
                f"\nProcessing fold {i + 1}/{len(self.splitter.subjects)}: Test={test_subj}, Val={val_subj}"
            )

            fold_results = self.train_fold(
                test_subj, val_subj, X_train, y_train, X_val, y_val, X_test, y_test
            )
            self.results["subjects"][test_subj] = fold_results

            for key in all_metrics:
                all_metrics[key].append(fold_results["metrics"][key])

            # Update results.json after each fold
            self.exp_manager.save_results()

        # Aggregate metrics from all folds
        self._aggregate_metrics(all_metrics)
        self.exp_manager.save_results()
        self._print_summary()

    def _aggregate_metrics(self, all_metrics: dict):
        """Aggregate metrics across all folds"""
        self.results["aggregate_metrics"] = {}
        for key in all_metrics:
            if all_metrics[key]:  # Only compute if we have metrics
                accs = [m["accuracy"] for m in all_metrics[key]]
                f1s = [m["f1_macro"] for m in all_metrics[key]]
                self.results["aggregate_metrics"][key] = {
                    "mean_accuracy": float(np.mean(accs)),
                    "std_accuracy": float(np.std(accs)),
                    "mean_f1_macro": float(np.mean(f1s)),
                    "std_f1_macro": float(np.std(f1s)),
                }

    def _print_summary(self):
        """Print experiment summary"""
        total_folds = len(self.results["subjects"])
        expected_folds = len(self.splitter.subjects)

        if total_folds == expected_folds:
            print("\nRobust LOSO Experiment Completed.")
            for key, metrics in self.results["aggregate_metrics"].items():
                print(
                    f"{key.capitalize()}: Mean Acc = {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}"
                )
        else:
            print(
                f"\nRobust LOSO Experiment Progress: {total_folds}/{expected_folds} folds completed."
            )
            print("Use --resume to continue training remaining folds.")
            if self.results["aggregate_metrics"]:
                for key, metrics in self.results["aggregate_metrics"].items():
                    print(
                        f"{key.capitalize()}: Mean Acc = {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}"
                    )
