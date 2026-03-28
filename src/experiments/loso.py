"""LOSO cross-validation experiment"""

import torch
import numpy as np
from typing import Optional, Union, Dict, Any
from omegaconf import DictConfig

from src.data.splitter import LOSOSplitter
from src.models.factory import create_model
from src.training.trainer import Trainer
from .experiment_manager import ExperimentManager
from .data_preparation import DataPreparator
from .evaluator import Evaluator


class LOSOExperiment:
    """Leave-One-Subject-Out cross-validation experiment"""

    def __init__(
        self,
        model_name,
        model_config,
        train_config,
        device=None,
        resume_dir=None,
    ):
        """
        Initialize LOSO experiment.

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

        # Extract random subject subset param if present
        random_subjects = train_config.get("random_subjects", None)
        if random_subjects is None and hasattr(train_config, "dataset"):
            random_subjects = train_config.dataset.get("random_subjects", None)
        seed = train_config.get("seed", 42)

        # Setup data splitter
        self.splitter = LOSOSplitter(train_config["data_dir"], random_subjects, seed)

        # Setup data preparator
        # Extract strategy config
        if isinstance(train_config, dict):
            strategy_config = DictConfig(train_config.get("strategy", {}))
        else:
            strategy_config = train_config.strategy

        self.data_preparator = DataPreparator(
            batch_size=train_config["batch_size"],
            strategy_config=strategy_config,
            num_workers=0,
        )

        self.model_name = self.exp_manager.model_name
        self.model_config = self.exp_manager.model_config
        self.train_config = self.exp_manager.train_config

        # Check configuration compatibility to prevent ignored cross-entropy dimension mismatches
        dataset_classes = len(np.unique(self.splitter.y))
        model_classes = self.model_config.get("nb_classes", self.model_config.get("num_classes", 8))
        if dataset_classes != model_classes:
            raise ValueError(f"Dataset mismatch: Model configured for {model_classes} classes, but the processed dataset contains exactly {dataset_classes} unique classes.")

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
        test_subject,
        val_subject,
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
            test_loaders,
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
        evaluator = Evaluator(model, trainer, self.device)
        metrics = evaluator.evaluate(
            test_loaders,
            checkpoint_path,
        )

        evaluator.print_results(test_subject, metrics)

        return {
            "subject": test_subject,
            "val_subject": val_subject,
            "history": history,
            "metrics": metrics,
        }

    def run_tuning(self):
        """
        Runs a single trial on a train/validation split optimized for hyperparameter tuning.
        Returns the evaluation macro F1 score.
        """
        print("\nStarting Hyperparameter Tuning Trial...")
        
        X_train, y_train, X_val, y_val = self.splitter.get_tuning_split(val_ratio=0.2, seed=self.train_config.get("seed", 42))
        
        # Prepare dataloaders
        # For tuning, we can reuse the test evaluation pipeline on the val split
        # so we pass X_val, y_val as both validation and test to prepare_loaders
        (
            train_loader,
            val_loader,
            test_loaders,
            _,
        ) = self.data_preparator.prepare_loaders(
            X_train, y_train, X_val, y_val, X_val, y_val
        )

        model = create_model(self.model_name, self.model_config)
        checkpoint_path = self.exp_manager.get_checkpoint_path("tuning_trial")
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.train_config["lr"],
            weight_decay=self.train_config.get("weight_decay", 0.0),
        )

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

        trainer.train(train_loader, val_loader, self.train_config["epochs"])

        evaluator = Evaluator(model, trainer, self.device)
        metrics = evaluator.evaluate(
            test_loaders,
            checkpoint_path,
        )

        # evaluator.evaluate returns a dict mapped by scenario name, typically 'clean' is first or main.
        # test_loaders is a dict with scenario names, e.g., {'clean': loader}
        # metrics dict structure: {'clean': {'accuracy': x, 'f1_macro': y}, ...}
        primary_scenario = list(metrics.keys())[0]
        f1_macro = metrics[primary_scenario]["f1_macro"]
        
        print(f"Tuning Trial Finished: F1 Macro = {f1_macro:.4f}")
        return float(f1_macro)

    def run(self, limit_folds: Optional[int] = None):
        """
        Run the experiment, skipping completed folds if resuming.
        Each fold trains an independent model with fresh, randomly initialized weights.

        Args:
            limit_folds: Limit number of folds to process (for debugging)
        """
        completed_subjects = self.exp_manager.get_completed_subjects()

        # Initialize metrics aggregation structure dynamically based on first fold results
        # But we need to know keys beforehand or aggregate lazily.
        # Let's aggregate lazily.
        all_metrics = {}

        # Collect metrics from already completed folds for aggregation
        for subject in completed_subjects:
            subject_result = self.results["subjects"].get(subject, {})
            metrics = subject_result.get("metrics", {})
            for key in metrics:
                if key not in all_metrics:
                    all_metrics[key] = []
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

            for key in fold_results["metrics"]:
                if key not in all_metrics:
                    all_metrics[key] = []
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
            print("\nLOSO Experiment Completed.")
            for key, metrics in self.results["aggregate_metrics"].items():
                print(
                    f"{key.capitalize()}: Mean Acc = {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}"
                )
        else:
            print(
                f"\nLOSO Experiment Progress: {total_folds}/{expected_folds} folds completed."
            )
            print("Use --resume to continue training remaining folds.")
            if self.results["aggregate_metrics"]:
                for key, metrics in self.results["aggregate_metrics"].items():
                    print(
                        f"{key.capitalize()}: Mean Acc = {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}"
                    )
