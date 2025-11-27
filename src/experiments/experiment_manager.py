"""Experiment directory and state management"""

import os
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Optional, Union
from src.utils.common import save_json, load_json


class ExperimentManager:
    """Manages experiment directory, configs, and results"""

    def __init__(
        self,
        model_name: str,
        model_config: Union[Dict[str, Any], DictConfig],
        train_config: Union[Dict[str, Any], DictConfig],
        resume_dir: Optional[str] = None,
    ):
        """
        Initialize experiment manager.

        Args:
            model_name: Name of the model
            model_config: Model configuration (dict or DictConfig)
            train_config: Training configuration (dict or DictConfig)
            resume_dir: Optional path to resume from existing experiment
        """
        self.model_name = model_name
        self.model_config = model_config
        self.train_config = train_config
        self.resume_dir = resume_dir

        if resume_dir:
            self._setup_resume_experiment()
        else:
            self._setup_new_experiment()

    def _setup_resume_experiment(self):
        """Setup for resuming an existing experiment"""
        self.experiment_dir = self.resume_dir
        if not os.path.exists(self.experiment_dir):
            raise ValueError(f"Resume directory does not exist: {self.resume_dir}")

        # Load existing results
        results_path = os.path.join(self.experiment_dir, "results.json")
        if os.path.exists(results_path):
            self.results = load_json(results_path)
            print(f"Resuming experiment: {self.experiment_dir}")
        else:
            # No results yet, start fresh
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results = {
                "model_name": self.model_name,
                "timestamp": timestamp,
                "subjects": {},
                "aggregate_metrics": {},
            }
            print(f"Resuming experiment (no previous results): {self.experiment_dir}")
        # Load existing configs to ensure consistent state after resume
        model_config_path = os.path.join(self.experiment_dir, "model_config.yaml")
        train_config_path = os.path.join(self.experiment_dir, "train_config.yaml")
        if os.path.exists(model_config_path):
            self.model_config = OmegaConf.load(model_config_path)
            print(f"Loaded model config from {model_config_path}")
        if os.path.exists(train_config_path):
            self.train_config = OmegaConf.load(train_config_path)
            print(f"Loaded train config from {train_config_path}")

            print(f"Found {len(self.results.get('subjects', {}))} completed folds")

    def _setup_new_experiment(self):
        """Setup for a new experiment"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Handle DictConfig for path construction
        results_dir = (
            self.train_config.get("results_dir", "results")
            if isinstance(self.train_config, dict)
            else self.train_config.results_dir
        )

        # Get strategy name from config, default to 'robust' if not present
        if isinstance(self.train_config, dict):
            strategy_name = self.train_config.get("strategy", {}).get("name", "robust")
        else:
            strategy_name = (
                self.train_config.strategy.name
                if "strategy" in self.train_config
                else "robust"
            )

        self.experiment_dir = os.path.join(
            results_dir, f"{self.model_name}_{strategy_name}_{timestamp}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Save configs
        self._save_config(self.model_config, "model_config.yaml")
        self._save_config(self.train_config, "train_config.yaml")

        self.results = {
            "model_name": self.model_name,
            "timestamp": timestamp,
            "subjects": {},
            "aggregate_metrics": {},
        }
        print(f"Experiment initialized: {self.experiment_dir}")

    def _save_config(self, config, filename):
        """Save configuration to file"""
        path = os.path.join(self.experiment_dir, filename)
        if isinstance(config, DictConfig):
            OmegaConf.save(config, path)
        else:
            # Fallback for dict
            import yaml

            with open(path, "w") as f:
                yaml.dump(config, f)

    def save_results(self):
        """Save current results to JSON file"""
        results_path = os.path.join(self.experiment_dir, "results.json")
        save_json(self.results, results_path)

    def get_checkpoint_path(self, test_subject: str) -> str:
        """Get path for model checkpoint"""
        return os.path.join(self.experiment_dir, f"best_model_{test_subject}.pt")

    def get_completed_subjects(self) -> set:
        """Get set of subjects that have already been completed"""
        return set(self.results.get("subjects", {}).keys())
