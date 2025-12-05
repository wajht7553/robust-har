"""Data preparation utilities for robust LOSO experiments"""

from torch.utils.data import DataLoader
from src.data.dataset import HARDataset
from src.data.transforms import (
    MixedDistributionTransform,
    MissingModalityTransform,
    NoiseInjectionTransform,
    ModalityDropoutTransform,
    SignalDegradationTransform,
)
from omegaconf import DictConfig


class DataPreparator:
    """Prepares datasets and loaders for LOSO experiments based on strategy"""

    def __init__(
        self, batch_size: int, strategy_config: DictConfig, num_workers: int = 0
    ):
        """
        Initialize data preparator.

        Args:
            batch_size: Batch size for data loaders
            strategy_config: Strategy configuration
            num_workers: Number of workers for data loading
        """
        self.batch_size = batch_size
        self.strategy_config = strategy_config
        self.num_workers = num_workers

    def prepare_loaders(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ):
        """
        Prepare all data loaders for a fold.

        Returns:
            tuple: (train_loader, val_loader, test_loaders_dict, normalization_stats)
        """
        # Train Transform
        train_transform = None
        if self.strategy_config.train_transform == "mixed":
            train_transform = MixedDistributionTransform()
        elif self.strategy_config.train_transform == "modality_dropout_10":
            train_transform = ModalityDropoutTransform(dropout_rate=0.1)
        elif self.strategy_config.train_transform == "modality_dropout_30":
            train_transform = ModalityDropoutTransform(dropout_rate=0.3)
        elif self.strategy_config.train_transform == "modality_dropout_50":
            train_transform = ModalityDropoutTransform(dropout_rate=0.5)
        elif self.strategy_config.train_transform == "signal_degradation":
            train_transform = SignalDegradationTransform()

        # Train Dataset
        train_dataset = HARDataset(
            X_train, y_train, normalize=True, transform=train_transform
        )
        mean, std = train_dataset.get_stats()

        # Validation Dataset (Clean)
        val_dataset = HARDataset(X_val, y_val, normalize=True, mean=mean, std=std)

        # Test Datasets based on scenarios
        test_loaders = {}
        scenarios = self.strategy_config.get("test_scenarios", ["clean"])

        for scenario in scenarios:
            transform = None
            if scenario == "noisy":
                transform = NoiseInjectionTransform(p=1.0)
            elif scenario == "dropout":
                transform = MissingModalityTransform(modality="gyro", p=1.0)

            dataset = HARDataset(
                X_test,
                y_test,
                normalize=True,
                mean=mean,
                std=std,
                transform=transform,
            )

            test_loaders[scenario] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Create train/val loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return (
            train_loader,
            val_loader,
            test_loaders,
            (mean, std),
        )
