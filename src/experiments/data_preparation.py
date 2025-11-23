"""Data preparation utilities for robust LOSO experiments"""

from torch.utils.data import DataLoader
from src.data.dataset import HARDataset
from src.data.transforms import (
    MixedDistributionTransform,
    MissingModalityTransform,
    NoiseInjectionTransform,
)


class RobustDataPreparator:
    """Prepares datasets and loaders for robust LOSO experiments"""

    def __init__(self, batch_size: int, num_workers: int = 0):
        """
        Initialize data preparator.

        Args:
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
        """
        self.batch_size = batch_size
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
            tuple: (train_loader, val_loader, test_clean_loader,
                   test_noisy_loader, test_dropout_loader, normalization_stats)
        """
        # Train: Mixed Distribution
        train_transform = MixedDistributionTransform()
        train_dataset = HARDataset(
            X_train, y_train, normalize=True, transform=train_transform
        )
        mean, std = train_dataset.get_stats()

        # Validation: Clean (for early stopping)
        val_dataset = HARDataset(X_val, y_val, normalize=True, mean=mean, std=std)

        # Test: 3 Variants
        test_clean_dataset = HARDataset(
            X_test, y_test, normalize=True, mean=mean, std=std
        )
        test_noisy_dataset = HARDataset(
            X_test,
            y_test,
            normalize=True,
            mean=mean,
            std=std,
            transform=NoiseInjectionTransform(p=1.0),
        )
        test_dropout_dataset = HARDataset(
            X_test,
            y_test,
            normalize=True,
            mean=mean,
            std=std,
            transform=MissingModalityTransform(modality="gyro", p=1.0),
        )

        # Create loaders
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
        test_clean_loader = DataLoader(
            test_clean_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        test_noisy_loader = DataLoader(
            test_noisy_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        test_dropout_loader = DataLoader(
            test_dropout_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return (
            train_loader,
            val_loader,
            test_clean_loader,
            test_noisy_loader,
            test_dropout_loader,
            (mean, std),
        )
