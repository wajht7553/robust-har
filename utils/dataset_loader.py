##################################################
# Dataset loader with LOSO (Leave-One-Subject-Out) support
##################################################

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HARDataset(Dataset):
    """Human Activity Recognition Dataset"""

    def __init__(self, X, y, normalize=True, mean=None, std=None):
        """
        Args:
            X: numpy array of shape (N, window_size, channels)
            y: numpy array of shape (N,)
            normalize: whether to normalize the data
            mean: pre-computed mean for normalization (if None, compute from X)
            std: pre-computed std for normalization (if None, compute from X)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.normalize = normalize

        if normalize:
            if mean is None or std is None:
                # Compute per-channel mean and std across all windows
                # X shape: (N, window_size, channels)
                self.mean = self.X.mean(dim=(0, 1), keepdim=True)
                self.std = self.X.std(dim=(0, 1), keepdim=True)
                # Avoid division by zero
                self.std[self.std == 0] = 1.0
            else:
                self.mean = torch.FloatTensor(mean).view(1, 1, -1)
                self.std = torch.FloatTensor(std).view(1, 1, -1)

            self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_stats(self):
        """Return mean and std for use in test set normalization"""
        if self.normalize:
            return self.mean.squeeze().numpy(), self.std.squeeze().numpy()
        return None, None


class LOSOSplitter:
    """Leave-One-Subject-Out cross-validation splitter"""

    def __init__(self, data_dir="dataset/processed_acc_gyr"):
        """
        Args:
            data_dir: directory containing X.npy, y.npy, and subject_index.json
        """
        self.data_dir = data_dir

        # Load data
        self.X = np.load(os.path.join(data_dir, "X.npy"))
        self.y = np.load(os.path.join(data_dir, "y.npy"))

        # Load subject index
        with open(os.path.join(data_dir, "subject_index.json"), "r") as f:
            self.subject_index = json.load(f)

        self.subjects = sorted(
            self.subject_index.keys(), key=lambda x: int(x.replace("proband", ""))
        )

        print(f"Loaded dataset: X.shape={self.X.shape}, y.shape={self.y.shape}")
        print(f"Number of subjects: {len(self.subjects)}")
        print(f"Subjects: {self.subjects}")

    def get_subject_data(self, subject):
        """Get data for a specific subject"""
        start_idx, end_idx = self.subject_index[subject]
        return self.X[start_idx:end_idx], self.y[start_idx:end_idx]

    def get_train_test_split(self, test_subject):
        """
        Get train/test split for LOSO with one subject held out

        Args:
            test_subject: subject to use as test set (e.g., 'proband1')

        Returns:
            X_train, y_train, X_test, y_test
        """
        if test_subject not in self.subjects:
            raise ValueError(f"Subject {test_subject} not found in dataset")

        # Collect training data from all subjects except test_subject
        X_train_list = []
        y_train_list = []

        for subject in self.subjects:
            X_subj, y_subj = self.get_subject_data(subject)
            if subject == test_subject:
                X_test, y_test = X_subj, y_subj
            else:
                X_train_list.append(X_subj)
                y_train_list.append(y_subj)

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        return X_train, y_train, X_test, y_test

    def get_loso_splits(self):
        """
        Generator that yields LOSO splits

        Yields:
            (test_subject, X_train, y_train, X_test, y_test)
        """
        for subject in self.subjects:
            X_train, y_train, X_test, y_test = self.get_train_test_split(subject)
            yield subject, X_train, y_train, X_test, y_test


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32, num_workers=0):
    """
    Create train and test dataloaders with proper normalization

    Args:
        X_train, y_train: training data
        X_test, y_test: test data
        batch_size: batch size for dataloaders
        num_workers: number of workers for data loading

    Returns:
        train_loader, test_loader, (mean, std) for normalization
    """
    # Create training dataset and get normalization stats
    train_dataset = HARDataset(X_train, y_train, normalize=True)
    mean, std = train_dataset.get_stats()

    # Create test dataset using training normalization stats
    test_dataset = HARDataset(X_test, y_test, normalize=True, mean=mean, std=std)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, (mean, std)
