##################################################
# Dataset loader with LOSO (Leave-One-Subject-Out) support
##################################################

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



class SensorFailureTransform:
    """
    Transform to simulate sensor failures and noise.
    Assumes input shape (window_size, channels).
    Channels 0-2: Accel (x,y,z)
    Channels 3-5: Gyro (x,y,z)
    """

    def __init__(self, p_dropout_gyro=0.0, p_noise=0.0, noise_std=0.1, p_channel_drop=0.0):
        self.p_dropout_gyro = p_dropout_gyro
        self.p_noise = p_noise
        self.noise_std = noise_std
        self.p_channel_drop = p_channel_drop

    def __call__(self, x):
        # x is a torch tensor (window_size, channels)
        x_aug = x.clone()

        # 1. Gyro Dropout (Missing Gyro)
        if self.p_dropout_gyro > 0 and torch.rand(1) < self.p_dropout_gyro:
            if x_aug.shape[1] >= 6:
                x_aug[:, 3:6] = 0.0

        # 2. Add Gaussian Noise
        if self.p_noise > 0 and torch.rand(1) < self.p_noise:
            noise = torch.randn_like(x_aug) * self.noise_std
            x_aug = x_aug + noise

        # 3. Random Channel Dropout
        if self.p_channel_drop > 0:
            # Create a mask for channels
            # shape (channels,)
            mask = torch.rand(x_aug.shape[1]) > self.p_channel_drop
            # Ensure at least one channel is kept? Or allow total blackout?
            # Let's allow total blackout for robustness testing, but typically we want some data.
            if mask.sum() == 0:
                mask[torch.randint(0, x_aug.shape[1], (1,))] = True
            
            x_aug = x_aug * mask.view(1, -1)

        return x_aug


class HARDataset(Dataset):
    """Human Activity Recognition Dataset"""

    def __init__(self, X, y, normalize=True, mean=None, std=None, transform=None):
        """
        Args:
            X: numpy array of shape (N, window_size, channels)
            y: numpy array of shape (N,)
            normalize: whether to normalize the data
            mean: pre-computed mean for normalization (if None, compute from X)
            std: pre-computed std for normalization (if None, compute from X)
            transform: callable transform to apply to the sample
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.normalize = normalize
        self.transform = transform

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
        sample = self.X[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.y[idx]

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


def create_dataloaders(
    X_train, y_train, X_test, y_test, batch_size=32, num_workers=0,
    train_transform=None, test_transform=None
):
    """
    Create train and test dataloaders with proper normalization

    Args:
        X_train, y_train: training data
        X_test, y_test: test data
        batch_size: batch size for dataloaders
        num_workers: number of workers for data loading
        train_transform: transform to apply to training data
        test_transform: transform to apply to test data

    Returns:
        train_loader, test_loader, (mean, std) for normalization
    """
    # Create training dataset and get normalization stats
    train_dataset = HARDataset(X_train, y_train, normalize=True, transform=train_transform)
    mean, std = train_dataset.get_stats()

    # Create test dataset using training normalization stats
    test_dataset = HARDataset(
        X_test, y_test, normalize=True, mean=mean, std=std, transform=test_transform
    )

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
