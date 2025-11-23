import os
import json
import numpy as np
from torch.utils.data import DataLoader
from .dataset import HARDataset


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
        """Legacy method for standard LOSO"""
        if test_subject not in self.subjects:
            raise ValueError(f"Subject {test_subject} not found in dataset")

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
        """Legacy generator"""
        for subject in self.subjects:
            X_train, y_train, X_test, y_test = self.get_train_test_split(subject)
            yield subject, X_train, y_train, X_test, y_test

    def get_robust_loso_splits(self):
        """
        Generator for Robust LOSO:
        Yields:
            (test_subj, val_subj, X_train, y_train, X_val, y_val, X_test, y_test)
        """
        num_subjects = len(self.subjects)
        
        for i, test_subject in enumerate(self.subjects):
            # Select validation subject (next one, cyclic)
            val_idx = (i + 1) % num_subjects
            val_subject = self.subjects[val_idx]
            
            X_train_list = []
            y_train_list = []
            
            X_test, y_test = self.get_subject_data(test_subject)
            X_val, y_val = self.get_subject_data(val_subject)
            
            for subject in self.subjects:
                if subject == test_subject or subject == val_subject:
                    continue
                
                X_subj, y_subj = self.get_subject_data(subject)
                X_train_list.append(X_subj)
                y_train_list.append(y_subj)
                
            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
            
            yield test_subject, val_subject, X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(
    X_train,
    y_train,
    X_test,
    y_test,
    batch_size=32,
    num_workers=0,
    train_transform=None,
    test_transform=None,
):
    """
    Create train and test dataloaders with proper normalization
    """
    # Create training dataset and get normalization stats
    train_dataset = HARDataset(
        X_train, y_train, normalize=True, transform=train_transform
    )
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
