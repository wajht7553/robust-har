import torch
from torch.utils.data import Dataset


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
