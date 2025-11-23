import torch
import numpy as np

class SensorFailureTransform:
    """
    Legacy wrapper for backward compatibility.
    """
    def __init__(self, p_dropout_gyro=0.0, p_noise=0.0, noise_std=0.1, p_channel_drop=0.0):
        self.transforms = []
        pass

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class MixedDistributionTransform:
    """
    Mixed Distribution for Robust Training:
    - 40% Clean
    - 30% Signal Degradation (Noise, Drift, Saturation)
    - 30% Modality Dropout (Missing Gyro)
    """
    def __init__(self):
        # Define specific transforms
        self.degradations = [
            DriftTransform(p=1.0),
            PacketLossTransform(p=1.0),
            SaturationTransform(p=1.0),
            NoiseInjectionTransform(p=1.0)
        ]
        self.dropout = MissingModalityTransform(modality='gyro', p=1.0)

    def __call__(self, x):
        rand = torch.rand(1).item()
        
        if rand < 0.4:
            # 40% Clean
            return x
        elif rand < 0.7:
            # 30% Signal Degradation
            # Pick one randomly
            t = self.degradations[torch.randint(0, len(self.degradations), (1,)).item()]
            return t(x)
        else:
            # 30% Modality Dropout
            return self.dropout(x)

class MissingModalityTransform:
    """1.1 Missing Entire Modality"""
    def __init__(self, modality='gyro', channels_map=None, p=1.0):
        self.modality = modality
        self.channels_map = channels_map or {'acc': [0, 1, 2], 'gyro': [3, 4, 5]}
        self.p = p

    def __call__(self, x):
        if torch.rand(1) > self.p: return x
        x_aug = x.clone()
        if self.modality in self.channels_map:
            idx = self.channels_map[self.modality]
            valid_idx = [i for i in idx if i < x_aug.shape[1]]
            if valid_idx:
                x_aug[:, valid_idx] = 0.0
        return x_aug

class DriftTransform:
    """Bias/Drift"""
    def __init__(self, axis_idx=5, drift_rate=0.001, p=1.0):
        self.axis_idx = axis_idx
        self.drift_rate = drift_rate
        self.p = p

    def __call__(self, x):
        if torch.rand(1) > self.p: return x
        x_aug = x.clone()
        if self.axis_idx < x_aug.shape[1]:
            t = torch.arange(x_aug.shape[0], device=x_aug.device).float()
            drift = t * self.drift_rate
            x_aug[:, self.axis_idx] += drift
        return x_aug

class PacketLossTransform:
    """Packet Loss"""
    def __init__(self, sampling_rate=50, min_duration=0.5, max_duration=2.0, p=1.0):
        self.min_samples = int(min_duration * sampling_rate)
        self.max_samples = int(max_duration * sampling_rate)
        self.p = p

    def __call__(self, x):
        if torch.rand(1) > self.p: return x
        x_aug = x.clone()
        seq_len = x_aug.shape[0]
        if self.max_samples > self.min_samples:
            dur = torch.randint(self.min_samples, self.max_samples, (1,)).item()
        else:
            dur = self.min_samples
        if seq_len - dur > 0:
            start = torch.randint(0, seq_len - dur, (1,)).item()
            x_aug[start:start+dur, :] = 0.0
        return x_aug

class SaturationTransform:
    """Saturation"""
    def __init__(self, threshold=15.0, channels=None, p=1.0):
        self.threshold = threshold
        self.channels = channels
        self.p = p

    def __call__(self, x):
        if torch.rand(1) > self.p: return x
        x_aug = x.clone()
        if self.channels:
            for c in self.channels:
                if c < x_aug.shape[1]:
                    x_aug[:, c] = torch.clamp(x_aug[:, c], -self.threshold, self.threshold)
        else:
            x_aug = torch.clamp(x_aug, -self.threshold, self.threshold)
        return x_aug

class NoiseInjectionTransform:
    """Noise Injection"""
    def __init__(self, noise_std=2.0, channels=[0, 1, 2], p=1.0):
        self.noise_std = noise_std
        self.channels = channels
        self.p = p

    def __call__(self, x):
        if torch.rand(1) > self.p: return x
        x_aug = x.clone()
        for c in self.channels:
            if c < x_aug.shape[1]:
                noise = torch.randn(x_aug.shape[0], device=x_aug.device) * self.noise_std
                x_aug[:, c] += noise
        return x_aug
