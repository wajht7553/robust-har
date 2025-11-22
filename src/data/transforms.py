import torch


class SensorFailureTransform:
    """
    Transform to simulate sensor failures and noise.
    Assumes input shape (window_size, channels).
    Channels 0-2: Accel (x,y,z)
    Channels 3-5: Gyro (x,y,z)
    """

    def __init__(
        self, p_dropout_gyro=0.0, p_noise=0.0, noise_std=0.1, p_channel_drop=0.0
    ):
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
