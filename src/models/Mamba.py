import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba = None


class MambaHAR(nn.Module):
    def __init__(self, config):
        super(MambaHAR, self).__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is not installed. Please install it to use MambaHAR."
            )

        self.input_dim = config["input_dim"]  # 113 (features)
        self.sequence_length = config["sequence_length"]  # 128
        self.num_classes = config["nb_classes"]  # 18

        # Mamba configuration
        d_model = config.get("d_model", 128)  # Hidden dimension
        n_layers = config.get("n_layers", 4)  # Number of Mamba layers
        d_state = config.get("d_state", 16)  # SSM state dimension
        d_conv = config.get("d_conv", 4)  # Convolution kernel size
        expand = config.get("expand", 2)  # Expansion factor

        # Input projection: map sensor features to d_model
        self.input_proj = nn.Linear(self.input_dim, d_model)

        # Stack of Mamba blocks
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
        )

        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_layers)]
        )

        # Global pooling and classification head
        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        self.fc = nn.Linear(d_model, self.num_classes)

        # Auxiliary classifier (after middle layer)
        self.use_aux_head = config.get("use_aux_head", False)
        self.aux_idx = n_layers // 2
        if self.use_aux_head:
            self.aux_fc = nn.Linear(d_model, self.num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        # Project input features to d_model
        x = self.input_proj(x)  # (B, L, d_model)

        # Pass through Mamba layers with residual connections
        aux_out = None
        for i, (mamba_layer, layer_norm) in enumerate(
            zip(self.mamba_layers, self.layer_norms)
        ):
            residual = x
            x = layer_norm(x)
            x = mamba_layer(x) + residual  # Residual connection

            if self.training and self.use_aux_head and i == self.aux_idx:
                # Global pooling for aux head
                x_aux = x.mean(dim=1)
                aux_out = self.aux_fc(x_aux)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (B, d_model)

        # Classification head
        x = self.dropout(x)
        logits = self.fc(x)  # (B, num_classes)

        if self.training:
            return logits, aux_out
        return logits
