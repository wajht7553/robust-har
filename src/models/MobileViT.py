##################################################
# MobileViT implementation for time-series HAR
# Adapted for 1D temporal data from MobileViT architecture
##################################################
# Based on: https://arxiv.org/abs/2110.02178
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNActivation(nn.Module):
    """Standard convolution block with BatchNorm and activation"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        activation=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual block for 1D"""

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # pointwise expansion
            layers.append(
                ConvBNActivation(in_channels, hidden_dim, kernel_size=1, padding=0)
            )

        # depthwise
        layers.extend(
            [
                nn.Conv1d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                # pointwise projection
                nn.Conv1d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm1d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder block"""

    def __init__(self, embed_dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, L, C)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """MobileViT block: Local representation (Conv) + Global representation (Transformer)"""

    def __init__(
        self,
        in_channels,
        out_channels,
        patch_size=2,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Local representation
        self.local_rep = nn.Sequential(
            ConvBNActivation(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            ),
            ConvBNActivation(in_channels, out_channels, kernel_size=1, padding=0),
        )

        # Global representation with transformer
        self.transformers = nn.ModuleList(
            [
                TransformerEncoder(out_channels, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        # Fusion
        self.fusion = nn.Sequential(
            ConvBNActivation(out_channels, in_channels, kernel_size=1, padding=0),
            ConvBNActivation(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            ),
        )

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape

        # Local representations
        local_x = self.local_rep(x)  # (B, out_channels, L)

        # Unfold patches for transformer
        # Create non-overlapping patches
        P = self.patch_size
        num_patches = L // P

        # Reshape to patches
        patches = local_x[:, :, : num_patches * P].reshape(
            B, -1, num_patches, P
        )  # (B, C, num_patches, P)
        patches = patches.permute(0, 2, 3, 1).reshape(
            B, num_patches * P, -1
        )  # (B, num_patches*P, C)

        # Apply transformers
        for transformer in self.transformers:
            patches = transformer(patches)

        # Reshape back
        patches = patches.reshape(B, num_patches, P, -1).permute(
            0, 3, 1, 2
        )  # (B, C, num_patches, P)
        global_x = patches.reshape(B, -1, num_patches * P)  # (B, C, L')

        # Pad if needed
        if global_x.shape[2] < L:
            pad_size = L - global_x.shape[2]
            global_x = F.pad(global_x, (0, pad_size))

        # Fusion
        out = self.fusion(global_x)

        return out


class MobileViT(nn.Module):
    """
    MobileViT for time-series Human Activity Recognition

    Args:
        config: dictionary containing:
            - window_size: length of input window
            - nb_channels: number of input channels (e.g., 6 for acc+gyr)
            - nb_classes: number of output classes
            - dims: list of channel dimensions for each stage
            - num_transformer_layers: list of transformer layers per MobileViT block
            - patch_size: patch size for MobileViT blocks
            - num_heads: number of attention heads
            - dropout: dropout probability
    """

    def __init__(self, config):
        super().__init__()
        self.window_size = config["window_size"]
        self.nb_channels = config["nb_channels"]
        self.nb_classes = config["nb_classes"]

        # Default architecture parameters
        dims = config.get("dims", [32, 64, 96, 128])
        num_transformer_layers = config.get("num_transformer_layers", [2, 4, 3])
        patch_size = config.get("patch_size", 2)
        num_heads = config.get("num_heads", 4)
        dropout = config.get("dropout", 0.1)

        # Stem
        self.stem = nn.Sequential(
            ConvBNActivation(
                self.nb_channels, dims[0], kernel_size=3, stride=2, padding=1
            ),
            InvertedResidual(dims[0], dims[0]),
        )

        # Stage 1
        self.stage1 = nn.Sequential(
            InvertedResidual(dims[0], dims[1], stride=2),
            InvertedResidual(dims[1], dims[1]),
        )

        # Stage 2 with MobileViT
        self.stage2 = nn.Sequential(
            InvertedResidual(dims[1], dims[2], stride=2),
            MobileViTBlock(
                dims[2],
                dims[2],
                patch_size=patch_size,
                num_layers=num_transformer_layers[0],
                num_heads=num_heads,
                dropout=dropout,
            ),
        )

        # Stage 3 with MobileViT
        self.stage3 = nn.Sequential(
            InvertedResidual(dims[2], dims[3], stride=2),
            MobileViTBlock(
                dims[3],
                dims[3],
                patch_size=patch_size,
                num_layers=num_transformer_layers[1],
                num_heads=num_heads,
                dropout=dropout,
            ),
        )

        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dims[3], self.nb_classes)

    def forward(self, x):
        # x: (B, L, C) -> need (B, C, L) for Conv1d
        x = x.permute(0, 2, 1)  # (B, C, L)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.pool(x).squeeze(-1)  # (B, C)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
