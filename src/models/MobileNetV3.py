import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv1d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm1d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, config):
        super(MobileNetV3, self).__init__()
        self.nb_channels = config["nb_channels"]
        self.nb_classes = config["nb_classes"]
        self.dropout_prob = config.get("drop_prob", 0.2)
        
        # MobileNetV3-Small configuration adapted for 1D
        cfgs = [
            # k, t, c, SE, HS, s 
            [3,    1,  16, 1, 0, 2],
            [3,  4.5,  24, 0, 0, 2],
            [3, 3.67,  24, 0, 0, 1],
            [5,    4,  40, 1, 1, 2],
            [5,    6,  40, 1, 1, 1],
            [5,    6,  40, 1, 1, 1],
            [5,    3,  48, 1, 1, 1],
            [5,    3,  48, 1, 1, 1],
            [5,    6,  96, 1, 1, 2],
            [5,    6,  96, 1, 1, 1],
            [5,    6,  96, 1, 1, 1],
        ]

        # building first layer
        input_channel = 16
        self.features = [nn.Sequential(
            nn.Conv1d(self.nb_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm1d(input_channel),
            h_swish()
        )]

        # building inverted residual blocks
        for k, t, c, use_se, use_hs, s in cfgs:
            output_channel = c
            hidden_channel = int(input_channel * t)
            self.features.append(InvertedResidual(input_channel, hidden_channel, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        
        self.features = nn.Sequential(*self.features)
        
        # Auxiliary classifier (attached somewhere in the middle, e.g., after 5th block)
        self.aux_idx = 5
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(40, self.nb_classes) # 40 is the channel count at block 5
        )

        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, 576, 1, 1, 0, bias=False),
            nn.BatchNorm1d(576),
            h_swish()
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            h_swish(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(1024, self.nb_classes),
        )

    def forward(self, x):
        # x shape: (batch, window_size, channels) -> (batch, channels, window_size)
        x = x.permute(0, 2, 1)
        
        aux_out = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.aux_idx and self.training:
                aux_out = self.aux_classifier(x)
        
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        
        if self.training:
            return out, aux_out
        return out
