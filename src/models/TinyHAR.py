import torch
import torch.nn as nn


class TinyHAR(nn.Module):
    def __init__(self, config):
        super(TinyHAR, self).__init__()
        self.window_size = config["window_size"]
        self.nb_channels = config["nb_channels"]
        self.nb_classes = config["nb_classes"]
        self.filter_width = config.get("filter_width", 5)
        self.nb_filters = config.get("nb_filters", 32)
        self.dropout_prob = config.get("dropout", 0.5)

        # Temporal Convolutional Layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.nb_channels, self.nb_filters, kernel_size=self.filter_width, padding="same"),
            nn.BatchNorm1d(self.nb_filters),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.nb_filters, self.nb_filters * 2, kernel_size=self.filter_width, padding="same"),
            nn.BatchNorm1d(self.nb_filters * 2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.nb_filters * 2, self.nb_filters * 4, kernel_size=self.filter_width, padding="same"),
            nn.BatchNorm1d(self.nb_filters * 4),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(self.nb_filters * 4, self.nb_filters * 8, kernel_size=self.filter_width, padding="same"),
            nn.BatchNorm1d(self.nb_filters * 8),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.nb_filters * 8, self.nb_filters * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.nb_filters * 4, self.nb_classes),
        )
        
        # Auxiliary Classifier (attached after conv2)
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.nb_filters * 2, self.nb_classes)
        )

    def forward(self, x):
        # x shape: (batch, window_size, channels) -> (batch, channels, window_size)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.conv2(x)
        
        # Auxiliary output
        aux_out = self.aux_classifier(x)
        
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        out = self.classifier(x)

        if self.training:
            return out, aux_out
        return out
