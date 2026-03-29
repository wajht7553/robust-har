##################################################
# MobileHART PyTorch Implementation
# Translated from official TensorFlow MobileHART
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    """Stochastic Depth DropPath natively for PyTorch"""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Work with different tensor dimensions (masking the batch dimension)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class ConvBNActivation(nn.Module):
    """Standard sequential 1D Convolution + BatchNorm + SWISH(SiLU) Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same", activation=True):
        super().__init__()
        if padding != "same":
            pad = padding
        else:
            pad = kernel_size // 2 if stride == 1 else (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, pad, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class PermuteLayerNorm(nn.Module):
    """Layer Normalization across the explicit channel dimension for 1D Temporal data"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
    def forward(self, x):
        # x: (B, C, L)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

class MLP2(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, expanded_channels, out_channels, stride=1):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        self.conv = nn.Sequential(
            # pw
            nn.Conv1d(in_channels, expanded_channels, 1, 1, 0, bias=False),
            nn.BatchNorm1d(expanded_channels),
            nn.SiLU(),
            # dw
            nn.Conv1d(expanded_channels, expanded_channels, 3, stride, 1 if stride == 1 else 1, groups=expanded_channels, bias=False) if stride == 1 else \
            nn.Conv1d(expanded_channels, expanded_channels, 3, stride, padding=1, groups=expanded_channels, bias=False),  
            nn.BatchNorm1d(expanded_channels),
            nn.SiLU(),
            # pw-linear
            nn.Conv1d(expanded_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm1d(out_channels)
        )
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MV2Block(nn.Module):
    def __init__(self, in_channels, expansion_factor, filterCounts):
        super().__init__()
        self.b1 = InvertedResidual(in_channels, filterCounts[0] * expansion_factor, filterCounts[1], stride=1)
        self.b2 = InvertedResidual(filterCounts[1], filterCounts[1] * expansion_factor, filterCounts[2], stride=2)
        self.b3 = InvertedResidual(filterCounts[2], filterCounts[2] * expansion_factor, filterCounts[2], stride=1)
        self.b4 = InvertedResidual(filterCounts[2], filterCounts[2] * expansion_factor, filterCounts[2], stride=1)
        self.b5 = InvertedResidual(filterCounts[2], filterCounts[2] * expansion_factor, filterCounts[3], stride=2)
        
    def forward(self, x):
        return self.b5(self.b4(self.b3(self.b2(self.b1(x)))))

class liteFormer(nn.Module):
    def __init__(self, start_index, stop_index, projection_size, kernel_size=16, attention_head=3, drop_path_rate=0.0):
        super().__init__()
        self.start_index = start_index
        self.stop_index = stop_index
        self.projection_size = projection_size
        self.attention_head = attention_head
        
        # Depthwise weights: (num_heads, 1, kernel_size)
        self.weight = nn.Parameter(torch.Tensor(attention_head, 1, kernel_size))
        nn.init.xavier_uniform_(self.weight)
        
        self.drop_path = DropPath(drop_path_rate)
        
    def forward(self, x):
        # x: (B, C, L)
        sub_x = x[:, self.start_index:self.stop_index, :] # (B, projection_size, L)
        B, C_sub, L = sub_x.shape
        reshaped = sub_x.reshape(-1, self.attention_head, L)
        
        # apply softmax to kernel weights
        norm_weight = F.softmax(self.weight, dim=2)
        
        conv_out = F.conv1d(reshaped, norm_weight, stride=1, padding="same", groups=self.attention_head)
        
        conv_out = self.drop_path(conv_out)
        out = conv_out.reshape(B, self.projection_size, L)
        return out

class SensorWiseMHA(nn.Module):
    def __init__(self, projection_quarter, num_heads, start_index, stop_index, dropout_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        self.start_index = start_index
        self.stop_index = stop_index
        self.mha = nn.MultiheadAttention(projection_quarter, num_heads, dropout=dropout_rate, batch_first=True)
        self.drop_path = DropPath(drop_path_rate)
        
    def forward(self, x):
        sub_x = x[:, self.start_index:self.stop_index, :]
        sub_x_t = sub_x.transpose(1, 2) # (B, L, C)
        
        attn_out, _ = self.mha(sub_x_t, sub_x_t, sub_x_t)
        attn_out = attn_out.transpose(1, 2) # (B, C, L)
        
        return self.drop_path(attn_out)

class SensorWiseTransformerBlock(nn.Module):
    def __init__(self, projection_dim, num_heads=2, kernel_size=4, dropout_rate=0.3, drop_path_rate=0.0):
        super().__init__()
        self.norm1 = PermuteLayerNorm(projection_dim)
        
        proj_q = projection_dim // 4
        proj_h = projection_dim // 2
        
        self.branch_acc_mha = SensorWiseMHA(proj_q, num_heads, 0, proj_q, dropout_rate, drop_path_rate)
        self.branch_lite = liteFormer(proj_q, proj_q + proj_h, proj_h, kernel_size, num_heads, drop_path_rate)
        self.branch_gyro_mha = SensorWiseMHA(proj_q, num_heads, proj_q + proj_h, projection_dim, dropout_rate, drop_path_rate)
        
        self.norm2 = PermuteLayerNorm(projection_dim)
        self.mlp = MLP2(projection_dim, projection_dim * 2, projection_dim, dropout_rate)
        self.drop_path = DropPath(drop_path_rate)
        
    def forward(self, x):
        x1 = self.norm1(x)
        b1 = self.branch_acc_mha(x1)
        b2 = self.branch_lite(x1)
        b3 = self.branch_gyro_mha(x1)
        
        attn = torch.cat([b1, b2, b3], dim=1) # Cat over channels
        x2 = attn + x
        x3 = self.norm2(x2)
        x3 = self.mlp(x3)
        x3 = self.drop_path(x3)
        return x3 + x2

class SensorWiseHART(nn.Module):
    def __init__(self, in_channels, projection_dim, num_blocks=2, kernel_size=4, dropout_rate=0.3):
        super().__init__()
        stride = 1
        
        # Acc branch
        self.local_acc1 = ConvBNActivation(in_channels, projection_dim//2, kernel_size=3, stride=stride)
        self.local_acc2 = ConvBNActivation(projection_dim//2, projection_dim//2, kernel_size=1, stride=1)
        
        # Gyro branch
        self.local_gyro1 = ConvBNActivation(in_channels, projection_dim//2, kernel_size=3, stride=stride)
        self.local_gyro2 = ConvBNActivation(projection_dim//2, projection_dim//2, kernel_size=1, stride=1)
        
        # Global Focus Modules
        drop_path_rates = torch.linspace(0, dropout_rate, num_blocks).tolist()
        self.transformers = nn.ModuleList([
            SensorWiseTransformerBlock(projection_dim, num_heads=2, kernel_size=kernel_size, 
                                       dropout_rate=dropout_rate, drop_path_rate=drop_path_rates[i])
            for i in range(num_blocks)
        ])
        
        self.fold_acc = ConvBNActivation(projection_dim//2, in_channels, kernel_size=1, stride=1)
        self.fuse_acc = ConvBNActivation(in_channels * 2, projection_dim//2, kernel_size=3, stride=stride)
        
        self.fold_gyro = ConvBNActivation(projection_dim//2, in_channels, kernel_size=1, stride=1)
        self.fuse_gyro = ConvBNActivation(in_channels * 2, projection_dim//2, kernel_size=3, stride=stride)

    def forward(self, x_acc, x_gyro):
        local_acc = self.local_acc2(self.local_acc1(x_acc))
        local_gyro = self.local_gyro2(self.local_gyro1(x_gyro))
        
        global_features = torch.cat([local_acc, local_gyro], dim=1) 
        for t in self.transformers:
            global_features = t(global_features)
            
        global_acc = global_features[:, :local_acc.shape[1], :]
        global_gyro = global_features[:, local_acc.shape[1]:, :]
        
        folded_acc = self.fold_acc(global_acc)
        cat_acc = torch.cat([x_acc, folded_acc], dim=1)
        fuse_acc = self.fuse_acc(cat_acc)
        
        folded_gyro = self.fold_gyro(global_gyro)
        cat_gyro = torch.cat([x_gyro, folded_gyro], dim=1)
        fuse_gyro = self.fuse_gyro(cat_gyro)
        
        return fuse_acc, fuse_gyro

class TransformerBlock(nn.Module):
    def __init__(self, projection_dim, num_heads=2, dropout_rate=0.3):
        super().__init__()
        self.norm1 = PermuteLayerNorm(projection_dim)
        self.mha = nn.MultiheadAttention(projection_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm2 = PermuteLayerNorm(projection_dim)
        self.mlp = MLP2(projection_dim, projection_dim * 2, projection_dim, dropout_rate)
        
    def forward(self, x):
        x1 = self.norm1(x)
        attn, _ = self.mha(x1.transpose(1, 2), x1.transpose(1, 2), x1.transpose(1, 2))
        x2 = attn.transpose(1, 2) + x
        x3 = self.mlp(self.norm2(x2))
        return x3 + x2

class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, projection_dim, num_blocks=2, stride=1, dropout_rate=0.3):
        super().__init__()
        self.local_conv1 = ConvBNActivation(in_channels, projection_dim, kernel_size=3, stride=stride)
        self.local_conv2 = ConvBNActivation(projection_dim, projection_dim, kernel_size=1, stride=1)
        
        self.transformers = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads=2, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])
        
        self.fold = ConvBNActivation(projection_dim, in_channels, kernel_size=1, stride=1)
        self.fuse = ConvBNActivation(in_channels * 2, projection_dim, kernel_size=3, stride=stride)
        
    def forward(self, x):
        local_features = self.local_conv2(self.local_conv1(x))
        global_features = local_features
        for t in self.transformers:
            global_features = t(global_features)
            
        folded = self.fold(global_features)
        
        # ensure folded has same resolution as x for concatenating
        if folded.shape[2] != x.shape[2]:
            folded = F.interpolate(folded, size=x.shape[2], mode='linear', align_corners=False)
            
        cat = torch.cat([x, folded], dim=1)
        out = self.fuse(cat)
        return out

class MLP_Head(nn.Module):
    def __init__(self, in_features, hidden_units, dropout_rate):
        super().__init__()
        layers = []
        last_dim = in_features
        for units in hidden_units:
            layers.extend([
                nn.Linear(last_dim, units),
                nn.SiLU(),
                nn.Dropout(dropout_rate)
            ])
            last_dim = units
        self.net = nn.Sequential(*layers)
        self.out_features = last_dim

    def forward(self, x):
        return self.net(x)

class MobileHART(nn.Module):
    """
    MobileHART translated directly from TensorFlow.
    Supports Dual-Variant Initialization : XS and XXS
    """
    def __init__(self, config):
        super().__init__()
        self.nb_classes = config.get("nb_classes", config.get("num_classes", 8))
        self.variant = config.get("variant", "XS").upper()
        self.dropout_rate = config.get("dropout", 0.3)
        self.mlp_head_units = config.get("mlp_head_units", [1024])
        self.use_aux_head = config.get("use_aux_head", False)
        
        if self.variant == "XXS":
            projectionDims = [64,80,96]
            filterCount = [8,8,12,24,64,80,320]
            expansion_factor = 2
        else: # XS Configuration
            projectionDims = [96,120,144]
            filterCount = [8,16,24,32,80,96,384]
            expansion_factor = 4
            
        self.acc_stem = ConvBNActivation(3, filterCount[0], kernel_size=3, stride=2)
        self.gyro_stem = ConvBNActivation(3, filterCount[0], kernel_size=3, stride=2)
        
        self.acc_mv2 = MV2Block(filterCount[0], expansion_factor, filterCount)
        self.gyro_mv2 = MV2Block(filterCount[0], expansion_factor, filterCount)
        
        self.sensor_wise = SensorWiseHART(filterCount[3], projectionDims[0], num_blocks=2, kernel_size=4, dropout_rate=self.dropout_rate)
        
        self.fc_swish = nn.Sequential(
            nn.Conv1d(projectionDims[0], projectionDims[0], 1),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        self.stage2_mv2 = InvertedResidual(projectionDims[0], projectionDims[0] * expansion_factor, filterCount[4], stride=2)
        self.stage2_mv = MobileViTBlock(filterCount[4], projectionDims[1], num_blocks=4, dropout_rate=self.dropout_rate)
        
        if self.use_aux_head:
            self.aux_pool = nn.AdaptiveAvgPool1d(1)
            self.aux_fc = nn.Linear(projectionDims[1], self.nb_classes)
            
        self.stage3_mv2 = InvertedResidual(projectionDims[1], projectionDims[1] * expansion_factor, filterCount[5], stride=2)
        self.stage3_mv = MobileViTBlock(filterCount[5], projectionDims[2], num_blocks=3, dropout_rate=self.dropout_rate)
        
        self.conv_out = ConvBNActivation(projectionDims[2], filterCount[6], kernel_size=1, stride=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp_head = MLP_Head(filterCount[6], self.mlp_head_units, self.dropout_rate)
        self.fc = nn.Linear(self.mlp_head.out_features, self.nb_classes)

    def forward(self, x):
        # Data is naturally (B, L, C) from pipeline but we strictly expect (B, C, L) 
        if x.shape[2] == 6 and x.shape[1] > 6:
            x = x.transpose(1, 2)
            
        x_acc = x[:, :3, :]
        x_gyro = x[:, 3:6, :]
        
        # Stems
        accX = self.acc_stem(x_acc)
        gyroX = self.gyro_stem(x_gyro)
        
        # MV2 Streams
        accX = self.acc_mv2(accX)
        gyroX = self.gyro_mv2(gyroX)
        
        # Cross Attention
        accX, gyroX = self.sensor_wise(accX, gyroX)
        
        x_joined = torch.cat([accX, gyroX], dim=1)
        x_joined = self.fc_swish(x_joined)
        
        # Stage 2
        x_joined = self.stage2_mv2(x_joined)
        x_joined = self.stage2_mv(x_joined)
        
        aux_out = None
        if self.training and self.use_aux_head:
            aux_out = self.aux_fc(self.aux_pool(x_joined).squeeze(-1))
            
        # Stage 3
        x_joined = self.stage3_mv2(x_joined)
        x_joined = self.stage3_mv(x_joined)
        
        x_joined = self.conv_out(x_joined)
        
        # Classification
        x_pool = self.pool(x_joined).squeeze(-1)
        x_features = self.mlp_head(x_pool)
        out = self.fc(x_features)
        
        if self.training and self.use_aux_head:
            return out, aux_out
        elif self.training:
            return out, None
        return out

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
