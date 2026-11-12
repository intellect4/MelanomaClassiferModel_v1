"""Deep Learning Model Architecture: MultiScaleAttentionMelanomaNet.

Combines a DenseNet-121 backbone with Convolutional Block Attention Module (CBAM)
and Dual Average/Max Pooling for robust Histopathological Melanoma Classification.
"""

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM) from CBAM.
    
    Explores inter-channel relationships using parallel average and max pooling.
    """
    def __init__(self, in_planes=1024, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM) from CBAM.
    
    Captures cellular loci across spatial locations using 7x7 convolution.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = float(drop_rate)

    def forward(self, init_features):
        if isinstance(init_features, torch.Tensor):
            prev_features = [init_features]
        else:
            prev_features = init_features
        concated_features = torch.cat(prev_features, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0.0):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet121Features(nn.Module):
    """DenseNet-121 Feature Extractor matching Torchvision's exact module layout.
    
    Target layer for Grad-CAM: denseblock4.denselayer16.conv2
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0.0):
        super().__init__()
        # First convolution
        self.conv0 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # DenseBlocks & Transitions
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            setattr(self, f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                setattr(self, f"transition{i + 1}", trans)
                num_features = num_features // 2

        # Final batch norm
        self.norm5 = nn.BatchNorm2d(num_features)

    def forward(self, x):
        out = self.pool0(self.relu0(self.norm0(self.conv0(x))))
        out = self.denseblock1(out)
        out = self.transition1(out)
        out = self.denseblock2(out)
        out = self.transition2(out)
        out = self.denseblock3(out)
        out = self.transition3(out)
        out = self.denseblock4(out)
        out = self.norm5(out)
        return out


class MultiScaleAttentionMelanomaNet(nn.Module):
    """Integrated Multiscale Attention Melanoma Network.
    
    Pairs DenseNet121 bottleneck features (1024-dim) with CBAM attention,
    dual avg/max pooling (2048-dim), and a multi-stage classification head.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = DenseNet121Features()
        self.channel_attn = ChannelAttention(in_planes=1024, ratio=16)
        self.spatial_attn = SpatialAttention(kernel_size=7)

        # Dual-branch feature pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Multi-layer classifier head matching majorprj.py exactly: 2048 -> 512 -> 128 -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.channel_attn(feat) * feat
        feat = self.spatial_attn(feat) * feat

        # Concatenate Average and Max pooling for rich morphological representation (1024 + 1024 = 2048)
        avg_feat = torch.flatten(self.avg_pool(feat), 1)
        max_feat = torch.flatten(self.max_pool(feat), 1)
        combined = torch.cat([avg_feat, max_feat], dim=1)

        return self.classifier(combined)

