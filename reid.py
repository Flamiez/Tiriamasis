import torch
import kagglehub

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Download latest version
path = kagglehub.dataset_download("pengcw1/market-1501")
print("Path to dataset files:", path)

import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.nn import init


class ResNet50ReID(nn.Module):
    def __init__(self):
        super(ResNet50ReID, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)

        # Remove classification layers, keep feature extractor
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        # Feature embedding size
        self.num_features = resnet.fc.in_features  # 2048
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)

        # Initialize batch normalization
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

    def forward(self, x):
        """
        x: Input sequence of images (batch_size, sequence_length, C, H, W)
        Returns: Feature embeddings (batch_size, feature_dim)
        """
        b, t, c, h, w = x.shape  # batch size, time steps, channels, height, width
        x = x.view(b * t, c, h, w)  # Flatten sequence dimension
        x = self.base(x)  # Feature extraction
        x = F.avg_pool2d(x, x.size()[2:])  # Global average pooling

        # Reshape to sequence format
        x = x.view(b, t, -1)  # (batch, sequence_length, feature_dim)

        # Normalize embeddings
        bn_x = self.feat_bn(x.view(b * t, -1)).view(b, t, -1)
        bn_x_norm = F.normalize(bn_x, p=2, dim=-1)

        # Aggregate across sequence
        bn_x_norm = bn_x_norm.permute(0, 2, 1)  # (batch, feature_dim, sequence_length)
        f_seq = F.avg_pool1d(bn_x_norm, t).view(b, self.num_features)

        if not self.training:
            return f_seq  # Inference mode: return final sequence feature

        return f_seq


# Instantiate model
model = ResNet50ReID()

# Example input: batch of 8 sequences, each with 10 frames of size 224x224
x = torch.randn(8, 10, 3, 224, 224)
features = model(x)

print(features.shape)  # Expected output: (8, 2048)
