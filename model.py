# model.py
import torch
import torch.nn as nn
import torchvision.models as models
from config import LATENT_DIM, JOINTS_NUM

class CausalConv1d(nn.Module):
    """
    1D Convolution that only looks at the past.
    Implements causality by padding only on the left.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.pad, dilation=dilation)

    def forward(self, x):
        # x shape: (Batch, Dim, Time)
        out = self.conv(x)
        return out[:, :, :-self.pad] # Slice off the "future" padding on the right

class ResidualBlock(nn.Module):
    """
    Paper Architecture: GroupNorm -> ReLU -> 1D Conv -> GroupNorm -> ReLU -> 1D Conv
    """
    def __init__(self, channels):
        super().__init__()
        self.gn1 = nn.GroupNorm(32, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = CausalConv1d(channels, channels, kernel_size=3)
        self.gn2 = nn.GroupNorm(32, channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=3)

    def forward(self, x):
        residual = x
        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + residual

class CausalTemporalEncoder(nn.Module):
    """
    f_movie: Encodes video features into 'Movie Strips' (Latent Space)
    """
    def __init__(self):
        super().__init__()
        # Paper uses 3 residual blocks
        self.blocks = nn.Sequential(
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM)
        )

    def forward(self, x):
        # x: (Batch, Time, Features) -> (Batch, Features, Time)
        x = x.permute(0, 2, 1)
        out = self.blocks(x)
        return out.permute(0, 2, 1) # Back to (Batch, Time, Features)

class Autoregressor(nn.Module):
    """
    f_AR: Predicts the NEXT movie strip given past movie strips.
    """
    def __init__(self):
        super().__init__()
        # Same architecture as Encoder
        self.blocks = nn.Sequential(
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM),
            ResidualBlock(LATENT_DIM)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.blocks(x)
        return out.permute(0, 2, 1)

class JointRegressor(nn.Module):
    """
    f_3D: Reads latent movie strip and outputs 3D Joints.
    MODIFIED: Outputs (J * 3) instead of SMPL params.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, JOINTS_NUM * 3) # Output: x,y,z for each joint
        )

    def forward(self, x):
        # x: (Batch, Time, 2048)
        batch, time, dim = x.shape
        out = self.fc(x)
        return out.view(batch, time, JOINTS_NUM, 3)

class PHDModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Image Feature Extractor (Frozen ResNet)
        resnet = models.resnet50(pretrained=True)
        # Remove classification layer (fc), keep up to avgpool
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False # Freeze weights

        # 2. Components
        self.f_movie = CausalTemporalEncoder()
        self.f_AR = Autoregressor()
        self.f_3D = JointRegressor()

    def extract_features(self, video_tensor):
        """
        video_tensor: (Batch, Time, C, H, W)
        Returns: (Batch, Time, 2048)
        """
        b, t, c, h, w = video_tensor.shape
        # Flatten time into batch for efficient ResNet processing
        images = video_tensor.view(b * t, c, h, w)
        with torch.no_grad():
            features = self.feature_extractor(images)
        features = features.view(b, t, LATENT_DIM)
        return features