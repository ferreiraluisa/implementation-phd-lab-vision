import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
"""
Attempted implementation of adapted PHD model for 3D joint prediction(in PyTorch).

Implemented by Luisa Ferreira (2025)
"""


# ============================================================
# RESIDUAL BLOCK
# ============================================================
# from the original paper, we have that both the causal temporal encoder fmovie and the autoregressive predictor fAR have the same architecture, consisting of 3 residual blocks. Each block consists of GroupNorm, ReLU, 1D Convolution, GroupNorm, ReLU, 1D Convolution, and each 1D Convolution uses a kernel size of 3 and a filter size of 2048.

# implementing causal convolution because nn.Conv1d does not support causality natively, output sees both past and future frames.
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,):
        super().__init__()
        self.left_pad = (kernel_size - 1) # to ensure causality, depend only on previous frames
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0
        )

    def forward(self, x):
        # x: (B, C, T)
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0), mode="replicate") # pad only on the left, replicating first values
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, groups=32, p_drop=0.2):
        super().__init__()
        self.gn1 = nn.GroupNorm(groups, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = CausalConv1d(channels, channels, kernel_size=3)
        self.drop1 = nn.Dropout1d(p_drop)

        self.gn2 = nn.GroupNorm(groups, channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=3)
        self.drop2 = nn.Dropout1d(p_drop)

    def forward(self, x):
        residual = x

        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.drop1(x)

        x = self.gn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.drop2(x)

        return x + residual



# ============================================================
# CAUSAL TEMPORAL ENCODER(fmovie) and AUTOREGRESSIVE PREDICTOR(fAR)
# ============================================================
# causal temporal encoder => encode 3D Human Dynamics, generating latent movie strips that captures temporal context of 3D motion until time t.
# autoregressive predictor => predict the next latent movie strip given the previous ones.
# from paper: "Predicting 3D Human Dynamics from Video":
# 3 residual blocks, each with kernel size 3, resulting in a receptive field of 13 frames.
class CausalTemporalNet(nn.Module):
    def __init__(self, latent_dim=2048, num_blocks=3):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(latent_dim) for _ in range(num_blocks)])

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B,D,T)
        x = self.blocks(x)
        return x.permute(0, 2, 1)  # (B,T,D)

# ============================================================
# 3D REGRESSOR (adapted from HMR's)
# ============================================================
# from the original HMR paper, the iterative 3D regressor consists of two fully-connected layers with 1024 neurons each with a dropout layer in between, followed by a final layer of 85D neurons.
#
# I've adapted it to output 17 joints * 3D = 51D. (instead of SMPL parameters Θ = {θ,β,R,t,s})
class JointRegressor(nn.Module):
    def __init__(self, latent_dim=2048, joints_num=17, iters=3, dropout=0.5, camera_params=False):
        super().__init__()
        self.joints_num = joints_num
        self.cam = 3 if camera_params else 0
        self.out_dim = joints_num * 3 + self.cam # 51D output + camera params (s, tx, ty) if needed
        self.iters = iters

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + self.out_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.out_dim),
        )

        # initial pose (all zeros)
        self.register_buffer("y0", torch.zeros(self.out_dim))

    def forward(self, phi):
        # phi: (B, T, 2048)
        B, T, D = phi.shape
        # y: (B, T, 51)
        y = self.y0.view(1, 1, -1).expand(B, T, -1).contiguous()

        # iterative error feedback, T = 3 iterations from the original HMR
        for _ in range(self.iters):
            inp = torch.cat([phi, y], dim=-1)
            dy = self.mlp(inp)
            y = y + dy

        return y.view(B, T, self.joints_num, 3)


# ============================================================
# PHD MODEL
# ============================================================
# overall architecture(inference pipeline ):
# first, a ResNet-50 backbone is used to extract image features from each frame in the input video.
# second, these features are passed through the causal temporal encoder fmovie to obtain latent movie strips.
# third, the autoregressive predictor fAR predicts the next latent movie strip based on previous ones
# finally, the joint regressor f3D takes both the latent movie strips and the predicted ones to output the 3D joint positions.
class PHDFor3DJoints(nn.Module):
    def __init__(self, latent_dim=2048, proj_dim=1024, joints_num=17):
        super().__init__()

        # Normalize original ResNet features
        self.norm_in = nn.LayerNorm(latent_dim)

        # 2048 -> 1024 projection (applied per-frame)
        self.in_proj = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(proj_dim),
        )

        # Temporal nets now run at proj_dim
        self.f_movie = CausalTemporalNet(latent_dim=proj_dim)
        self.f_AR = CausalTemporalNet(latent_dim=proj_dim)

        # Regressor also expects proj_dim
        self.f_3D = JointRegressor(latent_dim=proj_dim, joints_num=joints_num)

    def forward(self, feats, predict_future=False):
        # feats: (B, T, 2048)

        feats = self.norm_in(feats)
        feats = self.in_proj(feats)  # (B, T, 1024)

        phi = self.f_movie(feats)    # (B, T, 1024)

        ar_out = self.f_AR(phi)      # (B, T, 1024)
        phi_hat = torch.zeros_like(ar_out)
        phi_hat[:, 1:, :] = ar_out[:, :-1, :]

        joints_phi = self.f_3D(phi)

        joints_hat = None
        if predict_future:
            joints_hat = self.f_3D(phi_hat)

        return phi, phi_hat, joints_phi, joints_hat