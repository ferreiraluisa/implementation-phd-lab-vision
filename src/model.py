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
    # group norm + relu + causal conv1d + group norm + relu + causal conv1d + skip connection
    # [REGULARISATION] added Dropout1d (spatial/channel dropout) after each conv.
    # standard Dropout zeroes individual scalars which is too fine-grained for conv feature maps,
    # since adjacent time-steps are correlated. Dropout1d drops entire channels at once,
    # making it a much stronger regulariser for 1D conv sequences.
    def __init__(self, channels, groups=32, drop_prob=0.1):
        super().__init__()
        self.gn1   = nn.GroupNorm(groups, channels)
        self.conv1 = CausalConv1d(channels, channels, kernel_size=3)
        self.drop1 = nn.Dropout1d(drop_prob)   # drops full feature-map channels

        self.gn2   = nn.GroupNorm(groups, channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=3)
        self.drop2 = nn.Dropout1d(drop_prob)

    def forward(self, x):
        residual = x
        x = F.relu(self.gn1(x), inplace=True)
        x = self.drop1(self.conv1(x))
        x = F.relu(self.gn2(x), inplace=True)
        x = self.drop2(self.conv2(x))
        return x + residual



# ============================================================
# CAUSAL TEMPORAL ENCODER(fmovie) and AUTOREGRESSIVE PREDICTOR(fAR)
# ============================================================
# causal temporal encoder => encode 3D Human Dynamics, generating latent movie strips that captures temporal context of 3D motion until time t.
# autoregressive predictor => predict the next latent movie strip given the previous ones.
# from paper: "Predicting 3D Human Dynamics from Video":
# 3 residual blocks, each with kernel size 3, resulting in a receptive field of 13 frames.
#
# [REGULARISATION] added stochastic depth: each residual block is randomly skipped during
# training with probability `stochastic_depth_prob`, forcing the network not to rely on any
# single block. Probability increases linearly across blocks (later blocks dropped more often).
#
# [REGULARISATION] added a separate input linear projection for each CausalTemporalNet instance
# (f_movie and f_AR) so the two networks learn independent feature subspaces rather than
# co-adapting to the same raw backbone features.
class CausalTemporalNet(nn.Module):
    def __init__(self, in_dim=2048, latent_dim=2048, num_blocks=3,
                 drop_prob=0.1, stochastic_depth_prob=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, latent_dim) if in_dim != latent_dim else nn.Identity()
        self.blocks = nn.ModuleList(
            [ResidualBlock(latent_dim, drop_prob=drop_prob) for _ in range(num_blocks)]
        )
        # per-block drop probability for stochastic depth, linearly increasing
        self.sd_probs = [stochastic_depth_prob * (i / max(num_blocks - 1, 1))
                         for i in range(num_blocks)]

    def _stochastic_depth(self, x, block, drop_prob):
        if not self.training or drop_prob == 0.0:
            return block(x)
        # Bernoulli gate: keep the whole block output or fall back to identity
        keep = torch.rand(1, device=x.device).item() > drop_prob
        return block(x) if keep else x

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 1)   # (B,D,T)
        for block, prob in zip(self.blocks, self.sd_probs):
            x = self._stochastic_depth(x, block, prob)
        return x.permute(0, 2, 1)  # (B,T,D)

# ============================================================
# 3D REGRESSOR (adapted from HMR's)
# ============================================================
# from the original HMR paper, the iterative 3D regressor consists of two fully-connected layers with 1024 neurons each with a dropout layer in between, followed by a final layer of 85D neurons.
#
# I've adapted it to output 17 joints * 3D = 51D. (instead of SMPL parameters Θ = {θ,β,R,t,s})
#
# [REGULARISATION] added LayerNorm after each hidden layer. The original MLP had no normalisation
# at all; LayerNorm smooths the loss surface and reduces internal covariate shift, making the
# dropout that follows act on more stable activations. This combo consistently outperforms
# dropout alone on regression tasks.
class JointRegressor(nn.Module):
    def __init__(self, latent_dim=2048, joints_num=17, iters=3, dropout=0.3, camera_params=False):
        super().__init__()
        self.joints_num = joints_num
        self.cam = 3 if camera_params else 0
        self.out_dim = joints_num * 3 + self.cam # 51D output + camera params (s, tx, ty) if needed
        self.iters = iters

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + self.out_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
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
#
# [REGULARISATION] summary of all changes made to reduce overfitting:
#   1. Dropout1d (channel dropout) in every residual conv block.
#   2. Stochastic depth in CausalTemporalNet.
#   3. LayerNorm + Dropout in the joint regressor.
#   4. Separate input projections for f_movie and f_AR.
#   5. Gaussian feature noise injection during training (cheap latent-space augmentation).
#   6. build_optimizer() below for correct weight-decay param groups.
class PHDFor3DJoints(nn.Module):
    def __init__(self, latent_dim=2048, joints_num=17, freeze_backbone=True,
                 feat_noise_std=0.01,   # std of Gaussian noise added to backbone feats during training
                 drop_prob=0.1,         # channel dropout inside residual blocks
                 sd_prob=0.1,           # stochastic depth probability per block
                 reg_dropout=0.3):      # dropout inside joint regressor
        super().__init__()

        # ----------------------------------------------------
        # PER-FRAME FEATURE EXTRACTOR
        # ----------------------------------------------------
        # pretrained ResNet-50 + average pooling of last layer as 2048D feature extractor
        # resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # commented out bc we don't want to extract features here, only in preprocess_resnet_features.py

        # if freeze_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad = False

        self.latent_dim     = latent_dim
        self.feat_noise_std = feat_noise_std

        # [REGULARISATION] each net gets its own input projection (in_dim arg) so f_movie and
        # f_AR learn independent feature subspaces rather than co-adapting to the same raw feats.
        self.f_movie = CausalTemporalNet(latent_dim, latent_dim,
                                         drop_prob=drop_prob,
                                         stochastic_depth_prob=sd_prob)
        self.f_AR    = CausalTemporalNet(latent_dim, latent_dim,
                                         drop_prob=drop_prob,
                                         stochastic_depth_prob=sd_prob)
        self.f_3D    = JointRegressor(latent_dim, joints_num, dropout=reg_dropout)

    # @torch.no_grad() # resnet must not be trained
    # def extract_features(self, video):
    #     # video: (B, T, C, H, W) batch_size, frames, channels, height, width
    #     B, T, C, H, W = video.shape
    #     x = video.view(B * T, C, H, W)
    #     feats = self.backbone(x)
    #     feats = feats.flatten(1)
    #     return feats.view(B, T, -1)

    def _add_feat_noise(self, feats):
        # [REGULARISATION] light Gaussian noise in latent space during training.
        # acts like continuous stochastic perturbation of backbone features,
        # making the temporal nets more robust to backbone feature variance.
        if self.training and self.feat_noise_std > 0:
            feats = feats + torch.randn_like(feats) * self.feat_noise_std
        return feats

    # def forward(self, video, predict_future=False):
    def forward(self, feats, predict_future=False):
        # feats = self.extract_features(video) # preprocessed features input in preprocess_resnet_features.py
        # feats: (B, T, 2048)
        feats = self._add_feat_noise(feats)

        phi = self.f_movie(feats) # temporal causal encoder f_movie

        ar_out = self.f_AR(phi) # autoregressive predictor f_AR
        phi_hat = torch.zeros_like(ar_out)
        phi_hat[:, 1:, :] = ar_out[:, :-1, :]

        joints_phi = self.f_3D(phi) # 3D regressor f_3D

        joints_hat = None
        if predict_future:
            joints_hat = self.f_3D(phi_hat)

        # phi : teacher movie strips (B,T,D) batch_size, time, latent_dim
        # phi_hat : predicted movie strips (B,T,D) batch_size, time, latent_dim
        # joints_phi : joints from phi
        # joints_hat : joints from phi_hat (optional)
        return phi, phi_hat, joints_phi, joints_hat