""" Model architecture for Score Matching. A CNN with 6 Layers is used and Conditional Instance Norm within each layer. """
import torch
from torch import nn

class ConditionalInstanceNorm(nn.Module):
    def __init__(self, num_features, num_sigmas):
        super().__init__()
        """
        Conditional Instance Normalization layer.

        This layer applies instance normalization to each feature map in the input,
        and then rescales and shifts the normalized output using learnable affine parameters
        that depend on the conditioning variable (sigma level).

        Args:
            num_features (int): Number of feature channels in the input.
            num_sigmas (int): Number of discrete noise levels (used for conditioning).
        """
        # Learnable scale (gamma) and shift (beta) parameters for each sigma level.
        # Shape: (num_sigmas, num_features)
        self.gamma = nn.Parameter(torch.ones(num_sigmas, num_features))
        self.beta = nn.Parameter(torch.zeros(num_sigmas, num_features))
        # avoid division by zero
        self.eps = 1e-5

    def forward(self, x, sigma_idx):
        """
        Forward pass of the Conditional Instance Normalization.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            sigma_idx (LongTensor): Tensor of shape (B,) containing the sigma indices
                                    for each sample in the batch.

        Returns:
            Tensor: Conditionally normalized output of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        # Compute per-sample, per-channel mean and standard deviation (InstanceNorm)
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + self.eps
        # Normalize input
        x_norm = (x - mean) / std
        # Gather corresponding gamma and beta for each sample using sigma index
        gamma = self.gamma[sigma_idx].view(B, C, 1, 1)
        beta = self.beta[sigma_idx].view(B, C, 1, 1)
        return gamma * x_norm + beta


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_sigmas, dilation=1, stride=1):
        super().__init__()
        """
        Convolutional Block consisting of:
        - a convolutional layer
        - conditional instance normalization
        - a ReLU activation
        
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            num_sigmas (int): Number of discrete noise levels (used for conditioning).
            dilation (int): Dilation rate for the convolution (default: 1).
            stride (int): Stride for the convolution (default: 1).
        """
        # Convolutional layer with padding=dilation to preserve spatial resolution
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, stride=stride)
        # Conditional instance normalization layer, conditioned on noise level (sigma)
        self.norm = ConditionalInstanceNorm(out_ch, num_sigmas)
        # ReLU activation
        self.act = nn.ReLU()

    def forward(self, x, sigma_idx):
        # Apply convolution
        x = self.conv(x)
        # Apply conditional instance normalization (conditioned on sigma index)
        x = self.norm(x, sigma_idx)
        return self.act(x)


class ScoreNet(nn.Module):
    def __init__(self, dim, num_sigmas):
        super().__init__()
        """
        ScoreNet model for learning score functions s_theta(x, sigma).

        Args:
            dim (int): Output dimension, typically the same as the input dimension (e.g., 784 for MNIST).
            num_sigmas (int): Number of discrete noise levels used for conditioning (e.g., 10).
        """
        self.dim = dim

        # Convolutional blocks with increasing dilation to capture multi-scale features.
        # All ConvBlocks are conditioned on the noise level (sigma).
        self.conv1 = ConvBlock(1, 16, num_sigmas, dilation=1)
        self.conv2 = ConvBlock(16, 16, num_sigmas, dilation=2)
        self.conv3 = ConvBlock(16, 32, num_sigmas, dilation=4)
        self.conv4 = ConvBlock(32, 64, num_sigmas, dilation=4)
        self.conv5 = ConvBlock(64, 64, num_sigmas, dilation=8)
        self.conv6 = ConvBlock(64, 64, num_sigmas, dilation=8)

        # Flatten and map to desired output dimension
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 28 * 28, dim)

    def forward(self, x, sigma_idx):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (B, 784) or (B, 1, 28, 28)
            sigma_idx (LongTensor): Tensor of shape (B,) with sigma indices for each sample

        Returns:
            Tensor: Output tensor of shape (B, dim), representing the predicted score ∇_x log p_sigma(x)
        """
        # If input is flattened (e.g., from MNIST), reshape to image format
        if x.ndim == 2 and x.shape[1] == 784:
            x = x.view(-1, 1, 28, 28)

        # Pass through conditioned convolutional blocks
        x = self.conv1(x, sigma_idx)
        x = self.conv2(x, sigma_idx)
        x = self.conv3(x, sigma_idx)
        x = self.conv4(x, sigma_idx)
        x = self.conv5(x, sigma_idx)
        x = self.conv6(x, sigma_idx)

        # Flatten and map to output dimension
        x = self.flatten(x)
        x = self.linear(x)
        return x
