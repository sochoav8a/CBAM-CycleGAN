# src/models/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------
# Channel Attention Module (CAM) - Part of CBAM
# ------------------------------------
class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) for CBAM.
    Applies global average pooling and global max pooling, passes them through a shared MLP,
    adds the results, applies sigmoid, and scales the input feature map.
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        """
        Initialize ChannelAttention module.
        Args:
            gate_channels (int): Number of channels in the input feature map.
            reduction_ratio (int): Reduction ratio for the shared MLP hidden layer.
            pool_types (list): Types of pooling to use ('avg', 'max').
        """
        super(ChannelAttention, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            # Flatten layer might not be needed if using AdaptivePool2d output directly
            # nn.Flatten(), # Not strictly necessary if using adaptive pooling which outputs (B, C, 1, 1)
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels, kernel_size=1, bias=False)
        )
        self.pool_types = pool_types

        # Use AdaptiveAvgPool2d and AdaptiveMaxPool2d to handle any input size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        """
        Forward pass through the ChannelAttention module.
        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).
        Returns:
            torch.Tensor: Scaled feature map (B, C, H, W).
        """
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool_out = self.avg_pool(x) # Output shape (B, C, 1, 1)
                channel_att_raw = self.mlp(pool_out)
            elif pool_type == 'max':
                pool_out = self.max_pool(x) # Output shape (B, C, 1, 1)
                channel_att_raw = self.mlp(pool_out)
            else:
                # Handle cases where pool_type might be invalid, though constructor checks should prevent this
                continue # Or raise error

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # Apply sigmoid to get the attention map (scales between 0 and 1)
        # Shape: (B, C, 1, 1)
        scale = torch.sigmoid(channel_att_sum)

        # Multiply the input feature map by the computed scale
        # Broadcasting automatically applies the (B, C, 1, 1) scale to (B, C, H, W)
        return x * scale

# ------------------------------------
# Spatial Attention Module (SAM) - Part of CBAM
# ------------------------------------
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) for CBAM.
    Applies average pooling and max pooling along the channel axis, concatenates them,
    passes through a convolutional layer, applies sigmoid, and scales the input feature map.
    """
    def __init__(self, kernel_size=7):
        """
        Initialize SpatialAttention module.
        Args:
            kernel_size (int): Kernel size for the convolution layer. Must be odd.
        """
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd to maintain spatial dimensions with padding"
        # Padding = (kernel_size - 1) // 2
        padding = (kernel_size - 1) // 2

        # Input has 2 channels (from avg and max pooling along channel)
        # Output has 1 channel (the spatial attention map)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the SpatialAttention module.
        Args:
            x (torch.Tensor): Input feature map (B, C, H, W). Typically the output of CAM.
        Returns:
            torch.Tensor: Scaled feature map (B, C, H, W).
        """
        # Apply pooling along the channel dimension (dim=1)
        avg_out = torch.mean(x, dim=1, keepdim=True) # Shape: (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # Shape: (B, 1, H, W)

        # Concatenate the pooled features
        pooled = torch.cat([avg_out, max_out], dim=1) # Shape: (B, 2, H, W)

        # Apply convolution
        conv_out = self.conv(pooled) # Shape: (B, 1, H, W)

        # Apply sigmoid to get the spatial attention map
        scale = self.sigmoid(conv_out) # Shape: (B, 1, H, W)

        # Multiply the input feature map by the computed scale
        # Broadcasting automatically applies the (B, 1, H, W) scale to (B, C, H, W)
        return x * scale

# ------------------------------------
# CBAM Block
# ------------------------------------
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Combines Channel Attention Module (CAM) and Spatial Attention Module (SAM) sequentially.
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], spatial_kernel_size=7):
        """
        Initialize CBAM module.
        Args:
            gate_channels (int): Number of channels in the input feature map.
            reduction_ratio (int): Reduction ratio for CAM's MLP.
            pool_types (list): Pooling types for CAM ('avg', 'max').
            spatial_kernel_size (int): Kernel size for SAM's convolution.
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(gate_channels, reduction_ratio, pool_types)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        """
        Forward pass through the CBAM module.
        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).
        Returns:
            torch.Tensor: Attention-refined feature map (B, C, H, W).
        """
        # Apply Channel Attention first
        x_out = self.channel_attention(x)
        # Then apply Spatial Attention
        x_out = self.spatial_attention(x_out)
        return x_out

# ------------------------------------
# Self-Attention Module (Optional, e.g., SAGAN style)
# ------------------------------------
class SelfAttention(nn.Module):
    """
    Self-attention layer (similar to SAGAN).
    Computes attention between all pairs of spatial locations.
    """
    def __init__(self, in_channels, intermediate_channels=None):
        """
        Initialize SelfAttention module.
        Args:
            in_channels (int): Number of channels in the input feature map.
            intermediate_channels (int, optional): Number of channels for key/query.
                                                   Defaults to in_channels // 8.
        """
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        if intermediate_channels is None:
            self.intermediate_channels = in_channels // 8
            if self.intermediate_channels == 0: # Ensure intermediate channels > 0
                self.intermediate_channels = 1
        else:
             self.intermediate_channels = intermediate_channels

        # 1x1 Convolutions to generate query, key, value
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=self.intermediate_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=self.intermediate_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1) # Value often has full channels

        # Learnable scaling parameter (gamma) for the residual connection
        # Initialized to 0 so the block initially acts as an identity transformation
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1) # Apply softmax to the last dimension (over keys)

    def forward(self, x):
        """
        Forward pass through the SelfAttention module.
        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).
        Returns:
            torch.Tensor: Attention-refined feature map (B, C, H, W).
        """
        batch_size, C, height, width = x.size()
        N = height * width # Total number of spatial locations

        # Generate query, key, value
        proj_query = self.query_conv(x).view(batch_size, -1, N).permute(0, 2, 1) # B, N, C'
        proj_key = self.key_conv(x).view(batch_size, -1, N) # B, C', N
        proj_value = self.value_conv(x).view(batch_size, -1, N) # B, C, N

        # Calculate attention energy (scores)
        # energy = torch.bmm(proj_query, proj_key) # B, N, N
        # Use scaled dot-product attention for potentially better stability
        energy = torch.bmm(proj_query, proj_key) / (self.intermediate_channels**0.5) # B, N, N

        # Calculate attention map
        attention = self.softmax(energy) # B, N, N

        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B, C, N

        # Reshape back to image dimensions
        out = out.view(batch_size, C, height, width) # B, C, H, W

        # Apply learnable scale (gamma) and add residual connection
        out = self.gamma * out + x
        return out

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Example input tensor
    batch_size = 4
    channels = 64
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)

    print("--- Testing CBAM ---")
    cbam = CBAM(gate_channels=channels, reduction_ratio=8, spatial_kernel_size=7)
    output_cbam = cbam(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("CBAM Output shape:", output_cbam.shape)
    assert input_tensor.shape == output_cbam.shape

    print("\n--- Testing SelfAttention ---")
    self_attn = SelfAttention(in_channels=channels)
    output_self_attn = self_attn(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("SelfAttention Output shape:", output_self_attn.shape)
    assert input_tensor.shape == output_self_attn.shape

    # Test intermediate channel calculation edge case
    self_attn_low_ch = SelfAttention(in_channels=4)
    input_low_ch = torch.randn(batch_size, 4, height, width)
    output_low_ch = self_attn_low_ch(input_low_ch)
    print("\nSelfAttention Low Channel (4) Output shape:", output_low_ch.shape)
    assert input_low_ch.shape == output_low_ch.shape

    print("\nAttention modules test completed successfully.")
