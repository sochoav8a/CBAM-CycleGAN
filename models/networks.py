# src/models/networks.py

import torch
import torch.nn as nn
import functools
from .attention import CBAM, SelfAttention # Import attention modules
# from src import config # No importar config directamente aquí para evitar dependencia circular si utils importa networks
from src.utils import init_weights # For weight initialization

# ------------------------------------
# Helper Functions / Modules
# ------------------------------------

class Identity(nn.Module):
    """Identity layer"""
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer based on the name.

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics.
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        # Use functools.partial to create a BatchNorm2d layer factory
        # affine=True: learnable scale and shift parameters (gamma and beta)
        # track_running_stats=True: track mean and variance during training for use during evaluation
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        # Use functools.partial to create an InstanceNorm2d layer factory
        # affine=False: no learnable scale/shift (common in style transfer/CycleGAN)
        # track_running_stats=False: does not track running stats (normalization is instance-specific)
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        # Return a factory function that returns the Identity module
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError(f'Normalization layer [{norm_type}] is not found')
    return norm_layer

# ------------------------------------
# Generator Network (ResNet based with Attention)
# ------------------------------------

class ResnetBlock(nn.Module):
    """Define a ResNet block with optional integrated attention."""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, use_attention=False, attention_type='CBAM'):
        """Initialize the ResNet block.

        Parameters:
            dim (int) -- Number of channels in the convolutional layers.
            padding_type (str) -- Name of padding layer: reflect | replicate | zero.
            norm_layer -- Normalization layer type (e.g., nn.InstanceNorm2d).
            use_dropout (bool) -- Whether to use dropout layers.
            use_bias (bool) -- Whether the convolutional layers use bias.
            use_attention (bool) -- Flag to enable the attention module within the block.
            attention_type (str) -- Type of attention module to use ('CBAM' or 'SelfAttention').

        A ResNet block consists of two convolutional layers with a skip connection.
        Original ResNet paper: https://arxiv.org/pdf/1512.03385.pdf
        Attention is inserted after the second conv+norm layer, before the residual addition.
        """
        super(ResnetBlock, self).__init__()
        self.use_attention = use_attention
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

        if self.use_attention:
            if attention_type == 'CBAM':
                # Instantiate CBAM with default parameters suitable for the block's channel dimension
                self.attention = CBAM(gate_channels=dim)
            elif attention_type == 'SelfAttention':
                # Instantiate SelfAttention; often computationally heavier for ResBlocks
                # Default intermediate channels set within the SelfAttention class (e.g., dim // 8)
                self.attention = SelfAttention(in_channels=dim)
            else:
                raise NotImplementedError(f"Attention type [{attention_type}] is not implemented")
            # print(f"      - ResNet block using {attention_type} Attention") # Debug print

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct the main convolutional block pathway.

        Returns a sequential block containing:
        [Pad] -> Conv -> Norm -> ReLU -> [Dropout] -> [Pad] -> Conv -> Norm
        """
        conv_block = []
        p = 0 # Padding amount for 'zero' padding type

        # Determine padding layer based on padding_type
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)] # Pad by 1 pixel on each side
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1 # Use built-in padding for Conv2d
        else:
            raise NotImplementedError(f'Padding type [{padding_type}] is not implemented')

        # First Convolutional Layer + Normalization + ReLU
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)] # In-place ReLU saves memory

        # Optional Dropout
        if use_dropout:
            conv_block += [nn.Dropout(0.5)] # Dropout probability 0.5

        # Reset padding for the second convolutional layer
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'Padding type [{padding_type}] is not implemented')

        # Second Convolutional Layer + Normalization (Attention will be applied after this)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward pass through the ResNet block."""
        # Pass input through the main convolutional block sequence
        identity = x # Store the input for the residual connection
        conv_out = self.conv_block(x)

        # Apply attention module if enabled
        if self.use_attention:
            attn_out = self.attention(conv_out)
            # Add residual connection AFTER applying attention
            out = identity + attn_out
        else:
            # Add residual connection directly to the output of the conv block
            out = identity + conv_out

        return out


class ResnetGenerator(nn.Module):
    """ResNet-based generator architecture.

    Consists of an initial convolutional layer, downsampling layers,
    several ResNet blocks, upsampling layers, and a final output layer.
    Adapted from Johnson's neural style transfer project and CycleGAN paper.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect', use_attention=False, attention_type='CBAM'):
        """Construct a ResNet-based generator.

        Parameters:
            input_nc (int) -- Number of channels in input images.
            output_nc (int) -- Number of channels in output images.
            ngf (int) -- Number of filters in the first convolutional layer.
            norm_layer -- Normalization layer type.
            use_dropout (bool) -- Whether to use dropout in ResNet blocks.
            n_blocks (int) -- Number of ResNet blocks.
            padding_type (str) -- Padding type for convolutional layers.
            use_attention (bool) -- Flag to enable attention in ResNet blocks.
            attention_type (str) -- Type of attention module ('CBAM', 'SelfAttention').
        """
        assert n_blocks >= 0, "Number of ResNet blocks must be non-negative"
        super(ResnetGenerator, self).__init__()

        # Determine if bias should be used based on the normalization layer type
        # InstanceNorm2d can handle bias itself if affine=True, but usually set to False here
        if type(norm_layer) == functools.partial:
            # Check the function wrapped by partial
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            # Check the layer class directly
            use_bias = norm_layer == nn.InstanceNorm2d

        # --- Initial Convolution Block ---
        # Pad -> Conv(7x7) -> Norm -> ReLU
        model = [nn.ReflectionPad2d(3), # Pad 3 pixels for 7x7 kernel
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # --- Downsampling Layers ---
        n_downsampling = 2 # Number of downsampling steps
        for i in range(n_downsampling):
            mult = 2 ** i # Multiplier for number of filters (1, 2)
            # Conv(3x3, stride=2) -> Norm -> ReLU
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # Current multiplier after downsampling
        mult = 2 ** n_downsampling # Typically 4 (ngf * 4 channels)

        # --- ResNet Blocks ---
        # print(f"Building Generator with {n_blocks} ResNet blocks.")
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, # Channel dimension remains constant through blocks
                                  padding_type=padding_type,
                                  norm_layer=norm_layer,
                                  use_dropout=use_dropout,
                                  use_bias=use_bias,
                                  use_attention=use_attention, # Pass attention flags
                                  attention_type=attention_type)]

        # --- Upsampling Layers ---
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i) # Multiplier for number of filters (4, 2)
            # ConvTranspose(3x3, stride=2) -> Norm -> ReLU
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, # output_padding ensures correct output size
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        # --- Final Output Layer ---
        # Pad -> Conv(7x7) -> Tanh
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # Tanh activation maps the output to the range [-1, 1]
        model += [nn.Tanh()]

        # Combine all layers into a Sequential model
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward pass."""
        return self.model(input)

# ------------------------------------
# Discriminator Network (PatchGAN)
# ------------------------------------

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator architecture."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator.

        Parameters:
            input_nc (int) -- Number of channels in input images.
            ndf (int) -- Number of filters in the first convolutional layer.
            n_layers (int) -- Number of convolutional layers in the discriminator body.
            norm_layer -- Normalization layer type.
        """
        super(NLayerDiscriminator, self).__init__()

        # Determine if bias should be used
        if type(norm_layer) == functools.partial:
            # BatchNorm2d usually handles bias via affine parameters, InstanceNorm often doesn't
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4 # Kernel width for convolutional layers
        padw = 1 # Padding width

        # --- Initial Convolution Layer ---
        # Conv(4x4, stride=2) -> LeakyReLU (No normalization)
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)] # LeakyReLU with negative slope 0.2

        # --- Intermediate Convolutional Layers ---
        nf_mult = 1 # Filter multiplier
        nf_mult_prev = 1
        # Add 'n_layers - 1' intermediate layers
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            # Double the number of filters, capped at 8x ndf
            nf_mult = min(2 ** n, 8)
            # Conv(4x4, stride=2) -> Norm -> LeakyReLU
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # --- Layer before Final Output ---
        nf_mult_prev = nf_mult
        # Increase filters one last time if possible
        nf_mult = min(2 ** n_layers, 8)
        # Conv(4x4, stride=1) -> Norm -> LeakyReLU
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # --- Final Convolution Layer (Output) ---
        # Conv(4x4, stride=1) -> Output (1 channel prediction map)
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        # Note: No final activation (e.g., Sigmoid) is applied here.
        # This is suitable for LSGAN (which uses MSELoss) or
        # if using BCEWithLogitsLoss (which combines Sigmoid + BCELoss).

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward pass."""
        return self.model(input)

# ------------------------------------
# Factory Functions (for easy network creation)
# ------------------------------------
def define_G(input_nc, output_nc, ngf, netG_type, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], use_attention=False, attention_type='cbam'):
    """Create and initialize a generator network.
    Parameters mirror the ResnetGenerator constructor and add initialization options.
    gpu_ids are currently unused here but kept for potential future use (DDP handles device placement).
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    # Select generator architecture based on netG_type
    if netG_type == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, use_attention=use_attention, attention_type=attention_type)
    elif netG_type == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, use_attention=use_attention, attention_type=attention_type)
    elif netG_type == 'resnet_12blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=12, use_attention=use_attention, attention_type=attention_type)
    elif netG_type == 'resnet_15blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=15, use_attention=use_attention, attention_type=attention_type)
    # Add other generator types like UNet here if needed
    # elif netG_type == 'unet_256':
    #     net = UnetGenerator(...) # Assuming UnetGenerator is defined elsewhere
    else:
        raise NotImplementedError(f'Generator model name [{netG_type}] is not recognized')
    print(f"--- Generator [{netG_type}] created with: ---")
    print(f"    Input Channels: {input_nc}, Output Channels: {output_nc}")
    print(f"    Base Filters (ngf): {ngf}")
    print(f"    Normalization: {norm}")
    print(f"    Use Dropout: {use_dropout}")
    print(f"    Use Attention: {use_attention}, Type: {attention_type if use_attention else 'N/A'}")
    print(f"-------------------------------------------")
    # Initialize network weights
    init_weights(net, init_type, init_gain)
    return net

def define_D(input_nc, ndf, netD_type, n_layers_D=3, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create and initialize a discriminator network.

    Parameters mirror the NLayerDiscriminator constructor and add initialization options.
    gpu_ids are currently unused here.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    # Select discriminator architecture based on netD_type
    if netD_type == 'patchgan' or netD_type == 'n_layers': # Treat 'patchgan' as equivalent to 'n_layers'
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD_type == 'basic': # 'basic' is often used as an alias for PatchGAN with n_layers=3
        net = NLayerDiscriminator(input_nc, ndf, 3, norm_layer=norm_layer)
    # Add other discriminator types like PixelDiscriminator here if needed
    # elif netD_type == 'pixel':
    #     net = PixelDiscriminator(...) # Assuming PixelDiscriminator is defined elsewhere
    else:
        raise NotImplementedError(f'Discriminator model name [{netD_type}] is not recognized')

    print(f"--- Discriminator [{netD_type}] created with: ---")
    print(f"    Input Channels: {input_nc}")
    print(f"    Base Filters (ndf): {ndf}")
    print(f"    Number of Layers (n_layers_D): {n_layers_D if netD_type in ['patchgan', 'n_layers'] else ('3 (basic)' if netD_type == 'basic' else 'N/A')}")
    print(f"    Normalization: {norm}")
    print(f"-------------------------------------------")

    # Initialize network weights
    init_weights(net, init_type, init_gain)
    return net


# --- Example Usage Block (for testing network definitions) ---
if __name__ == '__main__':
    print("\n--- Testing Network Definitions ---")

    # Dummy configuration class to simulate config parameters
    class DummyConfig:
        INPUT_CHANNELS = 1
        OUTPUT_CHANNELS = 1
        IMG_HEIGHT = 256
        IMG_WIDTH = 256
        # Generator Params
        NGF = 64
        GEN_TYPE = 'resnet_9blocks'
        NORM_GEN = 'instance'
        USE_DROPOUT_GEN = False
        INIT_TYPE_GEN = 'normal'
        INIT_GAIN_GEN = 0.02
        USE_ATTENTION = True # Test WITH attention first
        ATTENTION_TYPE = 'CBAM'
        # Discriminator Params
        NDF = 64
        DISC_TYPE = 'patchgan'
        N_LAYERS_DISC = 3
        NORM_DISC = 'instance'
        INIT_TYPE_DISC = 'normal'
        INIT_GAIN_DISC = 0.02

    print("\n--- Testing Generator (with Attention) ---")
    cfg = DummyConfig()
    netG = define_G(cfg.INPUT_CHANNELS, cfg.OUTPUT_CHANNELS, cfg.NGF, cfg.GEN_TYPE,
                    norm=cfg.NORM_GEN, use_dropout=cfg.USE_DROPOUT_GEN,
                    init_type=cfg.INIT_TYPE_GEN, init_gain=cfg.INIT_GAIN_GEN, gpu_ids=[],
                    use_attention=cfg.USE_ATTENTION, attention_type=cfg.ATTENTION_TYPE)

    # Create a sample input tensor
    test_input_g = torch.randn(2, cfg.INPUT_CHANNELS, cfg.IMG_HEIGHT, cfg.IMG_WIDTH) # Batch size 2
    # Perform a forward pass
    test_output_g = netG(test_input_g)
    print("Generator Input shape:", test_input_g.shape)
    print("Generator Output shape:", test_output_g.shape)
    # Verify output shape matches input spatial dimensions and output channels
    assert test_output_g.shape == (2, cfg.OUTPUT_CHANNELS, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
    print("Generator output shape test PASSED.")

    print("\n--- Testing Generator (without Attention) ---")
    cfg.USE_ATTENTION = False # Disable attention
    netG_no_attn = define_G(cfg.INPUT_CHANNELS, cfg.OUTPUT_CHANNELS, cfg.NGF, cfg.GEN_TYPE,
                            norm=cfg.NORM_GEN, use_dropout=cfg.USE_DROPOUT_GEN,
                            init_type=cfg.INIT_TYPE_GEN, init_gain=cfg.INIT_GAIN_GEN, gpu_ids=[],
                            use_attention=cfg.USE_ATTENTION, attention_type=cfg.ATTENTION_TYPE)
    test_output_g_no_attn = netG_no_attn(test_input_g) # Use the same input
    print("Generator (No Attn) Output shape:", test_output_g_no_attn.shape)
    assert test_output_g_no_attn.shape == (2, cfg.OUTPUT_CHANNELS, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
    print("Generator (No Attn) output shape test PASSED.")


    print("\n--- Testing Discriminator ---")
    cfg = DummyConfig() # Reset config just in case
    netD = define_D(cfg.INPUT_CHANNELS, cfg.NDF, cfg.DISC_TYPE,
                    n_layers_D=cfg.N_LAYERS_DISC, norm=cfg.NORM_DISC,
                    init_type=cfg.INIT_TYPE_DISC, init_gain=cfg.INIT_GAIN_DISC, gpu_ids=[])

    # Create a sample input tensor
    test_input_d = torch.randn(2, cfg.INPUT_CHANNELS, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
    # Perform a forward pass
    test_output_d = netD(test_input_d)
    print("Discriminator Input shape:", test_input_d.shape)
    print("Discriminator Output shape:", test_output_d.shape)

    # Calculate expected PatchGAN output size for N=3 layers
    # Input: 256x256
    # L0: Conv(k=4,s=2,p=1) -> (256-4+2*1)/2 + 1 = 128
    # L1: Conv(k=4,s=2,p=1) -> (128-4+2*1)/2 + 1 = 64
    # L2: Conv(k=4,s=2,p=1) -> (64-4+2*1)/2 + 1 = 32
    # L3: Conv(k=4,s=1,p=1) -> (32-4+2*1)/1 + 1 = 31
    # L4: Conv(k=4,s=1,p=1) -> (31-4+2*1)/1 + 1 = 30
    # Expected output size: B x 1 x 30 x 30
    expected_shape = (2, 1, 30, 30)
    assert test_output_d.shape == expected_shape, f"Expected shape {expected_shape}, but got {test_output_d.shape}"
    print(f"Discriminator output shape test PASSED (Expected {expected_shape}).")

    print("\nNetwork definitions test completed successfully.")
