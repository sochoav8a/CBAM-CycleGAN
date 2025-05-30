# src/vgg_loss.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class VGG19PerceptualLoss(nn.Module):
    """
    Calculates VGG19 perceptual loss based on ReLU activation layers.

    Expects input tensors in the range [-1, 1] (CycleGAN standard output).
    Normalizes the input internally to ImageNet stats before feeding to VGG.
    Handles grayscale input by replicating channels.
    """
    def __init__(self, feature_layers=None, use_input_norm=True, requires_grad=False):
        """
        Args:
            feature_layers (list of int, optional): Indices of VGG19 feature layers to use.
                                                    Defaults to layers before pool1, pool2, pool3, pool4, pool5.
            use_input_norm (bool): If True, normalize inputs to ImageNet stats.
            requires_grad (bool): If True, VGG weights require gradients (usually False).
        """
        super(VGG19PerceptualLoss, self).__init__()

        # Load VGG19 features, pre-trained on ImageNet
        vgg19 = models.vgg19(pretrained=True).features
        print("Loaded pretrained VGG19 model for perceptual loss.")

        # Default layers are outputs of ReLU before max pooling layers
        if feature_layers is None:
            # Indices corresponding typically to relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
            # VGG19 structure: Conv -> ReLU -> [Conv -> ReLU] -> Pool ...
            # Layer indices can vary slightly depending on torchvision version, check model structure if needed.
            # These indices correspond to common choices for perceptual loss.
            self.feature_layers = [1, 6, 11, 20, 29] # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 (adjust if needed)
            # Alternative: [3, 8, 17, 26, 35] # Outputs after Conv layers
        else:
            self.feature_layers = feature_layers

        self.max_layer_idx = max(self.feature_layers)

        # Create slices of the VGG model up to the required layers
        self.vgg_slices = nn.ModuleList(
            [nn.Sequential(*list(vgg19.children())[:idx + 1]) for idx in self.feature_layers]
        )

        # Input normalization parameters (ImageNet)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            # Normalization specific to VGG's expected input
            # Input to this module is [-1, 1], we need to map to [0, 1] then normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        # Freeze VGG weights if requires_grad is False
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
            print(f"VGG19 weights frozen (requires_grad={requires_grad}).")
        else:
             print(f"VGG19 weights are trainable (requires_grad={requires_grad}).")


    def normalize_input(self, x):
        """ Normalizes tensor from [-1, 1] to ImageNet mean/std """
        if not self.use_input_norm:
            return x
        # Map [-1, 1] to [0, 1]
        x_norm = (x + 1.0) / 2.0
        # Apply ImageNet normalization
        x_norm = (x_norm - self.mean) / self.std
        return x_norm

    def forward(self, x):
        """
        Extract features from specified VGG layers.
        Args:
            x (torch.Tensor): Input tensor (B, C, H, W), range [-1, 1].
        Returns:
            list of torch.Tensor: List of feature maps from the specified layers.
        """
        # Handle grayscale input (B, 1, H, W) by repeating the channel
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1) # Repeat channel dim to get (B, 3, H, W)

        # Normalize input if required
        x_processed = self.normalize_input(x)

        # Extract features
        features = []
        current_input = x_processed
        last_slice_idx = 0
        for i, layer_idx in enumerate(self.feature_layers):
            # Pass through the next slice of VGG
            slice_model = self.vgg_slices[i]
            # We only need to compute up to the current layer idx
            # If we were smarter, we could avoid recomputing early layers,
            # but nn.Sequential makes this simple.
            # This approach recomputes, but is clear.
            # For efficiency, one could pass x through vgg19 once and hook layers.
            feature = slice_model(x_processed) # Pass original normalized input each time
            features.append(feature)

            # --- Alternative (More Efficient but complex indexing): ---
            # slice_to_run = nn.Sequential(*list(self.vgg_slices.children())[last_slice_idx:layer_idx+1])
            # current_input = slice_to_run(current_input)
            # features.append(current_input)
            # last_slice_idx = layer_idx + 1
            # ---

        return features

# --- Criterion Function ---
def vgg_perceptual_criterion(vgg_loss_module, x_features, y_features, criterion=nn.L1Loss()):
    """
    Calculates the perceptual loss based on extracted VGG features.
    Args:
        vgg_loss_module (VGG19PerceptualLoss): The module instance.
        x_features (list of torch.Tensor): Features from the first image.
        y_features (list of torch.Tensor): Features from the second image.
        criterion (nn.Module): Loss function to compare features (e.g., nn.L1Loss).
    Returns:
        torch.Tensor: The calculated perceptual loss.
    """
    if len(x_features) != len(y_features):
        raise ValueError("Feature lists must have the same length.")

    loss = 0.0
    for feat_x, feat_y in zip(x_features, y_features):
        # Detach y_features if they come from a target/real image to avoid backprop
        loss += criterion(feat_x, feat_y.detach())

    # Average loss over the number of feature layers used
    if len(x_features) > 0:
         loss /= len(x_features)

    return loss

# --- Example Usage ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing VGGPerceptualLoss on device: {device}")

    # Create dummy input tensors (Batch=2, Channels=1, Size=256x256, Range=[-1, 1])
    dummy_input1 = torch.rand(2, 1, 256, 256) * 2.0 - 1.0
    dummy_input2 = torch.rand(2, 1, 256, 256) * 2.0 - 1.0
    dummy_input1 = dummy_input1.to(device)
    dummy_input2 = dummy_input2.to(device)

    # Instantiate the loss module
    vgg_loss = VGG19PerceptualLoss().to(device)
    vgg_loss.eval() # Set to eval mode as it's not being trained

    # Extract features
    with torch.no_grad(): # No need for gradients during test
        features1 = vgg_loss(dummy_input1)
        features2 = vgg_loss(dummy_input2)

    print(f"Number of feature layers extracted: {len(features1)}")
    for i, feat in enumerate(features1):
        print(f"  Layer {vgg_loss.feature_layers[i]} feature shape: {feat.shape}")

    # Calculate loss between features
    criterionL1 = nn.L1Loss()
    loss_val = vgg_perceptual_criterion(vgg_loss, features1, features2, criterion=criterionL1)
    print(f"Calculated perceptual loss (L1 on features): {loss_val.item()}")

    print("VGG Loss test completed.")
