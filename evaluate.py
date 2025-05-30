# src/evaluate.py

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

# Import project modules
from src import config
from src.dataset import NpzImageDataset
from src.models import networks
from src.utils import tensor2im, save_image, is_main_process # is_main_process might not be needed if not DDP

# Helper function to load state dict, handling 'module.' prefix
def load_state_dict_helper(model, state_dict_path, device):
    """Loads a state dict from a .pth file, handling DDP's 'module.' prefix."""
    if not os.path.isfile(state_dict_path):
        raise FileNotFoundError(f"Checkpoint file not found: {state_dict_path}")

    print(f"Loading state dict from: {state_dict_path}")
    # Load to CPU first is safer
    state_dict = torch.load(state_dict_path, map_location='cpu')

    # Create new state_dict with 'module.' prefix removed if necessary
    new_state_dict = {}
    has_module_prefix = any(key.startswith('module.') for key in state_dict)

    if has_module_prefix:
        print("Removing 'module.' prefix from state dict keys...")
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v # remove `module.`
            else:
                new_state_dict[k] = v # Keep key as is if no prefix
    else:
        new_state_dict = state_dict

    # Load the cleaned state dict
    model.load_state_dict(new_state_dict, strict=True) # Use strict=True for evaluation
    print("State dict loaded successfully into the model.")
    model.to(device) # Move model to target device
    return model

def save_evaluation_comparison(lq_img, fake_hq_img, real_hq_img, filename, save_dir=config.EVAL_SAMPLES_DIR):
    """
    Saves a side-by-side comparison of LQ, Fake HQ, and Real HQ images.
    Assumes input images are numpy arrays (H, W) or (H, W, C) in [0, 255] uint8 format.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(9, 3)) # Adjust size as needed (width=9, height=3 for 3 images)
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0.05, hspace=0.05)

    titles = ['Input LQ', 'Generated HQ (Fake)', 'Ground Truth HQ (Real)']
    images = [lq_img, fake_hq_img, real_hq_img]

    for i in range(3):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(titles[i], fontsize=10)
        # Display grayscale correctly
        img_to_show = images[i]
        if img_to_show.ndim == 3 and img_to_show.shape[2] == 1:
             img_to_show = img_to_show.squeeze(axis=2) # Remove channel dim if present
        plt.imshow(img_to_show, cmap='gray')

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=150) # Use higher dpi for better quality
    plt.close(fig)
    # print(f"Saved comparison: {save_path}") # Can be verbose


def evaluate(args):
    """Main evaluation function."""

    # --- 1. Device Setup ---
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        print("Using GPU for evaluation.")
    else:
        device = torch.device("cpu")
        print("Using CPU for evaluation.")

    # --- 2. Load Model ---
    print("Defining Generator model...")
    # Use config for model architecture params
    netG_A2B = networks.define_G(config.INPUT_CHANNELS, config.OUTPUT_CHANNELS, config.NGF, config.GEN_TYPE,
                                 norm=config.NORM_GEN, use_dropout=config.USE_DROPOUT_GEN,
                                 init_type=config.INIT_TYPE_GEN, init_gain=config.INIT_GAIN_GEN, gpu_ids=[], # No GPU IDs needed here
                                 use_attention=config.USE_ATTENTION, attention_type=config.ATTENTION_TYPE)

    # Determine checkpoint path
    if args.checkpoint_path:
        checkpoint_path_g = args.checkpoint_path
    elif config.EVAL_CHECKPOINT_G_A2B:
        checkpoint_path_g = os.path.join(config.CHECKPOINT_DIR, config.EVAL_CHECKPOINT_G_A2B)
    else:
        raise ValueError("No checkpoint path specified via command line or config.py (EVAL_CHECKPOINT_G_A2B)")

    # Load state dict using the helper function
    netG_A2B = load_state_dict_helper(netG_A2B, checkpoint_path_g, device)

    # Set model to evaluation mode
    netG_A2B.eval()
    print("Generator loaded and set to evaluation mode.")

    # --- 3. Load Paired Validation Dataset ---
    print("Loading validation dataset (paired mode)...")
    val_dataset = NpzImageDataset(npz_path=config.VAL_DATA_PATH, mode='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size if args.batch_size else config.EVAL_BATCH_SIZE,
        shuffle=False, # No need to shuffle for evaluation
        num_workers=config.NUM_WORKERS, # Can use multiple workers
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"Validation dataset loaded with {len(val_dataset)} pairs.")
    print(f"Using batch size: {args.batch_size if args.batch_size else config.EVAL_BATCH_SIZE}")

    # --- 4. Evaluation Loop ---
    psnr_scores = []
    ssim_scores = []
    image_count = 0

    print("Starting evaluation...")
    with torch.no_grad(): # Disable gradient calculations
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            # Move data to device
            real_A = batch['A'].to(device, non_blocking=True) # LQ input
            real_B = batch['B'].to(device, non_blocking=True) # Real HQ target

            # Generate fake HQ image
            fake_B = netG_A2B(real_A)

            # Convert tensors to numpy images [0, 255], uint8 for metrics and saving
            # tensor2im returns uint8 by default. Ensure shape is (H, W) for grayscale metrics.
            batch_size_current = real_A.size(0)
            for j in range(batch_size_current):
                real_A_np = tensor2im(real_A[j]) # (H, W) or (H, W, C) uint8
                fake_B_np = tensor2im(fake_B[j]) # (H, W) or (H, W, C) uint8
                real_B_np = tensor2im(real_B[j]) # (H, W) or (H, W, C) uint8

                # Ensure grayscale images have shape (H, W) for skimage metrics
                if real_A_np.ndim == 3 and real_A_np.shape[2] == 1:
                    real_A_np = real_A_np.squeeze(axis=2)
                if fake_B_np.ndim == 3 and fake_B_np.shape[2] == 1:
                    fake_B_np = fake_B_np.squeeze(axis=2)
                if real_B_np.ndim == 3 and real_B_np.shape[2] == 1:
                    real_B_np = real_B_np.squeeze(axis=2)

                # --- Calculate Metrics ---
                # PSNR
                # Ensure images are not identical before calculating PSNR to avoid division by zero (inf)
                if np.all(fake_B_np == real_B_np):
                    current_psnr = float('inf') # Or a very high value like 100
                else:
                    current_psnr = psnr(real_B_np, fake_B_np, data_range=255)
                psnr_scores.append(current_psnr)

                # SSIM
                current_ssim = ssim(real_B_np, fake_B_np, data_range=255, multichannel=False, win_size=7) # Use multichannel=False for grayscale, win_size=7 is common
                ssim_scores.append(current_ssim)

                # --- Save Comparison Images ---
                if args.save_images and image_count < args.num_save_images:
                    filename = f"comparison_{image_count:04d}.png"
                    save_evaluation_comparison(real_A_np, fake_B_np, real_B_np, filename, save_dir=config.EVAL_SAMPLES_DIR)

                image_count += 1


    # --- 5. Aggregate and Report Results ---
    # Handle potential inf values in PSNR
    finite_psnr = [p for p in psnr_scores if p != float('inf')]
    if not finite_psnr: # All images were identical?
         avg_psnr = float('inf')
         print("Warning: All evaluated images were identical to ground truth!")
    else:
         avg_psnr = np.mean(finite_psnr)

    avg_ssim = np.mean(ssim_scores)

    print("-" * 50)
    print("Evaluation Complete")
    print(f"Checkpoint evaluated: {checkpoint_path_g}")
    print(f"Total image pairs evaluated: {image_count}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    if args.save_images:
        print(f"Saved {min(image_count, args.num_save_images)} comparison images to: {config.EVAL_SAMPLES_DIR}")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CycleGAN Model for Image Quality Enhancement")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to the generator (netG_A2B) checkpoint (.pth file). Overrides config.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for evaluation. Overrides config.")
    parser.add_argument("--save_images", action='store_true',
                        help="Save visual comparison images (LQ vs Fake HQ vs Real HQ).")
    parser.add_argument("--num_save_images", type=int, default=20,
                        help="Number of comparison images to save if --save_images is set.")
    parser.add_argument("--cpu", action='store_true', help="Force CPU usage for evaluation.")

    args = parser.parse_args()

    # Run the evaluation
    evaluate(args)
