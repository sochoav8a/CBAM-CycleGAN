# src/evaluate.py (CORREGIDO para guardar imágenes)

import torch
from torch.utils.data import DataLoader
import numpy as np
import os

from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') # Establecer el backend ANTES de importar pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import project modules
from src import config
from src.dataset import NpzImageDataset # Necesitamos el Dataset para cargar los datos pareados
from src.models import networks
from src.utils import tensor2im # Usaremos tensor2im para convertir a numpy

def evaluate_model():
    """
    Evaluates a trained generator checkpoint on the paired validation set.
    Calculates PSNR and SSIM metrics and saves comparison images.
    """
    print("--- Starting Evaluation ---")

    # --- 1. Configuration ---
    # Determine device
    if torch.cuda.is_available() and config.DEVICE.type == 'cuda':
        device = torch.device("cuda")
        print("Using GPU for evaluation.")
    else:
        device = torch.device("cpu")
        print("Using CPU for evaluation.")


    checkpoint_g_a2b_path = os.path.join(config.CHECKPOINT_DIR, config.EVAL_CHECKPOINT_G_A2B)
    if not os.path.isfile(checkpoint_g_a2b_path):
        print(f"Error: Generator checkpoint not found at {checkpoint_g_a2b_path}")
        return

    output_dir = config.EVAL_SAMPLES_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving comparison images to: {output_dir}")

    # --- 2. Load Model ---
    print("Defining Generator model...")
    netG_A2B = networks.define_G(config.INPUT_CHANNELS, config.OUTPUT_CHANNELS, config.NGF, config.GEN_TYPE,
                                 norm=config.NORM_GEN, use_dropout=config.USE_DROPOUT_GEN,
                                 init_type=config.INIT_TYPE_GEN, init_gain=config.INIT_GAIN_GEN, gpu_ids=[],
                                 use_attention=config.USE_ATTENTION, attention_type=config.ATTENTION_TYPE)

    print(f"Loading state dict from: {checkpoint_g_a2b_path}")
    # Load state dict to CPU first to avoid potential GPU OOM issues with model creation + loading
    state_dict = torch.load(checkpoint_g_a2b_path, map_location='cpu') # Use map_location='cpu'

    # Handle potential 'module.' prefix from DDP saving
    if list(state_dict.keys())[0].startswith('module.'):
        print("Removing 'module.' prefix from checkpoint state dict keys.")
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    netG_A2B.load_state_dict(state_dict)
    print("State dict loaded successfully into the model.")
    netG_A2B.to(device) # Move model to target device AFTER loading state dict
    netG_A2B.eval() # Set model to evaluation mode
    print("Generator loaded and set to evaluation mode.")


    # --- 3. Load Validation Dataset (Paired) ---
    print("Loading validation dataset (paired mode)...")
    try:
        val_dataset = NpzImageDataset(npz_path=config.VAL_DATA_PATH, mode='val') # Use 'val' mode for paired data
        print(f"Validation dataset loaded with {len(val_dataset)} pairs.")
    except FileNotFoundError:
        print(f"Error: Validation data file not found at {config.VAL_DATA_PATH}")
        return
    except Exception as e:
         print(f"Error loading validation dataset: {e}")
         return

    # Create DataLoader for evaluation
    print(f"Using batch size: {config.EVAL_BATCH_SIZE}")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE, # Use evaluation batch size
        shuffle=False, # No need to shuffle for evaluation
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False # pin_memory only for CUDA
    )

    # --- 4. Evaluation Loop ---
    total_psnr_fake_vs_real = 0.0
    total_ssim_fake_vs_real = 0.0
    total_psnr_fake_vs_lq = 0.0
    total_ssim_fake_vs_lq = 0.0
    total_psnr_real_vs_lq = 0.0
    total_ssim_real_vs_lq = 0.0
    num_images_processed = 0

    print("Starting evaluation...")
    for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        # Get data (LQ and Real HQ are paired)
        img_lq_batch = batch['A'].to(device) # Low Quality Input (Domain A)
        img_real_hq_batch = batch['B'].to(device) # Real High Quality Target (Domain B)

        # Generate Fake HQ image
        with torch.no_grad(): # No need to track gradients during evaluation
            img_fake_hq_batch = netG_A2B(img_lq_batch)

        # Convert tensors to NumPy images (uint8, [0, 255]) for metrics and saving
        img_lq_np_batch = tensor2im(img_lq_batch)       # Shape potentially (B, H, W)
        img_fake_hq_np_batch = tensor2im(img_fake_hq_batch) # Shape potentially (B, H, W)
        img_real_hq_np_batch = tensor2im(img_real_hq_batch) # Shape potentially (B, H, W)

        # Ensure consistent batch dimension handling (output of tensor2im can vary)
        if img_lq_np_batch.ndim == 2: # Handle case where batch size is 1 and tensor2im returns 2D
             img_lq_np_batch = img_lq_np_batch[np.newaxis, :, :]
             img_fake_hq_np_batch = img_fake_hq_np_batch[np.newaxis, :, :]
             img_real_hq_np_batch = img_real_hq_np_batch[np.newaxis, :, :]

        current_batch_size = img_lq_np_batch.shape[0]

        # Iterate through images in the current batch
        for j in range(current_batch_size):
            lq_single = img_lq_np_batch[j]          # Shape (H, W), uint8 [0, 255]
            fake_hq_single = img_fake_hq_np_batch[j] # Shape (H, W), uint8 [0, 255]
            real_hq_single = img_real_hq_np_batch[j] # Shape (H, W), uint8 [0, 255]

            # Calculate metrics
            data_range = 255 # For uint8 images

            # --- Fake vs Real ---
            try:
                psnr_fr = psnr(real_hq_single, fake_hq_single, data_range=data_range)
                ssim_fr = ssim(real_hq_single, fake_hq_single, data_range=data_range, channel_axis=None if real_hq_single.ndim == 2 else -1, win_size=7) # Added win_size for robustness
                total_psnr_fake_vs_real += psnr_fr
                total_ssim_fake_vs_real += ssim_fr
            except ValueError as e:
                 print(f"\nWarning: Skipping metrics for image {num_images_processed} (Fake vs Real) due to ValueError: {e}")
                 psnr_fr, ssim_fr = 0.0, 0.0 # Assign default values

            # --- Fake vs LQ ---
            try:
                psnr_fl = psnr(lq_single, fake_hq_single, data_range=data_range)
                ssim_fl = ssim(lq_single, fake_hq_single, data_range=data_range, channel_axis=None if lq_single.ndim == 2 else -1, win_size=7)
                total_psnr_fake_vs_lq += psnr_fl
                total_ssim_fake_vs_lq += ssim_fl
            except ValueError as e:
                 print(f"\nWarning: Skipping metrics for image {num_images_processed} (Fake vs LQ) due to ValueError: {e}")
                 psnr_fl, ssim_fl = 0.0, 0.0

            # --- Real vs LQ ---
            try:
                psnr_rl = psnr(lq_single, real_hq_single, data_range=data_range)
                ssim_rl = ssim(lq_single, real_hq_single, data_range=data_range, channel_axis=None if lq_single.ndim == 2 else -1, win_size=7)
                total_psnr_real_vs_lq += psnr_rl
                total_ssim_real_vs_lq += ssim_rl
            except ValueError as e:
                 print(f"\nWarning: Skipping metrics for image {num_images_processed} (Real vs LQ) due to ValueError: {e}")
                 psnr_rl, ssim_rl = 0.0, 0.0

            # --- Save comparison image --- <--- **LA PARTE QUE FALTABA**
            fig = plt.figure(figsize=(12, 4.5)) # Ajustado para incluir métricas en títulos
            gs = gridspec.GridSpec(1, 3, wspace=0.05, hspace=0.05)

            titles = [f'Input LQ\n(PSNR vs Real: {psnr_rl:.2f} dB)',
                      f'Generated HQ\n(PSNR vs Real: {psnr_fr:.2f} dB | SSIM: {ssim_fr:.3f})',
                      f'Real HQ']
            images = [lq_single, fake_hq_single, real_hq_single]

            for k in range(3):
                ax = plt.subplot(gs[k])
                ax.imshow(images[k], cmap='gray')
                ax.set_title(titles[k], fontsize=9) # Reducido tamaño fuente para mejor ajuste
                ax.axis('off')

            plt.suptitle(f'Comparison Image Index: {num_images_processed}', fontsize=11) # Título general
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar layout para el supertítulo

            save_path = os.path.join(output_dir, f'comparison_{num_images_processed:04d}.png')
            try:
                # Llamada a savefig que faltaba
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
            except Exception as e:
                print(f"\nWarning: Failed to save image {save_path}. Error: {e}")
            finally:
                 plt.close(fig) # Es importante cerrar la figura para liberar memoria
            # --- Fin de la parte de guardar imagen ---

            num_images_processed += 1

    # --- 5. Calculate and Print Average Metrics ---
    if num_images_processed > 0:
        avg_psnr_fr = total_psnr_fake_vs_real / num_images_processed
        avg_ssim_fr = total_ssim_fake_vs_real / num_images_processed
        avg_psnr_fl = total_psnr_fake_vs_lq / num_images_processed
        avg_ssim_fl = total_ssim_fake_vs_lq / num_images_processed
        avg_psnr_rl = total_psnr_real_vs_lq / num_images_processed
        avg_ssim_rl = total_ssim_real_vs_lq / num_images_processed

        print("\n--------------------------------------------------")
        print("Evaluation Complete")
        print(f"Checkpoint evaluated: {checkpoint_g_a2b_path}")
        print(f"Total image pairs evaluated: {num_images_processed}")
        print("\nAverage Metrics (Fake HQ vs Real HQ):")
        print(f"  PSNR: {avg_psnr_fr:.4f} dB")
        print(f"  SSIM: {avg_ssim_fr:.4f}")
        print("\nAverage Metrics (Fake HQ vs Original LQ):")
        print(f"  PSNR: {avg_psnr_fl:.4f} dB")
        print(f"  SSIM: {avg_ssim_fl:.4f}")
        print("\nAverage Metrics (Real HQ vs Original LQ - Baseline):")
        print(f"  PSNR: {avg_psnr_rl:.4f} dB")
        print(f"  SSIM: {avg_ssim_rl:.4f}")
        print("--------------------------------------------------")
    else:
        print("\nNo images were processed during evaluation.")

    print("--- Evaluation Finished ---")


if __name__ == "__main__":
    evaluate_model()
