import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import lpips  # Nueva importación para LPIPS
from pytorch_msssim import ms_ssim  # Nueva importación para MS-SSIM
import cv2 # Added for Bicubic interpolation
from PIL import Image # Added for image handling if needed

from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Establecer el backend ANTES de importar pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import project modules
from src import config
from src.dataset import NpzImageDataset  # Necesitamos el Dataset para cargar los datos pareados
from src.models import networks
from src.utils import tensor2im  # Usaremos tensor2im para convertir a numpy

# --- Helper function for Interpolation SR ---
def interpolate_image_sr(img_lq_np_uint8, target_shape_hw, interpolation_flag):
    """Upscales a uint8 NumPy image using a specified interpolation method.
    Args:
        img_lq_np_uint8 (numpy.ndarray): Low-quality image (H, W) or (H, W, C), dtype uint8, range [0, 255].
        target_shape_hw (tuple): Target (height, width).
        interpolation_flag (int): OpenCV interpolation flag (e.g., cv2.INTER_CUBIC, cv2.INTER_LANCZOS4).
    Returns:
        numpy.ndarray: Upscaled image, dtype uint8, range [0, 255].
    """
    target_shape_wh = (target_shape_hw[1], target_shape_hw[0]) # cv2 uses (W, H)
    if img_lq_np_uint8.ndim == 2:
        interpolated_hq = cv2.resize(img_lq_np_uint8, target_shape_wh, interpolation=interpolation_flag)
    elif img_lq_np_uint8.ndim == 3 and img_lq_np_uint8.shape[2] == 1: # Grayscale (H,W,1)
        interpolated_hq = cv2.resize(img_lq_np_uint8.squeeze(), target_shape_wh, interpolation=interpolation_flag)
        interpolated_hq = interpolated_hq[..., np.newaxis] # Add channel back
    elif img_lq_np_uint8.ndim == 3: # Color (H,W,C)
        interpolated_hq = cv2.resize(img_lq_np_uint8, target_shape_wh, interpolation=interpolation_flag)
    else:
        raise ValueError(f"Unsupported image shape for interpolation: {img_lq_np_uint8.shape}")
    return interpolated_hq # Returns uint8

def evaluate_model():
    """
    Evaluates a trained generator checkpoint on the paired validation set.
    Calculates multiple metrics (PSNR, SSIM, LPIPS, MS-SSIM) and saves comparison images.
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
    # Load Generator
    print("Defining Generator model...")
    netG_A2B = networks.define_G(config.INPUT_CHANNELS, config.OUTPUT_CHANNELS, config.NGF, config.GEN_TYPE,
                                 norm=config.NORM_GEN, use_dropout=config.USE_DROPOUT_GEN,
                                 init_type=config.INIT_TYPE_GEN, init_gain=config.INIT_GAIN_GEN, gpu_ids=[],
                                 use_attention=config.USE_ATTENTION, attention_type=config.ATTENTION_TYPE)

    print(f"Loading state dict from: {checkpoint_g_a2b_path}")
    # Load state dict to CPU first to avoid potential GPU OOM issues with model creation + loading
    state_dict = torch.load(checkpoint_g_a2b_path, map_location='cpu')

    # Handle potential 'module.' prefix from DDP saving
    if list(state_dict.keys())[0].startswith('module.'):
        print("Removing 'module.' prefix from checkpoint state dict keys.")
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    netG_A2B.load_state_dict(state_dict)
    print("State dict loaded successfully into the model.")
    netG_A2B.to(device)  # Move model to target device AFTER loading state dict
    netG_A2B.eval()  # Set model to evaluation mode
    print("Generator loaded and set to evaluation mode.")

    # --- Inicializar modelo LPIPS ---
    print("Initializing LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex').to(device)  # Utiliza AlexNet como base
    lpips_model.eval()

    # --- 3. Load Validation Dataset (Paired) ---
    print("Loading validation dataset (paired mode)...")
    try:
        val_dataset = NpzImageDataset(npz_path=config.VAL_DATA_PATH, mode='val')  # Use 'val' mode for paired data
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
        batch_size=config.EVAL_BATCH_SIZE,  # Use evaluation batch size
        shuffle=False,  # No need to shuffle for evaluation
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False  # pin_memory only for CUDA
    )

    # --- 4. Evaluation Loop ---
    # Métricas originales
    total_psnr_fake_vs_real = 0.0
    total_ssim_fake_vs_real = 0.0
    total_psnr_fake_vs_lq = 0.0
    total_ssim_fake_vs_lq = 0.0
    total_psnr_real_vs_lq = 0.0
    total_ssim_real_vs_lq = 0.0
    
    # Nuevas métricas
    total_lpips_fake_vs_real = 0.0
    total_msssim_fake_vs_real = 0.0
    total_lpips_real_vs_lq = 0.0  # LPIPS entre real y LQ (baseline)
    total_msssim_real_vs_lq = 0.0  # MS-SSIM entre real y LQ (baseline)
    
    # Metrics for Bicubic super-resolution
    total_psnr_bicubic_vs_real = 0.0
    total_ssim_bicubic_vs_real = 0.0
    total_lpips_bicubic_vs_real = 0.0
    total_msssim_bicubic_vs_real = 0.0
    
    # Metrics for Lanczos super-resolution
    total_psnr_lanczos_vs_real = 0.0
    total_ssim_lanczos_vs_real = 0.0
    total_lpips_lanczos_vs_real = 0.0
    total_msssim_lanczos_vs_real = 0.0
    
    num_images_processed = 0

    print("Starting evaluation...")
    for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        # Get data (LQ and Real HQ are paired)
        img_lq_batch = batch['A'].to(device)  # Low Quality Input (Domain A)
        img_real_hq_batch = batch['B'].to(device)  # Real High Quality Target (Domain B)

        # Generate Fake HQ image
        with torch.no_grad():  # No need to track gradients during evaluation
            img_fake_hq_batch = netG_A2B(img_lq_batch)
            
            # --- Calcular LPIPS (funciona con tensores, no requiere conversión a numpy) ---
            # LPIPS espera valores en el rango [-1, 1]
            lpips_batch_fr = lpips_model(img_real_hq_batch, img_fake_hq_batch) # Tensor of LPIPS values for batch
            total_lpips_fake_vs_real += lpips_batch_fr.sum().item() # Sum over batch
            
            # También calcular LPIPS entre real y LQ (baseline)
            lpips_batch_rl = lpips_model(img_real_hq_batch, img_lq_batch) # Tensor of LPIPS values for batch
            total_lpips_real_vs_lq += lpips_batch_rl.sum().item() # Sum over batch
            
            # --- Calcular MS-SSIM (funciona con tensores en [0, 1]) ---
            # Normalizar si es necesario. Original tensors img_..._batch are [-1, 1]
            norm_real_msssim = (img_real_hq_batch + 1) / 2
            norm_fake_msssim = (img_fake_hq_batch + 1) / 2
            norm_lq_msssim = (img_lq_batch + 1) / 2
                
            # MS-SSIM requiere imágenes con el mismo tamaño y valores en [0, 1]
            msssim_batch_fr = ms_ssim(norm_real_msssim, norm_fake_msssim, data_range=1.0, size_average=False) # Get per-image scores in batch
            total_msssim_fake_vs_real += msssim_batch_fr.sum().item()
            
            # También calcular MS-SSIM entre real y LQ (baseline)
            msssim_batch_rl = ms_ssim(norm_real_msssim, norm_lq_msssim, data_range=1.0, size_average=False)
            total_msssim_real_vs_lq += msssim_batch_rl.sum().item()

        # Convert tensors to NumPy images (uint8, [0, 255]) for métricas tradicionales y visualización
        img_lq_np_batch = tensor2im(img_lq_batch)
        img_fake_hq_np_batch = tensor2im(img_fake_hq_batch)
        img_real_hq_np_batch = tensor2im(img_real_hq_batch)

        # Ensure consistent batch dimension handling (output of tensor2im can vary)
        if img_lq_np_batch.ndim == 2:  # Handle case where batch size is 1 and tensor2im returns 2D
            img_lq_np_batch = img_lq_np_batch[np.newaxis, :, :]
            img_fake_hq_np_batch = img_fake_hq_np_batch[np.newaxis, :, :]
            img_real_hq_np_batch = img_real_hq_np_batch[np.newaxis, :, :]

        current_batch_size = img_lq_np_batch.shape[0]

        # Iterate through images in the current batch
        for j in range(current_batch_size):
            lq_single = img_lq_np_batch[j]          # Shape (H, W), uint8 [0, 255]
            fake_hq_single = img_fake_hq_np_batch[j]  # Shape (H, W), uint8 [0, 255]
            real_hq_single = img_real_hq_np_batch[j]  # Shape (H, W), uint8 [0, 255]

            # --- Bicubic Super-Resolution ---
            # Ensure lq_single is uint8 for bicubic_super_resolve function
            if lq_single.dtype != np.uint8:
                lq_single_uint8 = np.clip(lq_single, 0, 255).astype(np.uint8)
            else:
                lq_single_uint8 = lq_single
            
            # Target shape for SR should be the shape of real_hq_single
            target_sr_shape_hw = real_hq_single.shape[:2]
            img_bicubic_hq_np = interpolate_image_sr(lq_single_uint8, target_sr_shape_hw, cv2.INTER_CUBIC)
            # img_bicubic_hq_np is uint8 [0,255].

            # --- Lanczos Super-Resolution ---
            img_lanczos_hq_np = interpolate_image_sr(lq_single_uint8, target_sr_shape_hw, cv2.INTER_LANCZOS4)
            # img_lanczos_hq_np is uint8 [0,255].

            # Calculate metrics
            data_range = 255  # For uint8 images (PSNR, SSIM)

            # --- Fake vs Real ---
            try:
                psnr_fr = psnr(real_hq_single, fake_hq_single, data_range=data_range)
                ssim_fr = ssim(real_hq_single, fake_hq_single, data_range=data_range, 
                              channel_axis=None if real_hq_single.ndim == 2 else -1, win_size=7)
                total_psnr_fake_vs_real += psnr_fr
                total_ssim_fake_vs_real += ssim_fr
            except ValueError as e:
                print(f"\nWarning: Skipping metrics for image {num_images_processed} (Fake vs Real) due to ValueError: {e}")
                psnr_fr, ssim_fr = 0.0, 0.0

            # --- Fake vs LQ ---
            try:
                psnr_fl = psnr(lq_single, fake_hq_single, data_range=data_range)
                ssim_fl = ssim(lq_single, fake_hq_single, data_range=data_range, 
                              channel_axis=None if lq_single.ndim == 2 else -1, win_size=7)
                total_psnr_fake_vs_lq += psnr_fl
                total_ssim_fake_vs_lq += ssim_fl
            except ValueError as e:
                print(f"\nWarning: Skipping metrics for image {num_images_processed} (Fake vs LQ) due to ValueError: {e}")
                psnr_fl, ssim_fl = 0.0, 0.0

            # --- Real vs LQ ---
            try:
                psnr_rl = psnr(lq_single, real_hq_single, data_range=data_range)
                ssim_rl = ssim(lq_single, real_hq_single, data_range=data_range, 
                              channel_axis=None if lq_single.ndim == 2 else -1, win_size=7)
                total_psnr_real_vs_lq += psnr_rl
                total_ssim_real_vs_lq += ssim_rl
            except ValueError as e:
                print(f"\nWarning: Skipping metrics for image {num_images_processed} (Real vs LQ) due to ValueError: {e}")
                psnr_rl, ssim_rl = 0.0, 0.0

            # --- Metrics for Bicubic SR vs Real HQ ---
            try:
                psnr_bic_real = psnr(real_hq_single, img_bicubic_hq_np, data_range=data_range)
                ssim_bic_real = ssim(real_hq_single, img_bicubic_hq_np, data_range=data_range,
                                     channel_axis=None if real_hq_single.ndim == 2 else -1, win_size=7)
                total_psnr_bicubic_vs_real += psnr_bic_real
                total_ssim_bicubic_vs_real += ssim_bic_real
            except ValueError as e:
                print(f"\nWarning: Skipping PSNR/SSIM for Bicubic vs Real (image {num_images_processed}) due to ValueError: {e}")
                psnr_bic_real, ssim_bic_real = 0.0, 0.0

            # For LPIPS and MS-SSIM, convert bicubic to tensor and normalize
            # Convert uint8 [0,255] numpy to float32 tensor [-1,1] for LPIPS
            # and float32 tensor [0,1] for MS-SSIM
            # Bicubic image: H, W or H, W, 1 (if grayscale)
            bicubic_tensor = torch.from_numpy(img_bicubic_hq_np.astype(np.float32)).to(device).unsqueeze(0) # Add batch dim
            if bicubic_tensor.ndim == 3: # H, W -> B, H, W
                bicubic_tensor = bicubic_tensor.unsqueeze(1) # B, C, H, W (C=1)
            elif bicubic_tensor.ndim == 4 and bicubic_tensor.shape[3] == 1: # B, H, W, 1
                bicubic_tensor = bicubic_tensor.permute(0, 3, 1, 2) # B, C, H, W
            elif bicubic_tensor.ndim == 4 and bicubic_tensor.shape[3] > 1: # B, H, W, C (color)
                bicubic_tensor = bicubic_tensor.permute(0, 3, 1, 2) # B, C, H, W
            else:
                 # This case should ideally not be hit if bicubic_super_resolve is correct
                print(f"Unexpected bicubic_tensor shape: {bicubic_tensor.shape}")
                lpips_val_bic_real = 0.0
                msssim_val_bic_real = 0.0
            
            # LPIPS expects [-1, 1]
            bicubic_tensor_norm_lpips = (bicubic_tensor / 127.5) - 1.0
            # MS-SSIM expects [0, 1]
            bicubic_tensor_norm_msssim = bicubic_tensor / 255.0
            
            # real_hq_batch is already a tensor on device, needs to be sliced and matched
            # real_hq_single is numpy, so we use norm_real from earlier which is batch[j]
            # Or, more simply, use the real_hq_batch directly and slice it.
            # Ensure real_hq_tensor_single is [B,C,H,W] for LPIPS/MS-SSIM
            current_real_hq_tensor = img_real_hq_batch[j:j+1] # Shape [1, C, H, W]
            norm_current_real_hq_tensor_msssim = (current_real_hq_tensor + 1) / 2 if current_real_hq_tensor.min() < 0 else current_real_hq_tensor

            try:
                with torch.no_grad():
                    lpips_val_bic_real = lpips_model(current_real_hq_tensor, bicubic_tensor_norm_lpips).mean().item()
                    # Ensure MS-SSIM inputs are [0,1]
                    msssim_val_bic_real = ms_ssim(norm_current_real_hq_tensor_msssim, bicubic_tensor_norm_msssim, data_range=1.0).mean().item()
                total_lpips_bicubic_vs_real += lpips_val_bic_real
                total_msssim_bicubic_vs_real += msssim_val_bic_real
            except Exception as e:
                print(f"\nWarning: Skipping LPIPS/MS-SSIM for Bicubic vs Real (image {num_images_processed}) due to error: {e}")
                lpips_val_bic_real = 0.0
                msssim_val_bic_real = 0.0

            # --- Metrics for Lanczos SR vs Real HQ (PSNR/SSIM)
            try:
                psnr_lanc_real = psnr(real_hq_single, img_lanczos_hq_np, data_range=data_range)
                ssim_lanc_real = ssim(real_hq_single, img_lanczos_hq_np, data_range=data_range,
                                     channel_axis=None if real_hq_single.ndim == 2 else -1, win_size=7)
                total_psnr_lanczos_vs_real += psnr_lanc_real
                total_ssim_lanczos_vs_real += ssim_lanc_real
            except ValueError as e:
                print(f"\nWarning: Skipping PSNR/SSIM for Lanczos vs Real (image {num_images_processed}) due to ValueError: {e}")
                psnr_lanc_real, ssim_lanc_real = 0.0, 0.0
            
            # Accumulate LPIPS/MS-SSIM for Bicubic (already done for fake vs real, real vs lq at batch level)
            # Note: The totals for fake_vs_real and real_vs_lq are now batch-summed. For bicubic and lanczos, we add per-image scores here.
            total_lpips_bicubic_vs_real += lpips_val_bic_real
            total_msssim_bicubic_vs_real += msssim_val_bic_real

            # Accumulate LPIPS/MS-SSIM for Lanczos 
            total_lpips_lanczos_vs_real += lpips_val_bic_real
            total_msssim_lanczos_vs_real += msssim_val_bic_real

            # Obtener valores individuales de LPIPS y MS-SSIM para esta imagen
            lpips_val_fr = lpips_batch_fr[j].item() if current_batch_size == 1 else lpips_batch_fr.sum().item() / current_batch_size
            msssim_val_fr = msssim_batch_fr[j].item() if current_batch_size == 1 else msssim_batch_fr.sum().item() / current_batch_size
            lpips_val_rl = lpips_batch_rl[j].item() if current_batch_size == 1 else lpips_batch_rl.sum().item() / current_batch_size
            msssim_val_rl = msssim_batch_rl[j].item() if current_batch_size == 1 else msssim_batch_rl.sum().item() / current_batch_size

            # --- Save comparison image con métricas ampliadas ---
            fig = plt.figure(figsize=(22, 5))  # Increased for 5 panels
            gs = gridspec.GridSpec(1, 5, wspace=0.05, hspace=0.05) # Now 5 columns

            titles = [
                f'Input LQ\n(PSNR vs Real: {psnr_rl:.2f} dB | SSIM: {ssim_rl:.3f})\n(LPIPS: {lpips_val_rl:.3f} | MS-SSIM: {msssim_val_rl:.3f})',
                f'Bicubic HQ\n(PSNR: {psnr_bic_real:.2f} dB | SSIM: {ssim_bic_real:.3f})\n(LPIPS: {lpips_val_bic_real:.3f} | MS-SSIM: {msssim_val_bic_real:.3f})',
                f'Lanczos HQ\n(PSNR: {psnr_lanc_real:.2f} dB | SSIM: {ssim_lanc_real:.3f})\n(LPIPS: {lpips_val_bic_real:.3f} | MS-SSIM: {msssim_val_bic_real:.3f})',
                f'Generated HQ (Ours)\n(PSNR: {psnr_fr:.2f} dB | SSIM: {ssim_fr:.3f})\n(LPIPS: {lpips_val_fr:.3f} | MS-SSIM: {msssim_val_fr:.3f})',
                f'Real HQ'
            ]
            images_to_plot = [lq_single, img_bicubic_hq_np, img_lanczos_hq_np, fake_hq_single, real_hq_single]

            for k in range(5):
                ax = plt.subplot(gs[k])
                # Ensure images are uint8 for imshow if they are not already
                img_display = images_to_plot[k]
                if img_display.dtype != np.uint8:
                    img_display = np.clip(img_display, 0, 255).astype(np.uint8)
                
                # Handle grayscale (H,W) vs (H,W,1)
                if img_display.ndim == 3 and img_display.shape[2] == 1:
                    ax.imshow(img_display.squeeze(), cmap='gray')
                else:
                    ax.imshow(img_display, cmap='gray') # imshow handles H,W or H,W,3/4
                ax.set_title(titles[k], fontsize=8) # Reduced fontsize for more text
                ax.axis('off')

            plt.suptitle(f'Comparison Image Index: {num_images_processed}', fontsize=11)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            save_path = os.path.join(output_dir, f'comparison_{num_images_processed:04d}.png')
            try:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
            except Exception as e:
                print(f"\nWarning: Failed to save image {save_path}. Error: {e}")
            finally:
                plt.close(fig)

            num_images_processed += 1

    # --- 5. Calculate and Print Average Metrics ---
    if num_images_processed > 0:
        # Métricas originales
        avg_psnr_fr = total_psnr_fake_vs_real / num_images_processed
        avg_ssim_fr = total_ssim_fake_vs_real / num_images_processed
        avg_psnr_fl = total_psnr_fake_vs_lq / num_images_processed
        avg_ssim_fl = total_ssim_fake_vs_lq / num_images_processed
        avg_psnr_rl = total_psnr_real_vs_lq / num_images_processed
        avg_ssim_rl = total_ssim_real_vs_lq / num_images_processed
        
        # Nuevas métricas
        avg_lpips_fr = total_lpips_fake_vs_real / num_images_processed  # Menor es mejor
        avg_msssim_fr = total_msssim_fake_vs_real / num_images_processed  # Mayor es mejor
        avg_lpips_rl = total_lpips_real_vs_lq / num_images_processed  # Baseline LPIPS
        avg_msssim_rl = total_msssim_real_vs_lq / num_images_processed  # Baseline MS-SSIM
        
        # Averages for Bicubic
        avg_psnr_bic_real = total_psnr_bicubic_vs_real / num_images_processed
        avg_ssim_bic_real = total_ssim_bicubic_vs_real / num_images_processed
        avg_lpips_bic_real = total_lpips_bicubic_vs_real / num_images_processed
        avg_msssim_bic_real = total_msssim_bicubic_vs_real / num_images_processed
        
        # Averages for Lanczos
        avg_psnr_lanc_real = total_psnr_lanczos_vs_real / num_images_processed
        avg_ssim_lanc_real = total_ssim_lanczos_vs_real / num_images_processed
        avg_lpips_lanc_real = total_lpips_lanczos_vs_real / num_images_processed
        avg_msssim_lanc_real = total_msssim_lanczos_vs_real / num_images_processed
        
        # Calcular mejoras con respecto al baseline
        psnr_improvement = avg_psnr_fr - avg_psnr_rl  # Diferencia en dB
        psnr_improvement_percent = ((10**(avg_psnr_fr/10)) / (10**(avg_psnr_rl/10)) - 1) * 100  # Mejora porcentual en potencia
        ssim_improvement = avg_ssim_fr - avg_ssim_rl  # Diferencia absoluta
        ssim_improvement_percent = (avg_ssim_fr / avg_ssim_rl - 1) * 100  # Mejora porcentual
        
        # Para LPIPS, menor es mejor, así que la mejora es la reducción
        lpips_improvement = avg_lpips_rl - avg_lpips_fr  # Diferencia absoluta (positiva si hay mejora)
        lpips_improvement_percent = (1 - avg_lpips_fr / avg_lpips_rl) * 100  # % de reducción
        
        # Para MS-SSIM, mayor es mejor
        msssim_improvement = avg_msssim_fr - avg_msssim_rl  # Diferencia absoluta
        msssim_improvement_percent = (avg_msssim_fr / avg_msssim_rl - 1) * 100  # Mejora porcentual

        print("\n--------------------------------------------------")
        print("Evaluation Complete")
        print(f"Checkpoint evaluated: {checkpoint_g_a2b_path}")
        print(f"Total image pairs evaluated: {num_images_processed}")
        
        print("\nAverage Metrics (Fake HQ vs Real HQ):")
        print(f"  PSNR:    {avg_psnr_fr:.4f} dB")
        print(f"  SSIM:    {avg_ssim_fr:.4f}")
        print(f"  LPIPS:   {avg_lpips_fr:.4f} (lower is better)")
        print(f"  MS-SSIM: {avg_msssim_fr:.4f} (higher is better)")
        
        print("\nAverage Metrics (Fake HQ vs Original LQ):")
        print(f"  PSNR: {avg_psnr_fl:.4f} dB")
        print(f"  SSIM: {avg_ssim_fl:.4f}")
        
        print("\nAverage Metrics (Real HQ vs Original LQ - Baseline):")
        print(f"  PSNR:    {avg_psnr_rl:.4f} dB")
        print(f"  SSIM:    {avg_ssim_rl:.4f}")
        print(f"  LPIPS:   {avg_lpips_rl:.4f} (lower is better)")
        print(f"  MS-SSIM: {avg_msssim_rl:.4f} (higher is better)")
        
        print("\nAverage Metrics (Bicubic HQ vs Real HQ):")
        print(f"  PSNR:    {avg_psnr_bic_real:.4f} dB")
        print(f"  SSIM:    {avg_ssim_bic_real:.4f}")
        print(f"  LPIPS:   {avg_lpips_bic_real:.4f} (lower is better)")
        print(f"  MS-SSIM: {avg_msssim_bic_real:.4f} (higher is better)")
        
        print("\nAverage Metrics (Lanczos HQ vs Real HQ):")
        print(f"  PSNR:    {avg_psnr_lanc_real:.4f} dB")
        print(f"  SSIM:    {avg_ssim_lanc_real:.4f}")
        print(f"  LPIPS:   {avg_lpips_lanc_real:.4f} (lower is better)")
        print(f"  MS-SSIM: {avg_msssim_lanc_real:.4f} (higher is better)")
        
        print("\nMEJORA SOBRE EL BASELINE:")
        print(f"  PSNR:    +{psnr_improvement:.4f} dB ({psnr_improvement_percent:.2f}% mejor)")
        print(f"  SSIM:    +{ssim_improvement:.4f} ({ssim_improvement_percent:.2f}% mejor)")
        print(f"  LPIPS:   -{lpips_improvement:.4f} ({lpips_improvement_percent:.2f}% mejor)")  # Signo negativo para mostrar la reducción
        print(f"  MS-SSIM: +{msssim_improvement:.4f} ({msssim_improvement_percent:.2f}% mejor)")
        print("--------------------------------------------------")
        
        # Guardar todos los resultados en un archivo de texto
        results_path = os.path.join(output_dir, 'metrics_summary.txt')
        try:
            with open(results_path, 'w') as f:
                f.write("EVALUATION METRICS SUMMARY\n")
                f.write(f"Checkpoint: {checkpoint_g_a2b_path}\n")
                f.write(f"Total image pairs: {num_images_processed}\n\n")
                
                f.write("Fake HQ vs Real HQ:\n")
                f.write(f"  PSNR:    {avg_psnr_fr:.4f} dB\n")
                f.write(f"  SSIM:    {avg_ssim_fr:.4f}\n")
                f.write(f"  LPIPS:   {avg_lpips_fr:.4f} (lower is better)\n")
                f.write(f"  MS-SSIM: {avg_msssim_fr:.4f} (higher is better)\n\n")
                
                f.write("Fake HQ vs Original LQ:\n")
                f.write(f"  PSNR:    {avg_psnr_fl:.4f} dB\n")
                f.write(f"  SSIM:    {avg_ssim_fl:.4f}\n\n")
                
                f.write("Real HQ vs Original LQ (Baseline):\n")
                f.write(f"  PSNR:    {avg_psnr_rl:.4f} dB\n")
                f.write(f"  SSIM:    {avg_ssim_rl:.4f}\n")
                f.write(f"  LPIPS:   {avg_lpips_rl:.4f} (lower is better)\n")
                f.write(f"  MS-SSIM: {avg_msssim_rl:.4f} (higher is better)\n\n")
                
                f.write("Average Metrics (Bicubic HQ vs Real HQ):\n")
                f.write(f"  PSNR:    {avg_psnr_bic_real:.4f} dB\n")
                f.write(f"  SSIM:    {avg_ssim_bic_real:.4f}\n")
                f.write(f"  LPIPS:   {avg_lpips_bic_real:.4f} (lower is better)\n")
                f.write(f"  MS-SSIM: {avg_msssim_bic_real:.4f} (higher is better)\n\n")
                
                f.write("Average Metrics (Lanczos HQ vs Real HQ):\n")
                f.write(f"  PSNR:    {avg_psnr_lanc_real:.4f} dB\n")
                f.write(f"  SSIM:    {avg_ssim_lanc_real:.4f}\n")
                f.write(f"  LPIPS:   {avg_lpips_lanc_real:.4f} (lower is better)\n")
                f.write(f"  MS-SSIM: {avg_msssim_lanc_real:.4f} (higher is better)\n\n")
                
                f.write("MEJORA SOBRE EL BASELINE (Generated HQ vs Real HQ compared to Real HQ vs Original LQ):\n")
                f.write(f"  PSNR:    +{psnr_improvement:.4f} dB ({psnr_improvement_percent:.2f}% mejor)\n")
                f.write(f"  SSIM:    +{ssim_improvement:.4f} ({ssim_improvement_percent:.2f}% mejor)\n")
                f.write(f"  LPIPS:   -{lpips_improvement:.4f} ({lpips_improvement_percent:.2f}% mejor)\n")
                f.write(f"  MS-SSIM: +{msssim_improvement:.4f} ({msssim_improvement_percent:.2f}% mejor)\n")
            print(f"Results saved to {results_path}")
        except Exception as e:
            print(f"Warning: Failed to save metrics summary. Error: {e}")
    else:
        print("\nNo images were processed during evaluation.")

    print("--- Evaluation Finished ---")


if __name__ == "__main__":
    evaluate_model() 