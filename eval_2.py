# src/evaluate.py (ACTUALIZADO con LPIPS y MS-SSIM)

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import lpips  # Nueva importación para LPIPS
from pytorch_msssim import ms_ssim  # Nueva importación para MS-SSIM

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
            # LPIPS espera valores en el rango [-1, 1], verificar si necesitan ser normalizados
            lpips_value_fr = lpips_model(img_real_hq_batch, img_fake_hq_batch).mean()
            total_lpips_fake_vs_real += lpips_value_fr.item()
            
            # También calcular LPIPS entre real y LQ (baseline)
            lpips_value_rl = lpips_model(img_real_hq_batch, img_lq_batch).mean()
            total_lpips_real_vs_lq += lpips_value_rl.item()
            
            # --- Calcular MS-SSIM (funciona con tensores en [0, 1]) ---
            # Normalizar si es necesario (asumiendo que ya están en [0, 1])
            # Si los tensores están en [-1, 1], usar: (tensor + 1) / 2
            if img_real_hq_batch.min() < 0:
                norm_real = (img_real_hq_batch + 1) / 2
                norm_fake = (img_fake_hq_batch + 1) / 2
                norm_lq = (img_lq_batch + 1) / 2
            else:
                norm_real = img_real_hq_batch
                norm_fake = img_fake_hq_batch
                norm_lq = img_lq_batch
                
            # MS-SSIM requiere imágenes con el mismo tamaño y valores en [0, 1]
            msssim_value_fr = ms_ssim(norm_real, norm_fake, data_range=1.0)
            total_msssim_fake_vs_real += msssim_value_fr.item()
            
            # También calcular MS-SSIM entre real y LQ (baseline)
            msssim_value_rl = ms_ssim(norm_real, norm_lq, data_range=1.0)
            total_msssim_real_vs_lq += msssim_value_rl.item()

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

            # Calculate metrics
            data_range = 255  # For uint8 images

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

            # Obtener valores individuales de LPIPS y MS-SSIM para esta imagen
            lpips_val_fr = lpips_value_fr.item() if current_batch_size == 1 else lpips_value_fr.item() / current_batch_size
            msssim_val_fr = msssim_value_fr.item() if current_batch_size == 1 else msssim_value_fr.item() / current_batch_size
            lpips_val_rl = lpips_value_rl.item() if current_batch_size == 1 else lpips_value_rl.item() / current_batch_size
            msssim_val_rl = msssim_value_rl.item() if current_batch_size == 1 else msssim_value_rl.item() / current_batch_size

            # --- Save comparison image con métricas ampliadas ---
            fig = plt.figure(figsize=(14, 5))  # Aumentado para incluir más métricas
            gs = gridspec.GridSpec(1, 3, wspace=0.05, hspace=0.05)

            titles = [
                f'Input LQ\n(PSNR vs Real: {psnr_rl:.2f} dB | SSIM: {ssim_rl:.3f})\n(LPIPS: {lpips_val_rl:.3f} | MS-SSIM: {msssim_val_rl:.3f})',
                f'Generated HQ\n(PSNR: {psnr_fr:.2f} dB | SSIM: {ssim_fr:.3f})\n(LPIPS: {lpips_val_fr:.3f} | MS-SSIM: {msssim_val_fr:.3f})',
                f'Real HQ'
            ]
            images = [lq_single, fake_hq_single, real_hq_single]

            for k in range(3):
                ax = plt.subplot(gs[k])
                ax.imshow(images[k], cmap='gray')
                ax.set_title(titles[k], fontsize=9)
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
                
                f.write("MEJORA SOBRE EL BASELINE:\n")
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
