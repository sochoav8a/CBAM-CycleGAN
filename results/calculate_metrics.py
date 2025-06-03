import os
import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import lpips
import torchvision.transforms as transforms

def ensure_rgb(img):
    """Ensure image is in RGB format and a NumPy array."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)

# Function to prepare image for LPIPS
def prepare_img_for_lpips(img_pil, device='cpu'):
    """Converts a PIL image to a PyTorch tensor for LPIPS."""
    # LPIPS expects image in range [0,1] initially, then it internally normalizes to [-1,1]
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts PIL image (H, W, C) in range [0, 255] to (C, H, W) in range [0.0, 1.0]
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # LPIPS model does its own normalization
    ])
    tensor = transform(img_pil).unsqueeze(0) # Add batch dimension
    return tensor.to(device)

def calculate_metrics_for_pair(img1_path, img2_path, lpips_model=None, device='cpu'):
    """Calculates PSNR, SSIM, and LPIPS for a pair of images."""
    try:
        img1_pil = Image.open(img1_path)
        img2_pil = Image.open(img2_path)

        # Ensure img1 is resized to img2's size if they are different
        # This is crucial for metrics "vs LR" where img2 is LR.
        if img1_pil.size != img2_pil.size:
            # print(f"Warning: Images {os.path.basename(img1_path)} ({img1_pil.size}) and {os.path.basename(img2_path)} ({img2_pil.size}) have different sizes for metric calculation. Resizing {os.path.basename(img1_path)} to match {os.path.basename(img2_path)}.")
            img1_pil = img1_pil.resize(img2_pil.size, Image.Resampling.BICUBIC)

        img1_np = ensure_rgb(img1_pil)
        img2_np = ensure_rgb(img2_pil)

        # Calculate PSNR
        # data_range is max_pixel_value - min_pixel_value. For uint8 images, this is 255.
        current_psnr = psnr(img1_np, img2_np, data_range=255)

        # Calculate SSIM
        # For multichannel (RGB) images, set multichannel=True.
        # data_range is max_pixel_value - min_pixel_value. For uint8 images, this is 255.
        current_ssim = ssim(img1_np, img2_np, multichannel=True, data_range=255, channel_axis=-1)
        
        current_lpips = None
        if lpips_model:
            # Ensure PIL images are RGB before converting to tensor
            img1_pil_rgb = ensure_pil_rgb(img1_pil)
            img2_pil_rgb = ensure_pil_rgb(img2_pil)
            
            img1_tensor = prepare_img_for_lpips(img1_pil_rgb, device=device)
            img2_tensor = prepare_img_for_lpips(img2_pil_rgb, device=device)
            with torch.no_grad():
                current_lpips = lpips_model(img1_tensor, img2_tensor).item()

        return current_psnr, current_ssim, current_lpips

    except FileNotFoundError:
        print(f"Error: One or both image files not found for metric calculation: {img1_path}, {img2_path}")
        return None, None, None
    except Exception as e:
        print(f"Error calculating metrics for {img1_path} and {img2_path}: {e}")
        return None, None, None

# Helper to ensure PIL image is RGB before tensor conversion for LPIPS
def ensure_pil_rgb(img_pil):
    if img_pil.mode != 'RGB':
        return img_pil.convert('RGB')
    return img_pil

def upscale_image(image_path, target_size, method=Image.Resampling.BICUBIC):
    """Upscales an image to the target size using the specified method."""
    try:
        img = Image.open(image_path)
        img_upscaled = img.resize(target_size, method)
        return img_upscaled
    except FileNotFoundError:
        print(f"Error: Image file not found for upscaling: {image_path}")
        return None
    except Exception as e:
        print(f"Error upscaling {image_path}: {e}")
        return None

def get_font(font_name_list, size):
    """Tries to load a font from a list of names, falling back to default."""
    for font_name in font_name_list:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
    return ImageFont.load_default() # Returns a default bitmap font if no TTF found

def create_compilation_image(base_name, lr_path, g_path, hr_path, 
                               temp_bicubic_path, temp_lanczos_path, 
                               metrics_vs_lr_list, output_dir="compilation"):
    """Creates and saves the 5-image compilation with metrics."""
    try:
        lr_orig_pil = ensure_pil_rgb(Image.open(lr_path))
        g_orig_pil = ensure_pil_rgb(Image.open(g_path))
        hr_pil = ensure_pil_rgb(Image.open(hr_path))
        bic_hrsize_pil = ensure_pil_rgb(Image.open(temp_bicubic_path))
        lan_hrsize_pil = ensure_pil_rgb(Image.open(temp_lanczos_path))

        hr_size = hr_pil.size
        img_width, img_height = hr_size

        # Prepare display images (all at HR size)
        lr_display_pil = lr_orig_pil.resize(hr_size, Image.Resampling.BICUBIC)
        g_display_pil = g_orig_pil.resize(hr_size, Image.Resampling.BICUBIC) if g_orig_pil.size != hr_size else g_orig_pil
        
        display_images = [lr_display_pil, bic_hrsize_pil, lan_hrsize_pil, g_display_pil, hr_pil]

        text_labels = ["Input LR"] + [m[0] for m in metrics_vs_lr_list]
        metric_texts = [""] 
        for _, psnr_val, ssim_val, lpips_val in metrics_vs_lr_list:
            psnr_str = f"{psnr_val:.2f}" if psnr_val is not None and not np.isnan(psnr_val) else "N/A"
            ssim_str = f"{ssim_val:.4f}" if ssim_val is not None and not np.isnan(ssim_val) else "N/A"
            lpips_str = f"{lpips_val:.4f}" if lpips_val is not None and not np.isnan(lpips_val) else "N/A"
            metric_texts.append(f"PSNR: {psnr_str}\nSSIM: {ssim_str}\nLPIPS: {lpips_str}")

        font_candidates = ["DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"]
        title_font_size = 18 
        metric_font_size = 14
        title_font = get_font(font_candidates, title_font_size)
        metric_font = get_font(font_candidates, metric_font_size)
        
        text_area_height = 95 
        top_padding = 10
        image_top_offset = text_area_height 

        total_width = img_width * 5
        total_height = image_top_offset + img_height + top_padding # Add bottom padding

        composite_img = Image.new('RGB', (total_width, total_height), color='white')
        draw = ImageDraw.Draw(composite_img)

        for i, img_to_paste in enumerate(display_images):
            x_offset = i * img_width
            
            title_text = text_labels[i]
            try:
                # For Pillow 9.0.0+ textbbox is preferred
                # Check if textbbox is available
                if hasattr(draw, 'textbbox'):
                    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
                    title_w = title_bbox[2] - title_bbox[0]
                else: # Fallback for older Pillow versions (before 9.0.0)
                    title_w, _ = draw.textsize(title_text, font=title_font) # Deprecated
            except Exception: # Broad exception if font loading itself failed earlier and it's a basic font
                title_w, _ = draw.textsize(title_text, font=title_font) 
            
            text_x_centered = x_offset + (img_width - title_w) / 2
            draw.text((text_x_centered, top_padding), title_text, fill="black", font=title_font)

            if i > 0: 
                metric_text_content = metric_texts[i]
                metric_y_start = top_padding + title_font_size + 7 
                draw.multiline_text((x_offset + 10, metric_y_start), metric_text_content, fill="black", font=metric_font, spacing=4)
            
            composite_img.paste(img_to_paste, (x_offset, image_top_offset))

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{base_name}_benchmark_styled.png") 
        composite_img.save(save_path)
        print(f"Saved styled compilation: {save_path}")

    except FileNotFoundError as e:
        print(f"Error creating styled compilation for {base_name}: File not found - {e}")
    except Exception as e:
        print(f"Error creating styled compilation for {base_name}: {e} (Line: {e.__traceback__.tb_lineno if e.__traceback__ else 'N/A'})")

def main():
    hr_dir = "HR"
    lr_dir = "LR"
    g_dir = "G"
    compilation_output_dir = "compilation"
    os.makedirs(compilation_output_dir, exist_ok=True)
    
    lr_bicubic_dir = "LR_bicubic_upscaled"
    lr_lanczos_dir = "LR_lanczos_upscaled"
    os.makedirs(lr_bicubic_dir, exist_ok=True)
    os.makedirs(lr_lanczos_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for LPIPS: {device}")
    try:
        lpips_alex = lpips.LPIPS(net='alex').to(device)
        lpips_alex.eval()
    except Exception as e:
        print(f"Error loading LPIPS model: {e}. LPIPS scores will be N/A.")
        lpips_alex = None

    print(f"\nMetrics vs HR (High Resolution):")
    print(f"{'Image':<30} | {'Method':<15} | {'PSNR':<10} | {'SSIM':<10} | {'LPIPS':<10}")
    print("-" * 80)

    hr_files = sorted(glob.glob(os.path.join(hr_dir, "*_3.png")))
    if not hr_files:
        print(f"No HR images found in {hr_dir} with the pattern *_3.png.")
        return

    for hr_path in hr_files:
        base_name = os.path.basename(hr_path).replace("_3.png", "")
        lr_path = os.path.join(lr_dir, f"{base_name}_1.png")
        g_path = os.path.join(g_dir, f"{base_name}_2.png")

        if not (os.path.exists(lr_path) and os.path.exists(g_path) and os.path.exists(hr_path)):
            print(f"Skipping {base_name}: One or more source files (LR, G, HR) missing.")
            continue
        
        print(f"Processing: {base_name}")
        try:
            hr_img_pil_check = Image.open(hr_path)
            hr_size_check = hr_img_pil_check.size
        except Exception as e:
            print(f"Error opening HR image {hr_path} for size check: {e}. Skipping.")
            continue

        # --- Metrics vs HR (for console output) ---
        psnr_g_hr, ssim_g_hr, lpips_g_hr = calculate_metrics_for_pair(g_path, hr_path, lpips_alex, device)
        if psnr_g_hr is not None:
            lpips_g_hr_str = f"{lpips_g_hr:<10.4f}" if lpips_g_hr is not None else "N/A"
            print(f"{base_name:<30} | {'Generated (G)':<15} | {psnr_g_hr:<10.2f} | {ssim_g_hr:<10.4f} | {lpips_g_hr_str:<10}")

        temp_bicubic_path = os.path.join(lr_bicubic_dir, f"{base_name}_1_bicubic.png")
        lr_bicubic_pil = upscale_image(lr_path, hr_size_check, method=Image.Resampling.BICUBIC)
        if lr_bicubic_pil:
            lr_bicubic_pil.save(temp_bicubic_path)
            psnr_bic_hr, ssim_bic_hr, lpips_bic_hr = calculate_metrics_for_pair(temp_bicubic_path, hr_path, lpips_alex, device)
            if psnr_bic_hr is not None:
                lpips_bic_hr_str = f"{lpips_bic_hr:<10.4f}" if lpips_bic_hr is not None else "N/A"
                print(f"{base_name:<30} | {'LR_Bicubic (1)':<15} | {psnr_bic_hr:<10.2f} | {ssim_bic_hr:<10.4f} | {lpips_bic_hr_str:<10}")
        else: 
            print(f"Skipping Bicubic metrics for {base_name} due to upscaling failure.")
            continue 

        temp_lanczos_path = os.path.join(lr_lanczos_dir, f"{base_name}_1_lanczos.png")
        lr_lanczos_pil = upscale_image(lr_path, hr_size_check, method=Image.Resampling.LANCZOS)
        if lr_lanczos_pil:
            lr_lanczos_pil.save(temp_lanczos_path)
            psnr_lan_hr, ssim_lan_hr, lpips_lan_hr = calculate_metrics_for_pair(temp_lanczos_path, hr_path, lpips_alex, device)
            if psnr_lan_hr is not None:
                lpips_lan_hr_str = f"{lpips_lan_hr:<10.4f}" if lpips_lan_hr is not None else "N/A"
                print(f"{base_name:<30} | {'LR_Lanczos (1)':<15} | {psnr_lan_hr:<10.2f} | {ssim_lan_hr:<10.4f} | {lpips_lan_hr_str:<10}")
        else: 
            print(f"Skipping Lanczos metrics for {base_name} due to upscaling failure.")
            continue 
        print("-" * 80) 

        # --- Metrics vs LR (for compilation image) ---
        metrics_vs_lr_list = []
        # Ensure temp_bicubic_path and temp_lanczos_path exist before trying to use them for metrics
        methods_for_compilation = []
        if os.path.exists(temp_bicubic_path):
            methods_for_compilation.append(("Bicubic (vs LR)", temp_bicubic_path))
        else:
            metrics_vs_lr_list.append(("Bicubic (vs LR)", float('nan'), float('nan'), float('nan')))
            print(f"Cannot calculate Bicubic vs LR for {base_name}, upscaled file missing.")

        if os.path.exists(temp_lanczos_path):
            methods_for_compilation.append(("Lanczos (vs LR)", temp_lanczos_path))
        else:
            metrics_vs_lr_list.append(("Lanczos (vs LR)", float('nan'), float('nan'), float('nan')))
            print(f"Cannot calculate Lanczos vs LR for {base_name}, upscaled file missing.")
        
        methods_for_compilation.extend([
            ("Generated (vs LR)", g_path),
            ("HR (vs LR)", hr_path)
        ])

        for method_label, img1_path_for_metric in methods_for_compilation:
            # This loop now only processes valid paths from methods_for_compilation
            # For those that were pre-emptively added as NaN, they remain so.
            if not os.path.exists(img1_path_for_metric):
                 # This case should ideally be caught by the specific checks above
                if not any(m[0] == method_label for m in metrics_vs_lr_list):
                     metrics_vs_lr_list.append((method_label, float('nan'), float('nan'), float('nan')))
                continue

            psnr_val, ssim_val, lpips_val = calculate_metrics_for_pair(img1_path_for_metric, lr_path, lpips_alex, device)
            if psnr_val is not None:
                metrics_vs_lr_list.append((method_label, psnr_val, ssim_val, lpips_val))
            else:
                # Ensure a placeholder if calculate_metrics_for_pair returns None for all
                if not any(m[0] == method_label for m in metrics_vs_lr_list):
                    metrics_vs_lr_list.append((method_label, float('nan'), float('nan'), float('nan')))
        
        # Sort metrics_vs_lr_list to ensure consistent order for compilation if paths were skipped
        desired_order = ["Bicubic (vs LR)", "Lanczos (vs LR)", "Generated (vs LR)", "HR (vs LR)"]
        sorted_metrics = []
        for label in desired_order:
            found = False
            for m_label, m_psnr, m_ssim, m_lpips in metrics_vs_lr_list:
                if m_label == label:
                    sorted_metrics.append((m_label, m_psnr, m_ssim, m_lpips))
                    found = True
                    break
            if not found:
                 # This means it was never successfully calculated or path was bad
                sorted_metrics.append((label, float('nan'), float('nan'), float('nan')))
        metrics_vs_lr_list = sorted_metrics

        # --- Create and save compilation image ---
        if len(metrics_vs_lr_list) == 4: 
            create_compilation_image(base_name, lr_path, g_path, hr_path, 
                                   temp_bicubic_path, temp_lanczos_path, 
                                   metrics_vs_lr_list, output_dir=compilation_output_dir)
        else:
            print(f"Skipping compilation for {base_name} due to incomplete metrics vs LR (expected 4, got {len(metrics_vs_lr_list)}).")
            # print(f"Collected metrics for compilation: {metrics_vs_lr_list}") # Debugging line

        print("=" * 80) 

if __name__ == "__main__":
    main() 