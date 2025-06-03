import os
from PIL import Image

def split_image(image_path, output_dir1, output_dir2, output_dir3):
    """Splits an image into three equal horizontal parts and saves them."""
    try:
        img = Image.open(image_path)
        width, height = img.size
        crop_width = width // 3
        
        # Calculate the height to crop, removing the top portion with text
        # Assuming the text part is roughly 22% of the total height.
        # This might need adjustment if the text height varies significantly.
        crop_y_offset = int(height * 0.22)

        inner_lr_margin = 15 # Pixels to crop from left and right of each sub-ima
        inner_bottom_margin = 15 # Pixels to crop from the bottom of each sub-image

        # Crop 1
        img1_x0 = inner_lr_margin
        img1_y0 = crop_y_offset
        img1_x1 = crop_width - inner_lr_margin
        img1_y1 = height - inner_bottom_margin
        img1 = img.crop((img1_x0, img1_y0, img1_x1, img1_y1))
        
        # Crop 2
        img2_x0 = crop_width + inner_lr_margin
        img2_y0 = crop_y_offset
        img2_x1 = crop_width * 2 - inner_lr_margin
        img2_y1 = height - inner_bottom_margin
        img2 = img.crop((img2_x0, img2_y0, img2_x1, img2_y1))
        
        # Crop 3
        img3_x0 = crop_width * 2 + inner_lr_margin
        img3_y0 = crop_y_offset
        img3_x1 = width - inner_lr_margin # Use total original width for the rightmost boundary
        img3_y1 = height - inner_bottom_margin
        img3 = img.crop((img3_x0, img3_y0, img3_x1, img3_y1))

        base_filename = os.path.basename(image_path)
        name, ext = os.path.splitext(base_filename)

        img1.save(os.path.join(output_dir1, f"{name}_1{ext}"))
        img2.save(os.path.join(output_dir2, f"{name}_2{ext}"))
        img3.save(os.path.join(output_dir3, f"{name}_3{ext}"))
        print(f"Successfully split {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    source_dir = "."  # Current directory
    output_dir_image1 = "image1"
    output_dir_image2 = "image2"
    output_dir_image3 = "image3"

    # Create output directories if they don't exist
    os.makedirs(output_dir_image1, exist_ok=True)
    os.makedirs(output_dir_image2, exist_ok=True)
    os.makedirs(output_dir_image3, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.startswith("comparison_") and filename.endswith(".png"):
            image_path = os.path.join(source_dir, filename)
            split_image(image_path, output_dir_image1, output_dir_image2, output_dir_image3)

    print("Image splitting process complete.") 