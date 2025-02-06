import os
import numpy as np
import cv2
import tifffile as tiff

def images_to_tif(image_dir, output_tif_path, mode='stack'):
    """
    Combines multiple images into a single multi-page or multi-channel TIF file.

    Args:
        image_dir (str): Directory containing the image files.
        output_tif_path (str): Path to save the output TIF file.
        mode (str): 'stack' to create multi-page TIF (T, H, W), 'channels' to create multi-channel TIF (H, W, C).

    Returns:
        None
    """
    # Get sorted list of images
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))])

    assert len(image_files) > 0, "No images found in the directory!"

    images = []

    for img_file in image_files:
        # Load image in grayscale mode
        img = cv2.imread(os.path.join(image_dir, img_file), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Skipping {img_file}, unable to read!")
            continue
        
        images.append(img)

    # Convert list to numpy array
    images = np.array(images, dtype=np.uint8)  # Shape: (T, H, W)

    if mode == 'channels':
        # Stack images along the channel dimension (H, W, T)
        images = np.moveaxis(images, 0, -1)  # Change (T, H, W) → (H, W, C)

    # Save as a multi-page TIF
    tiff.imwrite(output_tif_path, images, imagej=True)

    print(f"Saved TIF file at {output_tif_path} with shape {images.shape}")

# Example usage:
images_to_tif(
    image_dir="benchmark_predictions_spine",
    output_tif_path="output_spine.tif",
    mode='stack'  # Change to 'channels' if you want multi-channel
)
