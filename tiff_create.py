#!/usr/bin/env python3
"""
Script to generate TIF stacks from mask images and create overlay visualizations
for all experiments in the results_sweep directory.

Usage:
    python generate_tiffs_and_overlays.py
"""
import os
import numpy as np
from glob import glob
from PIL import Image
import tifffile
import matplotlib.pyplot as plt
import cv2
import re
from tqdm import tqdm

def load_images_from_folder(folder_path, extensions=('png', 'tif', 'jpg', 'jpeg')):
    """
    Load and sort image file paths from a folder.
    
    Args:
        folder_path (str): Path to folder containing images.
        extensions (tuple): Allowed file extensions.
    
    Returns:
        List[str]: Sorted list of file paths.
    """
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(folder_path, f"*.{ext}")))
    return sorted(files)

def create_tiff_stack(image_paths, output_path):
    """
    Create a multi-page TIFF stack from a list of image files.
    
    Args:
        image_paths (List[str]): List of paths to images.
        output_path (str): Path to save the output TIFF stack.
    """
    # Load first image to get dimensions
    if not image_paths:
        print(f"Warning: No images found for {output_path}")
        return False
        
    sample_img = np.array(Image.open(image_paths[0]))
    
    # Prepare array for stack
    stack = np.zeros((len(image_paths), *sample_img.shape), dtype=np.uint8)
    
    # Load all images into stack
    for i, img_path in enumerate(image_paths):
        img = np.array(Image.open(img_path))
        stack[i] = img
    
    # Save as TIFF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(output_path, stack)
    return True

def create_overlay_images(gt_path, pred_folder, output_folder, experiment_name):
    """
    Create overlay images showing ground truth vs. predictions.
    
    Args:
        gt_path (str): Path to ground truth TIFF stack.
        pred_folder (str): Path to folder with prediction mask images.
        output_folder (str): Path to save the overlay images.
        experiment_name (str): Name of the experiment.
    """
    # Load ground truth stack
    gt_stack = tifffile.imread(gt_path)
    
    # Load predicted masks
    pred_paths = load_images_from_folder(pred_folder)
    
    # Check if we have predictions
    if not pred_paths:
        print(f"Warning: No mask images found in {pred_folder}")
        return False
    
    # Create output directory
    overlay_output = os.path.join(output_folder, experiment_name, "overlays")
    os.makedirs(overlay_output, exist_ok=True)
    
    # Create overlays for each slice
    num_slices = min(len(pred_paths), gt_stack.shape[0])
    
    for i in range(num_slices):
        # Get ground truth slice
        gt_slice = gt_stack[i].astype(np.uint8)
        gt_binary = gt_slice > 0
        
        # Get prediction slice
        pred_img = np.array(Image.open(pred_paths[i]))
        pred_binary = pred_img > 128
        
        # Create RGB overlay
        overlay = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
        
        # Green: True Positives (TP)
        overlay[gt_binary & pred_binary] = [0, 255, 0]
        
        # Red: False Negatives (FN) - in GT but not predicted
        overlay[gt_binary & ~pred_binary] = [255, 0, 0]
        
        # Blue: False Positives (FP) - predicted but not in GT
        overlay[~gt_binary & pred_binary] = [0, 0, 255]
        
        # Save overlay
        cv2.imwrite(os.path.join(overlay_output, f"overlay_slice_{i:03d}.png"), 
                   overlay)
    
    # Create a summary image with colored legend
    legend_img = np.zeros((100, 300, 3), dtype=np.uint8)
    # Add color keys with text
    legend_img[20:40, 10:30] = [0, 255, 0]  # Green
    legend_img[50:70, 10:30] = [255, 0, 0]  # Red
    legend_img[80:100, 10:30] = [0, 0, 255]  # Blue
    
    # Add text using OpenCV
    cv2.putText(legend_img, "True Positive", (40, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(legend_img, "False Negative (missed)", (40, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(legend_img, "False Positive (extra)", (40, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save legend
    cv2.imwrite(os.path.join(overlay_output, "legend.png"), legend_img)
    
    # Also create a collage of a few sample slices for quick reference
    create_sample_collage(gt_stack, pred_paths, overlay_output, experiment_name, samples=4)
    
    return True

def create_sample_collage(gt_stack, pred_paths, output_folder, experiment_name, samples=4):
    """
    Create a collage of sample slices with original, GT, prediction, and overlay.
    
    Args:
        gt_stack (ndarray): Ground truth stack.
        pred_paths (List[str]): List of prediction image paths.
        output_folder (str): Output folder path.
        experiment_name (str): Name of the experiment.
        samples (int): Number of sample slices to include.
    """
    # Determine slices to sample
    num_slices = min(gt_stack.shape[0], len(pred_paths))
    if num_slices < samples:
        samples = num_slices
    
    indices = np.linspace(0, num_slices-1, samples, dtype=int)
    
    # Create a large figure
    fig, axes = plt.subplots(samples, 4, figsize=(16, 4*samples))
    
    # Set title for the entire figure
    fig.suptitle(f"Sample Results for {experiment_name}", fontsize=16)
    
    for row, idx in enumerate(indices):
        # Original image (assuming original is in first channel of GT if multichannel)
        if gt_stack.ndim > 3:
            original = gt_stack[idx, 0]
        else:
            original = gt_stack[idx]
        
        # Ground truth mask
        gt_mask = gt_stack[idx] > 0
        
        # Prediction mask
        pred_img = np.array(Image.open(pred_paths[idx]))
        pred_mask = pred_img > 128
        
        # Create overlay
        overlay = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        # Green: TP, Red: FN, Blue: FP
        overlay[gt_mask & pred_mask] = [0, 255, 0]
        overlay[gt_mask & ~pred_mask] = [255, 0, 0]
        overlay[~gt_mask & pred_mask] = [0, 0, 255]
        
        # Plot in respective columns
        if samples > 1:
            ax1, ax2, ax3, ax4 = axes[row]
        else:
            ax1, ax2, ax3, ax4 = axes  # Handle case with only one sample
        
        # Original
        ax1.imshow(original, cmap='gray')
        ax1.set_title(f"Original {idx}")
        ax1.axis('off')
        
        # Ground truth
        ax2.imshow(gt_mask, cmap='gray')
        ax2.set_title(f"Ground Truth {idx}")
        ax2.axis('off')
        
        # Prediction
        ax3.imshow(pred_mask, cmap='gray')
        ax3.set_title(f"Prediction {idx}")
        ax3.axis('off')
        
        # Overlay
        ax4.imshow(overlay)
        ax4.set_title(f"Overlay {idx}")
        ax4.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the title
    plt.savefig(os.path.join(output_folder, f"sample_collage.png"), dpi=200, bbox_inches='tight')
    plt.close()

def parse_experiment_params(exp_name):
    """
    Parse patch size and overlap from experiment folder name.
    
    Args:
        exp_name (str): Experiment folder name like "128px_50ov"
        
    Returns:
        tuple: (patch_size, overlap_percent)
    """
    match = re.match(r"(\d+)px_(\d+)ov", exp_name)
    if match:
        patch_size = int(match.group(1))
        overlap = int(match.group(2))
        return patch_size, overlap
    return None, None

def main():
    # Path to ground truth
    gt_path = "Spine_U.tif"
    
    # Base directory for experiment results
    results_base_dir = "results_sweep"
    
    # Output directory for generated files
    output_base_dir = "tiff_outputs"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Collect experiment directories
    experiment_dirs = [d for d in os.listdir(results_base_dir) 
                     if os.path.isdir(os.path.join(results_base_dir, d))]
    
    print(f"Found {len(experiment_dirs)} experiment folders to process")
    
    # Processed indicators for summary
    tiff_generated = []
    overlay_generated = []
    
    # Process each experiment
    for exp_name in tqdm(sorted(experiment_dirs), desc="Processing experiments"):
        masks_dir = os.path.join(results_base_dir, exp_name, "masks")
        
        # Check if masks directory exists
        if not os.path.isdir(masks_dir):
            print(f"Skipping {exp_name}: No masks subfolder found")
            continue
        
        # Parse patch size and overlap from folder name
        patch_size, overlap = parse_experiment_params(exp_name)
        if patch_size is None:
            print(f"Skipping {exp_name}: Could not parse parameters")
            continue
        
        print(f"\nProcessing: {exp_name} (patch_size={patch_size}, overlap={overlap}%)")
        
        # 1. Create TIFF stack from masks
        mask_files = load_images_from_folder(masks_dir)
        tiff_output_path = os.path.join(output_base_dir, exp_name, f"{exp_name}_stack.tif")
        
        if create_tiff_stack(mask_files, tiff_output_path):
            print(f"✅ Created TIFF stack: {tiff_output_path}")
            tiff_generated.append(exp_name)
        else:
            print(f"❌ Failed to create TIFF stack for {exp_name}")
        
        # 2. Create overlay images for visual comparison
        if create_overlay_images(gt_path, masks_dir, output_base_dir, exp_name):
            print(f"✅ Created overlay images in: {os.path.join(output_base_dir, exp_name, 'overlays')}")
            overlay_generated.append(exp_name)
        else:
            print(f"❌ Failed to create overlays for {exp_name}")
    
    # Print summary
    print("\n===== PROCESSING SUMMARY =====")
    print(f"Total experiments: {len(experiment_dirs)}")
    print(f"Generated TIFF stacks: {len(tiff_generated)}")
    print(f"Generated overlays: {len(overlay_generated)}")
    
    if len(tiff_generated) < len(experiment_dirs) or len(overlay_generated) < len(experiment_dirs):
        print("\nThe following experiments had issues:")
        for exp in experiment_dirs:
            if exp not in tiff_generated:
                print(f"- {exp}: TIFF generation failed")
            if exp not in overlay_generated:
                print(f"- {exp}: Overlay generation failed")
    
    print("\nAll files have been saved to the following directory:")
    print(os.path.abspath(output_base_dir))
    
    # Generate a final report with links to all the outputs
    create_index_html(output_base_dir, experiment_dirs, tiff_generated, overlay_generated)
    print(f"\n✅ Created index.html in {output_base_dir} for easy navigation of results")

def create_index_html(output_dir, experiment_dirs, tiff_generated, overlay_generated):
    """Create an HTML index file to navigate all outputs."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Spine Segmentation Experiment Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            .experiment { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .experiment h2 { color: #3498db; }
            .links { margin-left: 20px; }
            .success { color: green; }
            .failure { color: red; }
            table { border-collapse: collapse; width: 100%; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            tr:hover { background-color: #f5f5f5; }
            th { background-color: #3498db; color: white; }
        </style>
    </head>
    <body>
        <h1>Spine Segmentation Experiment Results</h1>
        <p>Generated on: """ + f"{os.popen('date').read().strip()}" + """</p>
        
        <h2>Summary Table</h2>
        <table>
            <tr>
                <th>Experiment</th>
                <th>Patch Size</th>
                <th>Overlap</th>
                <th>TIFF Stack</th>
                <th>Overlays</th>
                <th>Sample Collage</th>
            </tr>
    """
    
    # Add rows for each experiment
    for exp_name in sorted(experiment_dirs):
        patch_size, overlap = parse_experiment_params(exp_name)
        
        tiff_status = "✅" if exp_name in tiff_generated else "❌"
        overlay_status = "✅" if exp_name in overlay_generated else "❌"
        
        tiff_link = f"<a href='{exp_name}/{exp_name}_stack.tif'>View TIFF</a>" if exp_name in tiff_generated else "N/A"
        collage_link = f"<a href='{exp_name}/overlays/sample_collage.png'>View Collage</a>" if exp_name in overlay_generated else "N/A"
        overlay_link = f"<a href='{exp_name}/overlays/'>View Overlays</a>" if exp_name in overlay_generated else "N/A"
        
        html_content += f"""
            <tr>
                <td>{exp_name}</td>
                <td>{patch_size}</td>
                <td>{overlap}%</td>
                <td>{tiff_status} {tiff_link}</td>
                <td>{overlay_status} {overlay_link}</td>
                <td>{collage_link}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Detailed Results</h2>
    """
    
    # Add detailed section for each experiment
    for exp_name in sorted(experiment_dirs):
        patch_size, overlap = parse_experiment_params(exp_name)
        
        html_content += f"""
        <div class="experiment">
            <h2>{exp_name}</h2>
            <p>Patch Size: {patch_size}px, Overlap: {overlap}%</p>
            
            <div class="links">
        """
        
        if exp_name in tiff_generated:
            html_content += f"""
                <p class="success">✅ TIFF Stack: <a href="{exp_name}/{exp_name}_stack.tif">Download</a></p>
            """
        else:
            html_content += """
                <p class="failure">❌ TIFF Stack: Generation failed</p>
            """
        
        if exp_name in overlay_generated:
            html_content += f"""
                <p class="success">✅ Overlays: <a href="{exp_name}/overlays/">View Folder</a></p>
                <p>Quick Preview: <a href="{exp_name}/overlays/sample_collage.png">
                    <img src="{exp_name}/overlays/sample_collage.png" width="600">
                </a></p>
            """
        else:
            html_content += """
                <p class="failure">❌ Overlays: Generation failed</p>
            """
        
        html_content += """
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html_content)

if __name__ == '__main__':
    main()