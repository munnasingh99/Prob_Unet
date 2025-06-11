import os
import numpy as np
import cv2
import torch
from torchvision.ops import nms
import tifffile
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import seaborn as sns
from datetime import timedelta

def calculate_metrics(pred_mask, gt_mask):
    """
    Calculate pixel-wise metrics (precision, recall, F1, IoU)
    
    Args:
        pred_mask: Binary prediction mask
        gt_mask: Binary ground truth mask
        
    Returns:
        dict: Dictionary with calculated metrics
    """
    # Ensure masks are binary
    pred_binary = pred_mask > 0
    gt_binary = gt_mask > 0
    
    # Calculate metrics
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    tp = intersection
    fp = pred_binary.sum() - tp
    fn = gt_binary.sum() - tp
    
    # Calculate metrics (handling division by zero)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = intersection / union if union > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def pad_to_multiple_of_32(image):
    """Pad image dimensions to be multiples of 32"""
    h, w = image.shape
    
    # Calculate padding amounts
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    
    # Pad image
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
        return padded, (h, w)  # Return padded image and original dimensions
    else:
        return image, (h, w)

def run_structured_tile_experiment(
    model_path,
    tiff_path,
    gt_tiff_path,
    output_dir,
    slice_idx=34,
    conf_threshold=0.25,
    iou_threshold=0.5,
    device="cuda:0"
):
    """
    Run experiment with strategically structured tiles for a 384×1440 image
    
    Args:
        model_path: Path to YOLO model
        tiff_path: Path to input TIFF stack
        gt_tiff_path: Path to ground truth TIFF stack
        output_dir: Directory to save results
        slice_idx: Slice index to test (default: 34)
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        device: Device to run on
        
    Returns:
        pd.DataFrame: Results dataframe with metrics for each configuration
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visual_compare"), exist_ok=True)
    
    # Load model
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    model.to(device)
    
    # Load image and ground truth
    print(f"Loading slice {slice_idx} from {tiff_path} and ground truth...")
    stack = tifffile.imread(tiff_path)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    
    gt_stack = tifffile.imread(gt_tiff_path)
    if gt_stack.ndim == 2:
        gt_stack = gt_stack[np.newaxis, ...]
    
    # Ensure slice_idx is valid
    if slice_idx >= len(stack):
        raise ValueError(f"Slice index {slice_idx} out of bounds (stack has {len(stack)} slices)")
    
    # Get slice and ground truth
    slice_img = stack[slice_idx]
    gt_mask = gt_stack[slice_idx] > 0
    
    # Ensure ground truth and image have the same shape
    if slice_img.shape != gt_mask.shape:
        print(f"Warning: Image shape {slice_img.shape} doesn't match ground truth shape {gt_mask.shape}")
        gt_mask = cv2.resize(gt_mask.astype(np.uint8), 
                             (slice_img.shape[1], slice_img.shape[0]), 
                             interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # Pad image and ground truth to multiple of 32
    padded_img, orig_dims = pad_to_multiple_of_32(slice_img)
    padded_gt, _ = pad_to_multiple_of_32(gt_mask)
    
    H, W = padded_img.shape
    orig_H, orig_W = orig_dims
    
    print(f"Original slice shape: {orig_H}×{orig_W}")
    print(f"Padded slice shape: {H}×{W}")
    
    # Define structured tile heights and widths
    # For 384×1440 padded image
    height_options = [96, 192, 288, 384]  # Divisible by 384
    width_options = [360, 720, 1080, 1440]  # Divisible by 1440
    
    # Define overlap options (as percentages)
    overlap_options = [0, 0.25, 0.5]
    
    # Generate all combinations of heights, widths, and overlaps
    tile_configs = []
    for h in height_options:
        for w in width_options:
            # Skip very large or very small tiles
            if h * w > 640 * 640 or h * w < 96 * 96:
                continue
                
            for overlap in overlap_options:
                # Skip if tile dimensions are too large for the image
                if h > H or w > W:
                    continue
                    
                tile_configs.append((h, w, overlap))
    
    print(f"Testing {len(tile_configs)} tile configurations...")
    
    # Initialize results storage
    results = []
    
    # Run experiments for each configuration
    for i, (tile_h, tile_w, overlap) in enumerate(tile_configs, 1):
        exp_start_time = time.time()
        print(f"Experiment {i}/{len(tile_configs)}: "
              f"Tile size = {tile_h}×{tile_w}, Overlap = {overlap:.2f}")
        
        # Calculate strides based on overlap
        h_stride = int(tile_h * (1 - overlap))
        w_stride = int(tile_w * (1 - overlap))
        
        # Ensure strides are at least 1
        h_stride = max(1, h_stride)
        w_stride = max(1, w_stride)
        
        # Initialize empty mask
        pred_mask = np.zeros((H, W), dtype=bool)
        
        # Process tiles
        tile_count = 0
        for y in range(0, H, h_stride):
            for x in range(0, W, w_stride):
                # Ensure tiles don't go beyond image boundaries
                y_end = min(y + tile_h, H)
                x_end = min(x + tile_w, W)
                
                # Adjust start position for edge tiles
                y_start = max(0, y_end - tile_h)
                x_start = max(0, x_end - tile_w)
                
                # Extract tile
                tile = padded_img[y_start:y_end, x_start:x_end]
                
                # Skip tiny tiles
                if tile.shape[0] < 10 or tile.shape[1] < 10:
                    continue
                
                tile_count += 1
                
                # Prepare the tile for model input
                
                # Normalize
                tile_norm = (tile.astype(np.float32) - tile.min()) / (tile.max() - tile.min() + 1e-9)
                
                # Convert to RGB
                tile_rgb = np.stack([tile_norm] * 3, axis=2) * 255
                tile_rgb = tile_rgb.astype(np.uint8)
                
                # Run inference
                with torch.no_grad():
                    try:
                        result = model.predict(
                            source=tile_rgb,
                            imgsz=tile_rgb.shape[:2],  # Use tile's own dimensions
                            conf=conf_threshold,
                            device=device,
                            verbose=False
                        )[0]
                    except Exception as e:
                        print(f"  Error during inference on tile at ({x_start}, {y_start}): {e}")
                        continue
                
                # Process masks
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data
                    
                    for mask_tensor in masks:
                        # Convert to numpy
                        mask_np = mask_tensor.cpu().numpy()
                        
                        # Handle different mask shapes
                        if mask_np.ndim == 3:  # [C, H, W]
                            mask = mask_np[0]
                        elif mask_np.ndim == 2:  # [H, W]
                            mask = mask_np
                        else:
                            continue
                        
                        # Ensure mask has the correct dimensions
                        if mask.shape != tile.shape:
                            try:
                                mask = cv2.resize(
                                    mask, 
                                    (tile.shape[1], tile.shape[0]), 
                                    interpolation=cv2.INTER_LINEAR
                                )
                            except Exception as e:
                                print(f"  Error resizing mask: {e}")
                                print(f"  Mask shape: {mask.shape}, Tile shape: {tile.shape}")
                                continue
                        
                        # Apply threshold and add to full mask
                        binary_mask = mask > 0.5
                        pred_mask[y_start:y_end, x_start:x_end] |= binary_mask
        
        # Crop prediction back to original size (remove padding)
        pred_mask = pred_mask[:orig_H, :orig_W]
        gt_crop = padded_gt[:orig_H, :orig_W]
        
        # Calculate metrics on original sized images
        metrics = calculate_metrics(pred_mask, gt_crop)
        
        # Add experiment parameters to metrics
        metrics['tile_height'] = tile_h
        metrics['tile_width'] = tile_w
        metrics['overlap'] = overlap
        metrics['tile_count'] = tile_count
        metrics['tile_area'] = tile_h * tile_w
        metrics['processing_time'] = time.time() - exp_start_time
        
        # Save metrics
        results.append(metrics)
        
        # Print metrics
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  IoU: {metrics['iou']:.4f}")
        print(f"  Tiles used: {tile_count}")
        print(f"  Processing time: {metrics['processing_time']:.2f}s")
        
        # Save visualization for this configuration
        # Convert to 8-bit for visualization
        if slice_img.dtype != np.uint8:
            slice_8bit = (255 * (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())).astype(np.uint8)
        else:
            slice_8bit = slice_img
            
        # Create RGB visualization
        vis_img = cv2.cvtColor(slice_8bit, cv2.COLOR_GRAY2BGR)
        
        # Add ground truth outline in blue
        gt_contours, _ = cv2.findContours(
            gt_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_img, gt_contours, -1, (255, 0, 0), 1)
        
        # Add prediction in transparent red
        red_overlay = np.zeros_like(vis_img)
        red_overlay[pred_mask] = (0, 0, 255)  # Red in BGR
        vis_img = cv2.addWeighted(vis_img, 0.7, red_overlay, 0.3, 0)
        
        # Add title with configuration and metrics
        title = f"Tile: {tile_h}×{tile_w}, Overlap: {overlap:.2f}"
        metrics_text = f"P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}, IoU: {metrics['iou']:.3f}"
        
        # Add text with metrics (white outline with black fill for visibility)
        y_pos = 30
        for text in [title, metrics_text]:
            cv2.putText(
                vis_img, text, (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                vis_img, text, (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1
            )
            y_pos += 30
        
        # Save visualization
        vis_path = os.path.join(
            output_dir, "visual_compare", 
            f"tile_{tile_h}x{tile_w}_overlap_{int(overlap*100)}.png"
        )
        cv2.imwrite(vis_path, vis_img)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, "experiment_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Create visualizations
    try:
        visualize_experiment_results(results_df, output_dir)
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Find and report best configurations
    print("\nTop 5 configurations by F1 score:")
    top_f1 = results_df.sort_values('f1', ascending=False).head(5)
    for i, (_, row) in enumerate(top_f1.iterrows(), 1):
        print(f"{i}. Tile: {row['tile_height']}×{row['tile_width']}, "
              f"Overlap: {row['overlap']:.2f}, "
              f"F1: {row['f1']:.4f}, IoU: {row['iou']:.4f}, "
              f"Time: {row['processing_time']:.2f}s")
    
    # Report best configuration
    best_config = results_df.loc[results_df['f1'].idxmax()]
    print("\nBest configuration by F1 score:")
    print(f"Tile size: {best_config['tile_height']}×{best_config['tile_width']}")
    print(f"Overlap: {best_config['overlap']:.2f}")
    print(f"F1 score: {best_config['f1']:.4f}")
    print(f"IoU: {best_config['iou']:.4f}")
    print(f"Processing time: {best_config['processing_time']:.2f}s")
    
    return results_df

def visualize_experiment_results(results, output_dir):
    """Create visualizations of experiment results with tiles"""
    plt.ioff()  # Turn off interactive mode
    
    # 1. Bubble Chart: F1 Score by Tile Dimensions and Overlap
    plt.figure(figsize=(14, 10))
    
    # Create scatter with size based on tile area and color based on overlap
    scatter = plt.scatter(
        results['tile_width'], 
        results['tile_height'],
        s=results['f1'] * 1000 + 50,  # Size based on F1 score, scaled up
        c=results['overlap'],  # Color based on overlap
        cmap='viridis',
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Overlap')
    
    # Add labels
    for i, row in results.iterrows():
        plt.annotate(
            f"F1: {row['f1']:.3f}",
            (row['tile_width'], row['tile_height']),
            fontsize=8,
            ha='center'
        )
    
    plt.xlabel('Tile Width')
    plt.ylabel('Tile Height')
    plt.title('F1 Score by Tile Dimensions (size = F1 score)')
    plt.grid(True)
    plt.tight_layout()
    
    try:
        plt.savefig(os.path.join(output_dir, 'f1_by_dimensions.png'), dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving f1_by_dimensions.png: {e}")
    
    plt.close()
    
    # 2. Scatter plot of F1 vs tile count (efficiency)
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(
        results['tile_count'],
        results['f1'],
        s=results['tile_area'] / 100,  # Size based on tile area, scaled down
        c=results['overlap'],  # Color based on overlap
        cmap='viridis',
        alpha=0.7
    )
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Overlap')
    
    # Add labels for top configurations
    top_configs = results.sort_values('f1', ascending=False).head(5)
    for i, row in top_configs.iterrows():
        plt.annotate(
            f"{row['tile_height']}×{row['tile_width']}, o:{row['overlap']:.1f}",
            (row['tile_count'], row['f1']),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    plt.xlabel('Number of Tiles')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Number of Tiles (size = tile area)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_vs_tiles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of F1 score by tile height and width (for each overlap)
    # First, ensure all combinations exist by creating a pivot table with fill_value
    for overlap in results['overlap'].unique():
        subset = results[results['overlap'] == overlap]
        
        # Create a pivot table
        pivot = subset.pivot_table(
            values='f1',
            index='tile_height',
            columns='tile_width',
            fill_value=0  # Fill missing values
        )
        
        plt.figure(figsize=(12, 9))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={'label': 'F1 Score'}
        )
        
        plt.title(f'F1 Score by Tile Dimensions (Overlap: {overlap:.2f})')
        plt.tight_layout()
        
        # Save with overlap in filename
        overlap_str = str(int(overlap * 100)).zfill(2)
        plt.savefig(os.path.join(output_dir, f'f1_heatmap_overlap_{overlap_str}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Precision-Recall Scatter
    plt.figure(figsize=(12, 10))
    
    scatter = plt.scatter(
        results['recall'],
        results['precision'],
        s=results['f1'] * 500 + 50,  # Size based on F1
        c=results['overlap'],  # Color based on overlap
        cmap='viridis',
        alpha=0.7
    )
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Overlap')
    
    # Add labels for top F1 configurations
    for i, row in top_configs.iterrows():
        plt.annotate(
            f"{row['tile_height']}×{row['tile_width']}, o:{row['overlap']:.1f}",
            (row['recall'], row['precision']),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    plt.xlim(0, max(results['recall']) * 1.1 + 0.05)
    plt.ylim(0, max(results['precision']) * 1.1 + 0.05)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall (size = F1 score)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. F1 vs Processing Time (Performance Tradeoff)
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(
        results['processing_time'],
        results['f1'],
        s=results['tile_count'] * 5,  # Size based on number of tiles
        c=results['overlap'],  # Color based on overlap
        cmap='viridis',
        alpha=0.7
    )
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Overlap')
    
    # Add annotations for top 5
    for i, row in top_configs.iterrows():
        plt.annotate(
            f"{row['tile_height']}×{row['tile_width']}, o:{row['overlap']:.1f}",
            (row['processing_time'], row['f1']),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    plt.xlabel('Processing Time (seconds)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Processing Time (size = tile count)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_vs_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Bar Chart of Top 10 Configurations
    plt.figure(figsize=(16, 10))
    
    # Get top 10 configurations
    top10 = results.sort_values('f1', ascending=False).head(10)
    
    # Create configuration labels
    config_labels = [
        f"{row['tile_height']}×{row['tile_width']}, o:{row['overlap']:.2f}"
        for _, row in top10.iterrows()
    ]
    
    x = np.arange(len(config_labels))
    width = 0.2
    
    # Plot bars for each metric
    plt.bar(x - width, top10['precision'], width, label='Precision', color='#ff9999')
    plt.bar(x, top10['recall'], width, label='Recall', color='#99ff99')
    plt.bar(x + width, top10['f1'], width, label='F1', color='#9999ff')
    
    plt.xlabel('Configuration')
    plt.ylabel('Score')
    plt.title('Top 10 Configurations by F1 Score')
    plt.xticks(x, config_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top10_configurations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. 3D Plot
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            results['tile_width'],
            results['tile_height'],
            results['f1'],
            s=results['f1'] * 500 + 50,
            c=results['overlap'],
            cmap='viridis',
            alpha=0.7
        )
        
        # Add colorbar
        cbar = fig.colorbar(scatter)
        cbar.set_label('Overlap')
        
        # Add best point annotation
        best_row = results.loc[results['f1'].idxmax()]
        ax.text(
            best_row['tile_width'],
            best_row['tile_height'],
            best_row['f1'],
            f"Best: {best_row['tile_height']}×{best_row['tile_width']}, o:{best_row['overlap']:.2f}",
            color='red'
        )
        
        ax.set_xlabel('Tile Width')
        ax.set_ylabel('Tile Height')
        ax.set_zlabel('F1 Score')
        ax.set_title('F1 Score in 3D Space (Tile Width, Tile Height, F1)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3d_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating 3D plot: {e}")
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    results = run_structured_tile_experiment(
        model_path="runs/segment/train18/weights/best.pt",
        tiff_path="dataset/DeepD3_Benchmark.tif",
        gt_tiff_path="Spine_U.tif",
        output_dir="structured_tile_experiment",
        slice_idx=34,
        conf_threshold=0.25,
        iou_threshold=0.5,
        device="cuda:0"
    )