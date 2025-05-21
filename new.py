#!/usr/bin/env python3
"""
Script to compare and visualize IoU and F1 scores across multiple experiments with
different patch sizes and overlap percentages.

Usage:
    python compare_experiments.py --gt path/to/ground_truth.tif --results path/to/results_sweep
"""
import os
import numpy as np
from glob import glob
from PIL import Image
import tifffile
import matplotlib.pyplot as plt
import re
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mticker

def compute_iou_f1(gt_mask: np.ndarray, pred_mask: np.ndarray):
    """
    Compute IoU and F1 score between two binary masks.
    Args:
        gt_mask (np.ndarray): Ground truth binary mask (bool or 0/1).
        pred_mask (np.ndarray): Predicted binary mask (bool or 0/1).
    Returns:
        iou (float): Intersection over Union.
        f1 (float): F1 score (Dice coefficient).
    """
    gt_bool = gt_mask.astype(bool)
    pred_bool = pred_mask.astype(bool)
    intersection = np.logical_and(gt_bool, pred_bool).sum()
    union = np.logical_or(gt_bool, pred_bool).sum()
    iou = intersection / union if union > 0 else 1.0
    
    # F1 score (Dice) = 2 * TP / (2*TP + FP + FN) = 2*intersection / (|gt| + |pred|)
    denom = gt_bool.sum() + pred_bool.sum()
    f1 = 2 * intersection / denom if denom > 0 else 1.0
    return iou, f1

def load_predicted_masks(pred_folder: str, extensions=('png', 'tif', 'jpg', 'jpeg')):
    """
    Load and sort predicted mask file paths from a folder.
    Args:
        pred_folder (str): Path to folder containing predicted mask images.
        extensions (tuple): Allowed file extensions.
    Returns:
        List[str]: Sorted list of file paths.
    """
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(pred_folder, f"*.{ext}")))
    return sorted(files)

def evaluate_experiment(gt_path, pred_folder):
    """
    Evaluate IoU and F1 for a single experiment folder.
    
    Args:
        gt_path (str): Path to ground truth TIFF stack.
        pred_folder (str): Path to folder with prediction masks.
        
    Returns:
        dict: Dictionary with average IoU and F1 scores.
    """
    # Load ground truth stack (shape: [slices, height, width])
    gt_stack = tifffile.imread(gt_path)
    num_slices = gt_stack.shape[0]
    
    # Load predicted mask file paths
    pred_files = load_predicted_masks(pred_folder)
    if len(pred_files) != num_slices:
        print(f"Warning: Number of predicted files ({len(pred_files)}) doesn't match ground truth slices ({num_slices}).")
    
    iou_scores = []
    f1_scores = []
    
    # Iterate over slices and predictions
    for idx in range(min(num_slices, len(pred_files))):
        gt_mask = gt_stack[idx]
        pred_img = Image.open(pred_files[idx])
        pred_mask = np.array(pred_img) > 128
        
        # Ensure boolean masks
        gt_binary = gt_mask > 0
        iou, f1 = compute_iou_f1(gt_binary, pred_mask)
        iou_scores.append(iou)
        f1_scores.append(f1)
    
    # Return average metrics
    return {
        "avg_iou": np.mean(iou_scores),
        "avg_f1": np.mean(f1_scores),
        "iou_scores": iou_scores,
        "f1_scores": f1_scores
    }

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

def visualize_results(results_df):
    """
    Create visualization of IoU and F1 scores across different settings.
    
    Args:
        results_df (pd.DataFrame): DataFrame with experiment results
    """
    # Set style for better visualization
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Set up the figure
    fig = plt.figure(figsize=(16, 12))
    
    # Create a 2x2 grid for different visualizations
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Heatmap of IoU by patch size and overlap
    ax1 = plt.subplot(gs[0, 0])
    iou_pivot = results_df.pivot(index="patch_size", columns="overlap", values="avg_iou")
    sns.heatmap(iou_pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax1, 
                cbar_kws={'label': 'IoU Score'})
    ax1.set_title("IoU Score by Patch Size and Overlap")
    ax1.set_xlabel("Overlap Percentage")
    ax1.set_ylabel("Patch Size (pixels)")
    
    # 2. Heatmap of F1 by patch size and overlap
    ax2 = plt.subplot(gs[0, 1])
    f1_pivot = results_df.pivot(index="patch_size", columns="overlap", values="avg_f1")
    sns.heatmap(f1_pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax2,
                cbar_kws={'label': 'F1 Score'})
    ax2.set_title("F1 Score by Patch Size and Overlap")
    ax2.set_xlabel("Overlap Percentage")
    ax2.set_ylabel("Patch Size (pixels)")
    
    # 3. Plot of IoU vs Patch Size grouped by overlap
    ax3 = plt.subplot(gs[1, 0])
    # Group by overlap
    for overlap in sorted(results_df["overlap"].unique()):
        subset = results_df[results_df["overlap"] == overlap].sort_values("patch_size")
        ax3.plot(subset["patch_size"], subset["avg_iou"], marker='o', linewidth=2, 
                 label=f"Overlap {overlap}%")
    
    ax3.set_title("IoU Score vs Patch Size")
    ax3.set_xlabel("Patch Size (pixels)")
    ax3.set_ylabel("IoU Score")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Format x-axis to show all patch sizes
    ax3.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax3.xaxis.set_tick_params(rotation=45)
    ax3.set_xticks(sorted(results_df["patch_size"].unique()))
    
    # 4. Top 5 best configurations
    ax4 = plt.subplot(gs[1, 1])
    
    # Calculate combined score (average of IoU and F1)
    results_df["combined_score"] = (results_df["avg_iou"] + results_df["avg_f1"]) / 2
    
    # Get top 5 configurations
    top5 = results_df.sort_values("combined_score", ascending=False).head(5)
    
    # Create bar plot for top 5
    x = np.arange(len(top5))
    width = 0.35
    
    # IoU bars
    ax4.bar(x - width/2, top5["avg_iou"], width, label="IoU", color="steelblue")
    
    # F1 bars
    ax4.bar(x + width/2, top5["avg_f1"], width, label="F1", color="forestgreen")
    
    # Add configuration labels
    labels = [f"{row.patch_size}px_{row.overlap}ov" for _, row in top5.iterrows()]
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45)
    
    ax4.set_title("Top 5 Performing Configurations")
    ax4.set_ylabel("Score")
    ax4.set_ylim(0, 1.0)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("experiment_comparison.png", dpi=300, bbox_inches="tight")
    
    # Additional 3D surface plot
    plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')
    
    X = results_df.pivot_table(index='patch_size', columns='overlap', values='avg_iou').columns.values
    Y = results_df.pivot_table(index='patch_size', columns='overlap', values='avg_iou').index.values
    X, Y = np.meshgrid(X, Y)
    Z_iou = results_df.pivot_table(index='patch_size', columns='overlap', values='avg_iou').values
    Z_f1 = results_df.pivot_table(index='patch_size', columns='overlap', values='avg_f1').values
    
    # Plot the surface for IoU
    surf = ax.plot_surface(X, Y, Z_iou, cmap='viridis', alpha=0.7, label='IoU')
    ax.set_xlabel('Overlap (%)')
    ax.set_ylabel('Patch Size (pixels)')
    ax.set_zlabel('Score')
    ax.set_title('3D Surface of IoU Scores')
    
    plt.savefig("experiment_surface.png", dpi=300, bbox_inches="tight")
    
    return fig

def main():
    # Path to ground truth
    gt_path = "Spine_U.tif"
    
    # Base directory for experiment results
    results_base_dir = "results_sweep"
    
    # Collect experiment directories
    experiment_dirs = [d for d in os.listdir(results_base_dir) 
                     if os.path.isdir(os.path.join(results_base_dir, d))]
    
    # Prepare results dataframe
    results = []
    
    print(f"Found {len(experiment_dirs)} experiment folders to analyze")
    
    # Process each experiment
    for exp_name in sorted(experiment_dirs):
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
        
        print(f"Evaluating: {exp_name} (patch_size={patch_size}, overlap={overlap}%)")
        
        # Calculate metrics for this experiment
        metrics = evaluate_experiment(gt_path, masks_dir)
        
        # Store results
        results.append({
            "experiment": exp_name,
            "patch_size": patch_size,
            "overlap": overlap,
            "avg_iou": metrics["avg_iou"],
            "avg_f1": metrics["avg_f1"]
        })
        
        print(f"  IoU: {metrics['avg_iou']:.4f}, F1: {metrics['avg_f1']:.4f}")
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv("experiment_metrics.csv", index=False)
    print(f"Results saved to experiment_metrics.csv")
    
    # Create visualizations
    fig = visualize_results(results_df)
    print(f"Visualizations saved to experiment_comparison.png and experiment_surface.png")
    
    # Print best configurations
    best_iou = results_df.loc[results_df["avg_iou"].idxmax()]
    best_f1 = results_df.loc[results_df["avg_f1"].idxmax()]
    
    print("\n===== BEST CONFIGURATIONS =====")
    print(f"Best for IoU: {best_iou.patch_size}px_{best_iou.overlap}ov (IoU: {best_iou.avg_iou:.4f}, F1: {best_iou.avg_f1:.4f})")
    print(f"Best for F1: {best_f1.patch_size}px_{best_f1.overlap}ov (IoU: {best_f1.avg_iou:.4f}, F1: {best_f1.avg_f1:.4f})")

if __name__ == '__main__':
    main()