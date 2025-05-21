import os
import time
import csv

import numpy as np
import tifffile
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt

def run_inference_patches(
    model_path, tiff_path, gt_tiff_path, output_dir,
    tile_size, overlap, confidence=0.25, device="cuda:0"
):
    """
    Runs segmentation on patches of size tile_size×tile_size with given overlap,
    computes mean IoU and Dice over all slices, and returns (time, mem, mean_iou, mean_dice).
    """
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)

    # load model
    model = YOLO(model_path).to(device)

    # load image stack
    stack = tifffile.imread(tiff_path)
    if stack.ndim == 2:
        stack = stack[None]
    S, H, W = stack.shape

    # load & binarize GT
    gt = tifffile.imread(gt_tiff_path)
    if gt.ndim == 2:
        gt = gt[None]
    gt = gt > 128  # bool array

    stride = int(tile_size * (1 - overlap))

    slice_metrics = []
    t0 = time.perf_counter()

    for idx in tqdm(range(S), desc=f"{tile_size}px, {int(overlap*100)}% ov"):
        img = stack[idx]
        pred_mask = np.zeros((H, W), dtype=bool)

        # patchwise inference
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y_end = min(y + tile_size, H)
                x_end = min(x + tile_size, W)
                y0 = max(0, y_end - tile_size)
                x0 = max(0, x_end - tile_size)

                patch = img[y0:y_end, x0:x_end]
                if patch.shape[0] < 32 or patch.shape[1] < 32:
                    continue

                # normalize & to 3-ch
                p = (patch.astype(np.float32) - patch.min()) / (patch.max() - patch.min() + 1e-9)
                rgb = (np.stack([p]*3, 2) * 255).astype(np.uint8)

                with torch.no_grad():
                    res = model.predict(source=rgb, conf=confidence, device=device, verbose=False)[0]

                if hasattr(res, "masks") and res.masks is not None:
                    masks = res.masks.data  # (n,1,h,w) or (n,h,w)
                    for m in masks:
                        m_np = m.cpu().numpy()
                        if m_np.ndim == 3:
                            m_np = m_np[0]
                        if m_np.shape != patch.shape:
                            m_np = cv2.resize(m_np, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LINEAR)
                        pred_mask[y0:y_end, x0:x_end] |= (m_np > 0.5)

        # save mask & overlay
        mask_path = os.path.join(output_dir, "masks", f"mask_{idx:03d}.png")
        cv2.imwrite(mask_path, (pred_mask * 255).astype(np.uint8))

        # overlay
        slice_8 = ((img - img.min()) / (img.max() - img.min() + 1e-9) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(slice_8, cv2.COLOR_GRAY2BGR)
        red = np.zeros_like(bgr); red[pred_mask] = (0,0,255)
        ov = cv2.addWeighted(bgr, 0.7, red, 0.3, 0)
        ov_path = os.path.join(output_dir, "overlays", f"overlay_{idx:03d}.png")
        cv2.imwrite(ov_path, ov)

        # compute IoU & Dice
        gt_mask = gt[idx]
        inter = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou  = inter / (union + 1e-9)
        dice = 2 * inter / (pred_mask.sum() + gt_mask.sum() + 1e-9)
        slice_metrics.append((iou, dice))

    total_time = time.perf_counter() - t0
    max_mem = torch.cuda.max_memory_allocated() if "cuda" in device else 0

    mean_iou  = np.mean([m[0] for m in slice_metrics])
    mean_dice = np.mean([m[1] for m in slice_metrics])

    return total_time, max_mem, mean_iou, mean_dice

def main():
    model_path   = "runs/segment/train18/weights/best.pt"
    tiff_path    = "dataset/DeepD3_Benchmark.tif"
    gt_tiff_path = "Spine_U.tif"
    base_out     = "results_sweep"

    tile_sizes = [32,64,96,128,160,192,224,256,288,320,352]
    overlaps   = [0.25, 0.5, 0.75]

    records = []

    for ov in overlaps:
        for ts in tile_sizes:
            tag = f"{ts}px_{int(ov*100)}ov"
            out = os.path.join(base_out, tag)
            os.makedirs(out, exist_ok=True)
            t, m, mi, md = run_inference_patches(
                model_path, tiff_path, gt_tiff_path,
                out, ts, ov
            )
            print(f"{tag} → time={t:.1f}s, mem={m/1e9:.2f}GB, IoU={mi:.3f}, F1={md:.3f}")
            records.append({
                "tile_size": ts,
                "overlap": ov,
                "time_s": t,
                "max_mem_bytes": int(m),
                "mean_iou": mi,
                "mean_f1": md
            })

    # write CSV
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(base_out, "sweep_results.csv"), index=False)

    # --- Plot 1: Mean IoU vs Tile Size (one line per overlap) ---
    plt.figure()
    for ov in overlaps:
        sub = df[df.overlap == ov]
        plt.plot(sub.tile_size, sub.mean_iou, label=f"{int(ov*100)}%")
    plt.xlabel("Tile Size (px)")
    plt.ylabel("Mean IoU")
    plt.title("Mean IoU vs Tile Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_out, "iou_vs_tilesize.png"))

    # --- Plot 2: Mean F1 vs Tile Size ---
    plt.figure()
    for ov in overlaps:
        sub = df[df.overlap == ov]
        plt.plot(sub.tile_size, sub.mean_f1, label=f"{int(ov*100)}%")
    plt.xlabel("Tile Size (px)")
    plt.ylabel("Mean F1 Score")
    plt.title("Mean F1 vs Tile Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_out, "f1_vs_tilesize.png"))

    # --- Plot 3: Mean IoU vs Overlap (averaged over tile sizes) ---
    avg_ov = df.groupby("overlap").mean().reset_index()
    plt.figure()
    plt.plot(avg_ov.overlap, avg_ov.mean_iou, marker="o")
    plt.xlabel("Overlap Fraction")
    plt.ylabel("Mean IoU")
    plt.title("Mean IoU vs Overlap (avg over all tile sizes)")
    plt.tight_layout()
    plt.savefig(os.path.join(base_out, "iou_vs_overlap.png"))

    # --- Plot 4: Mean F1 vs Overlap ---
    plt.figure()
    plt.plot(avg_ov.overlap, avg_ov.mean_f1, marker="o")
    plt.xlabel("Overlap Fraction")
    plt.ylabel("Mean F1 Score")
    plt.title("Mean F1 vs Overlap (avg over all tile sizes)")
    plt.tight_layout()
    plt.savefig(os.path.join(base_out, "f1_vs_overlap.png"))

    print("Sweep complete. CSV and plots in", base_out)

if __name__ == "__main__":
    main()
