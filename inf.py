import os
import torch
import numpy as np
import cv2
import tifffile as tiff
from tqdm import tqdm
import torchvision.transforms as transforms
from probabilistic_unet import ProbabilisticUnet
import imageio as io

def preprocess_image(image):
    """Normalize and pad image if needed"""
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    h, w = image.shape
    pad_h = (32 - h % 32) if h % 32 else 0
    pad_w = (32 - w % 32) if w % 32 else 0
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    return image, (h, w)

def save_outputs(image, predictions, slice_idx, output_dir, mode="dend"):
    """Save image and prediction outputs"""
    slice_dir = os.path.join(output_dir, f'slice_{slice_idx}')
    os.makedirs(slice_dir, exist_ok=True)
    
    # Save original image once
    cv2.imwrite(os.path.join(slice_dir, f'image.png'), image)
    
    # Save predictions for each sample
    for sample_idx, pred in enumerate(predictions):
        # Save probability map as numpy array
        np.save(os.path.join(slice_dir, f'{mode}_prob_map_sample{sample_idx}.npy'), pred)
        
        # Save visualization
        pred_binary = (pred > 0.5).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(slice_dir, f'{mode}_pred_sample{sample_idx}.png'), pred_binary)
    
    # Save mean and std
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    np.save(os.path.join(slice_dir, f'{mode}_mean_prob.npy'), mean_pred)
    np.save(os.path.join(slice_dir, f'{mode}_std_prob.npy'), std_pred)

def run_inference(tif_path, model_path, output_dir, num_samples=10):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProbabilisticUnet(input_channels=1, num_classes=1, latent_dim=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.ToTensor()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process TIF stack
    tif_stack = np.asarray(io.mimread(tif_path,memtest=False))
    if len(tif_stack.shape) == 2:
        tif_stack = tif_stack[np.newaxis, ...]
    
    with torch.no_grad():
        for slice_idx in tqdm(range(len(tif_stack)), desc="Processing slices"):
            # Preprocess slice
            processed_slice, (orig_h, orig_w) = preprocess_image(tif_stack[slice_idx])
            slice_tensor = transform(processed_slice.astype(np.float32)).unsqueeze(0).to(device)
            
            # Generate samples
            samples = []
            for _ in range(num_samples):
                model.forward(slice_tensor, None, training=False)
                pred = torch.sigmoid(model.sample())
                pred = pred.squeeze().cpu().numpy()[:orig_h, :orig_w]
                samples.append(pred)
            
            # Save outputs
            save_outputs(
                image=tif_stack[slice_idx],
                predictions=samples,
                slice_idx=slice_idx,
                output_dir=output_dir
            )

if __name__ == "__main__":
    config = {
        'tif_path': "DeepD3_Benchmark.tif",
        'model_path': "dend_model_epoch_20_iou_0.5976.pth",
        'output_dir': "inference_results",
        'num_samples': 10
    }
    run_inference(**config)