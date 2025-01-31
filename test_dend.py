import torch
import os
import numpy as np
import cv2
import tifffile as tiff
from tqdm import tqdm
from probabilistic_unet import ProbabilisticUnet
import torchvision.transforms as transforms

# 🔹 Paths
tif_file_path = "DeepD3_Benchmark.tif"  # Update with your TIF file path
model_path = "saved_models/dend_model_epoch_24_dice_0.7335.pth"  # Load trained model
output_dir = "benchmark_predictions_dend/"
os.makedirs(output_dir, exist_ok=True)

# 🔹 Model & Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProbabilisticUnet(input_channels=1, num_classes=1, latent_dim=12).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 🔹 Load TIF File (as NumPy array)
tif_stack = tiff.imread(tif_file_path).astype(np.float32)  # Convert to float32
tif_stack = tif_stack / np.max(tif_stack)   # Shape: (num_slices, height, width)

num_slices, height, width = tif_stack.shape
print(f"TIF Stack Loaded: {num_slices} slices, Original Size: ({height}, {width})")

# Compute Padded Size (closest multiple of 16)
pad_h = ((height // 16) + 1) * 16
pad_w = ((width // 16) + 1) * 16
print(f"Padding image to: {pad_h}x{pad_w}")

# Function to pad an image
def pad_image(image, target_h, target_w):
    pad_top = (target_h - image.shape[0]) // 2
    pad_bottom = target_h - image.shape[0] - pad_top
    pad_left = (target_w - image.shape[1]) // 2
    pad_right = target_w - image.shape[1] - pad_left
    return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')


# 🔹 Image Preprocessing (convert to tensor)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to [0,1] tensor
])

# 🔹 Dice Coefficient Function
def dice_coefficient(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    return (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)

# 🔹 Save Predictions
def save_image(image, filename):
    image = (image * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, filename), image)


# def center_crop(image, crop_size=512):
#     """
#     Crop the center of the image to a fixed square size.
#     """
#     h, w = image.shape
#     start_x = max((w - crop_size) // 2, 0)
#     start_y = max((h - crop_size) // 2, 0)
#     return image[start_y:start_y + crop_size, start_x:start_x + crop_size]

# 🔹 Inference Loop
total_dice = 0.0
num_slices = len(tif_stack)
print(f"Tif Stack shape is: {tif_stack[0].shape}")
with torch.no_grad():
    progress_bar = tqdm(enumerate(tif_stack), total=num_slices, desc="Benchmark Testing")

    for idx, slice_img in progress_bar:
        # Normalize & Convert to Tensor
        padded_slice = pad_image(slice_img, pad_h, pad_w)
        print(padded_slice.shape)
        slice_tensor = transform(padded_slice).unsqueeze(0).to(device)  # Add batch dimension
        # Forward Pass
        model.forward(slice_tensor,None,training=False)
        pred_logits = model.sample()

        # Compute Prediction
        pred_probs = torch.sigmoid(pred_logits) 
        print(pred_probs.shape) # Convert logits to probabilities
        pred_binary = (pred_probs > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()  # Convert to NumPy
        print(f"After numpy transfrom: {pred_binary.shape}")
        #image = slice_tensor.squeeze(0).cpu().detach().numpy()
        # Save Prediction
        save_image(pred_binary, f"benchmark_{idx}_prediction.png")
        #save_image(original,f"benchmark_{idx}_stack.png")

        # Update Progress
        progress_bar.set_postfix({"Saved": f"benchmark_{idx}_prediction.png"})

print("\n🔹 Benchmark Testing Completed! Predictions saved in 'benchmark_predictions_dend/'")
