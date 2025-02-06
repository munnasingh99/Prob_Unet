import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from datagen import DataGeneratorDataset
from probabilistic_unet import ProbabilisticUnet
import wandb

torch.manual_seed(42)
wandb.init(
    project="probabilistic-unet",
    config={
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "latent_dim": 16,
        "beta": 1
    },
)
config = wandb.config

output_dir = "predictions_spine/"
probability_dir = os.path.join(output_dir, "probabilities/")
cross_section_dir = os.path.join(output_dir, "cross_sections/")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(probability_dir, exist_ok=True)
os.makedirs(cross_section_dir, exist_ok=True)


output_dir = "sampling_probabilities_spine/"
os.makedirs(output_dir, exist_ok=True)

# Model & Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = ProbabilisticUnet(input_channels=1, num_classes=1, latent_dim=12).to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
#model.eval()

# Load dataset
val_dataset = DataGeneratorDataset(
    r"DeepD3_Validation.d3set",
    samples_per_epoch=64,
    size=(1, 128, 128),
    augment=False,
    shuffle=False,
)

# Create DataLoader
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)

# Load model
model = ProbabilisticUnet(input_channels=1, num_classes=1, latent_dim=config.latent_dim).to(device)
criterion = nn.BCEWithLogitsLoss()

# Load best model if available
best_model_path = r"spine_model_epoch_18_iou_0.4766.pth"
model.load_state_dict(torch.load( best_model_path, map_location=device))

def save_images(image, mask, pred, epoch, mode="train"):
    """
    Saves input image, ground truth mask, and predicted mask.

    Args:
        image (torch.Tensor): Input image
        mask (torch.Tensor): Ground truth mask
        pred (torch.Tensor): Predicted mask
        epoch (int): Current epoch index
        mode (str): "train" or "val" (used for naming)
    """
    image = image.squeeze(0).cpu().detach().numpy()
    mask = mask.squeeze(0).cpu().detach().numpy()
    pred = pred.squeeze(0).cpu().detach().numpy()

    # Normalize images (convert to binary if needed)
    image = (image * 255).astype(np.uint8)
    mask = (mask > 0.5).astype(np.uint8) * 255
    pred = (pred > 0.5).astype(np.uint8) * 255

    # Define filenames
    filename_i = f"{mode}_{epoch}_image.png"
    filename_m = f"{mode}_{epoch}_mask.png"
    filename_p = f"{mode}_{epoch}_prediction.png"

    # Save images
    cv2.imwrite(os.path.join(output_dir, filename_i), image)
    cv2.imwrite(os.path.join(output_dir, filename_m), mask)
    cv2.imwrite(os.path.join(output_dir, filename_p), pred)

    print(f"Saved: {filename_i}, {filename_m}, {filename_p}")

def dice_coefficient(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    return (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)

def iou_score(pred,target,epsilon=1e-6):

    intersection = (pred * target).sum()  # Intersection 
    union = pred.sum() + target.sum() - intersection  # Union

    return (intersection + epsilon) / (union - intersection + epsilon)  # A IoU B 

# Function to save probability maps
def save_probability_map(prob_map, epoch, batch_idx, sample_idx,dir):
    """
    Saves probability maps instead of binary predictions.
    """
    prob_map = prob_map.squeeze(0).cpu().detach().numpy()
    filename_p = f"prob_{epoch}_{batch_idx}_sample{sample_idx}.png"
    print(prob_map.min())
    # Normalize probability to 0-255
    prob_map = (prob_map * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(dir, filename_p), prob_map)
    print(f"Saved Probability Map: {filename_p}")

# # Function to save cross-sections
# def save_cross_section(image, mask, probabilities, epoch, batch_idx):
#     """
#     Extracts a vertical cross-section through the center of the image.
#     """
#     mid_x = image.shape[2] // 2  # Take center slice
    
#     # Extract cross-sections
#     cross_image = image[:, :, mid_x].squeeze(0).cpu().detach().numpy()
#     cross_mask = mask[:, :, mid_x].squeeze(0).cpu().detach().numpy()
#     cross_prob = probabilities[:, :, mid_x].squeeze(0).cpu().detach().numpy()
    
#     # Normalize
#     cross_image = (cross_image * 255).astype(np.uint8)
#     cross_mask = (cross_mask > 0.5).astype(np.uint8) * 255
#     cross_prob = (cross_prob * 255).astype(np.uint8)

#     # Save images
#     filename_i = f"cross_section_{epoch}_{batch_idx}_image.png"
#     filename_m = f"cross_section_{epoch}_{batch_idx}_mask.png"
#     filename_p = f"cross_section_{epoch}_{batch_idx}_prob.png"

#     cv2.imwrite(os.path.join(cross_section_dir, filename_i), cross_image)
#     cv2.imwrite(os.path.join(cross_section_dir, filename_m), cross_mask)
#     cv2.imwrite(os.path.join(cross_section_dir, filename_p), cross_prob)
    
#     print(f"Saved Cross-Sections: {filename_i}, {filename_m}, {filename_p}")

# **Validation Phase with Probabilities and Sampling**
model.eval()
val_loss = 0.0
val_kl = 0.0
val_iou = 0.0

with torch.no_grad():
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation")

    for batch_idx, (image, (dendrite_mask, spine_mask)) in progress_bar:
        image, mask = image.to(device), spine_mask.to(device)

        # **Sample multiple outputs (Professor's request)**
        sampled_probabilities = []

        for sample_idx in range(10):  # Sample 10 times
            model.forward(image, mask, training=False)
            #kl_div = model.kl_divergence()
            pred_logits = model.sample()
            recon_loss = criterion(pred_logits, mask)

            # Compute probability maps
            pred_probs = torch.sigmoid(pred_logits) 
            #print(pred_probs.min(),pred_probs.max()) # No binarization, raw probabilities
            sampled_probabilities.append(pred_probs.cpu().detach().numpy())

            # Compute Dice Score
            iou = iou_score(pred_probs, mask)
            
            # Compute total loss
            total_loss = recon_loss.mean()
            
            # Track metrics
            val_loss += total_loss.item()
            #val_kl += kl_div.mean().item()
            val_iou += iou.mean().item()

            # Save **probability maps** for analysis
                #save_probability_map(pred_probs, epoch=config.epochs, batch_idx=batch_idx, sample_idx=sample_idx,dir=probability_dir)

        # **Compute Mean & Variance Across 10 Samples**
        sampled_probabilities = np.array(sampled_probabilities)  # Shape: (10, B, 1, H, W)
        mean_probabilities = np.mean(sampled_probabilities, axis=0)  # Mean probability map
        variance_probabilities = np.var(sampled_probabilities, axis=0)  # Variance map

        # Save cross-section for a single image
        if batch_idx == 0:
            save_cross_section(image[0], mask[0], torch.tensor(mean_probabilities[0]), epoch=config.epochs, batch_idx=batch_idx)

        # Update progress bar
        progress_bar.set_postfix({
            "Loss": total_loss.item(),
            "iou": iou.mean().item()
        })

# Compute final validation metrics
avg_val_loss = val_loss / (len(val_loader) * 10)  # Averaged over all 10 samples per image
avg_val_iou = val_iou / (len(val_loader) * 10)

wandb.log({"Val Loss": avg_val_loss, "Val_iou_Loss": avg_val_iou, "Epoch": config.epochs})

print(f"\n[Final Validation] Loss: {avg_val_loss:.6f}, iou: {avg_val_iou:.6f}")

print("Validation Completed with Multi-Sampling & Probability Maps!")
