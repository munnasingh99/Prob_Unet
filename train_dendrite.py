import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import cv2
from torch.utils.data import DataLoader
from datagen import DataGeneratorDataset
from probabilistic_unet import ProbabilisticUnet

# Hyperparameters
num_epochs = 30
learning_rate = 0.0005
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory for saving predictions
output_dir = "predictions_dend/"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
train_dataset = DataGeneratorDataset(
    r"DeepD3_Training.d3set",
    samples_per_epoch=256 * 128,
    size=(1, 128, 128),
    augment=True,
    shuffle=True,
)

val_dataset = DataGeneratorDataset(
    r"DeepD3_Validation.d3set",
    samples_per_epoch=64 * 8,
    size=(1, 128, 128),
    augment=False,
    shuffle=False,
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

# Define model, loss, and optimizer
model = ProbabilisticUnet(input_channels=1, num_classes=1, latent_dim=12).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Dice Coefficient Calculation
def dice_coefficient(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    return (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)

# Function to save images (once per epoch)
def save_images(image, mask, pred, epoch):
    """
    Saves input image, ground truth mask, and predicted mask.

    Args:
        image (torch.Tensor): Input image
        mask (torch.Tensor): Ground truth mask
        pred (torch.Tensor): Predicted mask
        epoch (int): Current epoch index
    """
    image = image.squeeze(0).cpu().detach().numpy()
    mask = mask.squeeze(0).cpu().detach().numpy()
    pred = pred.squeeze(0).cpu().detach().numpy()

    # Normalize images (convert to binary if needed)
    image = (image * 255).astype(np.uint8)
    mask = (mask > 0.5).astype(np.uint8) * 255
    pred = (pred > 0.5).astype(np.uint8) * 255

    # Define filenames
    filename_i = f"dendrite_{epoch}_image.png"
    filename_m = f"dendrite_{epoch}_mask.png"
    filename_p = f"dendrite_{epoch}_prediction.png"

    # Save images
    cv2.imwrite(os.path.join(output_dir, filename_i), image)
    cv2.imwrite(os.path.join(output_dir, filename_m), mask)
    cv2.imwrite(os.path.join(output_dir, filename_p), pred)

    print(f"Saved: {filename_i}, {filename_m}, {filename_p}")

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_kl = 0.0
    epoch_dice = 0.0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, (image, (dendrite_mask, spine_mask)) in progress_bar:
        image, mask = image.to(device), dendrite_mask.to(device)

        # Forward pass
        model.forward(image, mask, training=True)  # Encode prior and posterior
        kl_div = model.kl_divergence()  # Compute KL Divergence
        pred_logits = model.sample()  # Sample segmentation output
        recon_loss = criterion(pred_logits, mask)  # Compute BCE loss

        # Compute Dice Score
        pred_probs = torch.sigmoid(pred_logits)  # Convert logits to probabilities
        pred_binary = (pred_probs > 0.5).float()  # Threshold to binary mask
        dice = dice_coefficient(pred_binary, mask)

        # Compute total loss (β controls KL regularization strength)
        beta = 1.0
        total_loss = (recon_loss + beta * kl_div).mean()

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track metrics
        epoch_loss += total_loss.item()
        epoch_kl += kl_div.mean().item()
        epoch_dice += dice.mean().item()

        # Save images only for the first batch of the epoch
        if batch_idx == 0:
            save_images(image[0], mask[0], pred_binary[0], epoch)

        # Update progress bar
        progress_bar.set_postfix({
            "Loss": total_loss.item(),
            "KL": kl_div.mean().item(),
            "Dice": dice.mean().item()
        })

    # Print summary at the end of the epoch
    avg_loss = epoch_loss / len(train_loader)
    avg_kl = epoch_kl / len(train_loader)
    avg_dice = epoch_dice / len(train_loader)

    print(f"\n[Epoch {epoch+1}] Avg Loss: {avg_loss:.6f}, Avg KL: {avg_kl:.6f}, Avg Dice: {avg_dice:.6f}")

    # Log latent space values (only once per epoch)
    print(f"Prior Mean: {model.prior_latent_space.base_dist.loc.mean().item()}, Prior Std: {model.prior_latent_space.base_dist.scale.mean().item()}")
    print(f"Posterior Mean: {model.posterior_latent_space.base_dist.loc.mean().item()}, Posterior Std: {model.posterior_latent_space.base_dist.scale.mean().item()}")

print("Training Completed!")
