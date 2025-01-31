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
import wandb

wandb.init(
    project="probabilistic-unet",
    config={
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "latent_dim": 12,
        "beta": 1
    },
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(42)
# Output directory for saving predictions & models
output_dir = "predictions_dend/"
model_dir = "saved_models/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
config=wandb.config
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
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)

# Define model, loss, and optimizer
model = ProbabilisticUnet(input_channels=1, num_classes=1, latent_dim=config.latent_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Best validation dice score tracking
best_val_dice = 0.0

# Dice Coefficient Calculation
def dice_coefficient(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    return (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)

# Function to save images (once per epoch)
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


# Training & Validation loop
for epoch in range(config.epochs):
    ### Training Phase ###
    model.train()
    epoch_loss = 0.0
    epoch_kl = 0.0
    epoch_dice = 0.0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs} [Train]")

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
        #beta = 1.0
        total_loss = (recon_loss + config.beta * kl_div).mean()

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
            save_images(image[0], mask[0], pred_binary[0], epoch, mode="train")

        # Update progress bar
        progress_bar.set_postfix({
            "Loss": total_loss.item(),
            "KL": kl_div.mean().item(),
            "Dice": dice.mean().item()
        })

    ### Validation Phase ###
    model.eval()
    val_loss = 0.0
    val_kl = 0.0
    val_dice = 0.0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1}/{config.epochs} [Validation]")

        for batch_idx, (image, (dendrite_mask, spine_mask)) in progress_bar:
            image, mask = image.to(device), dendrite_mask.to(device)

            # Forward pass
            model.forward(image,mask,training=False)
            kl_div = model.kl_divergence()
            pred_logits = model.sample()
            recon_loss = criterion(pred_logits, mask)

            # Compute Dice Score
            pred_probs = torch.sigmoid(pred_logits)
            pred_binary = (pred_probs > 0.5).float()
            dice = dice_coefficient(pred_binary, mask)

            # Compute total loss
            #beta = 1.0
            total_loss = (recon_loss + config.beta * kl_div).mean()

            # Track metrics
            val_loss += total_loss.item()
            val_kl += kl_div.mean().item()
            val_dice += dice.mean().item()

            # Save validation images once per epoch
            if batch_idx == 0:
                save_images(image[0], mask[0], pred_binary[0], epoch, mode="val")

            # Update progress bar
            progress_bar.set_postfix({
                "Loss": total_loss.item(),
                "KL": kl_div.mean().item(),
                "Dice": dice.mean().item()
            })

    # Compute average metrics
    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_kl = epoch_kl / len(train_loader)
    avg_train_dice = epoch_dice / len(train_loader)

    wandb.log({"Train Loss": avg_train_loss,"Train_KL_Loss":avg_train_kl, "Train_Dice_Loss":avg_train_dice, "Epoch": epoch + 1})

    avg_val_loss = val_loss / len(val_loader)
    avg_val_kl = val_kl / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    wandb.log({"Val Loss": avg_val_loss,"Val_KL_Loss":avg_val_kl, "Val_Dice_Loss":avg_val_dice , "Epoch": epoch + 1})

    print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f}, Train KL: {avg_train_kl:.6f}, Train Dice: {avg_train_dice:.6f}")
    print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.6f}, Val KL: {avg_val_kl:.6f}, Val Dice: {avg_val_dice:.6f}")


    # Save the best model based on validation Dice score
    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        best_model_path = os.path.join(model_dir, f"dend_model_epoch_{epoch+1}_dice_{best_val_dice:.4f}.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved: {best_model_path}")

print("Training & Validation Completed!")
