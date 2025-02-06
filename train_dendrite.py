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

torch.manual_seed(42)
wandb.init(
    project="probabilistic-unet",
    config={
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "latent_dim": 16,
        "beta": 1
    },
)
config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directories
output_dir = "predictions_dend/"
model_dir = "saved_models/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

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

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred_logits, target):
        pred_probs = torch.sigmoid(pred_logits)
        intersection = (pred_probs * target).sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (pred_probs.sum() + target.sum() + 1e-6)
        bce_loss = self.bce(pred_logits, target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

criterion = CombinedLoss(alpha=0.5)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
best_val_iou = 0.0

def iou_score(pred_logits, target, threshold=0.5, epsilon=1e-6):
    pred_probs = torch.sigmoid(pred_logits)
    pred_binary = (pred_probs > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    return (intersection + epsilon) / (union + epsilon)

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

# Training Loop
for epoch in range(config.epochs):
    model.train()
    epoch_loss, epoch_kl, epoch_iou = 0.0, 0.0, 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs} [Train]")

    for batch_idx, (image, (dendrite_mask, spine_mask)) in progress_bar:
        image, mask = image.to(device), dendrite_mask.to(device)

        model.forward(image, mask, training=True)
        kl_div = model.kl_divergence()
        pred_logits = model.sample()
        loss = criterion(pred_logits, mask) + config.beta * kl_div.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_kl += kl_div.mean().item()
        epoch_iou += iou_score(pred_logits, mask).item()

        # Save images only for the first batch of the epoch
        if batch_idx == 0:
            save_images(image[0], mask[0], pred_logits[0], epoch, mode="train")


    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_iou = epoch_iou / len(train_loader)
    wandb.log({"Train Loss": avg_train_loss, "Train IoU": avg_train_iou, "Epoch": epoch+1})

    # Validation
    model.eval()
    val_loss, val_iou = 0.0, 0.0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1}/{config.epochs} [Validation]")
        for batch_idx, (image, (dendrite_mask, spine_mask)) in progress_bar:
            image, mask = image.to(device), dendrite_mask.to(device)
            model.forward(image, mask, training=False)
            pred_logits = model.sample()
            loss = criterion(pred_logits, mask)
            val_loss += loss.item()
            val_iou += iou_score(pred_logits, mask).item()

            if batch_idx == 0:
                save_images(image[0], mask[0], pred_logits[0], epoch, mode="val")

    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    wandb.log({"Val Loss": avg_val_loss, "Val IoU": avg_val_iou, "Epoch": epoch+1})


    print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f},Train IoU: {avg_train_iou:.6f}")
    print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.6f},Val IoU: {avg_val_iou:.6f}")

    if avg_val_iou > best_val_iou:
        best_val_iou = avg_val_iou
        torch.save(model.state_dict(), f"dend_model_epoch_{epoch+1}_iou_{best_val_iou:.4f}.pth")

print("Training & Validation Completed!")
