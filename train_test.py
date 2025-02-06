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
output_dir = "predictions_dend_test/"
prob_dir = "probability_maps_dend/"
model_dir = "saved_models/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(prob_dir, exist_ok=True)
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

# Function to save images (binary mask & probability maps)
# def save_images(image, mask, pred_logits, epoch, mode="train"):
#     """
#     Saves:
#         - Input image
#         - Ground truth mask
#         - Binary predicted mask
#         - Probability maps (raw and scaled)
#     """
#     image = image.squeeze(0).cpu().detach().numpy()
#     mask = mask.squeeze(0).cpu().detach().numpy()
    
#     pred_probs = torch.sigmoid(pred_logits).squeeze(0).squeeze(0).cpu().detach().numpy()
#     pred_binary = (pred_probs > 0.5).astype(np.uint8) * 255

#     # Save binary prediction & probability map
#     cv2.imwrite(os.path.join(output_dir, f"{mode}_{epoch}_image.png"), (image * 255).astype(np.uint8))
#     cv2.imwrite(os.path.join(output_dir, f"{mode}_{epoch}_mask.png"), mask * 255)
#     cv2.imwrite(os.path.join(output_dir, f"{mode}_{epoch}_prediction.png"), pred_binary)

#     # Save probability maps (normalized 0-255 & raw numpy)
#     prob_map_scaled = (pred_probs * 255).astype(np.uint8)
#     cv2.imwrite(os.path.join(prob_dir, f"{mode}_{epoch}_prob_map.png"), prob_map_scaled)
#     np.save(os.path.join(prob_dir, f"{mode}_{epoch}_prob_map.npy"), pred_probs)

#     print(f"Saved: {mode}_{epoch}_images & probability maps")

def save_images(image, mask, pred_logits, epoch, sample_idx, mode="train", save_base=False):
    """
    save_base: If True, saves the base image and mask (only needed once per epoch)
    """
    if save_base:
        image_np = image.squeeze(0).cpu().detach().numpy()
        mask_np = mask.squeeze(0).cpu().detach().numpy()
        cv2.imwrite(os.path.join(output_dir, f"{mode}_{epoch}_image.png"), (image_np * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(output_dir, f"{mode}_{epoch}_mask.png"), mask_np * 255)
    
    pred_probs = torch.sigmoid(pred_logits).squeeze(0).squeeze(0).cpu().detach().numpy()
    pred_binary = (pred_probs > 0.5).astype(np.uint8) * 255
    
    cv2.imwrite(os.path.join(output_dir, f"{mode}_{epoch}_prediction_sample{sample_idx}.png"), pred_binary)
    cv2.imwrite(os.path.join(output_dir, f"{mode}_{epoch}_prob_map_sample{sample_idx}.png"), (pred_probs * 255).astype(np.uint8))
    np.save(os.path.join(output_dir, f"{mode}_{epoch}_prob_map_sample{sample_idx}.npy"), pred_probs)
    print(f"Saved: {mode}_{epoch}_images & probability maps")


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

    # Save images & probability maps for 10 samples
    model.eval()
    with torch.no_grad():
        for sample_idx in range(10):
            pred_logits = model.sample()  # Sample multiple times
            save_images(image[0], mask[0], pred_logits[0],epoch,sample_idx, mode="train",save_base=(sample_idx==0))

    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_iou = epoch_iou / len(train_loader)
    wandb.log({"Train Loss": avg_train_loss, "Train IoU": avg_train_iou, "Epoch": epoch+1})

    # Validation Loop
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

    # Save 10 probability maps during validation
    with torch.no_grad():
        for sample_idx in range(10):
            pred_logits = model.sample()
            save_images(image[0], mask[0], pred_logits[0], epoch, sample_idx, mode="val",save_base=(sample_idx==0))

    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    wandb.log({"Val Loss": avg_val_loss, "Val IoU": avg_val_iou, "Epoch": epoch+1})

    print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f}, Train IoU: {avg_train_iou:.6f}")
    print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.6f}, Val IoU: {avg_val_iou:.6f}")

    if avg_val_iou > best_val_iou:
        best_val_iou = avg_val_iou
        torch.save(model.state_dict(), f"dend_model_epoch_{epoch+1}_iou_{best_val_iou:.4f}.pth")

print("Training & Validation Completed!")
