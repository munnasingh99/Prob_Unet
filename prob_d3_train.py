import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import cv2
from torch.utils.data import DataLoader
from datagen import DataGeneratorDataset
from new_prob import ProbabilisticUnet
import wandb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# -------------------------------
# Setup & Configuration
# -------------------------------
torch.manual_seed(42)
wandb.init(
    project="prob-unet-run_exp_deepd3",
    config={
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "latent_dim": 16,
        "beta": 1,
        # You can adjust this if you want to sample more/less frequently:
        "sample_num": 10  # number of samples per final step
    },
)
config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directories for saved predictions/models, etc.
output_dir = "pred_prob_deepd3/"
prob_dir = "prob_prob_deepd3/"
model_dir = "saved_models/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(prob_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# -------------------------------
# Data Loading
# -------------------------------
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

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)

# -------------------------------
# Model, Loss, and Optimizer Setup
# -------------------------------
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
best_val_iou_spine, best_val_iou_dend = 0.0, 0.0

def iou_score(pred_logits, target, threshold=0.5, epsilon=1e-6):
    pred_probs = torch.sigmoid(pred_logits)
    pred_binary = (pred_probs > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    return (intersection + epsilon) / (union + epsilon)

# -------------------------------
# Fixed Images for Consistent Sampling
# -------------------------------
# Choose one fixed training image and one fixed validation image (same throughout training)
random_index = random.randint(0, len(val_dataset)-1)
fixed_train_image, _ = train_dataset[random_index]  # Using the first sample (change index if needed)
fixed_train_image = fixed_train_image.to(device)

fixed_val_image, _ = val_dataset[random_index]
fixed_val_image = fixed_val_image.to(device)

# -------------------------------
# Function to Perform Sampling and Save Results
# -------------------------------
def sample_and_save(model, fixed_img, epoch, mode="train", num_samples=10):
    """
    Given a fixed image, run a forward pass then sample num_samples times.
    Saves the sampled probability maps to disk.
    """
    samples_spine = []
    samples_dend = []
    filename = os.path.join(output_dir, f"{mode}_epoch{epoch}.png")
    cv2.imwrite(filename, (fixed_img.squeeze().cpu().numpy() * 255).astype(np.uint8))
    # Ensure the image has a batch dimension
    fixed_img_batch = fixed_img.unsqueeze(0)
    with torch.no_grad():
        # Run forward pass (if needed to update latent variables)
        model.forward(fixed_img_batch,None, training=False)
        for i in range(num_samples):
            # Sample the output. (Assumes model.sample accepts an input image.)
            sample_logits_dend,sample_logits_spine = model.sample()
            sample_prob_dend = torch.sigmoid(sample_logits_dend).cpu().numpy().squeeze() 
            sample_prob_spine = torch.sigmoid(sample_logits_spine).cpu().numpy().squeeze()  # shape (128, 128)
            samples_spine.append(sample_prob_dend)
            samples_dend.append(sample_prob_spine)
            # Save the sample image
            filename_spine = os.path.join(output_dir, f"{mode}_epoch{epoch}_spine_sample{i}.png")
            filename_dendrite = os.path.join(output_dir, f"{mode}_epoch{epoch}_dendrite_sample{i}.png")
            cv2.imwrite(filename_spine, (sample_prob_spine * 255).astype(np.uint8))
            cv2.imwrite(filename_dendrite, (sample_prob_dend * 255).astype(np.uint8))
            # if i == 0:
            #     np.save(f"{mode}_epoch{epoch}_sample{i}_dend.npy", sample_logits_dend.cpu().numpy())
            #     np.save(f"{mode}_epoch{epoch}_sample{i}_spine.npy", sample_logits_spine.cpu().numpy())
    return samples_dend, samples_spine

# -------------------------------
# Training Loop with Sampling at Final Step of Each Epoch
# -------------------------------
global_step = 0
for epoch in range(config.epochs):
    model.train()
    train_loss_epoch = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
    for batch_idx, (image, (dendrite_mask, spine_mask)) in progress_bar:
        image = image.to(device)
        d_mask = dendrite_mask.to(device)
        s_mask = spine_mask.to(device)
        segm = torch.cat([d_mask, s_mask], dim=1)

        # Forward pass and compute loss for training
        model.forward(image, segm, training=True)
        kl_div = model.kl_divergence()
        pred_logits_dend, pred_logits_spine = model.sample()
        loss = (criterion(pred_logits_dend, d_mask) +
                criterion(pred_logits_spine, s_mask) +
                config.beta * kl_div.mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item()
        global_step += 1
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss_epoch / len(train_loader)
    wandb.log({"Train Loss": avg_train_loss, "Epoch": epoch+1})
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f}")

    # At the end of training epoch, sample the fixed training image 10 times.
    train_samples_dend,train_samples_spine = sample_and_save(model, fixed_train_image, epoch+1, mode="train", num_samples=config.sample_num)
    # Optionally, you can log one of these sample images to wandb:
    #wandb.log({"Train Sample": wandb.Image(train_samples[0], caption=f"Train Sample Epoch {epoch+1}")})

    # ---------------------------
    # Validation Loop
    # ---------------------------
    model.eval()
    val_loss, val_iou_dend, val_iou_spine = 0.0, 0.0, 0.0
    with torch.no_grad():
        progress_bar_val = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1}/{config.epochs} [Validation]")
        for batch_idx, (image, (dendrite_mask, spine_mask)) in progress_bar_val:
            image = image.to(device)
            d_mask = dendrite_mask.to(device)
            s_mask = spine_mask.to(device)
            segm = torch.cat([d_mask, s_mask], dim=1)

            model.forward(image, segm, training=False)
            pred_logits_dend, pred_logits_spine = model.sample()
            loss = criterion(pred_logits_dend, d_mask) + criterion(pred_logits_spine, s_mask)
            val_loss += loss.item()
            val_iou_dend += iou_score(pred_logits_dend, d_mask).item()
            val_iou_spine += iou_score(pred_logits_spine, s_mask).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou_dend = val_iou_dend / len(val_loader)
        avg_val_iou_spine = val_iou_spine / len(val_loader)
        wandb.log({
            "Val Loss": avg_val_loss,
            "Val IoU Dendrite": avg_val_iou_dend,
            "Val IoU Spine": avg_val_iou_spine,
            "Epoch": epoch+1
        })
        print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.6f}, Val IoU Dendrite: {avg_val_iou_dend:.2f}, Val IoU Spine: {avg_val_iou_spine:.2f}")

    # At the end of validation epoch, sample the fixed validation image 10 times.
    val_samples_dend,val_samples_spine = sample_and_save(model, fixed_val_image, epoch+1, mode="val", num_samples=config.sample_num)
    #wandb.log({"Val Sample": wandb.Image(val_samples[0], caption=f"Val Sample Epoch {epoch+1}")})

    # Optionally save the model if performance improves.
    if avg_val_iou_spine > best_val_iou_spine or avg_val_iou_dend > best_val_iou_dend:
        best_val_iou_spine = max(best_val_iou_spine, avg_val_iou_spine)
        best_val_iou_dend = max(best_val_iou_dend, avg_val_iou_dend)
        torch.save(model.state_dict(), os.path.join(model_dir, f"model_epoch_{epoch+1}_val_loss_{avg_val_loss:.4f}.pth"))

print("Training Completed!")
