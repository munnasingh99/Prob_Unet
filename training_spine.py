import torch
import numpy as np
from torch.utils.data import DataLoader
from datagen import DataGeneratorDataset
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
from tqdm import tqdm
import os

# Set up device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize datasets
train_dataset = DataGeneratorDataset(
    r"DeepD3_Training.d3set",
    samples_per_epoch=256*256,
    size=(1, 128, 128),
    augment=True,
    shuffle=True,
)

val_dataset = DataGeneratorDataset(
    r"DeepD3_Validation.d3set",
    samples_per_epoch=64*8,
    size=(1, 128, 128),
    augment=False,
    shuffle=False,
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True)

# Initialize the network
net = ProbabilisticUnet(
    input_channels=1,
    num_classes=1,
    num_filters=[32, 64, 128, 192],
    latent_dim=6,
    no_convs_fcomb=4,
    beta=20,
)
net.to(device)

# Setup optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)

# Create directory for saving models
save_dir = 'model_checkpoints'
os.makedirs(save_dir, exist_ok=True)

# Dice coefficient function for evaluation
def dice_coefficient(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()  # Threshold predictions
    target = (target > 0.5).float()  # Ensure binary targets
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Training and validation loop
epochs = 10
best_val_loss = float('inf')

for epoch in range(epochs):
    net.train()
    epoch_loss = 0
    batch_count = 0

    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
    for step, (patch, (mask, _)) in enumerate(train_pbar):
        patch = patch.to(device)
        mask = mask.to(device)

        # Forward pass
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)

        # Calculate losses
        reg_loss = (
            l2_regularisation(net.posterior) +
            l2_regularisation(net.prior) +
            l2_regularisation(net.fcomb.layers)
        )
        loss = -elbo + 1e-5 * reg_loss

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1
        train_pbar.set_postfix({'avg_loss': f'{(epoch_loss / batch_count):.4f}'})

    avg_epoch_loss = epoch_loss / batch_count
    print(f"Epoch {epoch+1} Training Loss: {avg_epoch_loss:.4f}")

    # Validation loop
    net.eval()
    val_loss = 0
    dice_score = 0
    with torch.no_grad():
        for patch, (mask, _) in val_loader:
            patch, mask = patch.to(device), mask.to(device)

            net.forward(patch, mask, training=False)
            elbo = net.elbo(mask)

            reg_loss = (
                l2_regularisation(net.posterior) +
                l2_regularisation(net.prior) +
                l2_regularisation(net.fcomb.layers)
            )
            loss = -elbo + 1e-5 * reg_loss
            val_loss += loss.item()

            # Dice coefficient
            predictions = net.sample()
            dice_score += dice_coefficient(predictions, mask).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_dice_score = dice_score / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}, Dice Coefficient: {avg_dice_score:.4f}")

    # Save the model if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_path = os.path.join(save_dir, f'best_model_epoch_{epoch+1}_loss_{avg_val_loss:.4f}.pth')
        torch.save(net.state_dict(), save_path)
        print(f"Saved best model checkpoint to: {save_path}")

print("Training completed!")
