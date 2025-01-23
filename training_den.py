import torch
import numpy as np
from torch.utils.data import DataLoader
from datagen import DataGeneratorDataset
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
from tqdm import tqdm
import os
from datetime import datetime

# Set up device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize datasets
train_dataset = DataGeneratorDataset(
    r"DeepD3_Training.d3set",
    samples_per_epoch=50000,
    size=(1, 128, 128),
    augment=True,
    shuffle=True,
)

val_dataset = DataGeneratorDataset(
    r"DeepD3_Validation.d3set",
    samples_per_epoch=5000,
    size=(1, 128, 128),
    augment=False,
    shuffle=False,
)

# Create data loaders with progress bar support
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True)

# Initialize the network
net = ProbabilisticUnet(
    input_channels=1,
    num_classes=1,
    num_filters=[32, 64, 128, 192],
    latent_dim=2,
    no_convs_fcomb=4,
    beta=10.0
)
net.to(device)

# Setup optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)

# Create directory for saving models
save_dir = 'model_checkpoints'
os.makedirs(save_dir, exist_ok=True)

# Training configuration
epochs = 10
best_loss = float('inf')

# Training loop with progress tracking
for epoch in range(epochs):
    net.train()
    epoch_loss = 0
    batch_count = 0
    
    # Create progress bar for training batches
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
    
    for step, (patch, (mask,_)) in enumerate(train_pbar):
        # Move data to device
        patch = patch.to(device)
        mask = mask.to(device)
        #mask = torch.unsqueeze(mask, 1)
        
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
        
        # Update progress bar with current loss
        epoch_loss += loss.item()
        batch_count += 1
        avg_loss = epoch_loss / batch_count
        train_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
    
    # Calculate average epoch loss
    avg_epoch_loss = epoch_loss / batch_count
    
    # Save model if it's the best so far
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_dir, f'model_den_epoch_{epoch+1}_loss_{avg_epoch_loss:.4f}_{timestamp}.pth')
        
        # Save model state
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, save_path)
        print(f'\nSaved best model checkpoint to: {save_path}')
    
    print(f'Epoch {epoch+1} completed with average loss: {avg_epoch_loss:.4f}')

print('Training completed!')