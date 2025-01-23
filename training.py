import torch
from torch.utils.data import DataLoader
from probabilistic_unet import ProbabilisticUnet
from datagen import DataGeneratorDataset
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
data_path = r"DeepD3_Training.d3set"  # Update with your dataset path
batch_size = 32
epochs = 10
lr = 1e-4

# Load dataset
dataset = DataGeneratorDataset(fn=data_path, size=(1, 128, 128))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Probabilistic U-Net
model = ProbabilisticUnet(input_channels=1, num_classes=2, num_filters=[32, 64, 128], latent_dim=6)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
model.train()
for epoch in range(epochs):
    for images, (dendrite_masks, spine_masks) in dataloader:
        images, dendrite_masks, spine_masks = images.to(device), dendrite_masks.to(device), spine_masks.to(device)

        # Concatenate masks for multi-class segmentation
        model.forward(images, (dendrite_masks, spine_masks), training=True)
        elbo = model.elbo((dendrite_masks, spine_masks))
        loss = -elbo
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
