import numpy as np
import os
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time

class UNet(nn.Module):
    """
    U-Net architecture for Noise2Void denoising
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._block(512, 1024)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = self._block(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = self._block(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._block(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = self._block(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec1 = self.up1(bottleneck)
        dec1 = torch.cat((dec1, enc4), dim=1)
        dec1 = self.dec1(dec1)
        
        dec2 = self.up2(dec1)
        dec2 = torch.cat((dec2, enc3), dim=1)
        dec2 = self.dec2(dec2)
        
        dec3 = self.up3(dec2)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)
        
        dec4 = self.up4(dec3)
        dec4 = torch.cat((dec4, enc1), dim=1)
        dec4 = self.dec4(dec4)
        
        # Final layer
        return self.final(dec4)

class Noise2VoidDataset(Dataset):
    """
    Dataset for training Noise2Void model on image tiles
    """
    def __init__(self, images, patch_size=64, mask_probability=0.2, augment=True):
        """
        Initialize the dataset
        
        Parameters:
            images: List of 2D numpy arrays containing image data
            patch_size: Size of the patches to extract
            mask_probability: Probability of masking a pixel
            augment: Whether to use data augmentation
        """
        self.images = images
        self.patch_size = patch_size
        self.mask_probability = mask_probability
        self.augment = augment
        
        # Extract patches from images
        self.patches = self._extract_patches()
        print(f"Extracted {len(self.patches)} patches of size {patch_size}x{patch_size}")
        
    def _extract_patches(self):
        """Extract patches from images"""
        patches = []
        
        for image in self.images:
            h, w = image.shape
            
            # Define number of patches to extract based on image size
            n_h = max(1, (h - self.patch_size) // (self.patch_size // 2))
            n_w = max(1, (w - self.patch_size) // (self.patch_size // 2))
            
            for i in range(n_h):
                for j in range(n_w):
                    # Extract patch
                    y = i * (self.patch_size // 2)
                    x = j * (self.patch_size // 2)
                    patch = image[y:y+self.patch_size, x:x+self.patch_size]
                    
                    # Skip patches that are too small
                    if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                        continue
                    
                    patches.append(patch)
        
        return patches
    
    def __len__(self):
        return len(self.patches)
    
    def _augment(self, patch):
        """Apply random augmentations to a patch"""
        # Random rotation
        k = np.random.randint(0, 4)
        patch = np.rot90(patch, k)
        
        # Random flip
        if np.random.rand() > 0.5:
            patch = np.fliplr(patch)
        
        return patch
    
    def __getitem__(self, idx):
        # Get patch
        patch = self.patches[idx].copy()
        
        # Apply augmentation
        if self.augment:
            patch = self._augment(patch)
        
        # Normalize to [0, 1]
        if patch.max() > 0:
            patch = patch.astype(np.float32) / patch.max()
        
        # Create input and target tensors
        input_tensor = torch.from_numpy(patch).float().unsqueeze(0)  # Add channel dimension
        target_tensor = input_tensor.clone()
        
        # Create blind-spot mask (Noise2Void approach)
        mask = torch.rand_like(input_tensor) < self.mask_probability
        
        # Replace masked pixels with random neighbors (blind-spot)
        masked_input = input_tensor.clone()
        
        # For each masked pixel, replace with a random neighbor
        for i in range(1, masked_input.shape[1] - 1):
            for j in range(1, masked_input.shape[2] - 1):
                if mask[0, i, j]:
                    # Get random offset (-1, 0, or 1 for both dimensions)
                    offset_i = np.random.randint(-1, 2)
                    offset_j = np.random.randint(-1, 2)
                    
                    # If offset is (0,0), retry
                    if offset_i == 0 and offset_j == 0:
                        if np.random.rand() < 0.5:
                            offset_i = -1
                        else:
                            offset_i = 1
                    
                    # Replace with neighbor
                    masked_input[0, i, j] = input_tensor[0, i + offset_i, j + offset_j]
        
        return masked_input, target_tensor, mask

def train_noise2void(images, model_save_path=None, patch_size=64, batch_size=16, 
                    learning_rate=0.001, num_epochs=100, device=None):
    """
    Train Noise2Void model on a set of images
    
    Parameters:
        images: List of 2D numpy arrays containing image data
        model_save_path: Path to save the trained model
        patch_size: Size of the patches to extract
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        device: Device to use for training (cuda or cpu)
        
    Returns:
        model: Trained Noise2Void model
        losses: List of training losses
    """
    # Use GPU if available
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = Noise2VoidDataset(images, patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Create model
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_inputs, batch_targets, batch_masks in pbar:
            # Move to device
            inputs = batch_inputs.to(device)
            targets = batch_targets.to(device)
            masks = batch_masks.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss only on masked pixels (Noise2Void approach)
            loss = criterion(outputs * masks, targets * masks)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        # Update learning rate
        epoch_loss /= len(dataloader)
        scheduler.step(epoch_loss)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        
        # Save model periodically
        if model_save_path and (epoch + 1) % 10 == 0:
            checkpoint_path = model_save_path.replace('.pth', f'_e{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
    
    # Save final model
    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print(f"Final model saved to {model_save_path}")
    
    return model, losses

def denoise_image(model, image, device=None):
    """
    Denoise a single image using Noise2Void model
    
    Parameters:
        model: Trained Noise2Void model
        image: 2D numpy array containing image data
        device: Device to use for inference
        
    Returns:
        Denoised image as numpy array
    """
    # Use GPU if available
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Normalize to [0, 1]
    orig_max = image.max()
    if orig_max > 0:
        image_norm = image.astype(np.float32) / orig_max
    else:
        image_norm = image.astype(np.float32)
    
    # Create tensor
    input_tensor = torch.from_numpy(image_norm).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Move to device
    input_tensor = input_tensor.to(device)
    
    # Process image in tiles if it's large
    with torch.no_grad():
        h, w = image.shape
        tile_size = 512  # Adjust based on GPU memory
        
        if h <= tile_size and w <= tile_size:
            # Small image, process directly
            output_tensor = model(input_tensor)
            
            # Convert back to numpy
            denoised = output_tensor.squeeze().cpu().numpy()
        else:
            # Large image, process in tiles with overlap
            stride = tile_size // 2
            padding = 64  # Overlap padding to avoid boundary artifacts
            
            # Initialize output array
            denoised = np.zeros_like(image_norm)
            weight = np.zeros_like(image_norm)
            
            # Pad image
            padded = np.pad(image_norm, padding, mode='reflect')
            padded_tensor = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Process tiles
            for i in range(0, h, stride):
                for j in range(0, w, stride):
                    # Get tile coordinates with padding
                    i_start = i
                    j_start = j
                    i_end = min(i + tile_size, h)
                    j_end = min(j + tile_size, w)
                    
                    # Extract tile from padded image
                    tile_tensor = padded_tensor[:, :, 
                                              i_start:i_end+2*padding, 
                                              j_start:j_end+2*padding]
                    
                    # Process tile
                    output_tile = model(tile_tensor)
                    
                    # Remove padding
                    output_tile = output_tile[:, :, padding:-padding, padding:-padding]
                    
                    # Add to output with proper weight
                    tile_np = output_tile.squeeze().cpu().numpy()
                    denoised[i_start:i_end, j_start:j_end] += tile_np
                    weight[i_start:i_end, j_start:j_end] += 1
            
            # Average overlapping regions
            denoised = denoised / np.maximum(weight, 1e-8)
    
    # Rescale to original range
    denoised = denoised * orig_max
    
    return denoised

def process_tif_with_noise2void(tif_path, output_dir="denoised", model_path=None, 
                               train_slices=10, num_epochs=100, patch_size=64, 
                               batch_size=16, learning_rate=0.001, device=None):
    """
    Process a multi-stack TIF file with Noise2Void denoising
    
    Parameters:
        tif_path: Path to the TIF file
        output_dir: Directory to save the denoised TIF
        model_path: Path to save/load the model
        train_slices: Number of slices to use for training
        num_epochs: Number of training epochs
        patch_size: Size of the patches to extract
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        device: Device to use (cuda or cpu)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths
    output_tif_path = os.path.join(output_dir, os.path.basename(tif_path).replace('.tif', '_denoised.tif'))
    comparison_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # If model path is not specified, create one based on the TIF filename
    if model_path is None:
        model_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(tif_path))[0]}_n2v.pth")
    
    # Load TIF file
    print(f"Loading TIF file: {tif_path}")
    tif_data = tifffile.imread(tif_path)
    
    # Determine dimensions
    if len(tif_data.shape) == 3:  # Multi-stack grayscale
        num_stacks = tif_data.shape[0]
        is_grayscale = True
    elif len(tif_data.shape) == 4:  # Multi-stack RGB
        num_stacks = tif_data.shape[0]
        is_grayscale = False
    else:  # Single grayscale image
        num_stacks = 1
        is_grayscale = len(tif_data.shape) == 2
        # Reshape to add a dimension
        if is_grayscale:
            tif_data = np.expand_dims(tif_data, 0)
    
    print(f"TIF dimensions: {tif_data.shape}, Grayscale: {is_grayscale}")
    
    # Process RGB images
    if not is_grayscale:
        print("RGB TIF detected. Will process each channel separately.")
    
    # Create model on selected device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    # Train or load model
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Training new model on {train_slices} slices for {num_epochs} epochs")
        
        # Select training slices
        if num_stacks <= train_slices:
            train_indices = range(num_stacks)
        else:
            # Sample evenly from the stack
            train_indices = np.linspace(0, num_stacks-1, train_slices, dtype=int)
        
        print(f"Using slices {train_indices} for training")
        
        # Prepare training data
        train_images = []
        for i in train_indices:
            if is_grayscale:
                train_images.append(tif_data[i])
            else:
                # For RGB, convert to grayscale for simplicity
                # Convert all channels to grayscale
                for c in range(tif_data.shape[3]):
                    train_images.append(tif_data[i, :, :, c])
        
        # Train model
        model, losses = train_noise2void(
            images=train_images,
            model_save_path=model_path,
            patch_size=patch_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device=device
        )
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Process each slice
    print(f"Denoising {num_stacks} slices")
    denoised_tif = np.zeros_like(tif_data)
    
    # Process slices with progress bar
    for i in tqdm(range(num_stacks), desc="Denoising slices"):
        if is_grayscale:
            # Denoise the image
            denoised_tif[i] = denoise_image(model, tif_data[i], device=device)
            
            # Create comparison visualization
            create_comparison_visualization(tif_data[i], denoised_tif[i], i, comparison_dir)
        else:
            # Process each channel separately
            for c in range(tif_data.shape[3]):
                denoised_tif[i, :, :, c] = denoise_image(model, tif_data[i, :, :, c], device=device)
            
            # Create comparison visualization for RGB
            create_rgb_comparison_visualization(tif_data[i], denoised_tif[i], i, comparison_dir)
    
    # Save denoised TIF
    print(f"Saving denoised TIF to {output_tif_path}")
    tifffile.imwrite(output_tif_path, denoised_tif)
    
    # Create animated GIF of comparisons
    create_comparison_gif(comparison_dir, os.path.join(output_dir, "comparison_animation.gif"))
    
    print(f"Processing complete. Denoised TIF saved to {output_tif_path}")
    return output_tif_path

def create_comparison_visualization(original, denoised, slice_idx, output_dir):
    """
    Create and save a comparison visualization between original and denoised images
    
    Parameters:
        original: Original image
        denoised: Denoised image
        slice_idx: Index of the slice
        output_dir: Directory to save the comparison image
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original
    vmin = np.min(original)
    vmax = np.max(original)
    
    axes[0].imshow(original, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Plot denoised
    axes[1].imshow(denoised, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title("Denoised (Noise2Void)")
    axes[1].axis('off')
    
    # Plot difference
    diff = np.abs(original.astype(np.float32) - denoised.astype(np.float32))
    diff_normalized = diff / diff.max() if diff.max() > 0 else diff
    
    axes[2].imshow(diff_normalized, cmap='hot')
    axes[2].set_title("Difference")
    axes[2].axis('off')
    
    plt.suptitle(f"Slice {slice_idx}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_slice_{slice_idx:03d}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_rgb_comparison_visualization(original, denoised, slice_idx, output_dir):
    """
    Create and save a comparison visualization between original and denoised RGB images
    
    Parameters:
        original: Original RGB image
        denoised: Denoised RGB image
        slice_idx: Index of the slice
        output_dir: Directory to save the comparison image
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Plot denoised
    axes[1].imshow(denoised)
    axes[1].set_title("Denoised (Noise2Void)")
    axes[1].axis('off')
    
    # Plot difference
    diff = np.abs(original.astype(np.float32) - denoised.astype(np.float32))
    diff_sum = np.sum(diff, axis=2)
    diff_normalized = diff_sum / diff_sum.max() if diff_sum.max() > 0 else diff_sum
    
    axes[2].imshow(diff_normalized, cmap='hot')
    axes[2].set_title("Difference")
    axes[2].axis('off')
    
    plt.suptitle(f"Slice {slice_idx}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_slice_{slice_idx:03d}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_gif(input_dir, output_path, duration=200):
    """
    Create an animated GIF from all comparison images
    
    Parameters:
        input_dir: Directory containing comparison images
        output_path: Path to save the GIF
        duration: Duration of each frame in milliseconds
    """
    try:
        import imageio
        
        # Get all comparison PNG files
        png_files = sorted([f for f in os.listdir(input_dir) if f.startswith('comparison_') and f.endswith('.png')])
        
        if not png_files:
            print("No comparison PNG files found for GIF creation")
            return
            
        # Read all images
        images = [imageio.imread(os.path.join(input_dir, f)) for f in png_files]
        
        # Create GIF
        imageio.mimsave(output_path, images, duration=duration/1000)
        print(f"Created comparison GIF at {output_path}")
    except ImportError:
        print("imageio library not found. Install with: pip install imageio")
        print("GIF creation skipped.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Denoise a TIF file using Noise2Void")
    parser.add_argument("--tif", required=True, help="Path to the TIF file")
    parser.add_argument("--output", default="denoised", help="Directory to save the denoised TIF")
    parser.add_argument("--model", default="n2v", help="Path to save/load the model")
    parser.add_argument("--train-slices", type=int, default=10, help="Number of slices to use for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--patch-size", type=int, default=64, help="Size of the patches to extract")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (default: use any available GPU)")
    
    args = parser.parse_args()
    
    # Set GPU device if specified
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    else:
        device = None
    
    # Process the TIF file
    process_tif_with_noise2void(
        tif_path=args.tif,
        output_dir=args.output,
        model_path=args.model,
        train_slices=args.train_slices,
        num_epochs=args.epochs,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device
    )