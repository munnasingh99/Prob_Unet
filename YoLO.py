import os
import numpy as np
import cv2
import torch
from skimage.measure import label, regionprops
from torch.utils.data import Dataset
from ultralytics import YOLO
import flammkuchen as fl
import copy
from yolo_datagen import DataGeneratorDataset

class YOLOSpineDatasetPreparation:
    """Helper class to convert DeepD3 dataset to YOLOv8 format"""
    
    def __init__(self, d3set_path, output_dir, train_val_split=0.8):
        """
        Initialize the converter
        
        Args:
            d3set_path (str): Path to the DeepD3_Training.d3set file
            output_dir (str): Directory to save YOLOv8 formatted dataset
            train_val_split (float): Ratio of training samples (default: 0.8)
        """
        self.d3set_path = d3set_path
        self.output_dir = output_dir
        self.train_val_split = train_val_split
        
        # Create directory structure
        os.makedirs(f"{output_dir}/images/train", exist_ok=True)
        os.makedirs(f"{output_dir}/images/val", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/val", exist_ok=True)
        
        # Load the data
        self.d = fl.load(self.d3set_path)
        self.data = self.d['data']
        self.meta = self.d['meta']
        
    def generate_samples(self, num_samples=1000, size=(128, 128)):
        """
        Generate samples for YOLOv8 training
        
        Args:
            num_samples (int): Number of samples to generate
            size (tuple): Size of each sample (height, width)
        """
        # Use DataGeneratorDataset to generate samples
        dataset = DataGeneratorDataset(
            self.d3set_path, 
            samples_per_epoch=num_samples,
            size=size,
            augment=True
        )
        
        for i in range(num_samples):
            # Get a sample
            image, (dendrite, spines) = dataset[i]
            
            # Convert tensors to numpy arrays
            image_np = image.numpy().squeeze()
            spines_np = spines.numpy().squeeze()
            
            # Normalize image to 0-255 for saving
            image_uint8 = ((image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255).astype(np.uint8)
            
            # Create instance labels from spine mask using connected components
            spine_instances = label(spines_np > 0.5)
            
            # Split into train/val
            split = "train" if np.random.rand() < self.train_val_split else "val"
            
            # Save image
            cv2.imwrite(f"{self.output_dir}/images/{split}/{i:06d}.png", image_uint8)
            
            # Process each spine instance
            with open(f"{self.output_dir}/labels/{split}/{i:06d}.txt", 'w') as f:
                # Get properties of each labeled region
                regions = regionprops(spine_instances)
                
                for region in regions:
                    # Skip very small regions (optional)
                    if region.area < 10:  # Adjust threshold as needed
                        continue
                    
                    # Get bounding box (y_min, x_min, y_max, x_max)
                    y_min, x_min, y_max, x_max = region.bbox
                    
                    # Convert to YOLO format: class x_center y_center width height
                    # All values normalized by image dimensions
                    h, w = spines_np.shape
                    x_center = ((x_min + x_max) / 2) / w
                    y_center = ((y_min + y_max) / 2) / h
                    width = (x_max - x_min) / w
                    height = (y_max - y_min) / h
                    
                    # Save in YOLO format (class_id x_center y_center width height)
                    # Class 0 is for spines
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")
            
            # Progress update
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{num_samples} samples")
    
    def create_data_yaml(self):
        """Create the dataset.yaml file for YOLOv8 training"""
        yaml_content = f"""# Dataset configuration for YOLOv8 training
path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val

# Classes
nc: 1  # Number of classes
names: ['spine']  # Class names
"""
        
        with open(f"{self.output_dir}/dataset.yaml", 'w') as f:
            f.write(yaml_content)


class SpineInstanceSegmentationPipeline:
    """Pipeline that combines standard segmentation with YOLOv8 for instance segmentation"""
    
    def __init__(self, d3set_path, yolo_model_path):
        """
        Initialize the pipeline
        
        Args:
            d3set_path (str): Path to the DeepD3_Training.d3set file 
            yolo_model_path (str): Path to trained YOLOv8 model
        """
        self.data_generator = DataGeneratorDataset(d3set_path)
        self.yolo_model = YOLO(yolo_model_path)
    
    def process_sample(self, idx):
        """
        Process a single sample through both segmentation and instance detection
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (image, dendrite_mask, spine_mask, spine_instances)
        """
        # Get segmentation results
        image, (dendrite, spines) = self.data_generator[idx]
        
        # Convert to numpy for YOLOv8
        image_np = image.numpy().squeeze()
        
        # Normalize image to 0-255 for YOLOv8
        image_uint8 = ((image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255).astype(np.uint8)
        
        # Run YOLOv8 for instance detection
        results = self.yolo_model(image_uint8)
        
        # Get detected spine instances
        spine_instances = []
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            
            # Process each detected spine
            for i in range(len(boxes)):
                box = boxes[i].xyxy.cpu().numpy()[0]  # Get box coordinates (x1, y1, x2, y2)
                conf = boxes[i].conf.cpu().numpy()[0]  # Get confidence
                
                # Skip low confidence detections
                if conf < 0.25:  # Adjust threshold as needed
                    continue
                
                # Create a binary mask for this instance
                instance_mask = np.zeros_like(image_np, dtype=bool)
                
                # Convert box coordinates to integers
                x1, y1, x2, y2 = box.astype(int)
                
                # Apply segmentation mask to get the actual spine shape within the box
                # Only include pixels that are part of the spine segmentation
                spines_np = spines.numpy().squeeze()
                instance_mask[y1:y2, x1:x2] = spines_np[y1:y2, x1:x2] > 0.5
                
                spine_instances.append(instance_mask)
        
        return image, dendrite, spines, spine_instances


def train_yolo_for_spines(d3set_path, output_dir, epochs=100, img_size=640):
    """
    Train YOLOv8 model for spine instance segmentation
    
    Args:
        d3set_path (str): Path to the DeepD3_Training.d3set file
        output_dir (str): Directory to save YOLOv8 dataset and trained model
        epochs (int): Number of training epochs
        img_size (int): Image size for training
        
    Returns:
        str: Path to the trained model
    """
    # Prepare dataset
    print("Preparing dataset...")
    dataset_prep = YOLOSpineDatasetPreparation(d3set_path, f"{output_dir}/dataset")
    dataset_prep.generate_samples(num_samples=2000)  # Adjust number as needed
    dataset_prep.create_data_yaml()
    
    # Initialize YOLOv8 model
    print("Initializing YOLOv8 model...")
    model = YOLO('yolo11n.pt')  # Use smallest model to start
    
    # Train model
    print("Training YOLOv8 model...")
    model.train(
        data=f"{output_dir}/dataset/dataset.yaml",
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        patience=20,
        save=True
    )
    
    # Path to best model
    best_model_path = f"runs/detect/train/weights/best.pt"
    print(f"Training complete. Best model saved at: {best_model_path}")
    
    return best_model_path


def visualize_results(image, dendrite_mask, spine_mask, spine_instances, output_path=None):
    """
    Visualize the segmentation and instance results
    
    Args:
        image (np.ndarray): Original image
        dendrite_mask (np.ndarray): Dendrite segmentation mask
        spine_mask (np.ndarray): Spine segmentation mask
        spine_instances (list): List of spine instance masks
        output_path (str, optional): Path to save visualization
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    # Convert tensors to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.numpy().squeeze()
    if isinstance(dendrite_mask, torch.Tensor):
        dendrite_mask = dendrite_mask.numpy().squeeze()
    if isinstance(spine_mask, torch.Tensor):
        spine_mask = spine_mask.numpy().squeeze()
    
    # Normalize image for display
    image_norm = (image - image.min()) / (image.max() - image.min())
    
    # Create instance visualization
    instance_vis = np.zeros_like(image)
    colors = plt.cm.get_cmap('tab20', len(spine_instances) + 1)
    
    for i, instance in enumerate(spine_instances):
        instance_vis[instance] = i + 1
    
    # Create figure
    plt.figure(figsize=(18, 12))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(image_norm, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Dendrite mask
    plt.subplot(2, 2, 2)
    plt.imshow(image_norm, cmap='gray')
    plt.imshow(dendrite_mask, alpha=0.5, cmap='Blues')
    plt.title('Dendrite Segmentation')
    plt.axis('off')
    
    # Spine mask
    plt.subplot(2, 2, 3)
    plt.imshow(image_norm, cmap='gray')
    plt.imshow(spine_mask, alpha=0.5, cmap='Reds')
    plt.title('Spine Segmentation')
    plt.axis('off')
    
    # Spine instances
    plt.subplot(2, 2, 4)
    plt.imshow(image_norm, cmap='gray')
    plt.imshow(instance_vis, alpha=0.7, cmap=ListedColormap(['transparent'] + [colors(i) for i in range(len(spine_instances))]))
    plt.title(f'Spine Instances ({len(spine_instances)} detected)')
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    # Paths
    d3set_path = "dataset/DeepD3_Training.d3set"  # Path to your dataset
    output_dir = "train_spine_yolo"
    
    # Step 1: Train YOLOv8 model
    best_model_path = train_yolo_for_spines(d3set_path, output_dir)
    
    # Step 2: Initialize the pipeline
    pipeline = SpineInstanceSegmentationPipeline(d3set_path, best_model_path)
    
    # Step 3: Process a few samples
    for i in range(5):
        # Process sample
        image, dendrite, spines, spine_instances = pipeline.process_sample(i)
        
        # Visualize results
        visualize_results(
            image, dendrite, spines, spine_instances,
            output_path=f"{output_dir}/sample_{i}.png"
        )
        
    print("Done!")