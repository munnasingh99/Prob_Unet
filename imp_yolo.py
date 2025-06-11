import os
import numpy as np
import cv2
import torch
from skimage.measure import label, regionprops
from torch.utils.data import Dataset
from ultralytics import YOLO
import flammkuchen as fl
import copy
from datagen import DataGeneratorDataset
from bb_approach import optimized_hybrid_bb

# Set up GPU device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set CUDA visible devices if needed (useful for HPC environments)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Adjust this if you need specific GPUs

# Fix the import error - ensure correct import name
try:
    from ultralytics import YOLO
    print("Successfully imported YOLO from ultralytics")
except ImportError as e:
    print(f"Error importing YOLO: {e}")
    print("Checking available modules in ultralytics...")
    import ultralytics
    print(dir(ultralytics))

def preprocess_image(image_np):
    """
    Enhanced image preprocessing for better contrast in microscopy images
    
    Args:
        image_np (np.ndarray): Input image array
        
    Returns:
        np.ndarray: Preprocessed image (uint8, RGB)
    """
    # Handle NaN values
    image_np = np.nan_to_num(image_np)
    
    # Avoid division by zero
    if np.max(image_np) == np.min(image_np):
        image_uint8 = np.zeros_like(image_np, dtype=np.uint8)
    else:
        # Normalize to 0-255 range
        image_uint8 = (((image_np - np.min(image_np)) / 
                      (np.max(image_np) - np.min(image_np))) * 255).astype(np.uint8)
    
    # Apply CLAHE for better contrast in microscopy images
    if image_np.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_norm = clahe.apply(image_uint8)
        # Convert to RGB for YOLO
        image_rgb = cv2.cvtColor(image_norm, cv2.COLOR_GRAY2RGB)
    else:
        # For RGB images, apply CLAHE to each channel
        image_rgb = np.zeros_like(image_uint8, dtype=np.uint8)
        for i in range(3):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_rgb[:,:,i] = clahe.apply(image_uint8[:,:,i])
            
    return image_rgb

def improve_instance_segmentation(spine_mask, instance_boxes):
    """
    Use watershed algorithm to improve instance segmentation of spines
    
    Args:
        spine_mask (np.ndarray): Binary mask of all spines
        instance_boxes (list): List of bounding boxes from YOLO detection
        
    Returns:
        list: List of instance masks
    """
    # Create markers for watershed
    markers = np.zeros_like(spine_mask, dtype=np.int32)
    
    # Mark each instance with a unique ID
    for i, box in enumerate(instance_boxes):
        x1, y1, x2, y2 = box.astype(int)
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(spine_mask.shape[1] if spine_mask.ndim > 1 else spine_mask.shape[0], x2)
        y2 = min(spine_mask.shape[0], y2)
        
        # Create a small marker in the center of each box
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        markers[max(0, center_y-2):min(markers.shape[0], center_y+2), 
                max(0, center_x-2):min(markers.shape[1], center_x+2)] = i + 1
    
    # Prepare image for watershed
    # Convert spine mask to uint8
    spine_mask_uint8 = spine_mask.astype(np.uint8)
    
    # Distance transform to create watershed basins
    dist_transform = cv2.distanceTransform(spine_mask_uint8, cv2.DIST_L2, 3)
    
    # Apply watershed
    watershed_result = cv2.watershed(-dist_transform, markers)
    
    # Create instance masks
    instance_masks = []
    for i in range(1, watershed_result.max() + 1):
        mask = (watershed_result == i)
        # Only include if it overlaps with the original spine mask
        if np.any(mask & spine_mask):
            instance_masks.append(mask)
            
    return instance_masks

class YOLOSpineDatasetPreparation:
    """Helper class to convert DeepD3 dataset to YOLO segmentation format for instance segmentation"""
    
    def __init__(self, d3set_path, output_dir, train_val_split=0.8):
        """
        Initialize the converter
        
        Args:
            d3set_path (str): Path to the DeepD3_Training.d3set file
            output_dir (str): Directory to save YOLO formatted dataset
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
        
    def generate_samples(self, num_samples=1000, size=(128, 128), augment_intensity=1.5):
        """
        Generate samples for YOLO segmentation training with enhanced augmentation
        
        Args:
            num_samples (int): Number of samples to generate
            size (tuple): Size of each sample (height, width)
            augment_intensity (float): Intensity of augmentation (1.0 = normal)
        """
        # Use DataGeneratorDataset to generate samples
        dataset = DataGeneratorDataset(
            self.d3set_path, 
            samples_per_epoch=num_samples,
            size=size,
            augment=True  # Enable augmentation
        )
        
        # Sample counters
        train_count = 0
        val_count = 0
        
        for i in range(num_samples):
            # Get a sample
            image, (dendrite, spines) = dataset[i]
            
            # Convert tensors to numpy arrays
            image_np = image.numpy().squeeze()
            spines_np = spines.numpy().squeeze()
            
            # Preprocess image with enhanced preprocessing
            image_uint8 = preprocess_image(image_np.copy())
            
            # Create instance labels from spine mask using connected components
            #spine_instances = label(spines_np > 0.5)
            spine_instances, bboxes = optimized_hybrid_bb(image_np, spines_np, dendrite_np)
            num_instances = spine_instances.max()
            
            # If no instances found, skip this sample
            if num_instances == 0:
                continue
            
            # Split into train/val
            split = "train" if np.random.rand() < self.train_val_split else "val"
            
            # Update sample count
            if split == "train":
                train_count += 1
                sample_idx = train_count
            else:
                val_count += 1
                sample_idx = val_count
                
            # Save image
            img_path = f"{self.output_dir}/images/{split}/{sample_idx:06d}.png"
            cv2.imwrite(img_path, image_uint8)
            
            # For segmentation tasks, save both bounding boxes and polygon points
            label_path = f"{self.output_dir}/labels/{split}/{sample_idx:06d}.txt"
            
            # Process each spine instance and create a segmentation mask
            with open(label_path, 'w') as f:
                # Get properties of each labeled region
                regions = regionprops(spine_instances)
                
                for region in regions:
                    # Skip very small regions
                    if region.area < 10:
                        continue
                    
                    # Get bounding box (y_min, x_min, y_max, x_max)
                    y_min, x_min, y_max, x_max = region.bbox
                    
                    # Get the mask for this specific instance
                    instance_mask = (spine_instances == region.label)
                    
                    # Find contours for the mask to get polygon points
                    contours, _ = cv2.findContours(
                        instance_mask.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    # If no contours found, skip this instance
                    if not contours:
                        continue
                        
                    # Find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # If contour is too small, skip
                    if cv2.contourArea(largest_contour) < 10:
                        continue
                    
                    # Simplify the contour while preserving shape details
                    epsilon = 0.003 * cv2.arcLength(largest_contour, True)  # Reduced epsilon for better detail
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    # If we have too few points, use the original contour
                    if len(approx) < 6:  # Increased minimum points
                        # Subsample the original contour instead
                        step = max(1, len(largest_contour) // 20)
                        approx = largest_contour[::step]
                    
                    # Convert to YOLO segmentation format
                    # Format: class_id x1 y1 x2 y2 ... xn yn
                    
                    # First, calculate normalized bounding box center and dimensions (needed for YOLO)
                    h, w = instance_mask.shape
                    x_center = ((x_min + x_max) / 2) / w
                    y_center = ((y_min + y_max) / 2) / h
                    width = (x_max - x_min) / w
                    height = (y_max - y_min) / h
                    
                    # Start with class id (0 for spine)
                    line = "0"
                    
                    # Add normalized polygon points
                    for point in approx.reshape(-1, 2):
                        x, y = point
                        line += f" {x/w} {y/h}"
                    
                    f.write(line + "\n")
            
            # Progress update
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{num_samples} samples (Train: {train_count}, Val: {val_count})")
    
    def create_data_yaml(self):
        """Create the dataset.yaml file for YOLO segmentation training"""
        yaml_content = f"""# Dataset configuration for YOLO segmentation training
path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val

# Classes
nc: 1  # Number of classes
names: ['spine']  # Class names

# Task (important for segmentation)
task: segment
"""
        
        with open(f"{self.output_dir}/dataset.yaml", 'w') as f:
            f.write(yaml_content)


class SpineInstanceSegmentationPipeline:
    """Pipeline that combines standard segmentation with YOLO for instance segmentation"""
    
    def __init__(self, d3set_path, yolo_model_path):
        """
        Initialize the pipeline
        
        Args:
            d3set_path (str): Path to the DeepD3_Training.d3set file 
            yolo_model_path (str): Path to trained YOLO model
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
        
        # Convert to numpy for YOLO
        image_np = image.numpy().squeeze()
        spines_np = spines.numpy().squeeze()
        
        # Preprocess the image using the enhanced preprocessing function
        image_uint8 = preprocess_image(image_np.copy())
        
        # Run YOLO for instance segmentation
        results = self.yolo_model(image_uint8, device=0)
        
        # Get detected spine instances
        spine_instances = []
        
        if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
            # Process masks from segmentation model
            masks = results[0].masks
            boxes = results[0].boxes
            
            # Process each segmentation mask
            for i in range(len(masks)):
                # Get the mask data
                mask_data = masks[i].data.cpu().numpy()
                
                # Get confidence
                conf = boxes[i].conf.cpu().numpy()[0]
                
                # Skip low confidence detections
                if conf < 0.25:
                    continue
                
                # Create a binary mask for this instance
                if mask_data.ndim == 3:
                    instance_mask = mask_data[0] > 0.5
                else:
                    instance_mask = mask_data > 0.5
                
                # Resize mask to match the original image if needed
                if instance_mask.shape != spines_np.shape:
                    instance_mask = cv2.resize(
                        instance_mask.astype(np.uint8),
                        (spines_np.shape[1], spines_np.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                
                # Refine with the original spine segmentation
                refined_mask = instance_mask & (spines_np > 0.5)
                
                # Add to instances if not empty
                if np.any(refined_mask):
                    spine_instances.append(refined_mask)
        
        # If no masks found (fall back to boxes)
        elif len(results) > 0 and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            # Use watershed approach to get better instances
            if len(boxes) > 0:
                spine_instances = improve_instance_segmentation(spines_np > 0.5, boxes)
        
        # If no instances found, fall back to connected components
        if len(spine_instances) == 0:
            # Use connected components as fallback
            labeled_spines = label(spines_np > 0.5)
            for i in range(1, labeled_spines.max() + 1):
                instance_mask = labeled_spines == i
                if np.sum(instance_mask) > 10:  # Filter out very small regions
                    spine_instances.append(instance_mask)
        
        return image, dendrite, spines, spine_instances


def train_yolo_for_spines(d3set_path, output_dir, epochs=200, img_size=640):
    """
    Train YOLO model for spine instance segmentation with optimized parameters
    
    Args:
        d3set_path (str): Path to the DeepD3_Training.d3set file
        output_dir (str): Directory to save YOLO dataset and trained model
        epochs (int): Number of training epochs
        img_size (int): Image size for training
        
    Returns:
        str: Path to the trained model
    """
    # Prepare dataset
    print("Preparing dataset...")
    dataset_prep = YOLOSpineDatasetPreparation(d3set_path, f"{output_dir}/dataset")
    dataset_prep.generate_samples(num_samples=5000)  # Increased sample count for better training
    dataset_prep.create_data_yaml()
    
    # Initialize YOLO model - ensure it's the segmentation variant
    print("Initializing YOLO segmentation model...")
    
    # Check if a pre-trained model for segmentation exists
    model_name = 'yolo11x-seg.pt'  # Using YOLOv8 segmentation model
    try:
        model = YOLO(model_name)
        print(f"Successfully loaded {model_name}")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        # Fall back to a different model
        model_name = 'yolov8n-seg.pt'  # Smaller model as fallback
        try:
            model = YOLO(model_name)
            print(f"Falling back to {model_name}")
        except Exception as e:
            print(f"Error loading fallback model: {e}")
            raise
    
    # Set torch to use deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Train model with GPU acceleration and optimized parameters
    print(f"Training {model_name} segmentation model on GPU...")
    results = model.train(
        data=f"{output_dir}/dataset/dataset.yaml",
        epochs=epochs,
        imgsz=img_size,
        batch=8,           # Smaller batch size for better stability
        patience=30,       # More patience for convergence
        save=True,
        device=0,          # Use GPU device 0
        workers=4,         # Adjust number of workers for your environment
        rect=True,         # Use rectangular training for small objects
        mosaic=0.7,        # Increased mosaic for better augmentation
        mixup=0.2,         # Add mixup augmentation
        hsv_h=0.015,       # Hue augmentation
        hsv_s=0.7,         # Saturation augmentation
        hsv_v=0.4,         # Value augmentation
        translate=0.2,     # Translation augmentation
        scale=0.5,         # Scale augmentation
        fliplr=0.5,        # Horizontal flip
        flipud=0.2,        # Vertical flip (less common but can help)
        task='segment',    # Explicitly set task to segment
        verbose=True,      # Show detailed progress
        plots=True,        # Generate training plots
        save_period=20     # Save checkpoint every N epochs
    )
    
    best = model.best  # this is a pathlib.Path to the best.pt file
    print(f"✅ Training complete. Best model: {best}")
    return str(best)

def visualize_results(image, dendrite_mask, spine_mask, spine_instances, output_path=None):
    """
    Enhanced visualization of spine segmentation and instance detection results
    
    Args:
        image (torch.Tensor or np.ndarray): Original image
        dendrite_mask (torch.Tensor or np.ndarray): Dendrite segmentation mask
        spine_mask (torch.Tensor or np.ndarray): Spine segmentation mask
        spine_instances (list): List of instance masks
        output_path (str, optional): Path to save visualization
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    # Convert tensors to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.numpy().squeeze()
    if isinstance(dendrite_mask, torch.Tensor):
        dendrite_mask = dendrite_mask.numpy().squeeze()
    if isinstance(spine_mask, torch.Tensor):
        spine_mask = spine_mask.numpy().squeeze()

    # Normalize image for display
    image_norm = (image - image.min()) / (image.max() - image.min())

    # Create figure with more subplots for detailed analysis
    plt.figure(figsize=(20, 16))

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(image_norm, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Dendrite segmentation
    plt.subplot(2, 3, 2)
    plt.imshow(image_norm, cmap='gray')
    plt.imshow(dendrite_mask, alpha=0.5, cmap='Blues')
    plt.title('Dendrite Segmentation')
    plt.axis('off')

    # Spine segmentation
    plt.subplot(2, 3, 3)
    plt.imshow(image_norm, cmap='gray')
    plt.imshow(spine_mask, alpha=0.5, cmap='Reds')
    plt.title('Spine Segmentation')
    plt.axis('off')

    # Instance segmentation
    plt.subplot(2, 3, 4)
    # Build color map for instances
    base_cmap = plt.cm.get_cmap('tab20', len(spine_instances))
    rgba_list = [(0,0,0,0)] + [base_cmap(i) for i in range(len(spine_instances))]
    instance_cmap = ListedColormap(rgba_list)
    
    # Create instance map
    instance_vis = np.zeros_like(image_norm, dtype=int)
    for idx, inst in enumerate(spine_instances, start=1):
        instance_vis[inst] = idx
    
    plt.imshow(image_norm, cmap='gray')
    plt.imshow(instance_vis, alpha=0.7, cmap=instance_cmap)
    plt.title(f'Spine Instances ({len(spine_instances)} detected)')
    plt.axis('off')

    # Individual spine analysis
    plt.subplot(2, 3, 5)
    # Create a colored mask where each spine has a different color
    color_mask = np.zeros((*image_norm.shape, 3) if image_norm.ndim == 2 else (*image_norm.shape[:2], 3), dtype=np.float32)
    for i, mask in enumerate(spine_instances):
        color = plt.cm.tab20(i % 20)[:3]  # Get RGB from colormap
        for c in range(3):
            color_mask[..., c][mask] = color[c]
    
    plt.imshow(image_norm, cmap='gray')
    plt.imshow(color_mask, alpha=0.7)
    
    # Add spine numbers for analysis
    for i, mask in enumerate(spine_instances):
        # Find center of mass
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            cy, cx = np.mean(y_indices), np.mean(x_indices)
            plt.text(cx, cy, str(i+1), color='white', fontsize=8, 
                     ha='center', va='center', 
                     bbox=dict(facecolor='black', alpha=0.5, pad=0))
    
    plt.title('Numbered Spine Instances')
    plt.axis('off')

    # Spine morphology metrics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    plt.title('Spine Morphology Metrics')
    
    # Calculate and display metrics for each spine
    metrics_text = "ID    Area    Max Width\n"
    metrics_text += "----------------------\n"
    
    for i, mask in enumerate(spine_instances):
        # Calculate metrics
        area = np.sum(mask)
        
        # Find max width using distance transform
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        max_width = np.max(dist) * 2 if np.max(dist) > 0 else 0
        
        metrics_text += f"{i+1:<6}{area:<9}{max_width:.2f}\n"
    
    plt.text(0.1, 0.9, metrics_text, fontsize=10, family='monospace', 
             transform=plt.gca().transAxes, verticalalignment='top')

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Enhanced visualization saved to {output_path}")
    else:
        plt.show()

def evaluate_spine_detection(pipeline, test_indices, output_dir):
    """
    Evaluate spine detection performance on test samples
    
    Args:
        pipeline (SpineInstanceSegmentationPipeline): Initialized pipeline
        test_indices (list): List of sample indices to evaluate
        output_dir (str): Directory to save evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics dictionary
    metrics = {
        'sample_id': [],
        'spines_count': [],
        'detected_count': [],
        'avg_spine_size': []
    }
    
    # Process each sample
    for i, idx in enumerate(test_indices):
        # Process sample
        image, dendrite, spines, spine_instances = pipeline.process_sample(idx)
        
        # Count ground truth spines (from connected components)
        spines_np = spines.numpy().squeeze()
        gt_instances = label(spines_np > 0.5)
        gt_count = gt_instances.max()
        
        # Calculate metrics
        metrics['sample_id'].append(idx)
        metrics['spines_count'].append(gt_count)
        metrics['detected_count'].append(len(spine_instances))
        
        # Calculate average spine size
        if len(spine_instances) > 0:
            avg_size = np.mean([np.sum(mask) for mask in spine_instances])
        else:
            avg_size = 0
        metrics['avg_spine_size'].append(avg_size)
        
        # Visualize results
        output_path = f"{output_dir}/sample_{idx}.png"
        visualize_results(image, dendrite, spines, spine_instances, output_path)
        
        # Progress update
        print(f"Processed sample {idx} ({i+1}/{len(test_indices)}): "
              f"GT: {gt_count}, Detected: {len(spine_instances)}")
    
    # Save metrics to CSV
    import pandas as pd
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f"{output_dir}/evaluation_metrics.csv", index=False)
    
    # Calculate summary statistics
    detection_rate = metrics_df['detected_count'].sum() / metrics_df['spines_count'].sum()
    
    print(f"\nEvaluation Results:")
    print(f"Total samples: {len(test_indices)}")
    print(f"Total ground truth spines: {metrics_df['spines_count'].sum()}")
    print(f"Total detected spines: {metrics_df['detected_count'].sum()}")
    print(f"Overall detection rate: {detection_rate:.2f}")

# Example usage
if __name__ == "__main__":
    # Configure GPU memory usage
    # This helps prevent out-of-memory errors on HPC clusters
    torch.cuda.empty_cache()
    
    # Check if CUDA is available and print device properties
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Optional: Set memory growth to avoid allocating all GPU memory at once
        # This is useful for shared HPC environments
        for gpu_id in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(0.8, gpu_id)  # Use 80% of available memory
    else:
        print("WARNING: CUDA is not available. This will run much slower on CPU!")
    
    # Paths
    d3set_path = "dataset/DeepD3_Training.d3set"  # Path to your dataset
    output_dir = "spine_yolo_improved"  # Updated folder name
    
    # Step 1: Train YOLO model
    best_model_path = train_yolo_for_spines(d3set_path, output_dir, epochs=200)
    
    # Step 2: Initialize the pipeline
    pipeline = SpineInstanceSegmentationPipeline(d3set_path, best_model_path)
    
    # Step 3: Evaluate on test samples
    # Generate range of test indices that weren't used in training
    test_indices = list(range(5000, 5050))  # Example: 50 test samples
    evaluate_spine_detection(pipeline, test_indices, f"{output_dir}/evaluation")
    
    # Step 4: Process a few samples for visualization
    print("\nGenerating example visualizations...")
    for i in range(5):
        # Process sample
        image, dendrite, spines, spine_instances = pipeline.process_sample(i)
        
        # Visualize results
        visualize_results(
            image, dendrite, spines, spine_instances,
            output_path=f"{output_dir}/sample_{i}.png"
        )
    
    # Optional: Calculate and print memory usage for diagnostics
    if torch.cuda.is_available():
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated(0) / 1e9} GB")
        print(f"Max memory cached: {torch.cuda.max_memory_reserved(0) / 1e9} GB")
        
    print("Done!")