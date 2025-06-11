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

def adaptive_contour_extraction(instance_mask, min_area=10, min_points=4):
    """
    Extract contours with adaptive refinement based on spine size and shape
    
    Args:
        instance_mask (np.ndarray): Binary mask of the spine instance
        min_area (int): Minimum contour area to consider
        min_points (int): Minimum number of points in resulting contour
        
    Returns:
        np.ndarray: Extracted contour points or None if invalid
    """
    # Get raw contours
    contours, _ = cv2.findContours(
        instance_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_TC89_KCOS  # Better for biological shapes than SIMPLE
    )
    
    if not contours:
        return None
    
    # Find largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    
    if contour_area < min_area:
        return None
    
    # Adaptive epsilon based on spine size and perimeter
    perimeter = cv2.arcLength(largest_contour, True)
    area_factor = np.sqrt(contour_area) / 10  # Scale factor based on spine size
    
    # Smaller spines need gentler approximation
    if contour_area < 50:
        epsilon = 0.002 * perimeter * area_factor
    else:
        epsilon = 0.005 * perimeter * area_factor
    
    # Apply polygon approximation
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Ensure minimum number of points for YOLO
    if len(approx) < min_points:
        # For small contours, use more points from original
        if contour_area < 30:
            # Use more points for small spines to preserve shape
            epsilon = 0.001 * perimeter
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # If still too few points, add corners of bounding rect
            if len(approx) < min_points:
                x, y, w, h = cv2.boundingRect(largest_contour)
                corners = np.array([
                    [[x, y]],
                    [[x + w, y]],
                    [[x + w, y + h]],
                    [[x, y + h]]
                ])
                # Combine with existing points and remove duplicates
                approx = np.vstack([approx, corners])
                # Remove duplicate points
                approx = np.unique(approx.reshape(-1, 2), axis=0).reshape(-1, 1, 2)
    
    return approx

def get_attachment_aware_contour(instance_mask, dendrite_mask):
    """
    Extract contour with awareness of attachment point to dendrite
    
    Args:
        instance_mask (np.ndarray): Binary mask of the spine instance
        dendrite_mask (np.ndarray): Binary mask of the dendrite
        
    Returns:
        np.ndarray: Contour points considering dendrite attachment
    """
    # Basic contour extraction
    contour = adaptive_contour_extraction(instance_mask)
    if contour is None:
        return None
        
    # Find attachment region
    from skimage.morphology import binary_dilation
    attachment_region = instance_mask & binary_dilation(dendrite_mask)
    
    if not np.any(attachment_region):
        # No clear attachment, return regular contour
        return contour
    
    # Get contour of attachment region
    attach_contours, _ = cv2.findContours(
        attachment_region.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_TC89_KCOS
    )
    
    if not attach_contours:
        return contour
    
    # Find largest attachment contour
    attach_contour = max(attach_contours, key=cv2.contourArea)
    
    # Get centroid of attachment region
    M = cv2.moments(attach_contour)
    if M["m00"] == 0:
        return contour
        
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    attachment_point = np.array([[cx, cy]])
    
    # Find closest point on spine contour to attachment centroid
    min_dist = float('inf')
    closest_idx = 0
    
    for i, point in enumerate(contour.reshape(-1, 2)):
        dist = np.sum((point - attachment_point)**2)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    # Ensure attachment point is included in final contour
    # This helps preserve the biologically significant connection point
    final_contour = contour.reshape(-1, 1, 2)
    
    # If attachment point isn't already in contour, add it
    attachment_present = False
    for point in final_contour:
        if np.linalg.norm(point.reshape(-1) - attachment_point.reshape(-1)) < 2:
            attachment_present = True
            break
            
    if not attachment_present:
        # Insert attachment point near closest point
        new_contour = np.insert(
            final_contour, 
            closest_idx + 1, 
            attachment_point.reshape(-1, 1, 2), 
            axis=0
        )
        return new_contour
    
    return final_contour

def spine_aware_contour_extraction(image_np, instance_mask, dendrite_mask):
    """
    Extract contours with awareness of spine morphology and dendrite attachment
    
    Args:
        image_np (np.ndarray): Original image
        instance_mask (np.ndarray): Binary mask of the spine instance
        dendrite_mask (np.ndarray): Binary mask of the dendrite
        
    Returns:
        np.ndarray: Extracted contour points considering morphology
    """
    # Try attachment-aware approach first
    contour = get_attachment_aware_contour(instance_mask, dendrite_mask)
    if contour is not None and len(contour) >= 4:
        return contour
    
    # Fall back to adaptive extraction if attachment approach fails
    contour = adaptive_contour_extraction(instance_mask)
    if contour is not None and len(contour) >= 4:
        return contour
    
    # Last resort: basic contour extraction
    contours, _ = cv2.findContours(
        instance_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 10:
        return None
        
    # Ensure we have at least 4 points
    if len(largest_contour) < 4:
        # Add corners of bounding rect
        x, y, w, h = cv2.boundingRect(largest_contour)
        corners = np.array([
            [[x, y]],
            [[x + w, y]],
            [[x + w, y + h]],
            [[x, y + h]]
        ])
        return corners
    
    return largest_contour

class YOLOSpineDatasetPreparation:
    """Helper class to convert DeepD3 dataset to YOLOv11 segmentation format for instance segmentation"""
    
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
        Generate samples for YOLOv11 segmentation training with optimized instance separation
        """
        # Use DataGeneratorDataset to generate samples
        dataset = DataGeneratorDataset(
            self.d3set_path, 
            samples_per_epoch=num_samples,
            size=size,
            augment=True
        )
        
        # Create a bounding box cache to avoid redundant calculations
        bbox_cache = {}
        
        # Process samples in batches to leverage GPU parallelism
        batch_size = 16
        total_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            print(f"Processing batch {batch_idx+1}/{total_batches} (samples {start_idx}-{end_idx-1})")
            
            for i in range(start_idx, end_idx):
                # Get a sample
                sample = dataset[i % len(dataset)]
                
                if isinstance(sample, tuple) and len(sample) == 2:
                    # Handle different return formats from DataGeneratorDataset
                    image, (dendrite, spines) = sample
                else:
                    # Access expected keys
                    image = sample['image']
                    dendrite = sample['dendrite_mask']
                    spines = sample['spine_mask']
                
                # Convert tensors to numpy arrays
                image_np = image.numpy().squeeze() if isinstance(image, torch.Tensor) else image
                spines_np = spines.numpy().squeeze() if isinstance(spines, torch.Tensor) else spines
                dendrite_np = dendrite.numpy().squeeze() if isinstance(dendrite, torch.Tensor) else dendrite
                
                # Handle NaN values in the image
                if np.isnan(image_np).any():
                    image_np = np.nan_to_num(image_np)
                
                # Normalize image to 0-255 for saving
                image_min = np.nanmin(image_np) if not np.all(np.isnan(image_np)) else 0
                image_max = np.nanmax(image_np) if not np.all(np.isnan(image_np)) else 1
                
                # Avoid division by zero
                if image_max - image_min == 0:
                    image_uint8 = np.zeros_like(image_np, dtype=np.uint8)
                else:
                    image_uint8 = (((image_np - image_min) / (image_max - image_min)) * 255).astype(np.uint8)
                
                # Use the optimized hybrid approach for instance separation
                spine_instances, bboxes = optimized_hybrid_bb(image_np, spines_np, dendrite_np)
                
                # If no instances found, skip this sample
                if np.max(spine_instances) == 0:
                    continue
                    
                # Convert to RGB for YOLOv11
                if len(image_uint8.shape) == 2:
                    image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
                
                # Split into train/val
                split = "train" if np.random.rand() < self.train_val_split else "val"
                
                # Save image
                img_path = f"{self.output_dir}/images/{split}/{i:06d}.png"
                cv2.imwrite(img_path, image_uint8)
                
                # For segmentation tasks, save both bounding boxes and polygon points
                label_path = f"{self.output_dir}/labels/{split}/{i:06d}.txt"
                
                # Process each spine instance and create a segmentation mask
                with open(label_path, 'w') as f:
                    # Process each labeled region using the bboxes from optimized approach
                    for idx, bbox in enumerate(bboxes):
                        y_min, x_min, y_max, x_max = bbox
                        
                        # Get the mask for this specific instance
                        instance_mask = (spine_instances == idx+1)
                        
                        # Skip if mask is empty
                        if not np.any(instance_mask):
                            continue
                        
                        # Use improved spine-aware contour extraction
                        contour = spine_aware_contour_extraction(
                            image_np, 
                            instance_mask, 
                            dendrite_np
                        )
                        
                        # Skip if no valid contour found
                        if contour is None or len(contour) < 4:
                            continue
                            
                        # Get image dimensions
                        h, w = instance_mask.shape
                        
                        # Start with class id (0 for spine)
                        line = "0"
                        
                        # Add normalized polygon points
                        for point in contour.reshape(-1, 2):
                            x, y = point
                            # Ensure points are within image bounds
                            x = max(0, min(x, w-1))
                            y = max(0, min(y, h-1))
                            line += f" {x/w} {y/h}"
                        
                        f.write(line + "\n")
                
                # Progress update
                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1}/{num_samples} samples")
    
    def create_data_yaml(self):
        """Create the dataset.yaml file for YOLOv11 segmentation training"""
        yaml_content = f"""# Dataset configuration for YOLOv11 segmentation training
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
    """Pipeline that combines standard segmentation with YOLOv11 for instance segmentation"""
    
    def __init__(self, d3set_path, yolo_model_path):
        """
        Initialize the pipeline
        
        Args:
            d3set_path (str): Path to the DeepD3_Training.d3set file 
            yolo_model_path (str): Path to trained YOLOv11 model
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
        
        # Convert to numpy for YOLOv11
        image_np = image.numpy().squeeze()
        
        # Handle NaN values if present
        if np.isnan(image_np).any():
            image_np = np.nan_to_num(image_np)
            
        # Normalize image to 0-255 for YOLOv11 with NaN handling
        image_min = np.nanmin(image_np) if not np.all(np.isnan(image_np)) else 0
        image_max = np.nanmax(image_np) if not np.all(np.isnan(image_np)) else 1
        
        # Avoid division by zero
        if image_max - image_min == 0:
            image_uint8 = np.zeros_like(image_np, dtype=np.uint8)
        else:
            image_uint8 = (((image_np - image_min) / (image_max - image_min)) * 255).astype(np.uint8)
        
        # Convert grayscale to RGB if needed (YOLOv11 expects 3 channels)
        if image_uint8.ndim == 2:
            image_uint8 = np.stack([image_uint8]*3, axis=-1)  # Convert grayscale to RGB
        
        # Run YOLOv11 for instance segmentation (ensure it runs on GPU)
        results = self.yolo_model(image_uint8, device=0,imgsz=128)  # Use GPU device 0
        
        # Get detected spine instances - now handling segmentation masks
        spine_instances = []
        if len(results) > 0:
            # For YOLOv11 segmentation model, we need to check for masks
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                masks = results[0].masks
                
                # Process each segmentation mask
                for i in range(len(masks)):
                    # Get the mask data
                    mask_data = masks[i].data.cpu().numpy()
                    
                    # Get confidence
                    conf = results[0].boxes[i].conf.cpu().numpy()[0]
                    
                    # Skip low confidence detections
                    if conf < 0.25:
                        continue
                    
                    # Create a binary mask for this instance and ensure correct shape
                    if mask_data.ndim == 3:
                        instance_mask = mask_data[0] > 0.5  # First channel, threshold at 0.5
                    else:
                        instance_mask = mask_data > 0.5
                    
                    # Resize mask to match the original image if needed
                    if instance_mask.shape != image_np.shape:
                        instance_mask = cv2.resize(
                            instance_mask.astype(np.uint8),
                            (image_np.shape[1], image_np.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    
                    # Apply the original spine segmentation as a filter to refine the instance
                    spines_np = spines.numpy().squeeze()
                    refined_mask = instance_mask & (spines_np > 0.5)
                    
                    # Add to instances if the mask isn't empty
                    if np.any(refined_mask):
                        spine_instances.append(refined_mask)
            # If no masks found (fall back to boxes)
            elif hasattr(results[0], 'boxes'):
                boxes = results[0].boxes
                
                # Process each detected spine
                for i in range(len(boxes)):
                    box = boxes[i].xyxy.cpu().numpy()[0]  # Get box coordinates (x1, y1, x2, y2)
                    conf = boxes[i].conf.cpu().numpy()[0]  # Get confidence
                    
                    # Skip low confidence detections
                    if conf < 0.25:
                        continue
                    
                    # Create a binary mask for this instance
                    instance_mask = np.zeros_like(image_np, dtype=bool)
                    
                    # Convert box coordinates to integers
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Ensure coordinates are within image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(instance_mask.shape[1] if instance_mask.ndim > 1 else instance_mask.shape[0], x2)
                    y2 = min(instance_mask.shape[0], y2)
                    
                    # Apply segmentation mask to get the actual spine shape within the box
                    # Only include pixels that are part of the spine segmentation
                    spines_np = spines.numpy().squeeze()
                    
                    # Handle different dimensionality
                    if instance_mask.ndim == 2 and y2 > y1 and x2 > x1:
                        instance_mask[y1:y2, x1:x2] = spines_np[y1:y2, x1:x2] > 0.5
                    
                    # Add to instances if not empty
                    if np.any(instance_mask):
                        spine_instances.append(instance_mask)
        
        return image, dendrite, spines, spine_instances


def train_yolo_for_spines(d3set_path, output_dir, epochs=300, img_size=128):
    """
    Train YOLOv11 model for spine instance segmentation
    
    Args:
        d3set_path (str): Path to the DeepD3_Training.d3set file
        output_dir (str): Directory to save YOLOv11 dataset and trained model
        epochs (int): Number of training epochs
        img_size (int): Image size for training
        
    Returns:
        str: Path to the trained model
    """
    # Prepare dataset
    print("Preparing dataset...")
    dataset_prep = YOLOSpineDatasetPreparation(d3set_path, f"{output_dir}/dataset")
    dataset_prep.generate_samples(num_samples=6000)  # Increased sample count for better training
    dataset_prep.create_data_yaml()
    
    # Initialize YOLOv11 model - ensure it's the segmentation variant
    print("Initializing YOLOv11 segmentation model...")
    model = YOLO('yolo11l-seg.pt')  # Using YOLOv11 segmentation model as requested
    
    # Set torch to use deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Train model with GPU acceleration
    print("Training YOLOv11 segmentation model on GPU...")
    model.train(
        data=f"{output_dir}/dataset/dataset.yaml",
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        patience=20,
        save=True,
        device=0,  # Use GPU device 0
        workers=4,  # Adjust number of workers for your HPC environment
        rect=True,  # Use rectangular training for small objects like spines
        mosaic=0.5,  # Reduce mosaic augmentation to maintain small feature visibility
        task='segment'  # Explicitly set task to segment for instance segmentation
    )
    
    
    best = model.best  # this is a pathlib.Path to the best.pt file
    print(f"✅ Training complete. Best model: {best}")
    return str(best)

def visualize_results(image, dendrite_mask, spine_mask, spine_instances, output_path=None):
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

    # Build an RGBA colormap: first entry fully transparent, then tab20 colors
    base_cmap = plt.cm.get_cmap('tab20', len(spine_instances))
    # (0,0,0,0) means "nothing drawn" for background
    rgba_list = [(0, 0, 0, 0)] + [base_cmap(i) for i in range(len(spine_instances))]
    instance_cmap = ListedColormap(rgba_list)

    # Create an "instance map" where each pixel's value = instance index (0 = background)
    instance_vis = np.zeros_like(image, dtype=int)
    for idx, inst in enumerate(spine_instances, start=1):
        instance_vis[inst] = idx

    # Plot
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(image_norm, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(image_norm, cmap='gray')
    plt.imshow(dendrite_mask, alpha=0.5, cmap='Blues')
    plt.title('Dendrite Segmentation')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(image_norm, cmap='gray')
    plt.imshow(spine_mask, alpha=0.5, cmap='Reds')
    plt.title('Spine Segmentation')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(image_norm, cmap='gray')
    plt.imshow(instance_vis, alpha=0.7, cmap=instance_cmap)
    plt.title(f'Spine Instances ({len(spine_instances)} detected)')
    plt.axis('off')

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

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
    output_dir = "spine_yolo11_gpu"  # Updated folder name to reflect YOLOv11 usage
    
    # Step 1: Train YOLOv11 model
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
    
    # Optional: Calculate and print memory usage for diagnostics
    if torch.cuda.is_available():
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated(0) / 1e9} GB")
        print(f"Max memory cached: {torch.cuda.max_memory_reserved(0) / 1e9} GB")
        
    print("Done!")