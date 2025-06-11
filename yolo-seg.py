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
        Generate samples for YOLOv8 segmentation training
        
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
            dendrite_np = dendrite.numpy().squeeze()
            
            # Handle NaN values in the image
            if np.isnan(image_np).any():
                image_np = np.nan_to_num(image_np)
            
            # Normalize image to 0-255 for saving - with NaN handling
            image_min = np.nanmin(image_np) if not np.all(np.isnan(image_np)) else 0
            image_max = np.nanmax(image_np) if not np.all(np.isnan(image_np)) else 1
            
            # Avoid division by zero
            if image_max - image_min == 0:
                image_uint8 = np.zeros_like(image_np, dtype=np.uint8)
            else:
                image_uint8 = (((image_np - image_min) / (image_max - image_min)) * 255).astype(np.uint8)
            
            # Create instance labels from spine mask using connected components
            spine_instances, bboxes = optimized_hybrid_bb(image_np, spines_np,dendrite_np)
            num_instances = spine_instances.max()
            
            # If no instances found, skip this sample
            if num_instances == 0:
                continue
                
            # Convert to RGB for YOLOv8
            if len(image_uint8.shape) == 2:
                image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
            
            # Split into train/val
            split = "train" if np.random.rand() < self.train_val_split else "val"
            
            # Save image
            img_path = f"{self.output_dir}/images/{split}/{i:06d}.png"
            cv2.imwrite(img_path, image_uint8)
            
            # For segmentation tasks, we need to save both bounding boxes and polygon points
            label_path = f"{self.output_dir}/labels/{split}/{i:06d}.txt"
            
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
                    
                    # Simplify the contour to reduce the number of points
                    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    # If we have too few points, add some intermediate points
                    if len(approx) < 4:
                        approx = largest_contour
                    
                    # Convert to YOLOv8 segmentation format
                    # Format: class_id x1 y1 x2 y2 ... xn yn
                    
                    # Start with class id (0 for spine)
                    line = "0"
                    
                    # Add normalized polygon points
                    for point in approx.reshape(-1, 2):
                        x, y = point
                        line += f" {x/image_np.shape[1]} {y/image_np.shape[0]}"
                    
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
        # Load with device specification
        self.yolo_model = YOLO(yolo_model_path)
        if torch.cuda.is_available():
            print(f"Loading YOLO model on GPU: {torch.cuda.get_device_name(0)}")
            self.device = 0  # Use GPU
        else:
            print("Loading YOLO model on CPU")
            self.device = 'cpu'
    
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
        if len(image_uint8.shape) == 2:
            image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
        
        # Run YOLOv11 for instance segmentation (ensure it runs on GPU if available)
        results = self.yolo_model(image_uint8, device=self.device)
        
        # Get detected spine instances - properly handling segmentation masks
        spine_instances = []
        if len(results) > 0:
            # For YOLOv11 segmentation model
            for r in results:
                if hasattr(r, 'masks') and r.masks is not None:
                    masks = r.masks.data
                    boxes = r.boxes
                    
                    # Process each detection
                    for i in range(len(boxes)):
                        # Skip low confidence detections
                        conf = float(boxes.conf[i].item())
                        if conf < 0.25:
                            continue
                        
                        # Get mask for this instance
                        if i < len(masks):
                            mask = masks[i].cpu().numpy()
                            
                            # Resize if needed
                            if mask.shape != image_np.shape:
                                mask = cv2.resize(
                                    mask.astype(np.uint8), 
                                    (image_np.shape[1], image_np.shape[0]),
                                    interpolation=cv2.INTER_NEAREST
                                )
                            
                            # Refine mask with original spine segmentation
                            spines_np = spines.numpy().squeeze()
                            refined_mask = (mask > 0.5) & (spines_np > 0.5)
                            
                            # Add to instances if not empty
                            if np.any(refined_mask):
                                spine_instances.append(refined_mask)
                else:
                    # Fallback to bounding boxes if masks aren't available
                    for r in results:
                        if not hasattr(r, 'boxes'):
                            continue
                            
                        boxes = r.boxes
                        for i in range(len(boxes)):
                            box = boxes.xyxy[i].cpu().numpy()  # Get x1, y1, x2, y2
                            conf = float(boxes.conf[i].item())
                            
                            if conf < 0.25:
                                continue
                            
                            # Create binary mask
                            instance_mask = np.zeros_like(image_np, dtype=bool)
                            if len(instance_mask.shape) > 2:
                                instance_mask = instance_mask[:,:,0]  # Take first channel for binary mask
                                
                            # Get box coordinates
                            x1, y1, x2, y2 = box.astype(int)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2 = min(instance_mask.shape[1], x2)
                            y2 = min(instance_mask.shape[0], y2)
                            
                            # Apply segmentation inside the box
                            spines_np = spines.numpy().squeeze()
                            if x2 > x1 and y2 > y1:
                                instance_mask[y1:y2, x1:x2] = spines_np[y1:y2, x1:x2] > 0.5
                                
                                if np.any(instance_mask):
                                    spine_instances.append(instance_mask)
        
        return image, dendrite, spines, spine_instances


def train_yolo_for_spines(d3set_path, output_dir, epochs=300, img_size=640):
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
    model = YOLO('yolo11x-seg.pt')  # Using YOLOv11 segmentation model as requested
    
    # Set torch to use deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Train model with GPU acceleration
    print("Training YOLOv11 segmentation model on GPU...")
    results = model.train(
        data=f"{output_dir}/dataset/dataset.yaml",
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        patience=20,
        save=True,
        device=0 if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        workers=4,  # Adjust number of workers for your HPC environment
        rect=True,  # Use rectangular training for small objects like spines
        mosaic=0.5,  # Reduce mosaic augmentation to maintain small feature visibility
        task='segment'  # Explicitly set task to segment for instance segmentation
    )
    
    # Get the best model path from the results
    if hasattr(results, 'best'):
        best_model_path = results.best
    else:
        # Fallback to the standard path structure if 'best' attribute doesn't exist
        best_model_path = f"{output_dir}/train/weights/best.pt"
    
    print(f"✅ Training complete. Best model: {best_model_path}")
    return str(best_model_path)


def visualize_results(image, dendrite_mask, spine_mask, spine_instances, output_path=None):
    """
    Visualize segmentation and instance detection results
    
    Args:
        image: Original image
        dendrite_mask: Dendrite segmentation mask
        spine_mask: Spine segmentation mask
        spine_instances: List of individual spine instance masks
        output_path: Path to save visualization (if None, display instead)
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

    # Build an RGBA colormap: first entry fully transparent, then tab20 colors
    base_cmap = plt.cm.get_cmap('tab20', max(1, len(spine_instances)))
    # (0,0,0,0) means "nothing drawn" for background
    rgba_list = [(0, 0, 0, 0)] + [base_cmap(i) for i in range(min(len(spine_instances), 20))]
    instance_cmap = ListedColormap(rgba_list)

    # Create an "instance map" where each pixel's value = instance index (0 = background)
    instance_vis = np.zeros_like(image, dtype=int)
    for idx, inst in enumerate(spine_instances, start=1):
        if idx <= 20:  # Only visualize up to 20 instances (tab20 colormap limit)
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
        try:
            for gpu_id in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(0.8, gpu_id)  # Use 80% of available memory
        except Exception as e:
            print(f"Warning: Could not set memory fraction: {e}")
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