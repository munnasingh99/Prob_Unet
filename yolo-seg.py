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
import wandb


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
        
    def generate_samples(self, num_samples=3000, size=(128,128)):
        """
        Generate YOLOv11‐seg training data:
        - one line per spine instance
        - format: class xc yc w h x1 y1 x2 y2 … xn yn
        """
        import numpy as np, cv2
        from skimage.measure import label
        from datagen import DataGeneratorDataset

        ds = DataGeneratorDataset(self.d3set_path,
                                  samples_per_epoch=num_samples,
                                  size=size,
                                  augment=True)

        for i in range(num_samples):
            img_t, (_, spine_t) = ds[i]
            img = img_t.numpy().squeeze()
            mask = (spine_t.numpy().squeeze()>0.5).astype(np.uint8)

            # handle NaNs and normalize
            img = np.nan_to_num(img)
            mn, mx = img.min(), img.max()
            if mx-mn < 1e-6:
                img8 = np.zeros_like(img, np.uint8)
            else:
                img8 = ((img-mn)/(mx-mn)*255).astype(np.uint8)

            # RGB for YOLO
            img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)

            # your CC → each instance mask
            instances = label(mask)
            if instances.max()==0:
                continue

            split = "train" if np.random.rand()<self.train_val_split else "val"
            out_img = f"{self.output_dir}/images/{split}/{i:06d}.png"
            out_lbl = f"{self.output_dir}/labels/{split}/{i:06d}.txt"

            cv2.imwrite(out_img, img8)

            h,w = mask.shape
            lines=[]
            # find each instance via contour
            for inst_id in range(1, instances.max()+1):
                inst = (instances==inst_id).astype(np.uint8)
                cnts,_ = cv2.findContours(inst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c)<10:
                    continue

                # simplify polygon
                eps = 0.01*cv2.arcLength(c,True)
                poly = cv2.approxPolyDP(c, eps, True).reshape(-1,2)
                if poly.shape[0]<3:
                    continue

                # bounding rect
                x,y,ww,hh = cv2.boundingRect(c)
                xc, yc = (x+ww/2)/w, (y+hh/2)/h
                nw, nh = ww/w, hh/h

                # build line
                parts = [ "0",
                          f"{xc:.6f}", f"{yc:.6f}",
                          f"{nw:.6f}", f"{nh:.6f}" ]
                for px,py in poly:
                    parts += [ f"{px/w:.6f}", f"{py/h:.6f}" ]
                lines.append(" ".join(parts))

            # only write if at least one valid instance
            if lines:
                with open(out_lbl,"w") as f:
                    f.write("\n".join(lines))

            if (i+1)%100==0:
                print(f"Processed {i+1}/{num_samples}")


    
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
        drop_empty: False
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
        results = self.yolo_model(image_uint8, device=0)  # Use GPU device 0
        
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
    dataset_prep.generate_samples(num_samples=3000)  # Increased sample count for better training
    dataset_prep.create_data_yaml()
    
    # Initialize YOLOv11 model - ensure it's the segmentation variant
    print("Initializing YOLOv11 segmentation model...")
    model = YOLO('yolo11n-seg.pt')  # Using YOLOv11 segmentation model as requested
    
    # Set torch to use deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Train model with GPU acceleration
    print("Training YOLOv11 segmentation model on GPU...")
    model.train(
        data=f"{output_dir}/dataset/dataset.yaml",
        task='segment',
        epochs=250,
        imgsz=640,
        batch=64,
        lr0=0.005,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5,         # run name in W&B
        exist_ok=True,                # overwrite same project/name
        patience=20,
        rect=True,
        multi_scale=True,
        copy_paste=0.5,
        mixup=0.2,
        mosaic=0.5,  # Explicitly set task to segment for instance segmentation
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
    # (0,0,0,0) means “nothing drawn” for background
    rgba_list = [(0, 0, 0, 0)] + [base_cmap(i) for i in range(len(spine_instances))]
    instance_cmap = ListedColormap(rgba_list)

    # Create an "instance map" where each pixel’s value = instance index (0 = background)
    instance_vis = np.zeros_like(image, dtype=int)
    for idx, inst in enumerate(spine_instances, start=1):
        instance_vis[inst] = idx

    # Plot
    plt.figure(figsize=(18, 12))

    plt.subplot(1, 4, 1)
    plt.imshow(image_norm, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(image_norm, cmap='gray')
    plt.imshow(dendrite_mask, alpha=0.5, cmap='Blues')
    plt.title('Dendrite Segmentation')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(image_norm, cmap='gray')
    plt.imshow(spine_mask, alpha=0.5, cmap='Reds')
    plt.title('Spine Segmentation')
    plt.axis('off')

    plt.subplot(1, 4, 4)
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
    for i in range(100):
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