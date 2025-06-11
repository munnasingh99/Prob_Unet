import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, binary_dilation, remove_small_objects, binary_opening, binary_closing
from scipy import ndimage
from skimage.segmentation import watershed
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Import your existing modules
from yolo_datagen import DataGeneratorDataset
from model.prob_unet_deepd3 import ProbabilisticUnet
from bb_approach import optimized_hybrid_bb  # Your original method

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProbUNetPostProcessor:
    """Post-processing for Prob-UNet predictions before feeding to optimized_hybrid_bb"""
    
    def __init__(self, uncertainty_threshold=0.15, min_object_size=8, morphology_kernel_size=3):
        self.uncertainty_threshold = uncertainty_threshold
        self.min_object_size = min_object_size
        self.kernel = np.ones((morphology_kernel_size, morphology_kernel_size), np.uint8)
    
    def get_prob_unet_predictions(self, prob_unet_model, image_tensor, num_mc_samples=8):
        """
        Get post-processed predictions from Prob-UNet with uncertainty estimation
        """
        prob_unet_model.eval()
        
        with torch.no_grad():
            # Forward pass to set up latent spaces
            prob_unet_model.forward(image_tensor, None, training=False)
            
            # Generate multiple samples for uncertainty estimation
            dendrite_samples = []
            spine_samples = []
            
            for _ in range(num_mc_samples):
                dendrite_pred, spine_pred = prob_unet_model.sample(testing=True)
                dendrite_samples.append(torch.sigmoid(dendrite_pred).cpu().numpy())
                spine_samples.append(torch.sigmoid(spine_pred).cpu().numpy())
            
            # Calculate mean and uncertainty
            dendrite_samples = np.array(dendrite_samples)
            spine_samples = np.array(spine_samples)
            
            dendrite_mean = np.mean(dendrite_samples, axis=0).squeeze()
            spine_mean = np.mean(spine_samples, axis=0).squeeze()
            dendrite_uncertainty = np.var(dendrite_samples, axis=0).squeeze()
            spine_uncertainty = np.var(spine_samples, axis=0).squeeze()
        
        # Post-process predictions
        processed_dendrite = self.post_process_prediction(dendrite_mean, dendrite_uncertainty)
        processed_spine = self.post_process_prediction(spine_mean, spine_uncertainty)
        
        return processed_dendrite, processed_spine, spine_uncertainty
    
    def post_process_prediction(self, prediction, uncertainty):
        """Post-process a single prediction using uncertainty information"""
        # Uncertainty-guided filtering
        reliable_mask = uncertainty < self.uncertainty_threshold
        filtered_prediction = prediction * reliable_mask
        
        # Adaptive thresholding
        base_threshold = 0.5
        adaptive_threshold = base_threshold + (uncertainty * 0.3)
        binary_mask = filtered_prediction > adaptive_threshold
        
        # Morphological refinement
        cleaned = remove_small_objects(binary_mask, min_size=self.min_object_size)
        opened = binary_opening(cleaned, self.kernel)
        refined = binary_closing(opened, self.kernel)
        
        return refined.astype(np.float32)


class YOLOTrainingDataGenerator:
    """Generate YOLO training data using optimized_hybrid_bb with Prob-UNet predictions"""
    
    def __init__(self, d3set_path, prob_unet_model_path):
        """
        Initialize the training data generator
        
        Args:
            d3set_path: Path to your dataset
            prob_unet_model_path: Path to your trained Prob-UNet model
        """
        self.d3set_path = d3set_path
        self.prob_unet_model_path = prob_unet_model_path
        
        # Load your trained Prob-UNet
        self.prob_unet = self.load_prob_unet()
        
        # Initialize post-processor
        self.post_processor = ProbUNetPostProcessor(
            uncertainty_threshold=0.15,
            min_object_size=8,
            morphology_kernel_size=3
        )
        
        # Initialize data generator
        self.data_generator = DataGeneratorDataset(d3set_path, samples_per_epoch=8000,size=(480,480))
        
        print("✅ YOLO training data generator initialized with Prob-UNet!")
    
    def load_prob_unet(self):
        """Load your trained Prob-UNet model"""
        model = ProbabilisticUnet(
            input_channels=1,
            num_classes=1,
            latent_dim=16
        ).to(device)
        
        model.load_state_dict(torch.load(self.prob_unet_model_path, map_location=device))
        model.eval()
        return model
    
    def generate_instances_with_optimized_hybrid_bb(self, image):
        """
        Use optimized_hybrid_bb with Prob-UNet predictions to generate instances
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            instance_labels, bounding_boxes, quality_score
        """
        # Prepare image for Prob-UNet
        if np.isnan(image).any():
            image = np.nan_to_num(image)
        
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Get post-processed predictions from Prob-UNet
        dendrite_pred, spine_pred, spine_uncertainty = self.post_processor.get_prob_unet_predictions(
            self.prob_unet, image_tensor, num_mc_samples=8
        )
        
        # Use YOUR original optimized_hybrid_bb method with Prob-UNet predictions
        instance_labels, bounding_boxes = optimized_hybrid_bb(image, spine_pred, dendrite_pred)
        
        # Calculate quality score based on uncertainty and prediction confidence
        if len(bounding_boxes) > 0:
            # Average uncertainty in spine regions
            spine_regions = spine_pred > 0.5
            if np.any(spine_regions):
                avg_uncertainty = np.mean(spine_uncertainty[spine_regions])
                quality_score = 1.0 - min(avg_uncertainty, 1.0)  # Convert uncertainty to quality
            else:
                quality_score = 0.0
        else:
            quality_score = 0.0
        
        return instance_labels, bounding_boxes, quality_score
    
    def generate_yolo_dataset(self, output_dir, num_samples=6000, quality_threshold=0.3):
        """
        Generate YOLO training dataset using optimized_hybrid_bb with Prob-UNet
        
        Args:
            output_dir: Directory to save the dataset
            num_samples: Number of samples to generate
            quality_threshold: Minimum quality score to include sample
        """
        print(f"🔄 Generating YOLO dataset using optimized_hybrid_bb + Prob-UNet...")
        print(f"Target samples: {num_samples}, Quality threshold: {quality_threshold}")
        
        # Create directories
        os.makedirs(f"{output_dir}/images/train", exist_ok=True)
        os.makedirs(f"{output_dir}/images/val", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/val", exist_ok=True)
        
        generated_samples = 0
        quality_stats = {'high': 0, 'medium': 0, 'low': 0, 'rejected': 0}
        
        progress_bar = tqdm(range(num_samples * 2), desc="Generating YOLO dataset")  # Generate more to filter
        
        for i in progress_bar:
            try:
                # Get sample from data generator
                sample_idx = i % len(self.data_generator)
                image, (dendrite_gt, spine_gt) = self.data_generator[sample_idx]
                image_np = image.numpy().squeeze()
                
                # Generate instances using optimized_hybrid_bb + Prob-UNet
                instance_labels, bboxes, quality_score = self.generate_instances_with_optimized_hybrid_bb(image_np)
                
                # Quality filtering
                if quality_score < quality_threshold or len(bboxes) == 0:
                    quality_stats['rejected'] += 1
                    continue
                
                # Categorize quality
                if quality_score > 0.7:
                    quality_stats['high'] += 1
                elif quality_score > 0.5:
                    quality_stats['medium'] += 1
                else:
                    quality_stats['low'] += 1
                
                # Prepare image for saving
                image_norm = (image_np - image_np.min()) / (image_np.max() - image_np.min())
                image_uint8 = (image_norm * 255).astype(np.uint8)
                
                # Convert to RGB for YOLO
                if len(image_uint8.shape) == 2:
                    image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
                
                # Train/val split
                split = "train" if np.random.rand() < 0.8 else "val"
                
                # Save image
                img_path = f"{output_dir}/images/{split}/{generated_samples:06d}.png"
                cv2.imwrite(img_path, image_uint8)
                
                # Save YOLO segmentation labels
                label_path = f"{output_dir}/labels/{split}/{generated_samples:06d}.txt"
                self.save_yolo_segmentation_labels(instance_labels, bboxes, label_path)
                
                generated_samples += 1
                
                # Update progress
                progress_bar.set_postfix({
                    'Generated': generated_samples,
                    'Quality': f"{quality_score:.3f}",
                    'High_Q': quality_stats['high']
                })
                
                # Stop if we have enough samples
                if generated_samples >= num_samples:
                    break
                
            except Exception as e:
                print(f"Error in sample {i}: {e}")
                continue
        
        # Create dataset.yaml for YOLO
        yaml_content = f"""# YOLO dataset generated using optimized_hybrid_bb + Prob-UNet
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# Classes
nc: 1
names: ['spine']

# Task
task: segment
"""
        
        with open(f"{output_dir}/dataset.yaml", 'w') as f:
            f.write(yaml_content)
        
        # Save generation statistics
        stats = {
            'total_generated': generated_samples,
            'quality_distribution': quality_stats,
            'quality_threshold': quality_threshold,
            'method': 'optimized_hybrid_bb + Prob-UNet'
        }
        
        with open(f"{output_dir}/generation_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ Generated {generated_samples} high-quality samples")
        print(f"📊 Quality distribution: {quality_stats}")
        
        return f"{output_dir}/dataset.yaml"
    
    def save_yolo_segmentation_labels(self, instance_labels, bboxes, label_path):
        """Save YOLO segmentation labels from instance labels"""
        with open(label_path, 'w') as f:
            for idx, bbox in enumerate(bboxes):
                # Get the mask for this specific instance
                instance_mask = (instance_labels == idx + 1)
                
                if not np.any(instance_mask):
                    continue
                
                # Find contours for the instance
                contours, _ = cv2.findContours(
                    instance_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if not contours:
                    continue
                
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Skip if contour is too small
                if cv2.contourArea(largest_contour) < 10:
                    continue
                
                # Simplify the contour
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                if len(approx) < 4:
                    approx = largest_contour
                
                # Convert to YOLO segmentation format
                h, w = instance_mask.shape
                line = "0"  # Class 0 for spine
                
                for point in approx.reshape(-1, 2):
                    x, y = point
                    line += f" {x/w} {y/h}"
                
                f.write(line + "\n")
    
    def train_yolo_model(self, dataset_yaml_path, epochs=200, batch_size=12):
        """Train YOLO model on the generated dataset"""
        print("🚀 Training YOLO model on optimized_hybrid_bb + Prob-UNet data...")
        
        # Initialize YOLO segmentation model
        model = YOLO('yolo11l-seg.pt')
        
        # Train with optimized parameters
        results = model.train(
            data=dataset_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            patience=25,
            device=0,
            workers=4,
            rect=True,          # Rectangular training for better small object detection
            mosaic=0.3,         # Reduced mosaic for small objects
            copy_paste=0.1,     # Copy-paste augmentation
            task='segment',
            project="yolo_optimized_hybrid_bb",
            name="prob_unet_enhanced",
            save_period=25
        )
        
        best_model_path = str(model.trainer.best)
        print(f"✅ YOLO training complete! Best model: {best_model_path}")
        
        return best_model_path
    
    def visualize_training_sample(self, sample_idx, output_path=None):
        """Visualize how a training sample is generated"""
        # Get sample
        image, (dendrite_gt, spine_gt) = self.data_generator[sample_idx]
        image_np = image.numpy().squeeze()
        
        if np.isnan(image_np).any():
            image_np = np.nan_to_num(image_np)
        
        # Get Prob-UNet predictions
        image_tensor = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0).to(device)
        dendrite_pred, spine_pred, spine_uncertainty = self.post_processor.get_prob_unet_predictions(
            self.prob_unet, image_tensor
        )
        
        # Generate instances using optimized_hybrid_bb
        instance_labels, bboxes, quality_score = self.generate_instances_with_optimized_hybrid_bb(image_np)
        
        # Also get GT instances for comparison
        gt_instances, gt_bboxes = optimized_hybrid_bb(image_np, spine_gt.numpy().squeeze(), dendrite_gt.numpy().squeeze())
        
        # Create visualization
        plt.figure(figsize=(20, 12))
        
        # Normalize image
        image_norm = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        # Original image
        plt.subplot(2, 4, 1)
        plt.imshow(image_norm, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # GT spine
        plt.subplot(2, 4, 2)
        plt.imshow(image_norm, cmap='gray')
        plt.imshow(spine_gt.numpy().squeeze(), alpha=0.5, cmap='Reds')
        plt.title('GT Spine Mask')
        plt.axis('off')
        
        # Prob-UNet spine prediction
        plt.subplot(2, 4, 3)
        plt.imshow(image_norm, cmap='gray')
        plt.imshow(spine_pred, alpha=0.5, cmap='Reds')
        plt.title('Prob-UNet Spine Pred')
        plt.axis('off')
        
        # Uncertainty map
        plt.subplot(2, 4, 4)
        plt.imshow(spine_uncertainty, cmap='viridis')
        plt.colorbar()
        plt.title('Spine Uncertainty')
        plt.axis('off')
        
        # GT instances (using optimized_hybrid_bb with GT)
        plt.subplot(2, 4, 5)
        plt.imshow(image_norm, cmap='gray')
        plt.imshow(gt_instances, alpha=0.7, cmap='tab20')
        plt.title(f'GT + optimized_hybrid_bb\n({len(gt_bboxes)} instances)')
        plt.axis('off')
        
        # Prob-UNet instances (using optimized_hybrid_bb with Prob-UNet)
        plt.subplot(2, 4, 6)
        plt.imshow(image_norm, cmap='gray')
        plt.imshow(instance_labels, alpha=0.7, cmap='tab20')
        plt.title(f'Prob-UNet + optimized_hybrid_bb\n({len(bboxes)} instances)')
        plt.axis('off')
        
        # Quality and statistics
        plt.subplot(2, 4, 7)
        plt.text(0.1, 0.8, f"Quality Score: {quality_score:.3f}", fontsize=12)
        plt.text(0.1, 0.6, f"GT Instances: {len(gt_bboxes)}", fontsize=12)
        plt.text(0.1, 0.4, f"Pred Instances: {len(bboxes)}", fontsize=12)
        plt.text(0.1, 0.2, f"Avg Uncertainty: {np.mean(spine_uncertainty):.3f}", fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Statistics')
        plt.axis('off')
        
        # Bounding boxes overlay
        plt.subplot(2, 4, 8)
        plt.imshow(image_norm, cmap='gray')
        
        # Draw bounding boxes
        for bbox in bboxes:
            y_min, x_min, y_max, x_max = bbox
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               linewidth=2, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
        
        plt.title(f'Bounding Boxes\n(Quality: {quality_score:.3f})')
        plt.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Training sample visualization saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def run_complete_pipeline(self, output_dir):
        """Run the complete pipeline: data generation -> YOLO training"""
        print("🚀 Starting complete YOLO training pipeline with optimized_hybrid_bb + Prob-UNet...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Visualize a few training samples
        print("📸 Generating sample visualizations...")
        for i in range(3):
            self.visualize_training_sample(i, f"{output_dir}/training_sample_{i}.png")
        
        # Step 2: Generate YOLO training dataset
        dataset_yaml = self.generate_yolo_dataset(f"{output_dir}/yolo_dataset", num_samples=6000)
        
        # Step 3: Train YOLO model
        yolo_model_path = self.train_yolo_model(dataset_yaml, epochs=200)
        
        print(f"\n✅ Complete pipeline finished!")
        print(f"📁 Results saved in: {output_dir}")
        print(f"🎯 YOLO model: {yolo_model_path}")
        print(f"📊 Dataset: {dataset_yaml}")
        
        return {
            'yolo_model': yolo_model_path,
            'dataset_yaml': dataset_yaml,
            'output_dir': output_dir
        }


# Simple usage example
if __name__ == "__main__":
    # Configure GPU
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # YOUR PATHS - MODIFY THESE
    d3set_path = "dataset/DeepD3_Training.d3set"
    prob_unet_model_path = "final_models_path/model_epoch_19_val_loss_0.3692.pth"  # ← Your trained Prob-UNet
    output_dir = "yolo_optimized_hybrid_bb_results"
    
    # Initialize the training data generator
    yolo_trainer = YOLOTrainingDataGenerator(d3set_path, prob_unet_model_path)
    
    # Run the complete pipeline
    results = yolo_trainer.run_complete_pipeline(output_dir)
    
    print(f"\n🎉 Training complete!")
    print(f"Your YOLO model trained on optimized_hybrid_bb + Prob-UNet data: {results['yolo_model']}")