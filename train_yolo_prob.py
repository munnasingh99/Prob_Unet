import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, binary_opening, binary_closing
from scipy import ndimage
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from collections import defaultdict

# Import your existing modules
from yolo_datagen import DataGeneratorDataset
from model.prob_unet_deepd3 import ProbabilisticUnet
from bb_approach import optimized_hybrid_bb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UncertaintyGuidedYOLOTrainer:
    """Enhanced YOLO training with uncertainty-based quality curation"""
    
    def __init__(self, d3set_path, prob_unet_path):
        """
        Initialize uncertainty-guided YOLO trainer
        
        Args:
            d3set_path: Path to training dataset
            prob_unet_path: Path to trained Prob-UNet model
        """
        self.d3set_path = d3set_path
        self.prob_unet_path = prob_unet_path
        
        # Load Prob-UNet
        self.prob_unet = self.load_prob_unet()
        
        # Data generator
        self.data_generator = DataGeneratorDataset(d3set_path, samples_per_epoch=8000)
        
        # Uncertainty thresholds for quality curation
        self.uncertainty_thresholds = {
            'gold_standard': 0.08,      # Very low uncertainty - highest quality
            'high_quality': 0.15,       # Low uncertainty - good quality
            'medium_quality': 0.25,     # Medium uncertainty - acceptable
            'hard_negative': 0.40       # High uncertainty - potential noise
        }
        
        print("✅ Uncertainty-guided YOLO trainer initialized")
        print(f"🎯 Quality thresholds: {self.uncertainty_thresholds}")
    
    def load_prob_unet(self):
        """Load Prob-UNet with automatic architecture detection"""
        checkpoint = torch.load(self.prob_unet_path, map_location=device)
        
        # Detect latent_dim
        if 'prior.conv_layer.bias' in checkpoint:
            latent_dim = checkpoint['prior.conv_layer.bias'].shape[0] // 2
        else:
            latent_dim = 6
        
        model = ProbabilisticUnet(
            input_channels=1,
            num_classes=1,
            latent_dim=16
        ).to(device)
        
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    
    def generate_uncertainty_enhanced_dataset(self, output_dir, num_samples=8000, 
                                            mc_samples=15, enable_hard_negatives=True):
        """
        Generate YOLO dataset with uncertainty-based quality curation
        
        Args:
            output_dir: Output directory for dataset
            num_samples: Target number of samples
            mc_samples: Monte Carlo samples for uncertainty estimation
            enable_hard_negatives: Whether to include hard negative mining
        """
        print("🔬 Generating uncertainty-enhanced YOLO dataset...")
        print(f"🎲 Monte Carlo samples: {mc_samples}")
        print(f"🎯 Target samples: {num_samples}")
        
        # Create directories
        for split in ['train', 'val']:
            for quality in ['gold', 'high', 'medium', 'hard_neg']:
                os.makedirs(f"{output_dir}/images/{split}/{quality}", exist_ok=True)
                os.makedirs(f"{output_dir}/labels/{split}/{quality}", exist_ok=True)
        
        # Statistics tracking
        quality_stats = defaultdict(int)
        instance_stats = defaultdict(list)
        generated_samples = {
            'gold_standard': 0,
            'high_quality': 0,
            'medium_quality': 0,
            'hard_negative': 0,
            'rejected': 0
        }
        
        sample_idx = 0
        attempts = 0
        max_attempts = num_samples * 3  # Allow more attempts to get good samples
        
        progress_bar = tqdm(total=num_samples, desc="Generating enhanced dataset")
        
        while sample_idx < num_samples and attempts < max_attempts:
            attempts += 1
            
            try:
                # Get sample from data generator
                data_idx = attempts % len(self.data_generator)
                image, (dendrite_gt, spine_gt) = self.data_generator[data_idx]
                image_np = image.numpy().squeeze()
                
                if np.isnan(image_np).any():
                    image_np = np.nan_to_num(image_np)
                
                # Generate instances with uncertainty analysis
                result = self.generate_instances_with_uncertainty(image_np, mc_samples)
                
                if result is None or len(result['instances']) == 0:
                    quality_stats['no_instances'] += 1
                    continue
                
                # Categorize instances by uncertainty
                categorized_instances = self.categorize_instances_by_uncertainty(result)
                
                # Skip if no good quality instances
                if not any(categorized_instances.values()):
                    quality_stats['all_poor_quality'] += 1
                    continue
                
                # Determine primary quality level for this sample
                primary_quality = self.determine_primary_quality(categorized_instances)
                
                if primary_quality == 'reject':
                    generated_samples['rejected'] += 1
                    continue
                
                # Save sample in appropriate quality folder
                success = self.save_enhanced_sample(
                    image_np, result, categorized_instances, 
                    primary_quality, output_dir, sample_idx
                )
                
                if success:
                    generated_samples[primary_quality] += 1
                    instance_stats[primary_quality].extend([
                        inst['uncertainty'] for inst in result['instances']
                        if inst['quality'] == primary_quality
                    ])
                    
                    sample_idx += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'Gold': generated_samples['gold_standard'],
                        'High': generated_samples['high_quality'],
                        'Med': generated_samples['medium_quality'],
                        'Quality': f"{primary_quality[:4]}"
                    })
            
            except Exception as e:
                print(f"Error in sample {attempts}: {e}")
                continue
        
        progress_bar.close()
        
        # Create dataset configuration
        self.create_enhanced_dataset_config(output_dir, generated_samples, enable_hard_negatives)
        
        # Save generation statistics
        self.save_generation_stats(output_dir, generated_samples, instance_stats, quality_stats)
        
        print(f"\n✅ Enhanced dataset generation complete!")
        print(f"📊 Quality distribution: {dict(generated_samples)}")
        
        return f"{output_dir}/dataset.yaml"
    
    def generate_instances_with_uncertainty(self, image, mc_samples=15):
        """Generate instances with detailed uncertainty analysis"""
        # Prepare image
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Get Prob-UNet predictions with uncertainty
        with torch.no_grad():
            self.prob_unet.forward(image_tensor, None, training=False)
            
            # Generate multiple samples for uncertainty estimation
            dendrite_samples = []
            spine_samples = []
            
            for _ in range(mc_samples):
                dendrite_pred, spine_pred = self.prob_unet.sample(testing=True)
                dendrite_samples.append(torch.sigmoid(dendrite_pred).cpu().numpy())
                spine_samples.append(torch.sigmoid(spine_pred).cpu().numpy())
            
            # Calculate statistics
            dendrite_samples = np.array(dendrite_samples)
            spine_samples = np.array(spine_samples)
            
            dendrite_mean = np.mean(dendrite_samples, axis=0).squeeze()
            spine_mean = np.mean(spine_samples, axis=0).squeeze()
            dendrite_uncertainty = np.var(dendrite_samples, axis=0).squeeze()
            spine_uncertainty = np.var(spine_samples, axis=0).squeeze()
        
        # Post-process predictions
        dendrite_clean = self.clean_prediction(dendrite_mean, dendrite_uncertainty)
        spine_clean = self.clean_prediction(spine_mean, spine_uncertainty)
        
        # Generate instances using optimized_hybrid_bb
        try:
            instance_labels, bboxes = optimized_hybrid_bb(image, spine_clean, dendrite_clean)
        except Exception as e:
            print(f"Error in optimized_hybrid_bb: {e}")
            return None
        
        if len(bboxes) == 0:
            return None
        
        # Analyze each instance
        instances = []
        for idx, bbox in enumerate(bboxes):
            instance_mask = (instance_labels == idx + 1)
            
            if not np.any(instance_mask):
                continue
            
            # Calculate instance-specific uncertainty metrics
            instance_spine_uncertainty = spine_uncertainty[instance_mask]
            instance_dendrite_uncertainty = dendrite_uncertainty[instance_mask]
            
            # Instance quality metrics
            avg_spine_uncertainty = np.mean(instance_spine_uncertainty)
            max_spine_uncertainty = np.max(instance_spine_uncertainty)
            uncertainty_std = np.std(instance_spine_uncertainty)
            
            # Prediction confidence in this region
            instance_spine_confidence = spine_mean[instance_mask]
            avg_confidence = np.mean(instance_spine_confidence)
            
            # Shape and size metrics
            props = regionprops(instance_mask.astype(int))
            if not props:
                continue
            
            prop = props[0]
            
            instances.append({
                'id': idx + 1,
                'bbox': bbox,
                'mask': instance_mask,
                'size': np.sum(instance_mask),
                'uncertainty': avg_spine_uncertainty,
                'max_uncertainty': max_spine_uncertainty,
                'uncertainty_std': uncertainty_std,
                'confidence': avg_confidence,
                'eccentricity': prop.eccentricity,
                'solidity': prop.solidity,
                'area': prop.area
            })
        
        return {
            'instances': instances,
            'instance_labels': instance_labels,
            'spine_mean': spine_mean,
            'dendrite_mean': dendrite_mean,
            'spine_uncertainty': spine_uncertainty,
            'dendrite_uncertainty': dendrite_uncertainty,
            'global_spine_uncertainty': np.mean(spine_uncertainty[spine_clean > 0.5]) if np.any(spine_clean > 0.5) else 1.0
        }
    
    def categorize_instances_by_uncertainty(self, result):
        """Categorize instances based on uncertainty levels"""
        categorized = {
            'gold_standard': [],
            'high_quality': [],
            'medium_quality': [],
            'hard_negative': []
        }
        
        for instance in result['instances']:
            uncertainty = instance['uncertainty']
            confidence = instance['confidence']
            size = instance['size']
            
            # Additional quality filters
            min_size = 8
            min_confidence = 0.3
            max_eccentricity = 0.95
            
            # Size and shape filters
            if (size < min_size or 
                confidence < min_confidence or 
                instance['eccentricity'] > max_eccentricity):
                continue
            
            # Categorize by uncertainty
            if uncertainty < self.uncertainty_thresholds['gold_standard']:
                instance['quality'] = 'gold_standard'
                categorized['gold_standard'].append(instance)
            elif uncertainty < self.uncertainty_thresholds['high_quality']:
                instance['quality'] = 'high_quality'
                categorized['high_quality'].append(instance)
            elif uncertainty < self.uncertainty_thresholds['medium_quality']:
                instance['quality'] = 'medium_quality'
                categorized['medium_quality'].append(instance)
            elif uncertainty < self.uncertainty_thresholds['hard_negative']:
                instance['quality'] = 'hard_negative'
                categorized['hard_negative'].append(instance)
        
        return categorized
    
    def determine_primary_quality(self, categorized_instances):
        """Determine primary quality level for sample"""
        # Priority: Gold > High > Medium > Hard Negative
        if len(categorized_instances['gold_standard']) >= 1:
            return 'gold_standard'
        elif len(categorized_instances['high_quality']) >= 2:
            return 'high_quality'
        elif len(categorized_instances['medium_quality']) >= 1:
            return 'medium_quality'
        elif len(categorized_instances['hard_negative']) >= 1:
            return 'hard_negative'
        else:
            return 'reject'
    
    def save_enhanced_sample(self, image, result, categorized_instances, 
                           primary_quality, output_dir, sample_idx):
        """Save sample with quality-based organization"""
        try:
            # Prepare image for saving
            image_norm = (image - image.min()) / (image.max() - image.min())
            image_uint8 = (image_norm * 255).astype(np.uint8)
            
            # Convert to RGB
            if len(image_uint8.shape) == 2:
                image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
            
            # Determine split
            split = "train" if np.random.rand() < 0.8 else "val"
            
            # Map quality to folder name
            quality_folder_map = {
                'gold_standard': 'gold',
                'high_quality': 'high',
                'medium_quality': 'medium',
                'hard_negative': 'hard_neg'
            }
            
            quality_folder = quality_folder_map[primary_quality]
            
            # Save image
            img_path = f"{output_dir}/images/{split}/{quality_folder}/{sample_idx:06d}.png"
            cv2.imwrite(img_path, image_uint8)
            
            # Save labels (focus on instances of the primary quality and higher)
            label_path = f"{output_dir}/labels/{split}/{quality_folder}/{sample_idx:06d}.txt"
            
            # Select instances to include in labels
            instances_to_save = []
            
            if primary_quality == 'gold_standard':
                instances_to_save = categorized_instances['gold_standard']
            elif primary_quality == 'high_quality':
                instances_to_save = (categorized_instances['gold_standard'] + 
                                   categorized_instances['high_quality'])
            elif primary_quality == 'medium_quality':
                instances_to_save = (categorized_instances['gold_standard'] + 
                                   categorized_instances['high_quality'] + 
                                   categorized_instances['medium_quality'])
            elif primary_quality == 'hard_negative':
                # For hard negatives, include all but mark them differently
                instances_to_save = (categorized_instances['gold_standard'] + 
                                   categorized_instances['high_quality'] + 
                                   categorized_instances['medium_quality'])
            
            # Save YOLO labels
            self.save_yolo_labels_enhanced(instances_to_save, result['instance_labels'], label_path)
            
            return True
            
        except Exception as e:
            print(f"Error saving sample {sample_idx}: {e}")
            return False
    
    def save_yolo_labels_enhanced(self, instances_to_save, instance_labels, label_path):
        """Save YOLO labels with quality information"""
        with open(label_path, 'w') as f:
            for instance in instances_to_save:
                instance_mask = instance['mask']
                
                # Find contours
                contours, _ = cv2.findContours(
                    instance_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if not contours:
                    continue
                
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) < 10:
                    continue
                
                # Simplify contour
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                if len(approx) < 4:
                    approx = largest_contour
                
                # YOLO format: class_id + normalized polygon points
                h, w = instance_mask.shape
                line = "0"  # Class 0 for spine
                
                for point in approx.reshape(-1, 2):
                    x, y = point
                    line += f" {x/w} {y/h}"
                
                f.write(line + "\n")
    
    def clean_prediction(self, prediction, uncertainty, threshold=0.15):
        """Clean prediction using uncertainty"""
        reliable = uncertainty < threshold
        filtered = prediction * reliable
        binary = filtered > 0.5
        cleaned = remove_small_objects(binary, min_size=5)
        opened = binary_opening(cleaned, np.ones((3, 3), np.uint8))
        refined = binary_closing(opened, np.ones((3, 3), np.uint8))
        return refined.astype(np.float32)
    
    def create_enhanced_dataset_config(self, output_dir, generated_samples, enable_hard_negatives):
        """Create enhanced dataset configuration"""
        # Create main dataset.yaml
        yaml_content = f"""# Uncertainty-Enhanced YOLO Dataset
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# Classes
nc: 1
names: ['spine']

# Task
task: segment

# Quality-based training strategy
quality_levels:
  gold_standard: {generated_samples['gold_standard']}
  high_quality: {generated_samples['high_quality']}
  medium_quality: {generated_samples['medium_quality']}
  hard_negative: {generated_samples['hard_negative']}

# Training recommendations
training_strategy:
  stage_1: "Train on gold + high quality samples first"
  stage_2: "Fine-tune with medium quality samples"
  stage_3: "Hard negative mining with hard_negative samples"
"""
        
        with open(f"{output_dir}/dataset.yaml", 'w') as f:
            f.write(yaml_content)
        
        # Create quality-specific configs only if we have samples
        config_paths = []
        
        # Gold + High quality config
        if generated_samples['gold_standard'] > 0 or generated_samples['high_quality'] > 0:
            quality_folders = []
            if generated_samples['gold_standard'] > 0:
                quality_folders.append('gold')
            if generated_samples['high_quality'] > 0:
                quality_folders.append('high')
            
            if quality_folders:
                config_path = self.create_quality_specific_config(output_dir, 'gold_high', quality_folders)
                config_paths.append(config_path)
                print(f"✅ Created gold+high config: {config_path}")
        
        # Medium quality config
        if generated_samples['medium_quality'] > 0:
            config_path = self.create_quality_specific_config(output_dir, 'medium', ['medium'])
            config_paths.append(config_path)
            print(f"✅ Created medium config: {config_path}")
        
        # Hard negative config
        if enable_hard_negatives and generated_samples['hard_negative'] > 0:
            config_path = self.create_quality_specific_config(output_dir, 'hard_neg', ['hard_neg'])
            config_paths.append(config_path)
            print(f"✅ Created hard negative config: {config_path}")
        
        return config_paths
    
    def create_quality_specific_config(self, output_dir, config_name, quality_folders):
        """Create configuration for specific quality levels"""
        # Create symbolic links or copy files for quality-specific training
        base_path = os.path.abspath(output_dir)
        
        # Create a quality-specific dataset directory
        quality_dataset_dir = f"{output_dir}/quality_datasets/{config_name}"
        os.makedirs(f"{quality_dataset_dir}/images/train", exist_ok=True)
        os.makedirs(f"{quality_dataset_dir}/images/val", exist_ok=True)
        os.makedirs(f"{quality_dataset_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{quality_dataset_dir}/labels/val", exist_ok=True)
        
        # Create symbolic links to the quality folders
        for split in ['train', 'val']:
            for quality_folder in quality_folders:
                src_img = f"{base_path}/images/{split}/{quality_folder}"
                src_lbl = f"{base_path}/labels/{split}/{quality_folder}"
                
                if os.path.exists(src_img):
                    dst_img = f"{quality_dataset_dir}/images/{split}"
                    dst_lbl = f"{quality_dataset_dir}/labels/{split}"
                    
                    # Copy files instead of symlinks for better compatibility
                    self.copy_quality_files(src_img, dst_img, quality_folder)
                    self.copy_quality_files(src_lbl, dst_lbl, quality_folder)
        
        # Create the YAML config
        yaml_content = f"""# Quality-Specific Dataset: {config_name}
path: {os.path.abspath(quality_dataset_dir)}
train: images/train
val: images/val

nc: 1
names: ['spine']
task: segment
"""
        
        config_path = f"{output_dir}/dataset_{config_name}.yaml"
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        return config_path
    
    def copy_quality_files(self, src_dir, dst_dir, quality_folder):
        """Copy files from quality folder to combined directory"""
        if not os.path.exists(src_dir):
            return
        
        import shutil
        
        # Copy all files from src to dst
        for filename in os.listdir(src_dir):
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(dst_dir, f"{quality_folder}_{filename}")
            
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
    
    def save_generation_stats(self, output_dir, generated_samples, instance_stats, quality_stats):
        """Save detailed generation statistics"""
        stats = {
            'generation_summary': {
                'total_samples': sum(generated_samples.values()),
                'quality_distribution': generated_samples,
                'uncertainty_thresholds': self.uncertainty_thresholds
            },
            'instance_uncertainty_stats': {},
            'generation_issues': dict(quality_stats)
        }
        
        # Calculate uncertainty statistics for each quality level
        for quality, uncertainties in instance_stats.items():
            if uncertainties:
                stats['instance_uncertainty_stats'][quality] = {
                    'mean': float(np.mean(uncertainties)),
                    'std': float(np.std(uncertainties)),
                    'min': float(np.min(uncertainties)),
                    'max': float(np.max(uncertainties)),
                    'count': len(uncertainties)
                }
        
        with open(f"{output_dir}/generation_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create uncertainty distribution plot
        self.plot_uncertainty_distribution(instance_stats, f"{output_dir}/uncertainty_distribution.png")
    
    def plot_uncertainty_distribution(self, instance_stats, output_path):
        """Plot uncertainty distribution across quality levels"""
        plt.figure(figsize=(12, 8))
        
        colors = ['gold', 'green', 'orange', 'red']
        quality_names = ['Gold Standard', 'High Quality', 'Medium Quality', 'Hard Negative']
        
        for idx, (quality, uncertainties) in enumerate(instance_stats.items()):
            if uncertainties:
                plt.hist(uncertainties, bins=30, alpha=0.7, 
                        label=f'{quality_names[idx]} (n={len(uncertainties)})',
                        color=colors[idx])
        
        # Add threshold lines
        for name, threshold in self.uncertainty_thresholds.items():
            plt.axvline(threshold, linestyle='--', alpha=0.7, 
                       label=f'{name}: {threshold}')
        
        plt.xlabel('Uncertainty')
        plt.ylabel('Frequency')
        plt.title('Instance Uncertainty Distribution by Quality Level')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_progressive_yolo(self, dataset_yaml, epochs_per_stage=[100, 50, 25]):
        """
        Progressive YOLO training using uncertainty-curated data
        
        Args:
            dataset_yaml: Path to main dataset.yaml
            epochs_per_stage: Epochs for each training stage
        """
        print("🚀 Starting progressive YOLO training...")
        
        dataset_dir = os.path.dirname(dataset_yaml)
        models_dir = f"{dataset_dir}/models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Check what quality levels we actually have
        available_configs = []
        
        gold_high_config = f"{dataset_dir}/dataset_gold_high.yaml"
        medium_config = f"{dataset_dir}/dataset_medium.yaml"
        hard_neg_config = f"{dataset_dir}/dataset_hard_neg.yaml"
        
        # Stage 1: Gold + High Quality (if available)
        if os.path.exists(gold_high_config):
            print("\n📍 Stage 1: Training on Gold + High Quality samples")
            
            model = YOLO('yolo11l-seg.pt')
            try:
                results = model.train(
                    data=gold_high_config,
                    epochs=epochs_per_stage[0],
                    imgsz=640,
                    batch=16,
                    patience=25,
                    device=0,
                    project=models_dir,
                    name="stage1_gold_high",
                    save_period=25,
                    exist_ok=True
                )
                stage1_model = str(model.trainer.best)
                print(f"✅ Stage 1 complete: {stage1_model}")
            except Exception as e:
                print(f"⚠️ Stage 1 failed: {e}")
                print("🔄 Proceeding with base model...")
                stage1_model = 'yolo11l-seg.pt'
        else:
            print("⚠️ No gold/high quality samples found, starting with full dataset")
            stage1_model = 'yolo11l-seg.pt'
        
        # Stage 2: Full dataset or medium quality
        print("\n📍 Stage 2: Training on full/medium quality dataset")
        
        # Use medium config if available, otherwise use full dataset
        stage2_config = medium_config if os.path.exists(medium_config) else dataset_yaml
        
        model = YOLO(stage1_model)
        try:
            results = model.train(
                data=stage2_config,
                epochs=epochs_per_stage[1],
                imgsz=640,
                batch=12,
                patience=20,
                device=0,
                project=models_dir,
                name="stage2_main_training",
                save_period=15,
                exist_ok=True
            )
            stage2_model = str(model.trainer.best)
            print(f"✅ Stage 2 complete: {stage2_model}")
        except Exception as e:
            print(f"❌ Stage 2 failed: {e}")
            return stage1_model
        
        # Stage 3: Hard Negative Mining (optional)
        if os.path.exists(hard_neg_config) and epochs_per_stage[2] > 0:
            print("\n📍 Stage 3: Hard negative mining")
            
            model = YOLO(stage2_model)
            try:
                results = model.train(
                    data=hard_neg_config,
                    epochs=epochs_per_stage[2],
                    imgsz=640,
                    batch=8,
                    patience=15,
                    device=0,
                    project=models_dir,
                    name="stage3_hard_negatives",
                    save_period=10,
                    exist_ok=True
                )
                final_model = str(model.trainer.best)
                print(f"✅ Stage 3 complete: {final_model}")
            except Exception as e:
                print(f"⚠️ Stage 3 failed: {e}")
                print("Using Stage 2 model as final model")
                final_model = stage2_model
        else:
            print("⚠️ No hard negative samples or stage 3 disabled, using stage 2 model")
            final_model = stage2_model
        
        print(f"\n✅ Progressive training complete!")
        print(f"🎯 Final model: {final_model}")
        
        return final_model


# Usage function
def train_uncertainty_enhanced_yolo(d3set_path, prob_unet_path, output_dir):
    """
    Complete uncertainty-enhanced YOLO training pipeline
    
    Args:
        d3set_path: Path to training dataset
        prob_unet_path: Path to trained Prob-UNet
        output_dir: Output directory for enhanced dataset and models
    """
    # Initialize trainer
    trainer = UncertaintyGuidedYOLOTrainer(d3set_path, prob_unet_path)
    
    # Generate uncertainty-enhanced dataset
    dataset_yaml = trainer.generate_uncertainty_enhanced_dataset(
        output_dir=f"{output_dir}/enhanced_dataset",
        num_samples=6000,
        mc_samples=15,
        enable_hard_negatives=True
    )
    
    # Train YOLO progressively
    final_model = trainer.train_progressive_yolo(
        dataset_yaml=dataset_yaml,
        epochs_per_stage=[120, 60, 30]  # Adjust as needed
    )
    
    print(f"\n🎉 Uncertainty-enhanced training complete!")
    print(f"📁 Enhanced dataset: {dataset_yaml}")
    print(f"🎯 Final YOLO model: {final_model}")
    
    return final_model, dataset_yaml


# Example usage
if __name__ == "__main__":
    # YOUR PATHS
    d3set_path = "dataset/DeepD3_Training.d3set"
    prob_unet_path = "final_models_path/model_epoch_19_val_loss_0.3692.pth"
    output_dir = "uncertainty_enhanced_yolo"
    
    print("🔬 Starting uncertainty-enhanced YOLO training...")
    
    final_model, dataset_yaml = train_uncertainty_enhanced_yolo(
        d3set_path=d3set_path,
        prob_unet_path=prob_unet_path,
        output_dir=output_dir
    )
    
    print("🎊 Training pipeline complete!")