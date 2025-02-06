import os
import torch
import numpy as np
import cv2
import tifffile as tiff
from tqdm import tqdm
import torchvision.transforms as transforms
from probabilistic_unet import ProbabilisticUnet

class ProbabilisticUNetInference:
    def __init__(self, spine_model_path, dend_model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_spine = self._load_model(spine_model_path)
        self.model_dend = self._load_model(dend_model_path)
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def _load_model(self, model_path):
        model = ProbabilisticUnet(input_channels=1, num_classes=1, latent_dim=16).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def preprocess_image(self, image):
        epsilon = 1e-8
        image = (image - image.min()) / (image.max() - image.min() + epsilon) * 2 - 1
        h, w = image.shape
        pad_h = (32 - h % 32) if h % 32 else 0
        pad_w = (32 - w % 32) if w % 32 else 0
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        return image, (h, w)

    def process_stack(self, image_stack, num_samples=10):
        results = {
            'spine_samples': [],
            'dend_samples': [],
            'spine_mean': None,
            'dend_mean': None,
            'spine_std': None,
            'dend_std': None
        }
        
        with torch.no_grad():
            for slice_idx in range(image_stack.shape[0]):
                processed_slice, (orig_h, orig_w) = self.preprocess_image(image_stack[slice_idx])
                slice_tensor = self.transform(processed_slice).unsqueeze(0).to(self.device)
                
                spine_samples = []
                dend_samples = []
                
                for _ in range(num_samples):
                    self.model_spine.forward(slice_tensor, None, training=False)
                    self.model_dend.forward(slice_tensor, None, training=False)
                    
                    spine_prob = torch.sigmoid(self.model_spine.sample())
                    dend_prob = torch.sigmoid(self.model_dend.sample())
                    
                    spine_prob = spine_prob.squeeze().cpu().numpy()[:orig_h, :orig_w]
                    dend_prob = dend_prob.squeeze().cpu().numpy()[:orig_h, :orig_w]
                    
                    spine_samples.append(spine_prob)
                    dend_samples.append(dend_prob)
                
                spine_samples = np.stack(spine_samples)
                dend_samples = np.stack(dend_samples)
                
                results['spine_samples'].append(spine_samples)
                results['dend_samples'].append(dend_samples)
                results['spine_mean'] = np.mean(spine_samples, axis=0)
                results['dend_mean'] = np.mean(dend_samples, axis=0)
                results['spine_std'] = np.std(spine_samples, axis=0)
                results['dend_std'] = np.std(dend_samples, axis=0)
                
        return results

def save_heatmap(data, path):
    heatmap = (data * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(path, heatmap)

def run_inference(tif_path, spine_model_path, dend_model_path, output_dir, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)
    tif_stack = tiff.imread(tif_path).astype(np.float32)
    inferencer = ProbabilisticUNetInference(spine_model_path, dend_model_path)
    
    for stack_idx in tqdm(range(len(tif_stack)), desc="Processing stacks"):
        results = inferencer.process_stack(tif_stack[stack_idx:stack_idx+1], num_samples)
        stack_dir = os.path.join(output_dir, f'stack_{stack_idx}')
        os.makedirs(stack_dir, exist_ok=True)
        
        # Save heatmaps
        save_heatmap(results['spine_mean'], os.path.join(stack_dir, 'spine_mean_prob.png'))
        save_heatmap(results['dend_mean'], os.path.join(stack_dir, 'dend_mean_prob.png'))
        save_heatmap(results['spine_std'], os.path.join(stack_dir, 'spine_uncertainty.png'))
        save_heatmap(results['dend_std'], os.path.join(stack_dir, 'dend_uncertainty.png'))
        
        # Save samples
        samples_dir = os.path.join(stack_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        for sample_idx in range(num_samples):
            save_heatmap(results['spine_samples'][0][sample_idx], 
                        os.path.join(samples_dir, f'spine_sample_{sample_idx}.png'))
            save_heatmap(results['dend_samples'][0][sample_idx], 
                        os.path.join(samples_dir, f'dend_sample_{sample_idx}.png'))

if __name__ == "__main__":
    config = {
        'tif_path': "DeepD3_Benchmark.tif",
        'spine_model_path': "spine_model_epoch_18_iou_0.4766.pth",
        'dend_model_path': "dend_model_epoch_20_iou_0.5976.pth",
        'output_dir': "inference_results_png",
        'num_samples': 10
    }
    run_inference(**config)