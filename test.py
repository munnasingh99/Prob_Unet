import torch
import numpy as np
import matplotlib.pyplot as plt
from probabilistic_unet import ProbabilisticUnet  # Ensure this is your model implementation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model_path = 'model_checkpoints/best_model_epoch_43_loss_7.3387_20250124_133230.pth'
net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32, 64, 128, 192], latent_dim=6, no_convs_fcomb=4, beta=20)
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()  # Set to evaluation mode
net.to(device)

# Prepare input data
input_data = np.random.rand(1, 1, 128, 128).astype(np.float32)  # Dummy data
input_tensor = torch.from_numpy(input_data).to(device)

# Generate predictions
with torch.no_grad():
    predictions = net.sample()  # Stochastic predictions
    reconstructed = net.reconstruct(use_posterior_mean=True)  # Deterministic predictions

# Post-process (binary segmentation example)
binary_predictions = (torch.sigmoid(reconstructed) > 0.5).float()

# Visualize predictions
plt.imshow(binary_predictions.cpu().numpy()[0, 0], cmap='gray')
plt.title('Binary Prediction')
plt.show()
