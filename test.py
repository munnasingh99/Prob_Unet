import torch
import numpy as np
import matplotlib.pyplot as plt
from test_train import ProbabilisticUnet  # Ensure this is your model implementation
import imageio as io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


stack = np.asarray(io.mimread(r'DeepD3_Benchmark.tif'))
# Load the trained model
model_path = 'model_checkpoints/best_model_dend_epoch_2_loss_13878.9867.pth'
net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32, 64, 128, 192], latent_dim=12, no_convs_fcomb=4, beta=20)
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint)
net.eval()  # Set to evaluation mode
net.to(device)

# Prepare input data
input_data = stack[34].astype(np.float32)  # Dummy data
input_tensor = torch.from_numpy(input_data).to(device)
input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

print(input_tensor.shape)
# Dummy segmentation mask (since your model requires both input and segmentation during training)
segm_data = np.random.rand(1, 1, 128, 128).astype(np.float32)  # Dummy data
segm_tensor = torch.from_numpy(segm_data).to(device)

# Run forward pass to initialize latent spaces
with torch.no_grad():
    net.forward(input_tensor, segm_tensor, training=False)  # Use training=False for prior latent space only


# Generate predictions
with torch.no_grad():
    predictions = net.sample(testing=True)  # Stochastic predictions
    reconstructed = net.reconstruct(training=False)  # Deterministic predictions

print("Here")
# Post-process (binary segmentation example)
binary_predictions = (torch.sigmoid(reconstructed) > 0.5).float()
predictions_squeezed = binary_predictions.squeeze().cpu().numpy()  # Shape: (128, 128)

# Visualize predictions
plt.imshow(predictions_squeezed, cmap='gray')
plt.title('Binary Prediction')
plt.colorbar()
plt.savefig("predictions.png")
plt.show()
