import torch
from probabilistic_unet import ProbabilisticUnet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the .pth file
checkpoint = torch.load("model_checkpoints/best_den_model_epoch_10_loss_731692.8438_20250126_151044.pth")
# Assume 'model' is your model class instance



# Iterate over the state_dict to see stored parameters
state_dict = checkpoint if "model_state_dict" not in checkpoint else checkpoint["model_state_dict"]

for name, param in state_dict.items():
    print(f"Parameter Name: {name}, Shape: {param.shape}")


