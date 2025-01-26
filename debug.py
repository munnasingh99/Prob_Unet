import torch
from probabilistic_unet import ProbabilisticUnet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ProbabilisticUnet()
input_patch = torch.randn(1, 1, 128, 128).to(device)  # Example input
segmentation = torch.randn(1, 1, 128, 128).to(device)  # Example target

model.forward(input_patch, segmentation, training=True)
model.sample(testing=False)
kl_div = model.kl_divergence(analytic=True)
