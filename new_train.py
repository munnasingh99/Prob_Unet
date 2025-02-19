import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
import torch.nn.functional as F

# Make sure these modules are in your PYTHONPATH or the same directory
from new_model import DeepD3Model  
from unet_blocks import *  # if needed
from utils import init_weights, init_weights_orthogonal_normal, l2_regularisation

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- (Re)define AxisAlignedConvGaussian and Fcomb if not imported ---
# (Assuming you have these classes in your file, or they are imported from your probabilistic_unet.py)

# For brevity, we assume you have already defined the modified versions of Encoder, AxisAlignedConvGaussian, Fcomb,
# and ProbabilisticUnet as per your integration code.

# Here is a simplified test script using your integrated ProbabilisticUnet:
from new_prob import ProbabilisticUnet  # update the import to point to your file

if __name__ == '__main__':
    # Instantiate the model
    model = ProbabilisticUnet(
        input_channels=1,
        num_classes=1,
        num_filters=[32, 64, 128, 192],
        latent_dim=6,
        no_convs_fcomb=4,
        beta=1
    ).to(device)
    
    # Create dummy input image (patch): batch size 2, one channel, 128x128 image
    patch = torch.randn(2, 1, 128, 128).to(device)
    
    # Create dummy segmentation ground truth:
    # For the posterior network we expect a segmentation tensor with 2 channels (one for dendrites, one for spines)
    segm = torch.randn(2, 2, 128, 128).to(device)
    
    # Forward pass (training mode)
    model.forward(patch, segm, training=True)
    
    # Sample segmentation predictions (this will fuse the latent sample with U-Net features)
    dendrites_pred, spines_pred = model.sample(testing=False)
    print("Dendrites prediction shape:", dendrites_pred.shape)
    print("Spines prediction shape:", spines_pred.shape)
    
    # Compute the ELBO (loss)
    # In this example, we assume that segm's channel 0 corresponds to dendrites and channel 1 to spines.
    loss = model.elbo(segm_d=segm[:, 0:1, ...], segm_s=segm[:, 1:2, ...])
    print("ELBO loss:", loss.item())
