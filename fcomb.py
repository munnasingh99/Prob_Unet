import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from axis_align import Encoder
from deepd3 import DeepD3
from utils import init_weights, init_weights_orthogonal_normal, l2_regularisation
from torch.distributions import Normal, Independent, kl
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    Modified to handle both dendrite and spine segmentation inputs for the posterior.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        
        # For posterior, we need to account for two segmentation masks (dendrites and spines)
        if self.posterior:
            self.name = 'Posterior'
            self.input_channels += 0  # Add channels for both dendrites and spines
        else:
            self.name = 'Prior'
            
        # Create encoder for distribution parameters
        self.encoder = Encoder(self.input_channels, self.num_filters, 
                             self.no_convs_per_block, initializers, 
                             posterior=self.posterior)
        
        # Final 1x1 convolution to get mu and log_sigma
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):
        """
        Forward pass
        Args:
            input: Input image
            segm: For posterior, tuple of (dendrites, spines) segmentation masks
        """
        if segm is not None:
            # Concatenate both segmentation masks for posterior
            dendrites, spines = segm
            input = torch.cat([input, dendrites, spines], dim=1)

        # Get encoding
        encoding = self.encoder(input)

        # Global average pooling
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        # Get mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        # Split into mu and log_sigma
        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        # Create and return distribution
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist

class Fcomb(nn.Module):
    """
    Modified Fcomb to produce dual outputs for dendrites and spines.
    Combines the sample from latent space with UNet features.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb, self).__init__()
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.channel_axis = 1
        self.spatial_axes = [2, 3]
        self.use_tile = use_tile
        
        if self.use_tile:
            # Create two separate processing paths for dendrites and spines
            # Dendrites path
            self.dendrites_layers = self._create_processing_path(
                num_filters[0] + latent_dim,
                num_filters[0],
                no_convs_fcomb
            )
            self.dendrites_final = nn.Conv2d(num_filters[0], num_output_channels, kernel_size=1)
            
            # Spines path
            self.spines_layers = self._create_processing_path(
                num_filters[0] + latent_dim,
                num_filters[0],
                no_convs_fcomb
            )
            self.spines_final = nn.Conv2d(num_filters[0], num_output_channels, kernel_size=1)
            
            # Initialize weights
            self._initialize_weights(initializers)

    def _create_processing_path(self, in_channels, hidden_channels, no_convs):
        """Helper function to create processing path."""
        layers = []
        
        # First layer handles different input dimensions
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=1))
        layers.append(nn.ReLU(inplace=True))
        
        # Additional layers
        for _ in range(no_convs - 2):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, initializers):
        """Initialize network weights."""
        def init_function(m):
            if isinstance(m, nn.Conv2d):
                if initializers['w'] == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight)
                elif initializers['w'] == 'he_normal':
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                
                if m.bias is not None:
                    if initializers['b'] == 'normal':
                        torch.nn.init.normal_(m.bias, mean=0, std=0.001)
                    elif initializers['b'] == 'zeros':
                        torch.nn.init.zeros_(m.bias)
    
    def tile(self, a, dim, n_tile):
        """
        Tile function for repeating tensor along a dimension.
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        
        # Move order_index to the same device as a
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
        ).to(a.device)  # Ensure order_index is on the same device
        
        return torch.index_select(a, dim, order_index)


    def forward(self, feature_map, z):
        """
        Forward pass producing both dendrite and spine predictions.
        Args:
            feature_map: Output features from the UNet
            z: Latent space sample
        Returns:
            tuple: (dendrites_output, spines_output)
        """
        if self.use_tile:
            # Expand z to match feature map dimensions
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])
            
            # Concatenate feature map with z
            combined = torch.cat([feature_map, z], dim=self.channel_axis)
            
            # Process through both paths
            dendrites = self.dendrites_layers(combined)
            dendrites = self.dendrites_final(dendrites)
            
            spines = self.spines_layers(combined)
            spines = self.spines_final(spines)
            
            return dendrites, spines

class ProbabilisticUnet(nn.Module):
    """
    Modified Probabilistic U-Net that uses DeepD3 architecture and handles dual outputs
    for dendrite and spine segmentation.
    
    Args:
        input_channels (int): Number of input image channels
        num_filters (list): List of filter numbers for each level
        latent_dim (int): Dimension of the latent space
        no_convs_fcomb (int): Number of convs in the Fcomb network
        beta (float): KL divergence weight in the ELBO loss
        deepd3_model: Instance of DeepD3 model
    """
    def __init__(self, input_channels=1, num_filters=[32,64,128,192], 
                 latent_dim=6, no_convs_fcomb=4, beta=10.0, deepd3_model=None):
        super(ProbabilisticUnet, self).__init__()
        
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3  # fixed parameter from original implementation
        self.no_convs_fcomb = no_convs_fcomb
        self.beta = beta
        
        # Initialize DeepD3 model if not provided
        if deepd3_model is None:
            raise ValueError("DeepD3 model must be provided")
        self.deepd3 = deepd3_model
        
        # Initialize the ProbabilisticUnet components
        self.prior = AxisAlignedConvGaussian(
            input_channels=self.input_channels,
            num_filters=self.num_filters,
            no_convs_per_block=self.no_convs_per_block,
            latent_dim=self.latent_dim,
            initializers={'w': 'he_normal', 'b': 'normal'},
        )
        
        self.posterior = AxisAlignedConvGaussian(
            input_channels=self.input_channels,
            num_filters=self.num_filters,
            no_convs_per_block=self.no_convs_per_block,
            latent_dim=self.latent_dim,
            initializers={'w': 'he_normal', 'b': 'normal'},
            posterior=True
        )
        
        self.fcomb = Fcomb(
            num_filters=self.num_filters,
            latent_dim=self.latent_dim,
            num_output_channels=1,  # single channel output for each mask
            no_convs_fcomb=self.no_convs_fcomb,
            initializers={'w': 'orthogonal', 'b': 'normal'},
            use_tile=True
        )

    def forward(self, patch, segm=None, training=True):
        """
        Forward pass of the model.
        
        Args:
            patch (torch.Tensor): Input image
            segm (tuple): Tuple of (dendrites_mask, spines_mask) if training
            training (bool): Whether in training mode
        """
        # Get DeepD3 features
        if training:
            self.unet_features = self.deepd3(patch)[0]  # Get features before final layer
            if segm is not None:
                # Handle both segmentation masks for posterior
                dendrites_mask, spines_mask = segm
                self.posterior_latent_space = self.posterior(patch, (dendrites_mask, spines_mask))
        else:
            with torch.no_grad():
                self.unet_features = self.deepd3(patch)[0]
        
        # Get prior distribution
        self.prior_latent_space = self.prior(patch)

    def sample(self, testing=False):
        """
        Sample from the prior distribution and generate segmentation.
        
        Args:
            testing (bool): Whether in testing mode
            
        Returns:
            tuple: (dendrites_prediction, spines_prediction)
        """
        if testing:
            z_prior = self.prior_latent_space.sample()
        else:
            z_prior = self.prior_latent_space.rsample()
        
        # Generate both segmentation masks
        dendrites, spines = self.fcomb(self.unet_features, z_prior)
        return dendrites, spines

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct segmentation using posterior distribution.
        
        Args:
            use_posterior_mean (bool): Use posterior mean instead of sampling
            calculate_posterior (bool): Calculate new posterior sample
            z_posterior (torch.Tensor): Use provided posterior sample
            
        Returns:
            tuple: (dendrites_reconstruction, spines_reconstruction)
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        
        return self.fcomb(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate KL divergence between posterior and prior distributions.
        
        Args:
            analytic (bool): Use analytic or sampling-based KL
            calculate_posterior (bool): Calculate new posterior sample
            z_posterior (torch.Tensor): Use provided posterior sample
            
        Returns:
            torch.Tensor: KL divergence value
        """
        if analytic:
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound (ELBO) loss.
        
        Args:
            segm (tuple): Tuple of (dendrites_mask, spines_mask)
            analytic_kl (bool): Use analytic KL calculation
            reconstruct_posterior_mean (bool): Use posterior mean for reconstruction
            
        Returns:
            torch.Tensor: ELBO loss value
        """
        dendrites_mask, spines_mask = segm
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # Calculate KL divergence
        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl))
        
        # Generate reconstructions
        dendrites_recon, spines_recon = self.reconstruct(
            use_posterior_mean=reconstruct_posterior_mean
        )
        
        # Calculate reconstruction losses for both outputs
        reconstruction_loss_dendrites = criterion(dendrites_recon, dendrites_mask)
        reconstruction_loss_spines = criterion(spines_recon, spines_mask)
        
        # Combine losses
        self.reconstruction_loss = torch.sum(reconstruction_loss_dendrites) + torch.sum(reconstruction_loss_spines)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss_dendrites) + torch.mean(reconstruction_loss_spines)
        
        return -(self.reconstruction_loss + self.beta * self.kl)

# Example usage and testing
def test_probabilistic_unet():
    # Create DeepD3 model
    deepd3 = DeepD3(filters=32, input_shape=(128, 128, 1))
    
    # Create Probabilistic UNet with DeepD3
    prob_unet = ProbabilisticUnet(
        input_channels=1,
        num_filters=[32, 64, 128, 192],
        latent_dim=6,
        deepd3_model=deepd3
    )
    
    # Create dummy data
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 128)
    dendrites = torch.randint(0, 2, (batch_size, 1, 128, 128)).float()
    spines = torch.randint(0, 2, (batch_size, 1, 128, 128)).float()
    
    # Forward pass
    prob_unet.forward(x, (dendrites, spines), training=True)
    
    # Test sampling
    d_sample, s_sample = prob_unet.sample()
    print(f"Sample shapes - Dendrites: {d_sample.shape}, Spines: {s_sample.shape}")
    
    # Test reconstruction
    d_recon, s_recon = prob_unet.reconstruct()
    print(f"Reconstruction shapes - Dendrites: {d_recon.shape}, Spines: {s_recon.shape}")
    
    # Test ELBO
    elbo_loss = prob_unet.elbo((dendrites, spines))
    print(f"ELBO loss: {elbo_loss.item()}")

if __name__ == "__main__":
    test_probabilistic_unet()

    self.dendrites_layers.apply(init_function)
    self.spines_layers.apply(init_function)
    self.dendrites_final.apply(init_function)
    self.spines_final.apply(init_function)
