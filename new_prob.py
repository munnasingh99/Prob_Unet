import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

# Import your custom unet (the PyTorch version you ported)
from new_model import DeepD3Model  
from unet_blocks import *  # if needed for other utilities
from utils import init_weights, init_weights_orthogonal_normal, l2_regularisation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Keep the definitions for AxisAlignedConvGaussian and Fcomb unchanged.
# ... (AxisAlignedConvGaussian and Fcomb remain as in your original probabilistic_unet.py)
class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, segm_channels,padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += segm_channels

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.BatchNorm2d(output_dim)) 
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.BatchNorm2d(output_dim))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, segm_channels,posterior=False,):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.segm_channels = segm_channels
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, self.segm_channels,posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            #print(input.shape,segm.shape)
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        #We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]
        log_sigma = torch.clamp(log_sigma, min=-10, max=10)

        # Debugging mean and log variance
        #print(f"{self.name} Latent Space:")
        #print(f"  Mean (mu): {mu}")
        #print(f"  Log Variance (log_sigma): {log_sigma}")

        # mu.register_hook(debug_hook)
        # log_sigma.register_hook(debug_hook)

        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        #dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma + 1e-6)), 1) 
        return dist

class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb 
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.num_filters[0]+self.latent_dim, self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z,3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)


class ProbabilisticUnet(nn.Module):
    """
    Modified Probabilistic U-Net that integrates the custom U-Net with dual decoders.
    """
    def __init__(self, input_channels=1, num_classes=1, num_filters=[32,64,128,192], 
                 latent_dim=6, no_convs_fcomb=4, beta=1):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.beta = beta
        self.z_prior_sample = 0

        # Instead of using the vanilla Unet, we now instantiate the custom unet.
        # Adjust the parameters as needed (here, we assume base_filters is num_filters[0]
        # and number of layers is the length of num_filters).
        self.unet = DeepD3Model(
            in_channels=self.input_channels,
            base_filters=self.num_filters[0],
            num_layers=len(self.num_filters),
            activation="swish",
            use_batchnorm=True,apply_last_layer=False
        ).to(device)

        # Create two separate fcomb modules, one for each decoder output.
        self.fcomb_dendrites = Fcomb(
            self.num_filters, self.latent_dim, self.input_channels, 
            self.num_classes, self.no_convs_fcomb, 
            {'w': 'orthogonal', 'b': 'normal'}, use_tile=True
        ).to(device)
        self.fcomb_spines = Fcomb(
            self.num_filters, self.latent_dim, self.input_channels, 
            self.num_classes, self.no_convs_fcomb, 
            {'w': 'orthogonal', 'b': 'normal'}, use_tile=True
        ).to(device)

        # Prior and posterior networks remain the same.
        self.prior = AxisAlignedConvGaussian(
            self.input_channels, self.num_filters, self.no_convs_per_block, 
            self.latent_dim, self.initializers, posterior=False,
        segm_channels=1).to(device)
        self.posterior = AxisAlignedConvGaussian(
            self.input_channels, self.num_filters, self.no_convs_per_block, 
            self.latent_dim, self.initializers, posterior=True,
        segm_channels=2).to(device)

    def forward(self, patch, segm, training=True):
        """
        Run patch through prior/posterior networks and the custom U-Net.
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
            print(f"Posterior Mean: {self.posterior_latent_space.base_dist.loc.mean().item()}, "
                  f"Posterior Std: {self.posterior_latent_space.base_dist.scale.mean().item()}")

        self.prior_latent_space = self.prior.forward(patch)
        print(f"Prior Mean: {self.prior_latent_space.base_dist.loc.mean().item()}, "
              f"Prior Std: {self.prior_latent_space.base_dist.scale.mean().item()}")

        # Run patch through the custom U-Net.
        # The custom unet returns two outputs: one from the dendrite decoder and one from the spine decoder.
        dendrite_features, spine_features = self.unet(patch)
        self.dendrite_features = dendrite_features
        self.spine_features = spine_features
        print(f"Dendrite Features: {dendrite_features.shape}, Spine Features: {spine_features.shape}")

    def sample(self, testing=False):
        """
        Sample a segmentation by fusing a latent sample with the U-Net features.
        """
        if not testing:
            z_prior = self.prior_latent_space.rsample()
        else:
            z_prior = self.prior_latent_space.sample()
        self.z_prior_sample = z_prior

        # Fuse the latent sample with each set of features using the separate fcomb modules.
        dendrites = self.fcomb_dendrites(self.dendrite_features, z_prior)
        spines = self.fcomb_spines(self.spine_features, z_prior)
        return dendrites, spines

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None, training=True):
        """
        Reconstruct segmentation from latent space.
        """
        if self.posterior_latent_space is not None:
            if use_posterior_mean:
                z_posterior = self.posterior_latent_space.loc
            elif calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        else:
            print("Warning: Posterior latent space unavailable. Falling back to prior latent space.")
            z_posterior = self.prior_latent_space.rsample()

        dendrites = self.fcomb_dendrites(self.dendrite_features, z_posterior)
        spines = self.fcomb_spines(self.spine_features, z_posterior)
        return dendrites, spines

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior distributions.
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

    def elbo(self, segm_d,segm_s, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        z_posterior = self.posterior_latent_space.rsample()

        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))
        dendrites_rec, spines_rec = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean,
                                                     calculate_posterior=False, z_posterior=z_posterior)
        # You need to define how to split segm into dendrite and spine targets.
        # For illustration, assume segm has two channels.
        segm_dendrites = segm_d
        segm_spines = segm_s

        loss_dendrites = criterion(dendrites_rec, segm_dendrites)
        loss_spines = criterion(spines_rec, segm_spines)
        reconstruction_loss = loss_dendrites + loss_spines

        epsilon = 1e-7  # small epsilon
        self.reconstruction_loss = torch.sum(reconstruction_loss + epsilon)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss + epsilon)

        return -(self.reconstruction_loss + self.beta * self.kl)
