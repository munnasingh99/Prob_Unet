import unittest
import torch
from probabilistic_unet import ProbabilisticUnet
from unet import Unet_TwoHeads
from encoder import Encoder
from axis_align import AxisAlignedConvGaussian
from fcomb import FcombDendrite, FcombSpine
from utils import *  # Add this if you are using utils.py file

# ... (Your model code: probabilistic_unet.py, unet_two_heads.py, encoder.py, etc.) ...

batch_size = 2
input_channels = 1
num_classes = 1
height = 64
width = 64
latent_dim = 6
num_filters = [32, 64, 128, 192]

# Dummy input image
dummy_input = torch.randn(batch_size, input_channels, height, width)
# Dummy segmentation masks for dendrites and spines
dummy_segm_dendrite = torch.randint(0, 2, (batch_size, num_classes, height, width)).float()
dummy_segm_spine = torch.randint(0, 2, (batch_size, num_classes, height, width)).float()
# Dummy latent vector (for testing Fcomb and reconstruct)
dummy_z = torch.randn(batch_size, latent_dim)

class TestProbabilisticUNet(unittest.TestCase):
    def setUp(self):
        self.model = ProbabilisticUnet(input_channels=input_channels, num_classes=num_classes,
                                       num_filters=num_filters, latent_dim=latent_dim)

    def test_forward_pass(self):
        self.model.forward(dummy_input, dummy_segm_dendrite, dummy_segm_spine)
        # Add assertions here to check any intermediate values if needed

    def test_sample(self):
        sample_dendrite, sample_spine = self.model.sample()
        self.assertEqual(sample_dendrite.shape, (batch_size, num_classes, height, width))
        self.assertEqual(sample_spine.shape, (batch_size, num_classes, height, width))
        self.assertEqual(sample_dendrite.dtype, torch.float32)
        self.assertEqual(sample_spine.dtype, torch.float32)
        self.assertFalse(torch.isnan(sample_dendrite).any())
        self.assertFalse(torch.isnan(sample_spine).any())

    def test_reconstruct(self):
        reconstruction_dendrite, reconstruction_spine = self.model.reconstruct()
        self.assertEqual(reconstruction_dendrite.shape, (batch_size, num_classes, height, width))
        self.assertEqual(reconstruction_spine.shape, (batch_size, num_classes, height, width))
        self.assertEqual(reconstruction_dendrite.dtype, torch.float32)
        self.assertEqual(reconstruction_spine.dtype, torch.float32)
        self.assertFalse(torch.isnan(reconstruction_dendrite).any())
        self.assertFalse(torch.isnan(reconstruction_spine).any())

    def test_kl_divergence(self):
        kl_div = self.model.kl_divergence()
        self.assertTrue(isinstance(kl_div, torch.Tensor))
        self.assertEqual(kl_div.numel(), 1)  # Should be a scalar

    def test_elbo(self):
        elbo = self.model.elbo(dummy_segm_dendrite, dummy_segm_spine)
        self.assertTrue(isinstance(elbo, torch.Tensor))
        self.assertEqual(elbo.numel(), 1)  # Should be a scalar
        elbo.backward()  # Check if gradients can flow

    def test_Unet_TwoHeads(self):
        unet = Unet_TwoHeads(input_channels=input_channels, num_classes=num_classes, num_filters=num_filters,
                             initializers={'w': 'he_normal', 'b': 'normal'})
        output = unet(dummy_input, False)
        self.assertEqual(len(output), 2)  # Should output two feature maps
        self.assertEqual(output[0].shape, (batch_size, num_filters[0], height, width))
        self.assertEqual(output[1].shape, (batch_size, num_filters[0], height, width))

    def test_Encoder(self):
        encoder = Encoder(input_channels=input_channels, num_filters=num_filters, no_convs_per_block=3,
                          initializers={'w': 'he_normal', 'b': 'normal'}, posterior=True)
        output = encoder(torch.cat((dummy_input, dummy_segm_dendrite, dummy_segm_spine), dim=1))  # Concatenate masks for posterior
        self.assertEqual(output.shape, (batch_size, num_filters[-1], height // (2 ** (len(num_filters) - 1)),
                                        width // (2 ** (len(num_filters) - 1))))

    def test_AxisAlignedConvGaussian(self):
        model = AxisAlignedConvGaussian(input_channels=input_channels, num_filters=num_filters,
                                        no_convs_per_block=3, latent_dim=latent_dim,
                                        initializers={'w': 'he_normal', 'b': 'normal'}, posterior=True)
        output = model(dummy_input, torch.cat((dummy_segm_dendrite, dummy_segm_spine), dim=1))
        self.assertTrue(isinstance(output, Independent))
        self.assertEqual(output.base_dist.loc.shape, (batch_size, latent_dim))
        self.assertEqual(output.base_dist.scale.shape, (batch_size, latent_dim))

    def test_Fcomb(self):
        fcomb_dendrite = FcombDendrite(num_filters=num_filters, latent_dim=latent_dim,
                                      num_output_channels=input_channels, num_classes=num_classes, no_convs_fcomb=4,
                                      initializers={'w': 'orthogonal', 'b': 'normal'}, use_tile=True)
        fcomb_spine = FcombSpine(num_filters=num_filters, latent_dim=latent_dim,
                                num_output_channels=input_channels, num_classes=num_classes, no_convs_fcomb=4,
                                initializers={'w': 'orthogonal', 'b': 'normal'}, use_tile=True)
        dummy_feature_map = torch.randn(batch_size, num_filters[0], height, width)
        output_dendrite = fcomb_dendrite(dummy_feature_map, dummy_z)
        output_spine = fcomb_spine(dummy_feature_map, dummy_z)
        self.assertEqual(output_dendrite.shape, (batch_size, num_classes, height, width))
        self.assertEqual(output_spine.shape, (batch_size, num_classes, height, width))

if __name__ == '__main__':
    unittest.main()