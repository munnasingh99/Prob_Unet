import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Basic building blocks
# ---------------------------

class EncoderBlock(nn.Module):
    """
    One encoder block:
    - Computes a 1x1 convolution (the identity/residual branch)
    - Applies two 3x3 conv layers (with BatchNorm and activation)
    - Adds the identity to the output of the second conv layer
    - Returns the output (to be concatenated in the decoder) and the pooled output.
    """
    def __init__(self, in_channels, out_channels, activation, use_batchnorm=True):
        super().__init__()
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        self.activation = activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        identity = self.identity_conv(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        # Add the identity (residual connection)
        x = self.activation(x + identity)
        pooled = self.pool(x)
        return x, pooled

class DecoderBlock(nn.Module):
    """
    One decoder block that applies two 3x3 conv layers (with BatchNorm and activation).
    """
    def __init__(self, in_channels, out_channels, activation, use_batchnorm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x

# ---------------------------
# Decoder Module
# ---------------------------

class Decoder(nn.Module):
    """
    Decoder module that upsamples the latent features and
    concatenates corresponding encoder features at each level.
    
    The structure follows:
      for each decoder block:
          1. Upsample by a factor of 2
          2. Concatenate the corresponding encoder feature map
          3. Apply two convolutional layers (DecoderBlock)
      Finally, apply a 1x1 convolution and a sigmoid to obtain a single-channel output.
    """
    def __init__(self, num_layers, base_filters, activation, use_batchnorm=True):
        super().__init__()
        self.num_layers = num_layers
        self.blocks = nn.ModuleList()
        # For each decoder block, compute in/out channel numbers.
        # Let L = num_layers and F = base_filters.
        # The latent feature map has channels F * 2**L.
        # For the first block, after upsampling and concatenation with the encoder feature (which has F * 2**(L-1) channels),
        # the number of input channels is: F*2**L + F*2**(L-1), and we want to output F*2**(L-1) channels.
        # For subsequent blocks, a similar pattern holds.
        for i in range(num_layers):
            k = num_layers - 1 - i  # k goes from L-1 down to 0
            if i == 0:
                in_ch = base_filters * (2 ** num_layers) + base_filters * (2 ** (num_layers - 1))
            else:
                in_ch = base_filters * (2 ** (k + 1)) + base_filters * (2 ** k)
            out_ch = base_filters * (2 ** k)
            self.blocks.append(DecoderBlock(in_ch, out_ch, activation, use_batchnorm))
        self.last_layer_features = None
        self.final_conv = nn.Conv2d(base_filters, 1, kernel_size=1)
        self.final_activation = nn.Sigmoid()
        
    def forward(self, x, encoder_features):
        # encoder_features is a list of features from the encoder;
        # they should be used in reverse order (deepest first)
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            # Pop the last saved encoder feature
            enc_feat = encoder_features.pop()
            x = torch.cat([x, enc_feat], dim=1)
            x = block(x)
        self.last_layer_features = x
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

# ---------------------------
# Main Model
# ---------------------------

class DeepD3Model(nn.Module):
    """
    PyTorch implementation of your custom U-Net with one encoder and dual decoders.
    
    - The encoder comprises several EncoderBlocks.
    - A latent convolution is applied after the encoder.
    - Two decoders (for dendrites and spines) process the latent representation,
      each concatenating with the same saved encoder features.
      
    Args:
        in_channels (int): Number of channels in the input image.
        base_filters (int): Base number of filters (multiplied at each encoder level).
        num_layers (int): Depth of the network (number of encoder/decoder blocks).
        activation (str): Activation function to use; defaults to "swish" (implemented as nn.SiLU).
        use_batchnorm (bool): Whether to use BatchNorm (default True).
    """
    def __init__(self, in_channels=1, base_filters=32, num_layers=4, activation="swish", use_batchnorm=True,apply_last_layer=True):
        super().__init__()
        # Choose activation function
        self.apply_last_layer=apply_last_layer
        if activation == "swish":
            act = nn.SiLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)
        self.activation = act
        self.num_layers = num_layers
        self.base_filters = base_filters
        self.use_batchnorm = use_batchnorm
        
        # Build encoder blocks
        self.encoder_blocks = nn.ModuleList()
        current_in = in_channels
        for i in range(num_layers):
            out_channels = base_filters * (2 ** i)
            self.encoder_blocks.append(EncoderBlock(current_in, out_channels, self.activation, use_batchnorm))
            current_in = out_channels  # pooling does not change channel count
            
        # Latent convolution:
        # Input channels: output of the last encoder block (before pooling)
        # Output channels: base_filters * 2**(num_layers)
        latent_in = base_filters * (2 ** (num_layers - 1))
        latent_out = base_filters * (2 ** num_layers)
        self.latent_conv = nn.Sequential(
            nn.Conv2d(latent_in, latent_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_out) if use_batchnorm else nn.Identity(),
            self.activation
        )
        
        # Two decoders: one for dendrites and one for spines.
        self.decoder_dendrites = Decoder(num_layers, base_filters, self.activation, use_batchnorm)
        self.decoder_spines = Decoder(num_layers, base_filters, self.activation, use_batchnorm)
        
    def forward(self, x):
        encoder_features = []
        # Encoder: save features for skip connections.
        for block in self.encoder_blocks:
            feat, x = block(x)
            encoder_features.append(feat)
        
        # Make copies of encoder features for each decoder
        enc_feats_d = encoder_features.copy()
        enc_feats_s = encoder_features.copy()
        
        # Latent representation
        x_latent = self.latent_conv(x)
        
        # Obtain decoder features (before the final 1x1 conv)
        dendrites_features = self.decoder_dendrites.forward(x_latent,enc_feats_d)  # you need to implement this method
        spines_features = self.decoder_spines.forward(x_latent, enc_feats_s)
        
        if self.apply_last_layer:
            # Optionally apply final conv to get segmentation
            return dendrites_features, spines_features
        else:
            # Return the intermediate features (which should have self.num_filters[0] channels, e.g. 32)
            return self.decoder_dendrites.last_layer_features, self.decoder_spines.last_layer_features

# ---------------------------
# Testing the model
# ---------------------------

if __name__ == '__main__':
    # Create a random input tensor (batch_size, channels, height, width)
    x = torch.randn(1, 1, 48, 48)  # for example, a 48x48 grayscale image
    model = DeepD3Model(in_channels=1, base_filters=32, num_layers=4, activation="swish",apply_last_layer=False)
    dendrites, spines = model(x)
    print("Dendrites output shape:", dendrites.shape)
    print("Spines output shape:", spines.shape)
