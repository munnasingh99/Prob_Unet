import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    """PyTorch equivalent of the TensorFlow convlayer function"""
    def __init__(self, in_channels, out_channels, activation="swish", use_batchnorm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.activation = F.silu if activation == "swish" else getattr(F, activation)

    def forward(self, x, residual=None):
        out = self.conv(x)
        out = self.bn(out)
        if residual is not None:
            out = out + residual
        return self.activation(out)

class Identity(nn.Module):
    """PyTorch equivalent of the TensorFlow identity function"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.conv(x)

class Decoder(nn.Module):
    """PyTorch equivalent of the TensorFlow decoder function"""
    def __init__(self, in_channels, filters, layers, activation="swish"):
        super().__init__()
        self.layers = layers
        self.filters = filters
        self.activation = activation
        
        # Create decoder blocks - corrected channel dimensions
        self.conv_blocks = nn.ModuleList()
        for i in range(layers):
            # Calculate input channels (upsampled + skip connection)
            if i == 0:
                in_ch = in_channels + filters * 2**(layers-1)
            else:
                in_ch = filters * 2**(layers-i) + filters * 2**(layers-1-i)
            
            out_ch = filters * 2**(layers-1-i)
            
            layer_block = nn.ModuleList([
                ConvLayer(in_ch, out_ch, activation),
                ConvLayer(out_ch, out_ch, activation)
            ])
            self.conv_blocks.append(layer_block)
        
        # Final 1x1 conv
        self.final_conv = nn.Conv2d(filters, 1, kernel_size=1, padding=0)
        
    def forward(self, x, skip_connections):
        for i in range(self.layers):
            # Upsample
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            
            # Concatenate skip connection
            x = torch.cat([x, skip_connections[-(i+1)]], dim=1)
            
            # Apply conv blocks
            x = self.conv_blocks[i][0](x)
            x = self.conv_blocks[i][1](x)
            
        # Final 1x1 conv with sigmoid
        x = torch.sigmoid(self.final_conv(x))
        return x

class DeepD3(nn.Module):
    """PyTorch implementation of DeepD3_Model"""
    def __init__(self, filters=32, input_shape=(128, 128, 1), layers=4, activation="swish"):
        super().__init__()
        self.filters = filters
        self.layers = layers
        self.activation = activation
        
        # Calculate input channels from input_shape
        in_channels = input_shape[2]
        
        # Create encoder layers
        self.encoder_identities = nn.ModuleList([
            Identity(in_channels if i == 0 else filters*2**(i-1), filters*2**i)
            for i in range(layers)
        ])
        
        self.encoder_conv1 = nn.ModuleList([
            ConvLayer(in_channels if i == 0 else filters*2**(i-1), filters*2**i, activation)
            for i in range(layers)
        ])
        
        self.encoder_conv2 = nn.ModuleList([
            ConvLayer(filters*2**i, filters*2**i, activation)
            for i in range(layers)
        ])
        
        self.pool = nn.MaxPool2d(2)
        
        # Latent conv
        self.latent_conv = ConvLayer(filters*2**(layers-1), filters*2**layers, activation)
        
        # Create decoders with correct input channels
        latent_channels = filters*2**layers
        self.dendrites_decoder = Decoder(latent_channels, filters, layers, activation)
        self.spines_decoder = Decoder(latent_channels, filters, layers, activation)

    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Encoder
        for i in range(self.layers):
            identity = self.encoder_identities[i](x)
            x = self.encoder_conv1[i](x)
            x = self.encoder_conv2[i](x, identity)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Latent space
        x = self.latent_conv(x)
        
        # Decoders
        dendrites = self.dendrites_decoder(x, skip_connections)
        spines = self.spines_decoder(x, skip_connections)
        
        return dendrites, spines

def test_deepd3():
    # Create model
    model = DeepD3(filters=8, input_shape=(48, 48, 1))
    
    # Create dummy input
    x = torch.randn(4, 1, 48, 48)
    
    # Forward pass
    dendrites, spines = model(x)
    
    # Print shapes for debugging
    print(f"Input shape: {x.shape}")
    print(f"Dendrites output shape: {dendrites.shape}")
    print(f"Spines output shape: {spines.shape}")
    
    # Check output shapes
    assert dendrites.shape == (4, 1, 48, 48)
    assert spines.shape == (4, 1, 48, 48)
    print("All tests passed!")

if __name__ == "__main__":
    test_deepd3()