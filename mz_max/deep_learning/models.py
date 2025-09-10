"""
Pre-built neural network architectures for MZ Max

This module provides ready-to-use neural network architectures for
various deep learning tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import math


class CNN(nn.Module):
    """
    Convolutional Neural Network for image classification.
    """
    
    def __init__(self, input_channels: int = 3, num_classes: int = 10,
                 conv_layers: List[int] = [32, 64, 128],
                 kernel_size: int = 3, dropout: float = 0.5):
        """
        Initialize CNN.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            conv_layers: List of channel sizes for conv layers
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super(CNN, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in conv_layers:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2)
                )
            )
            in_channels = out_channels
        
        # Calculate the size after convolutions (assuming square input)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(conv_layers[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for ResNet."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet architecture for image classification.
    """
    
    def __init__(self, num_classes: int = 10, num_blocks: List[int] = [2, 2, 2, 2]):
        """
        Initialize ResNet.
        
        Args:
            num_classes: Number of output classes
            num_blocks: Number of blocks in each layer
        """
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        """Create a layer with multiple residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 num_layers: int = 6, d_ff: int = 2048, max_len: int = 5000,
                 dropout: float = 0.1, num_classes: Optional[int] = None):
        """
        Initialize Transformer.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
            num_classes: Number of classes (for classification tasks)
        """
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        if num_classes is not None:
            self.classifier = nn.Linear(d_model, num_classes)
        else:
            self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        # Output projection
        if hasattr(self, 'classifier'):
            # For classification: use [CLS] token or mean pooling
            x = x.mean(dim=1)  # Mean pooling
            x = self.classifier(x)
        else:
            x = self.output_projection(x)
        
        return x


class LSTM(nn.Module):
    """
    LSTM model for sequence modeling.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 num_classes: Optional[int] = None, dropout: float = 0.0,
                 bidirectional: bool = False):
        """
        Initialize LSTM.
        
        Args:
            input_size: Input feature size
            hidden_size: Hidden state size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        if num_classes is not None:
            self.classifier = nn.Linear(lstm_output_size, num_classes)
        else:
            self.output_layer = nn.Linear(lstm_output_size, input_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                        batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                        batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Output projection
        if hasattr(self, 'classifier'):
            # For classification: use last output
            out = self.classifier(out[:, -1, :])
        else:
            # For sequence generation: use all outputs
            out = self.output_layer(out)
        
        return out


class Generator(nn.Module):
    """Generator network for GAN."""
    
    def __init__(self, latent_dim: int, img_shape: Tuple[int, int, int]):
        super(Generator, self).__init__()
        
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    """Discriminator network for GAN."""
    
    def __init__(self, img_shape: Tuple[int, int, int]):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class GAN(nn.Module):
    """
    Generative Adversarial Network.
    """
    
    def __init__(self, latent_dim: int = 100, img_shape: Tuple[int, int, int] = (1, 28, 28)):
        """
        Initialize GAN.
        
        Args:
            latent_dim: Latent space dimension
            img_shape: Image shape (channels, height, width)
        """
        super(GAN, self).__init__()
        
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, img_shape)
        self.discriminator = Discriminator(img_shape)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator."""
        return self.generator(z)


class Encoder(nn.Module):
    """Encoder for VAE."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var


class Decoder(nn.Module):
    """Decoder for VAE."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))


class VAE(nn.Module):
    """
    Variational Autoencoder.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 400, latent_dim: int = 20):
        """
        Initialize VAE.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
        """
        super(VAE, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, log_var = self.encoder(x.view(x.size(0), -1))
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var
