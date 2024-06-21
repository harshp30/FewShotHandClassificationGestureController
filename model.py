'''
Define model architecture compatible for a low-shot approach
'''

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        Initializes the Transformer block.
        
        Parameters:
        embed_dim (int): Dimensionality of the embedding space.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network.
        dropout (float): Dropout rate.
        """
        super(TransformerBlock, self).__init__()
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the Transformer block.
        
        Parameters:
        x (Tensor): Input tensor of shape (seq_len, batch_size, embed_dim).
        
        Returns:
        Tensor: Output tensor of the same shape as input.
        """
        # Apply multi-head attention
        attn_output, _ = self.attention(x, x, x)
        # Apply residual connection and layer normalization
        x = self.norm1(x + self.dropout(attn_output))
        # Apply feed-forward network
        ff_output = self.ff(x)
        # Apply residual connection and layer normalization
        x = self.norm2(x + self.dropout(ff_output))
        return x

class HybridResNetTransformer(nn.Module):
    def __init__(self, num_classes=1, num_transformer_layers=1, embed_dim=256, num_heads=4, ff_dim=512):
        """
        Initializes the Hybrid ResNet-Transformer model.
        
        Parameters:
        num_classes (int): Number of output classes.
        num_transformer_layers (int): Number of Transformer layers.
        embed_dim (int): Dimensionality of the embedding space.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network.
        """
        super(HybridResNetTransformer, self).__init__()
        # Load a pre-trained ResNet18 model and remove the last two layers
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        # 1x1 convolution to adjust the channel dimension to embed_dim
        self.conv = nn.Conv2d(512, embed_dim, kernel_size=1)
        # Stack multiple Transformer blocks
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_transformer_layers)]
        )
        # Adaptive average pooling to reduce the spatial dimensions to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer for classification
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass for the Hybrid ResNet-Transformer model.
        
        Parameters:
        x (Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
        Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Pass input through the ResNet18 CNN
        x = self.cnn(x)
        # Apply 1x1 convolution to adjust the channel dimension
        x = self.conv(x)
        b, c, h, w = x.size()  # Batch size, channel, height, width
        # Reshape and permute the tensor for the Transformer
        x = x.view(b, c, h * w).permute(2, 0, 1)  # Shape: (seq_len, batch_size, embed_dim)
        # Pass through the Transformer
        x = self.transformer(x)
        # Reshape and permute back to the original format
        x = x.permute(1, 2, 0).view(b, c, h, w)  # Shape: (batch_size, embed_dim, height, width)
        # Apply adaptive average pooling and flatten the tensor
        x = self.pool(x).view(b, -1)  # Shape: (batch_size, embed_dim)
        # Pass through the fully connected layer for classification
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Create an instance of the HybridResNetTransformer model and print its architecture
    model = HybridResNetTransformer()
    print(model)
