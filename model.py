import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=1, num_transformer_layers=1, embed_dim=256, num_heads=4, ff_dim=512):
        super(HybridCNNTransformer, self).__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        self.conv = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_transformer_layers)]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.conv(x)
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(2, 0, 1)  # Shape: (seq_len, batch, embed_dim)
        x = self.transformer(x)
        x = x.permute(1, 2, 0).view(b, c, h, w)  # Shape: (batch, embed_dim, h, w)
        x = self.pool(x).view(b, -1)  # Shape: (batch, embed_dim)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = HybridCNNTransformer()
    print(model)
