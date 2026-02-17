import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, dropout_rate=0.1, use_gelu=True):
        super(SpatialAttention, self).__init__()
        # Paper specifies a 7x7 convolution [cite: 1762]
        padding = kernel_size // 2
        activation = nn.GELU() if use_gelu else nn.ReLU(inplace=True)
        
        # Enhanced spatial attention with multi-layer convolution [cite: 1762]
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(16),
            activation,
            nn.Dropout(dropout_rate),
            nn.Conv2d(16, 8, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(8),
            activation,
            nn.Dropout(dropout_rate),
            nn.Conv2d(8, 1, kernel_size, padding=padding, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Concatenate mean and max pooling along channel dimension [cite: 1761, 1764]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        
        # Pass through multi-layer conv and Sigmoid [cite: 1762]
        x = self.conv(x)
        return self.sigmoid(x)