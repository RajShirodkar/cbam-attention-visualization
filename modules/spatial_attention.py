import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Paper specifies a 7x7 convolution [cite: 1762]
        padding = kernel_size // 2
        
        # Simple 2-layer spatial attention with GELU activation [cite: 1762]
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size, padding=padding, bias=True),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size, padding=padding, bias=True)
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