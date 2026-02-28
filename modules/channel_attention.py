import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Paper utilizes mean pooling and max pooling [cite: 1755]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        hidden_planes = max(in_planes // ratio, 8)
        
        # Simple 2-layer MLP with GELU activation [cite: 1756]
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_planes, in_planes, 1, bias=True)
        )
        # Sigmoid activation function to generate weight map [cite: 1757]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Calculating weights as per Formula (2) [cite: 1758]
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)