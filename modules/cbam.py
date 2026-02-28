import torch
import torch.nn as nn
from .channel_attention import ChannelAttention
from .spatial_attention import SpatialAttention

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7, ca_first=True):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        self.ca_first = ca_first

    def forward(self, x):
        # Original input x [cite: 1719]
        identity = x 
        
        if self.ca_first:
            # Step 1: F' = Mc(F) * F [cite: 1753]
            out = x * self.ca(x)
            
            # Step 2: F'' = Ms(F') * F' [cite: 1753]
            out = out * self.sa(out)
        else:
            # Alternative: Apply spatial attention first
            out = x * self.sa(x)
            out = out * self.ca(out)
        
        # Step 3: K(x) = x + F'' (Skip connection) [cite: 1753]
        return out + identity