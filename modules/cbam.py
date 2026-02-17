import torch
import torch.nn as nn
from .channel_attention import ChannelAttention
from .spatial_attention import SpatialAttention

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7, dropout_rate=0.1, 
                 use_gelu=True, use_layer_norm=False, ca_first=True):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio, dropout_rate, use_gelu)
        self.sa = SpatialAttention(kernel_size, dropout_rate, use_gelu)
        self.use_layer_norm = use_layer_norm
        self.ca_first = ca_first
        
        if use_layer_norm:
            self.ln1 = nn.GroupNorm(32, in_planes) if in_planes >= 32 else nn.GroupNorm(1, in_planes)
            self.ln2 = nn.GroupNorm(32, in_planes) if in_planes >= 32 else nn.GroupNorm(1, in_planes)

    def forward(self, x):
        # Original input x [cite: 1719]
        identity = x 
        
        if self.ca_first:
            # Step 1: F' = Mc(F) * F [cite: 1753]
            out = x * self.ca(x)
            if self.use_layer_norm:
                out = self.ln1(out)
            
            # Step 2: F'' = Ms(F') * F' [cite: 1753]
            out = out * self.sa(out)
            if self.use_layer_norm:
                out = self.ln2(out)
        else:
            # Alternative: Apply spatial attention first
            out = x * self.sa(x)
            if self.use_layer_norm:
                out = self.ln1(out)
            
            out = out * self.ca(out)
            if self.use_layer_norm:
                out = self.ln2(out)
        
        # Step 3: K(x) = x + F'' (Skip connection) [cite: 1753]
        return out + identity