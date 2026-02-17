import torch
import sys
import os

# Adjust path to access modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.cbam import CBAM

def test_cbam_standalone():
    # Simulation: Batch size 1, 30 Channels, 256x256 resolution [cite: 1725, 1798]
    dummy_input = torch.randn(1, 30, 256, 256)
    
    # Initialize the CBAM module
    model = CBAM(in_planes=30)
    
    # Process the tensor
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Test Input Shape: {dummy_input.shape}")
    print(f"Test Output Shape: {output.shape}")
    
    if dummy_input.shape == output.shape:
        print("Success: CBAM module successfully processed input while maintaining resolution.")

if __name__ == "__main__":
    test_cbam_standalone()