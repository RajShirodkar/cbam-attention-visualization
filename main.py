import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from modules.cbam import CBAM

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------
# Save attention heatmap visualization
# ----------------------------------------------------
def save_attention_visualization(output_tensor, filename='attention_result.png'):
    """
    Converts CBAM output into a heatmap and saves it.
    """

    # Remove batch dimension → [3, H, W]
    tensor = output_tensor.squeeze(0)

    # Mean across channels → [H, W]
    heatmap = torch.mean(tensor, dim=0).cpu().numpy()

    # Normalize safely
    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()

    if heatmap_max - heatmap_min != 0:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

    # Plot heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='viridis')
    plt.colorbar()
    plt.title("CBAM Refined Feature Map (Heatmap)")

    # Save image
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    print(f"✅ Visualization saved as: {filename}")


# ----------------------------------------------------
# Main CBAM image processing
# ----------------------------------------------------
def process_image_option_1(image_path):

    # Check image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: {image_path} not found.")
        return

    # Load image
    img = Image.open(image_path).convert('RGB')

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Initialize CBAM
    model = CBAM(in_planes=3)
    model.eval()  # Set to evaluation mode for inference

    # Run CBAM
    with torch.no_grad():
        output = model(img_tensor)

    # Save visualization
    save_attention_visualization(output)

    # Console info
    print(f"\n✅ Successfully processed: {image_path}")
    print(f"Input Shape:  {img_tensor.shape}")
    print(f"Output Shape: {output.shape}")
    print("✔ Attention weight maps applied to RGB channels.\n")


# ----------------------------------------------------
# Entry point
# ----------------------------------------------------
if __name__ == "__main__":

    # Change path if needed
    image_path = 'images/test.jpg'

    process_image_option_1(image_path)
