# CBAM (Convolutional Block Attention Module) - Enhanced Implementation

A powerful PyTorch implementation of CBAM with enhanced architecture for improved performance. This module combines channel attention and spatial attention mechanisms to refine feature maps in neural networks.

---

## Table of Contents
1. [Clone](#clone)
2. [Install](#install)
3. [Run](#run)
4. [Understand](#understand)
5. [Modify](#modify)

---

## Clone

Clone the repository to your local machine:

```bash
git clone <repository-url>
cd cbam
```

Or if working from an existing directory:
```bash
cd path/to/cbam
```

**Repository Structure:**
```
cbam/
├── main.py                          # Main script for image processing
├── README.md                        # This file
├── images/
│   └── test.jpg                    # Sample image for testing
├── modules/
│   ├── __init__.py                 # Module initialization
│   ├── cbam.py                     # Main CBAM module
│   ├── channel_attention.py        # Channel attention implementation
│   └── spatial_attention.py        # Spatial attention implementation
└── tests/
    └── test_cbam.py                # Unit tests
```

---

## Install

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Step 1: Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate          # On Windows
source venv/bin/activate       # On Mac/Linux
```

### Step 2: Install Dependencies
```bash
pip install torch torchvision pillow matplotlib numpy
```

Or install from requirements (if you have requirements.txt):
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

---

## Run

### Option 1: Process a Single Image
Run the main script to process an image and generate attention heatmaps:

```bash
python main.py
```

**What it does:**
- Loads `images/test.jpg` (or specify your own image path in `main.py`)
- Applies CBAM attention mechanisms
- Generates a heatmap visualization as `attention_result.png`
- Prints input/output shapes and processing status

### Option 2: Run Unit Tests
Test the CBAM module with dummy data:

```bash
python tests/test_cbam.py
```

**Expected Output:**
```
Test Input Shape: torch.Size([1, 30, 256, 256])
Test Output Shape: torch.Size([1, 30, 256, 256])
Success: CBAM module successfully processed input while maintaining resolution.
```

### Option 3: Custom Image Processing
Edit `main.py` and change the image path:

```python
image_path = 'path/to/your/image.jpg'
process_image_option_1(image_path)
```

---

## Understand

### Architecture Overview

#### **1. Channel Attention Module** (`channel_attention.py`)
- **Purpose:** Adaptively recalibrate channel-wise feature responses
- **How it works:**
  - Uses global average pooling and max pooling
  - Passes pooled features through a 3-layer MLP
  - Outputs channel attention weights (values 0-1)
- **Key Features:**
  - Deeper MLP (3 layers instead of 2)
  - GroupNorm for stable training
  - Dropout for regularization
  - GELU activation (better than ReLU)

```
Input [B, C, H, W]
  ↓
Avg Pool + Max Pool → [B, C, 1, 1]
  ↓
3-Layer MLP (C → C/ratio → C)
  ↓
Sigmoid → Channel Attention [B, C, 1, 1]
```

#### **2. Spatial Attention Module** (`spatial_attention.py`)
- **Purpose:** Adaptively recalibrate pixel-wise feature relationships
- **How it works:**
  - Concatenates mean and max pooling across channels
  - Passes through multi-layer convolutions
  - Outputs spatial attention weights
- **Key Features:**
  - Multi-layer conv (2→16→8→1)
  - BatchNorm for training stability
  - Dropout for regularization

```
Input [B, C, H, W]
  ↓
Mean + Max Pool along C → [B, 2, H, W]
  ↓
Multi-Layer Conv (2→16→8→1)
  ↓
Sigmoid → Spatial Attention [B, 1, H, W]
```

#### **3. CBAM Module** (`cbam.py`)
- **Purpose:** Combines channel and spatial attention with skip connections
- **How it works:**
  1. Apply channel attention: `F' = Mc(F) × F`
  2. Apply spatial attention: `F'' = Ms(F') × F'`
  3. Add skip connection: `Output = F'' + F`

**Configuration Options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_planes` | - | Number of input channels (required) |
| `ratio` | 16 | Compression ratio in channel attention |
| `kernel_size` | 7 | Kernel size for spatial attention convolution |
| `dropout_rate` | 0.1 | Dropout rate for regularization |
| `use_gelu` | True | Use GELU activation (vs ReLU) |
| `use_layer_norm` | False | Add GroupNorm between attention modules |
| `ca_first` | True | Apply channel attention before spatial (vs spatial first) |

### Usage Example

```python
import torch
from modules.cbam import CBAM

# Create CBAM for 3-channel input (RGB image)
model = CBAM(in_planes=3)
model.eval()

# Process image
x = torch.randn(1, 3, 256, 256)  # Batch size 1, RGB, 256x256
y = model(x)  # Output shape: (1, 3, 256, 256)

# With custom parameters
model = CBAM(
    in_planes=64,
    ratio=8,
    kernel_size=5,
    dropout_rate=0.2,
    use_gelu=True,
    use_layer_norm=True,
    ca_first=False  # Spatial attention first
)
```

### Weight Visualization
Run `python main.py` to generate attention heatmaps that show:
- Which regions of the image the model focuses on
- How spatial and channel attention modify feature responses
- Visualization saved as `attention_result.png`

---

## Modify

### Making Changes to the Modules

#### **1. Modify Channel Attention Depth**
Edit `modules/channel_attention.py` to change MLP layers:

```python
# Example: Add 4th layer for deeper feature extraction
self.fc = nn.Sequential(
    nn.Conv2d(in_planes, hidden_planes, 1, bias=False),
    nn.GroupNorm(1, hidden_planes),
    activation,
    nn.Dropout(dropout_rate),
    nn.Conv2d(hidden_planes, hidden_planes, 1, bias=False),
    nn.GroupNorm(1, hidden_planes),
    activation,
    nn.Dropout(dropout_rate),
    nn.Conv2d(hidden_planes, hidden_planes, 1, bias=False),  # NEW LAYER
    nn.GroupNorm(1, hidden_planes),
    activation,
    nn.Dropout(dropout_rate),
    nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
)
```

#### **2. Modify Spatial Attention Channels**
Edit `modules/spatial_attention.py` to change feature channels:

```python
# Current: 2→16→8→1
# Change to: 2→32→16→1 for more capacity
self.conv = nn.Sequential(
    nn.Conv2d(2, 32, kernel_size, padding=padding, bias=False),  # Changed to 32
    nn.BatchNorm2d(32),
    activation,
    nn.Dropout(dropout_rate),
    nn.Conv2d(32, 16, kernel_size, padding=padding, bias=False),  # Changed to 16
    nn.BatchNorm2d(16),
    activation,
    nn.Dropout(dropout_rate),
    nn.Conv2d(16, 1, kernel_size, padding=padding, bias=False)
)
```

#### **3. Add Custom Attention Variants**
Create a new file `modules/custom_attention.py`:

```python
import torch
import torch.nn as nn

class CustomAttention(nn.Module):
    def __init__(self, in_planes):
        super(CustomAttention, self).__init__()
        # Your custom attention implementation
        pass
    
    def forward(self, x):
        # Your custom forward pass
        return x
```

Then import and use in `cbam.py`:
```python
from .custom_attention import CustomAttention

class CBAM(nn.Module):
    def __init__(self, in_planes, ...):
        # ... existing code ...
        self.custom = CustomAttention(in_planes)
```

#### **4. Change Activation Functions**
Replace GELU with alternatives in both modules:

```python
# Use LeakyReLU instead
activation = nn.LeakyReLU(0.1, inplace=True)

# Use Mish activation
activation = nn.Mish()

# Use SiLU (Swish)
activation = nn.SiLU()
```

#### **5. Modify Main Script** (`main.py`)
Change image processing pipeline:

```python
# Process multiple images
image_paths = ['images/test1.jpg', 'images/test2.jpg']
for img_path in image_paths:
    process_image_option_1(img_path)

# Use different CBAM configurations
model = CBAM(in_planes=3, ratio=1, dropout_rate=0.2, use_layer_norm=True)

# Apply to different input sizes
img_tensor = transform(img).unsqueeze(0)  # Change resolution in transforms.Resize()
```

#### **6. Add Batch Processing**
Create a batch processing function:

```python
def process_batch(image_dir, model):
    results = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(image_dir, filename)
            # Process image
            results.append(img_path)
    return results
```

### Testing Your Changes

After modifications, verify functionality:

```bash
# Run unit tests
python tests/test_cbam.py

# Test with custom parameters
python -c "
from modules.cbam import CBAM
import torch
model = CBAM(in_planes=64, ratio=8, dropout_rate=0.2)
x = torch.randn(2, 64, 224, 224)
y = model(x)
print(f'Input: {x.shape}, Output: {y.shape}')
"
```

### Best Practices for Modifications

1. **Test incrementally** - Make one change, test it, then move to the next
2. **Keep backups** - Use Git to version control your changes
3. **Document changes** - Add comments explaining why you modified something
4. **Validate shapes** - Ensure input/output shapes match expectations
5. **Use eval() mode** - Call `model.eval()` before inference
6. **Check gradient flow** - For training, verify gradients propagate correctly

---

## Common Issues & Solutions

### Issue: "Expected more than 1 value per channel when training"
**Solution:** Call `model.eval()` before inference or use GroupNorm instead of BatchNorm

### Issue: "Module not found" when importing
**Solution:** Ensure you're in the project root directory and run: `python -m modules.cbam`

### Issue: Image not found error
**Solution:** Verify the image path exists: `images/test.jpg` or update the path in `main.py`

### Issue: Out of memory (OOM)
**Solution:** 
- Reduce batch size
- Decrease image resolution in `transforms.Resize()`
- Reduce `dropout_rate` or `ratio` parameters

---

## Additional Resources

- **CBAM Paper:** [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- **PyTorch Docs:** https://pytorch.org/docs/stable/index.html
- **Attention Mechanisms:** Learn about channel and spatial attention

---

## Support

For questions or issues:
1. Check the "Common Issues & Solutions" section
2. Review the code comments marked with `[cite: ####]`
3. Run tests to verify the implementation
4. Check the main branch for latest updates

---

**Last Updated:** February 17, 2026  
**Version:** 2.0 (Enhanced Performance)
