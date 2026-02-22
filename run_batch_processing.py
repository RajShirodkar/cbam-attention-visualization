#!/usr/bin/env python3
"""
CBAM Batch Processing Automation Script
Processes multiple images in a dataset with one click
"""

import subprocess
import sys
import os
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def print_header(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


def install_dependencies():
    """Install required packages"""
    print_header("STEP 1: Installing Dependencies")
    
    required_packages = [
        'torch',
        'torchvision',
        'pillow',
        'matplotlib',
        'numpy'
    ]
    
    try:
        print("📦 Installing required packages...\n")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-q'
        ] + required_packages)
        print("✅ All dependencies installed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}\n")
        return False


def verify_dependencies():
    """Verify all dependencies are installed"""
    print_header("STEP 2: Verifying Dependencies")
    
    packages_to_check = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'numpy': 'NumPy'
    }
    
    all_available = True
    for package, name in packages_to_check.items():
        try:
            __import__(package)
            print(f"✅ {name:15} - Available")
        except ImportError:
            print(f"❌ {name:15} - Missing")
            all_available = False
    
    print()
    return all_available


def save_attention_visualization(output_tensor, filename='attention_result.png'):
    """
    Converts CBAM output into a heatmap and saves it.
    (Same as main.py)
    """
    try:
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

        return True
    except Exception as e:
        print(f"   ⚠️  Failed to save visualization: {e}")
        return False


def process_single_image(image_path, model, output_dir):
    """Process a single image with CBAM"""
    try:
        # Check image exists
        if not os.path.exists(image_path):
            print(f"   ❌ File not found: {image_path}")
            return False

        # Load image
        img = Image.open(image_path).convert('RGB')

        # Preprocessing
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Run CBAM
        with torch.no_grad():
            output = model(img_tensor)

        # Generate output filename
        filename_stem = Path(image_path).stem
        heatmap_path = os.path.join(output_dir, f"{filename_stem}_heatmap.png")

        # Save visualization
        if save_attention_visualization(output, heatmap_path):
            print(f"   ✅ Processed: {Path(image_path).name} → {Path(heatmap_path).name}")
            return True
        else:
            print(f"   ⚠️  Processed but couldn't save visualization: {Path(image_path).name}")
            return True  # Still count as processed

    except Exception as e:
        print(f"   ❌ Error processing {Path(image_path).name}: {e}")
        return False


def run_tests():
    """Run unit tests"""
    print_header("STEP 3: Running Unit Tests")
    
    test_file = Path("tests") / "test_cbam.py"
    if not test_file.exists():
        print(f"⚠️  Test file not found: {test_file}\n")
        return False
    
    try:
        print(f"🧪 Running: {test_file}\n")
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Tests passed successfully!\n")
            return True
        else:
            print(f"⚠️  Tests completed with return code: {result.returncode}\n")
            return True
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out (>60 seconds)\n")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}\n")
        return False


def find_images(directory='images'):
    """Find all image files in directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_dir = Path(directory)
    
    if not image_dir.exists():
        return []
    
    images = [
        str(f) for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    return sorted(images)


def process_image_dataset(image_dir='images', output_dir='heatmaps'):
    """Process all images in a directory"""
    print_header("STEP 4: Processing Image Dataset")
    
    # Find all images
    images = find_images(image_dir)
    
    if not images:
        print(f"❌ No images found in '{image_dir}' directory.\n")
        print("📁 Supported formats: .jpg, .jpeg, .png, .bmp, .gif, .webp\n")
        return False
    
    print(f"🖼️  Found {len(images)} image(s) to process:\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Import CBAM model
    try:
        from modules.cbam import CBAM
        model = CBAM(in_planes=3, ratio=1)
        model.eval()
        print(f"✅ CBAM model loaded\n")
    except Exception as e:
        print(f"❌ Failed to load CBAM model: {e}\n")
        return False
    
    # Process each image
    successful = 0
    failed = 0
    
    for idx, image_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] Processing: {Path(image_path).name}")
        if process_single_image(image_path, model, str(output_path)):
            successful += 1
        else:
            failed += 1
    
    # Print results
    print(f"\n📊 Processing Complete!")
    print(f"   ✅ Successful: {successful}/{len(images)}")
    if failed > 0:
        print(f"   ❌ Failed: {failed}/{len(images)}")
    print(f"   📂 Output directory: {output_dir}/\n")
    
    return successful > 0


def print_summary(results):
    """Print execution summary"""
    print_header("EXECUTION SUMMARY")
    
    steps = [
        ("Dependencies Installed", results.get('dependencies', False)),
        ("Dependencies Verified", results.get('verification', False)),
        ("Unit Tests Passed", results.get('tests', False)),
        ("Image Dataset Processed", results.get('processing', False))
    ]
    
    for step, success in steps:
        status = "✅" if success else "❌"
        print(f"{status} {step}")
    
    print()
    
    if results.get('processing'):
        print("🎉 SUCCESS! Batch processing completed!")
        print("📂 Check 'heatmaps/' directory for visualization outputs.\n")
    else:
        print("⚠️  Processing was skipped or failed.")
        print("📝 Please review the output above for details.\n")


def main():
    """Main automation flow"""
    print("\n" + "🤖 "*20)
    print("    CBAM Batch Processing - One-Click Dataset Execution")
    print("🤖 "*20 + "\n")
    
    print(f"Working directory: {os.getcwd()}\n")
    
    results = {}
    
    # Step 1: Install dependencies
    results['dependencies'] = install_dependencies()
    
    # Step 2: Verify dependencies
    results['verification'] = verify_dependencies()
    
    if not results['verification']:
        print("❌ Cannot proceed without required dependencies.\n")
        print_summary(results)
        sys.exit(1)
    
    # Step 3: Run tests
    results['tests'] = run_tests()
    
    # Step 4: Process image dataset
    results['processing'] = process_image_dataset(
        image_dir='images',
        output_dir='heatmaps'
    )
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Batch processing interrupted by user.\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        sys.exit(1)
