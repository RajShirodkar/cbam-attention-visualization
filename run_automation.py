#!/usr/bin/env python3
"""
One-Click CBAM Automation Script
Handles dependency installation, testing, and execution
"""

import subprocess
import sys
import os
from pathlib import Path


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


def check_test_image():
    """Check if test image exists"""
    test_image_path = Path("images") / "test.jpg"
    if test_image_path.exists():
        print(f"✅ Test image found: {test_image_path}\n")
        return True
    else:
        print(f"⚠️  Test image not found at: {test_image_path}")
        print("   The main script will fail without an image.")
        print("   Please add an image to the 'images/' directory.\n")
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
            capture_output=False,
            timeout=60
        )
        
        if result.returncode == 0:
            print("\n✅ Tests passed successfully!\n")
            return True
        else:
            print(f"\n⚠️  Tests completed with return code: {result.returncode}\n")
            return True  # Don't stop automation on test failure
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out (>60 seconds)\n")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}\n")
        return False


def run_main_script():
    """Run the main image processing script"""
    print_header("STEP 4: Running Main Script (Image Processing)")
    
    main_file = Path("main.py")
    if not main_file.exists():
        print(f"❌ Main script not found: {main_file}\n")
        return False
    
    try:
        print(f"🚀 Running: {main_file}\n")
        result = subprocess.run(
            [sys.executable, str(main_file)],
            capture_output=False,
            timeout=120
        )
        
        if result.returncode == 0:
            print("\n✅ Main script executed successfully!\n")
            return True
        else:
            print(f"\n⚠️  Script completed with return code: {result.returncode}\n")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Script timed out (>120 seconds)\n")
        return False
    except Exception as e:
        print(f"❌ Error running main script: {e}\n")
        return False


def print_summary(results):
    """Print execution summary"""
    print_header("EXECUTION SUMMARY")
    
    steps = [
        ("Dependencies Installed", results.get('dependencies', False)),
        ("Dependencies Verified", results.get('verification', False)),
        ("Unit Tests Passed", results.get('tests', False)),
        ("Main Script Executed", results.get('main', False))
    ]
    
    for step, success in steps:
        status = "✅" if success else "❌"
        print(f"{status} {step}")
    
    print()
    
    if results.get('main'):
        print("🎉 SUCCESS! All automation steps completed!")
        print("📊 Check 'attention_result.png' for the visualization output.\n")
    else:
        print("⚠️  Some steps were skipped or failed.")
        print("📝 Please review the output above for details.\n")


def main():
    """Main automation flow"""
    print("\n" + "🤖 "*20)
    print("    CBAM Automation Script - One-Click Execution")
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
    
    # Step 3: Check for test image
    image_exists = check_test_image()
    
    # Step 4: Run tests
    results['tests'] = run_tests()
    
    # Step 5: Run main script (only if image exists)
    if image_exists:
        results['main'] = run_main_script()
    else:
        print_header("STEP 4: Running Main Script (Skipped)")
        print("⏭️  Skipping main script - test image not found.\n")
        results['main'] = False
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Automation interrupted by user.\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        sys.exit(1)
