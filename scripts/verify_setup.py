#!/usr/bin/env python3
"""
Verification script to check if the HPC environment is set up correctly.
Run this after setup_hpc.sh to verify all dependencies are installed.
"""

import sys
import subprocess

def check_module(module_name, import_name=None):
    """Check if a Python module can be imported."""
    if import_name is None:
        import_name = module_name

    try:
        __import__(import_name)
        print(f"✓ {module_name} is installed")
        return True
    except ImportError:
        print(f"✗ {module_name} is NOT installed")
        return False

def check_cuda():
    """Check if CUDA is available via PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available (version: {torch.version.cuda})")
            print(f"  - {torch.cuda.device_count()} GPU(s) detected")
            for i in range(torch.cuda.device_count()):
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("✗ CUDA is NOT available")
            print("  Note: This is expected on login nodes. CUDA will be available on compute nodes.")
            return True  # Not a fatal error
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def check_command(command):
    """Check if a command is available."""
    try:
        result = subprocess.run([command, "--version"],
                              capture_output=True,
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            print(f"✓ {command} is available")
            return True
        else:
            print(f"✗ {command} is NOT available")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"✗ {command} is NOT available")
        return False

def main():
    print("=" * 50)
    print("Environment Verification Script")
    print("=" * 50)
    print()

    all_checks_passed = True

    # Check Python version
    print("Python Version:")
    print(f"  {sys.version}")
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 8:
        print(f"✓ Python version is acceptable (3.{minor})")
    else:
        print(f"✗ Python version should be 3.8 or higher")
        all_checks_passed = False
    print()

    # Check critical packages
    print("Critical Packages:")
    critical_packages = [
        "torch",
        "transformers",
        "trl",
        "datasets",
        "accelerate",
        "peft",
    ]

    for package in critical_packages:
        if not check_module(package):
            all_checks_passed = False
    print()

    # Check optional but recommended packages
    print("Optional Packages:")
    optional_packages = [
        ("wandb", "wandb"),
        ("bitsandbytes", "bitsandbytes"),
        ("deepspeed", "deepspeed"),
    ]

    for package_name, import_name in optional_packages:
        check_module(package_name, import_name)
    print()

    # Check CUDA
    print("CUDA Check:")
    if not check_cuda():
        print("  Note: CUDA availability will be checked again on compute nodes")
    print()

    # Check torchrun command
    print("Command Availability:")
    if not check_command("torchrun"):
        print("  ERROR: torchrun not found! PyTorch may not be installed correctly.")
        all_checks_passed = False
    print()

    # Print summary
    print("=" * 50)
    if all_checks_passed:
        print("✓ All critical checks passed!")
        print("  Your environment is ready for training.")
        return 0
    else:
        print("✗ Some checks failed!")
        print("  Please review the errors above and fix them before training.")
        print("  You may need to re-run setup_hpc.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())
