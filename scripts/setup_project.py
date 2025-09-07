"""
Project setup script for quantum fraud detection.
Run this after cloning the repository.
"""

import os
import sys
import subprocess
from pathlib import Path


def create_directories():
    """Create necessary project directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/synthetic",
        "results",
        "logs",
        "models",
        "checkpoints"
    ]
    
    project_root = Path(__file__).parent.parent
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def install_requirements():
    """Install Python requirements."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Installed requirements.txt")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")


def setup_git_hooks():
    """Setup pre-commit hooks."""
    try:
        subprocess.check_call(["pre-commit", "install"])
        print("✓ Installed pre-commit hooks")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ Pre-commit not available, skipping hooks setup")


def test_imports():
    """Test essential imports."""
    print("Testing essential imports...")
    
    try:
        import qiskit
        print(f"✓ Qiskit {qiskit.__version__}")
    except ImportError:
        print("✗ Qiskit not available")
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError:
        print("✗ Scikit-learn not available")
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError:
        print("✗ Pandas not available")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not available")


def main():
    """Main setup function."""
    print("🚀 Setting up Quantum Fraud Detection project...")
    print()
    
    create_directories()
    print()
    
    install_requirements()
    print()
    
    setup_git_hooks()
    print()
    
    test_imports()
    print()
    
    print("✅ Project setup complete!")
    print()
    print("Next steps:")
    print("1. Activate your virtual environment (if not already active)")
    print("2. Run: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("3. Start implementing Phase 1 components")
    print()
    print("For development:")
    print("- Install dev dependencies: pip install -r requirements-dev.txt")
    print("- Run tests: pytest")
    print("- Format code: black src/ tests/")


if __name__ == "__main__":
    main()
