from setuptools import setup, find_packages

setup(
    name="quantum-fraud-detection",
    version="0.1.0",
    description="Hybrid Quantum-Classical Machine Learning for Financial Fraud Detection",
    author="Your Name",
    author_email="your.email@domain.com",
    packages=find_packages(),
    install_requires=[
        "qiskit>=0.45.0",
        "pennylane>=0.33.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
        "xgboost>=1.7.0",
        "imbalanced-learn>=0.11.0",
        "plotly>=5.15.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
