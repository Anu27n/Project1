"""Configuration settings for the quantum fraud detection project."""

import os
from typing import Dict, Any

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Quantum circuit configuration
QUANTUM_CONFIG = {
    "n_qubits": 8,
    "n_layers": 4,
    "entanglement": "linear",
    "optimizer": "COBYLA",
    "max_iter": 100,
    "shots": 1024,
}

# Classical model configuration
CLASSICAL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
    },
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "cv_folds": 5,
    "test_size": 0.2,
    "random_state": 42,
    "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
}

# Data preprocessing configuration
PREPROCESSING_CONFIG = {
    "scaler": "StandardScaler",
    "feature_selection": "SelectKBest",
    "k_features": 10,
    "handle_imbalance": "SMOTE",
}
