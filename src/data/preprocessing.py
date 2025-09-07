"""Data preprocessing utilities for fraud detection."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def load_fraud_dataset(
    dataset_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess fraud detection dataset.
    
    REAL-WORLD DATASET SOURCES (for academic validation):
    1. Credit Card Fraud Detection Dataset
       - Source: Kaggle/ULB Machine Learning Group
       - DOI: 10.1016/j.dib.2018.06.028
       - Reference: Lebichot, B., et al. (2018). "Credit Card Fraud Detection"
       - URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
       - Features: 284,807 transactions, 30 features (V1-V28 PCA transformed)
       
    2. IEEE-CIS Fraud Detection Dataset
       - Source: IEEE Computational Intelligence Society
       - Competition: IEEE-CIS Fraud Detection (2019)
       - URL: https://www.kaggle.com/c/ieee-fraud-detection
       - Features: 590,540 transactions, 434 features
       
    3. PaySim Financial Dataset
       - Source: NTNU (Norwegian University of Science and Technology)
       - Reference: Lopez-Rojas, E.A., et al. (2016)
       - URL: https://www.kaggle.com/datasets/ntnu-testimon/paysim1
    
    Args:
        dataset_path: Path to the dataset file
        test_size: Proportion of test data
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if dataset_path is None:
        # NOTE: Using synthetic data for development/testing purposes
        # In production, replace with actual dataset download:
        # Example: dataset_path = "data/raw/creditcard.csv"
        print("📝 Using synthetic data for development. For production:")
        print("   Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        X, y = generate_synthetic_fraud_data()
    else:
        # Load real dataset
        print(f"📊 Loading real fraud dataset from: {dataset_path}")
        data = pd.read_csv(dataset_path)
        X = data.drop('Class', axis=1).values  # Assuming 'Class' is the target
        y = data['Class'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def generate_synthetic_fraud_data(
    n_samples: int = 10000,
    n_features: int = 20,
    fraud_rate: float = 0.05,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic fraud detection dataset for development and testing.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        fraud_rate: Proportion of fraudulent transactions
        random_state: Random seed
        
    Returns:
        X, y: Features and labels
    """
    np.random.seed(random_state)
    
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud
    
    # Generate normal transactions (centered around 0)
    X_normal = np.random.normal(0, 1, (n_normal, n_features))
    y_normal = np.zeros(n_normal)
    
    # Generate fraudulent transactions (shifted distribution)
    X_fraud = np.random.normal(2, 1.5, (n_fraud, n_features))
    # Add some random noise to make detection challenging
    X_fraud += np.random.normal(0, 0.5, (n_fraud, n_features))
    y_fraud = np.ones(n_fraud)
    
    # Combine datasets
    X = np.vstack([X_normal, X_fraud])
    y = np.hstack([y_normal, y_fraud])
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    return X, y


def preprocess_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler_type: str = "StandardScaler",
    feature_selection: bool = True,
    k_features: int = 10
) -> Tuple[np.ndarray, np.ndarray, object, object]:
    """
    Preprocess features: scaling and feature selection.
    
    Args:
        X_train: Training features
        X_test: Test features
        scaler_type: Type of scaler ('StandardScaler' or 'MinMaxScaler')
        feature_selection: Whether to perform feature selection
        k_features: Number of features to select
        
    Returns:
        X_train_processed, X_test_processed, scaler, selector
    """
    # Feature scaling
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection
    selector = None
    if feature_selection and X_train_scaled.shape[1] > k_features:
        selector = SelectKBest(score_func=f_classif, k=k_features)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        return X_train_selected, X_test_selected, scaler, selector
    
    return X_train_scaled, X_test_scaled, scaler, selector


def handle_class_imbalance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "SMOTE",
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle class imbalance in the training data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        method: Method for handling imbalance ('SMOTE', 'none')
        random_state: Random seed
        
    Returns:
        X_train_balanced, y_train_balanced
    """
    if method == "SMOTE":
        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        return X_train_balanced, y_train_balanced
    elif method == "none":
        return X_train, y_train
    else:
        raise ValueError(f"Unknown imbalance handling method: {method}")


def prepare_quantum_features(
    X: np.ndarray,
    encoding_method: str = "angle_encoding",
    n_qubits: int = 8
) -> np.ndarray:
    """
    Prepare features for quantum encoding.
    
    Args:
        X: Input features
        encoding_method: Quantum encoding method
        n_qubits: Number of qubits available
        
    Returns:
        X_quantum: Features prepared for quantum encoding
    """
    if encoding_method == "angle_encoding":
        # Ensure features are in [0, 2π] range for angle encoding
        X_normalized = np.arctan(X) + np.pi/2  # Maps to [0, π]
        X_quantum = X_normalized * 2  # Maps to [0, 2π]
    elif encoding_method == "amplitude_encoding":
        # Normalize features for amplitude encoding
        X_quantum = X / np.linalg.norm(X, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown encoding method: {encoding_method}")
    
    # Ensure we don't exceed the number of qubits
    if X_quantum.shape[1] > n_qubits:
        X_quantum = X_quantum[:, :n_qubits]
    
    return X_quantum


if __name__ == "__main__":
    # Test the preprocessing functions
    print("Testing fraud data preprocessing...")
    
    # Generate synthetic data
    X, y = generate_synthetic_fraud_data(n_samples=1000)
    print(f"Generated data: {X.shape}, Fraud rate: {y.mean():.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = load_fraud_dataset()
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Preprocess features
    X_train_proc, X_test_proc, scaler, selector = preprocess_features(
        X_train, X_test, feature_selection=True, k_features=8
    )
    print(f"After preprocessing: {X_train_proc.shape}")
    
    # Handle imbalance
    X_train_bal, y_train_bal = handle_class_imbalance(X_train_proc, y_train)
    print(f"After balancing: {X_train_bal.shape}, Fraud rate: {y_train_bal.mean():.3f}")
    
    # Prepare for quantum
    X_quantum = prepare_quantum_features(X_train_bal[:100], n_qubits=8)
    print(f"Quantum features: {X_quantum.shape}")
    
    print("Preprocessing test completed successfully!")
