# Quantum Machine Learning for Financial Fraud Detection

## Project Overview

This repository contains a comprehensive implementation of quantum machine learning techniques applied to the detection of fraudulent transactions in financial data. The project explores the potential advantages of quantum computing approaches over classical methods in identifying subtle patterns and anomalies associated with fraudulent activities.

## Motivation

Financial fraud detection presents significant challenges due to class imbalance, the subtle nature of fraud indicators, and the massive volume of transaction data. While classical machine learning techniques have been effective, quantum computing offers promising theoretical advantages:

1. **Quantum Superposition**: The ability to process multiple states simultaneously
2. **Quantum Entanglement**: Enhanced feature correlation analysis
3. **Quantum Interference**: Potential for improved pattern recognition in high-dimensional data

This project investigates whether these quantum advantages can be leveraged to improve fraud detection accuracy and efficiency.

## Dataset Information

The project utilizes the **Bank Account Fraud Dataset** from Kaggle (NeurIPS 2022) with the following characteristics:

**Dataset Source**: [Bank Account Fraud Dataset - NeurIPS 2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)

**Dataset Characteristics:**
- **Dataset Size**: 1,000,000 transaction records
- **Features**: 32 attributes including financial, behavioral, demographic, and digital features
- **Target Variable**: Binary fraud indicator (`fraud_bool`)
- **Class Distribution**: Highly imbalanced (~1.1% fraud cases)
- **File Size**: ~200MB (CSV format)
- **Original Purpose**: Created for NeurIPS 2022 research on fraud detection

**Key Feature Categories:**
- **Financial**: Income, credit risk score, proposed credit limit, intended balance amount
- **Behavioral**: Transaction velocities (6h, 24h, 4w), session length, device usage patterns
- **Demographic**: Customer age, employment status, housing status, address history
- **Digital**: Device OS, email characteristics, phone validation, session behavior
- **Temporal**: Days since request, monthly patterns, historical activity metrics

*This dataset represents a realistic simulation of bank account opening fraud scenarios and is specifically designed for machine learning research in financial fraud detection.*

## Repository Structure

```
Project1/
├── dataset/                  # Financial transaction dataset
│   ├── Base.csv              # Main dataset file (1M records)
│   └── Base.csv.zip          # Compressed version of dataset
├── docs/                     # Project documentation
│   └── Project-1-Zero_review_QML.pdf  # Technical review document
├── notebooks/                # Jupyter notebooks for analysis and models
│   ├── 01_data_exploration.ipynb      # Data exploration and preprocessing
│   ├── 02_qml.ipynb                   # Quantum ML implementation
│   └── 02_qml copy.ipynb              # Backup of quantum implementation
└── requirements.txt          # Project dependencies
```

## Detailed Notebook Documentation

### 1. Data Exploration (`01_data_exploration.ipynb`)

This notebook provides comprehensive exploratory data analysis of the financial transaction dataset, focusing on the following key aspects:

- **Dataset Loading and Initial Inspection**: Analysis of dataset dimensions, feature types, and basic statistics
- **Class Distribution Analysis**: Examination of the fraud/non-fraud class imbalance ratio (approximately 90:1)
- **Feature Distribution Visualization**: Multivariate analysis of feature distributions across fraud and non-fraud classes
- **Correlation Analysis**: Identification of features most strongly correlated with fraudulent transactions
- **Feature Engineering Potential**: Assessment of potential transformations and derived features
- **Outlier Detection and Handling**: Implementation of multiple outlier detection methods (statistical tests, Isolation Forest) with a careful evaluation of their impact on fraud detection
- **Data Quality Assessment**: Analysis of missing values, data types, and feature ranges to ensure data integrity
- **Preprocessing Strategy Development**: Formulation of a comprehensive preprocessing pipeline for the quantum ML model

Key findings include a significant class imbalance, approximately 8.5% genuine outliers using conservative statistical methods, and several features with moderate correlation to fraudulent activities.

### 2. Quantum Machine Learning (`02_qml.ipynb`)

This notebook implements a quantum machine learning approach to fraud detection using PennyLane, a cross-platform Python library for quantum computing. The implementation follows these critical steps:

- **Data Preprocessing**:
  - Categorical feature encoding using one-hot encoding
  - Numerical feature standardization to ensure equal scale representation
  - Dimensionality reduction through Principal Component Analysis (PCA)
  
- **Dataset Partitioning**:
  - Creation of stratified train/test splits to maintain class balance
  - Generation of reduced-size datasets for quantum simulation efficiency

- **Classical Baseline Model**:
  - Implementation of a balanced Logistic Regression model for performance benchmarking
  - Evaluation using accuracy, ROC-AUC, and precision-recall metrics

- **Quantum Model Development**:
  - Construction of a Variational Quantum Classifier (VQC) with angle embedding
  - Implementation of a parameterized quantum circuit with entangling layers
  - Integration of classical optimization methods with quantum circuit evaluation

- **Advanced Quantum Training Techniques**:
  - Implementation of sigmoid readout and binary cross-entropy loss
  - Application of class weighting to address imbalance
  - Integration of minibatch training for improved convergence
  - Exploration of balanced minibatch sampling strategies

- **Model Evaluation and Comparison**:
  - Performance assessment using accuracy, ROC-AUC, and precision-recall metrics
  - Comparative analysis of classical and quantum approaches

The notebook demonstrates the implementation of increasingly sophisticated quantum machine learning techniques for fraud detection, with careful attention to the challenges of class imbalance and model evaluation.

## Technical Implementation

The project utilizes the following core technologies:

- **Quantum Computing Framework**: PennyLane for quantum circuit design and execution
- **Classical ML Libraries**: scikit-learn for classical models and preprocessing
- **Data Processing**: Pandas and NumPy for data manipulation
- **Visualization**: Matplotlib and Seaborn for data visualization

Quantum circuits are designed to process reduced-dimensionality financial transaction data, with careful attention to qubit utilization efficiency. The implementation follows best practices for variational quantum algorithms, including:

1. Efficient feature encoding using angle embedding
2. Parameterized quantum circuit design with entangling layers
3. Hybrid classical-quantum optimization approach
4. Careful management of the classical-quantum interface

## Execution Requirements

To run this project, the following dependencies are required:

```
# Core dependencies
pandas
numpy
matplotlib
seaborn
plotly
pennylane
pennylane-qiskit
scikit-learn
torch
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Running the Project

1. Clone the repository
2. Install dependencies from `requirements.txt`
3. Execute notebooks in sequence:
   - First, run `01_data_exploration.ipynb` to understand the dataset
   - Then, run `02_qml.ipynb` to implement and evaluate the quantum models

The notebooks are designed to be self-contained and include detailed documentation of each step.

## Conclusions and Future Work

This project demonstrates the application of quantum machine learning techniques to financial fraud detection, with a focus on addressing the challenges of class imbalance and feature selection. While quantum approaches show promise, several areas for future work include:

1. Exploration of more complex quantum feature maps
2. Implementation of quantum kernel methods for fraud detection
3. Scaling to larger datasets with more qubits as quantum hardware improves
4. Integration of quantum anomaly detection techniques
5. Hybrid classical-quantum ensemble methods

## References

- **Dataset**: Bank Account Fraud Dataset - NeurIPS 2022. Available at: [https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
- PennyLane Documentation: [https://pennylane.ai/](https://pennylane.ai/)
- Quantum Machine Learning: [https://arxiv.org/abs/1611.09347](https://arxiv.org/abs/1611.09347)
- Financial Fraud Detection with Machine Learning: [https://arxiv.org/abs/1911.05109](https://arxiv.org/abs/1911.05109)
