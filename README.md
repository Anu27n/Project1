# Quantum Machine Learning for Financial Fraud Detection

**A hybrid quantum-classical approach to enhance fraud detection accuracy and reduce false positives in financial transactions.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-Latest-purple)](https://qiskit.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Project Overview

This project implements a hybrid quantum-classical machine learning framework for financial fraud detection, aiming to achieve >95% accuracy while reducing false positive rates to <1.5%.

### 🎯 Key Objectives
- **Accuracy Target:** >95% fraud detection accuracy (vs. 85-92% classical baseline)
- **False Positive Reduction:** <1.5% false positive rate (vs. 3-5% current)
- **Quantum Advantage:** Demonstrate measurable improvement using quantum feature mapping
- **Real-world Feasibility:** Assess deployment potential for banking systems

### 🔬 Research Gaps Addressed
1. **Scalability Gap:** Testing on large-scale datasets (100K+ transactions)
2. **Deployment Gap:** Production-ready QML framework for fraud detection
3. **Optimization Gap:** Optimal classical-quantum hybrid balance

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install required packages
pip install -r requirements.txt
```

### Installation
```bash
git clone <repository-url>
cd quantum-fraud-detection
pip install -e .
```

### Basic Usage
```python
from src.models.hybrid_qml import HybridQuantumClassifier
from src.data.preprocessing import load_fraud_dataset

# Load data
X_train, X_test, y_train, y_test = load_fraud_dataset()

# Initialize hybrid model
model = HybridQuantumClassifier(n_qubits=8, n_layers=4)

# Train and evaluate
model.fit(X_train, y_train)
accuracy = model.evaluate(X_test, y_test)
```

## 📊 Project Structure

```
quantum-fraud-detection/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── data/
│   ├── raw/                 # Raw datasets
│   ├── processed/           # Cleaned datasets
│   └── synthetic/           # Generated test data
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py # Data cleaning & feature engineering
│   │   └── dataset_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical_baselines.py
│   │   ├── quantum_circuits.py
│   │   └── hybrid_qml.py   # Main hybrid model
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── statistical_tests.py
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py
│       └── config.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_classical_baselines.ipynb
│   ├── 03_quantum_circuits.ipynb
│   └── 04_hybrid_model.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
├── docs/
│   ├── methodology.md
│   ├── results.md
│   └── api_reference.md
└── scripts/
    ├── train_baselines.py
    ├── train_quantum.py
    └── run_evaluation.py
```

## 📅 Implementation Timeline

### Phase 1: Foundation & Data Preparation (Weeks 1-2)
- [x] Project setup and environment configuration
- [x] Dataset acquisition and initial exploration
- [x] Classical preprocessing pipeline
- [x] Baseline model implementation
- [x] Performance benchmarking

### Phase 2: Quantum Circuit Design (Weeks 3-4)
- [x] Quantum encoding strategy implementation
- [x] Variational Quantum Classifier design
- [x] Circuit optimization and testing
- [x] Simulator validation

### Phase 3: Hybrid Model Integration (Weeks 5-6)
- [ ] Classical-quantum interface development
- [ ] Hybrid optimization implementation
- [ ] Parameter tuning framework
- [ ] Cross-validation setup

### Phase 4: Evaluation & Analysis (Weeks 7-8)
- [ ] Comprehensive performance testing
- [ ] Statistical significance analysis
- [ ] Comparative study documentation
- [ ] Final report preparation

## 🔧 Technical Architecture

### Hybrid Model Design
```
Classical Preprocessing → Quantum Feature Mapping → Classical Decision Layer
     ↓                         ↓                        ↓
Feature Engineering      Variational Circuits      Final Classification
Normalization           Entanglement Layers       Probability Output
Dimensionality          Parameter Optimization     
```

### Quantum Components
- **Encoding:** Angle embedding for continuous features
- **Circuit:** 4-6 layer variational quantum classifier
- **Qubits:** 8-12 qubits for proof-of-concept
- **Optimizer:** COBYLA/SPSA for quantum parameters

### Classical Components
- **Preprocessing:** Feature scaling, selection, engineering
- **Baselines:** Random Forest, XGBoost, Neural Networks
- **Optimizer:** Adam for classical parameters

## 📈 Evaluation Metrics

### Primary Metrics
- **Accuracy:** Overall classification accuracy
- **Precision:** True positive rate for fraud detection
- **Recall:** Fraud case detection rate
- **F1-Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Area under receiver operating characteristic

### Performance Targets
| Metric | Current State | Target | Quantum Goal |
|--------|---------------|---------|--------------|
| Accuracy | 85-92% | >95% | >97% |
| False Positive Rate | 3-5% | <1.5% | <1% |
| Precision | 80-85% | >90% | >95% |
| Recall | 75-85% | >90% | >95% |

## 🛠️ Development Setup

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_models.py
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## 📚 Literature Foundation

### Classical ML Limitations
- High false positive rates (2-5%)
- Inability to detect novel attack patterns
- Computational bottlenecks with large-scale data
- Fixed feature representation limitations

### Quantum Advantages
- Enhanced pattern recognition in high-dimensional spaces
- Non-linear feature mapping capabilities
- Potential for exponential speedup in specific tasks
- Novel optimization landscapes

### Research Progression
```
Classical ML → Quantum Feature Maps → Hybrid Optimization → Real-world Deployment
                                                          ↑
                                                   (Current Gap)
```

## 🎯 Success Criteria

### Technical Milestones
- [ ] Successful quantum circuit simulation
- [ ] Hybrid model convergence
- [ ] Statistical significance in performance improvement (p < 0.05)
- [ ] Scalability demonstration on 100K+ transactions

### Performance Benchmarks
- [ ] >95% accuracy achievement
- [ ] <1.5% false positive rate
- [ ] Measurable quantum advantage demonstration
- [ ] Computational efficiency analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Project Lead:** [Your Name]  
**Email:** [your.email@domain.com]  
**Institution:** [Your Institution]

## 🙏 Acknowledgments

- Quantum computing frameworks: Qiskit, PennyLane
- Dataset providers: IEEE-CIS, Kaggle
- Research community contributions
- Academic advisors and collaborators

## 📖 References

### Real-World Dataset Sources
1. **Lebichot, B., et al.** (2018). "Credit Card Fraud Detection Dataset" 
   - *Data in Brief*, Volume 19, pp. 1-5
   - DOI: 10.1016/j.dib.2018.06.028
   - Available: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. **IEEE Computational Intelligence Society** (2019). "IEEE-CIS Fraud Detection"
   - Competition Dataset, 590K+ transactions
   - Available: https://www.kaggle.com/c/ieee-fraud-detection

3. **Lopez-Rojas, E.A., et al.** (2016). "PaySim: A financial mobile money simulator for fraud detection"
   - *Lecture Notes in Computer Science*, Volume 10718
   - Available: https://www.kaggle.com/datasets/ntnu-testimon/paysim1

### Literature References
4. **Schuld, M., & Petruccione, F.** (2018). *Supervised learning with quantum computers*
   - Springer Nature, ISBN: 978-3-319-96424-9

5. **Biamonte, J., et al.** (2017). "Quantum machine learning"
   - *Nature*, 549(7671), 195-202

6. **Havlíček, V., et al.** (2019). "Supervised learning with quantum-enhanced feature spaces"
   - *Nature*, 567(7747), 209-212

### Academic Validation Note
This project references established, peer-reviewed datasets and follows academic standards for reproducible research in quantum machine learning applications.

---

**Note:** This project is part of academic research. For production deployment, additional security and compliance considerations are required.
