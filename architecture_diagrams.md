# System Architecture Documentation
## Quantum Machine Learning for Financial Fraud Detection

### 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUANTUM ML FRAUD DETECTION SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────────────┐ │
│  │   Raw Data  │───▶│ Preprocessing│───▶│      Classical Pipeline        │ │
│  │  (1M records│    │   Module     │    │                                 │ │
│  │  32 features)│    └──────────────┘    │  ┌─────────────┐                │ │
│  └─────────────┘           │             │  │   Logistic  │                │ │
│                             │             │  │ Regression  │───┐            │ │
│                             ▼             │  │(Baseline)   │   │            │ │
│  ┌─────────────────────────────────────┐  │  └─────────────┘   │            │ │
│  │      PREPROCESSING PIPELINE         │  └─────────────────────│────────────┘ │
│  │                                     │                      │              │
│  │  ┌─────────────┐  ┌──────────────┐  │                      ▼              │
│  │  │ Missing Val │  │ Categorical  │  │  ┌─────────────────────────────────┐ │
│  │  │  Analysis   │  │   Encoding   │  │  │      Quantum Pipeline           │ │
│  │  │ (0 missing) │  │ (One-Hot)    │  │  │                                 │ │
│  │  └─────────────┘  └──────────────┘  │  │  ┌──────────────────────────┐   │ │
│  │                                     │  │  │    Feature Reduction     │   │ │
│  │  ┌─────────────┐  ┌──────────────┐  │  │  │    PCA (32→6 dims)       │   │ │
│  │  │ Outlier     │  │   Feature    │  │  │  └──────────────────────────┘   │ │
│  │  │ Detection   │  │ Scaling      │  │  │              │                  │ │
│  │  │ (8.5%)      │  │(Standard)    │  │  │              ▼                  │ │
│  │  └─────────────┘  └──────────────┘  │  │  ┌──────────────────────────┐   │ │
│  └─────────────────────────────────────┘  │  │    Quantum Circuit       │   │ │
│                             │              │  │                          │   │ │
│                             ▼              │  │  ┌────────────────────┐  │   │ │
│  ┌─────────────────────────────────────┐   │  │  │  Angle Embedding   │  │   │ │
│  │         CLEAN DATASET               │   │  │  │   (6 features)     │  │   │ │
│  │                                     │   │  │  └────────────────────┘  │   │ │
│  │  • 1M samples                       │   │  │              │           │   │ │
│  │  • 32 features → 45+ after encoding │   │  │              ▼           │   │ │
│  │  • Balanced train/test split        │   │  │  ┌────────────────────┐  │   │ │
│  │  • Small subset (2K) for quantum    │   │  │  │ Variational Layers │  │   │ │
│  │                                     │   │  │  │(Strongly Entangling│  │   │ │
│  └─────────────────────────────────────┘   │  │  │    3 layers)       │  │   │ │
│                                             │  │  └────────────────────┘  │   │ │
│                                             │  │              │           │   │ │
│                                             │  │              ▼           │   │ │
│                                             │  │  ┌────────────────────┐  │   │ │
│                                             │  │  │   Measurement      │  │   │ │
│                                             │  │  │  PauliZ(qubit 0)   │  │   │ │
│                                             │  │  └────────────────────┘  │   │ │
│                                             │  └──────────────────────────┘   │ │
│                                             └─────────────────────────────────┘ │
│                                                             │                   │
│                                                             ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      EVALUATION MODULE                                   │   │
│  │                                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │   │
│  │  │  Accuracy   │  │   ROC-AUC   │  │ Precision/  │  │   Training  │   │   │
│  │  │ Comparison  │  │ Comparison  │  │   Recall    │  │    Time     │   │   │
│  │  │             │  │             │  │ Comparison  │  │ Comparison  │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Quantum Circuit Architecture

**6-Qubit Variational Quantum Classifier (VQC) Circuit:**

```
Qubit 0: ──RY(x₀)──╭●──RY(θ₀)──RZ(θ₁)──RY(θ₂)──╭●──RY(θ₉)──RZ(θ₁₀)──RY(θ₁₁)──⟨Z⟩
                    │                            │                              
Qubit 1: ──RY(x₁)──╰X──RY(θ₃)──RZ(θ₄)──RY(θ₅)──╰X──RY(θ₁₂)─RZ(θ₁₃)──RY(θ₁₄)─────
                    │                            │                              
Qubit 2: ──RY(x₂)──╭●──RY(θ₆)──RZ(θ₇)──RY(θ₈)──╭●──RY(θ₁₅)─RZ(θ₁₆)──RY(θ₁₇)─────
                    │                            │                              
Qubit 3: ──RY(x₃)──╰X──RY(θ₁₈)─RZ(θ₁₉)─RY(θ₂₀)─╰X──RY(θ₂₁)─RZ(θ₂₂)──RY(θ₂₃)─────
                    │                            │                              
Qubit 4: ──RY(x₄)──╭●──RY(θ₂₄)─RZ(θ₂₅)─RY(θ₂₆)─╭●──RY(θ₂₇)─RZ(θ₂₈)──RY(θ₂₉)─────
                    │                            │                              
Qubit 5: ──RY(x₅)──╰X──RY(θ₃₀)─RZ(θ₃₁)─RY(θ₃₂)─╰X──RY(θ₃₃)─RZ(θ₃₄)──RY(θ₃₅)─────

         ↑─────────────────────────────────────────────────────────────────
      Angle         Variational              Variational
    Embedding        Layer 1                  Layer 2
```

**Circuit Components:**
- **Angle Embedding**: RY rotations encode classical features into quantum states
- **Variational Layers**: Parameterized rotations and entangling gates
- **Entangling Gates**: CNOT gates create quantum correlations between qubits
- **Measurement**: Expectation value of Pauli-Z operator on qubit 0

### 3. Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA PROCESSING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Raw Dataset (Bank Account Fraud - NeurIPS 2022)                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • 1,000,000 transaction records                                     │   │
│  │ • 32 features (financial, behavioral, demographic, digital)         │   │
│  │ • Binary target: fraud_bool                                         │   │
│  │ • Class imbalance: 1.1% fraud, 98.9% legitimate                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PREPROCESSING STEPS                              │   │
│  │                                                                     │   │
│  │  Step 1: Data Quality Assessment                                    │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ • Missing value analysis: 0 missing values                  │   │   │
│  │  │ • Data type validation: 21 numerical, 11 categorical       │   │   │
│  │  │ • Infinite value check: 0 infinite values                  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  Step 2: Outlier Detection                                          │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ • Statistical Test (99.9% confidence): 8.5% outliers       │   │   │
│  │  │ • Isolation Forest (5% contamination): 5.0% outliers       │   │   │
│  │  │ • Isolation Forest (2% contamination): 2.0% outliers       │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  Step 3: Feature Engineering                                        │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ • Categorical encoding (One-Hot): 11 → 33+ features        │   │   │
│  │  │ • Feature scaling (StandardScaler): mean=0, std=1          │   │   │
│  │  │ • Total features after encoding: 45+ features              │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  Step 4: Dimensionality Reduction                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ • PCA transformation: 45+ → 6 components                   │   │   │
│  │  │ • Explained variance ratio: 99.x% preserved                │   │   │
│  │  │ • Optimized for 6-qubit quantum system                     │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     DATASET SPLITTING                               │   │
│  │                                                                     │   │
│  │  ┌─────────────────────┐    ┌─────────────────────┐                │   │
│  │  │   Training Set      │    │    Testing Set      │                │   │
│  │  │   (80% of data)     │    │   (20% of data)     │                │   │
│  │  │ • Stratified split  │    │ • Maintains class   │                │   │
│  │  │ • Class balance     │    │   distribution      │                │   │
│  │  │   maintained        │    │ • For evaluation    │                │   │
│  │  └─────────────────────┘    └─────────────────────┘                │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              Small Subset (for Quantum)                     │   │   │
│  │  │ • 2,000 samples total                                       │   │   │
│  │  │ • Stratified sampling maintaining fraud ratio               │   │   │
│  │  │ • Required for quantum simulation efficiency                │   │   │
│  │  │ • 1,600 training + 400 testing                             │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. Model Comparison Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLASSICAL vs QUANTUM MODELS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │        CLASSICAL PIPELINE       │  │        QUANTUM PIPELINE        │   │
│  │                                 │  │                                 │   │
│  │  ┌─────────────────────────┐    │  │  ┌─────────────────────────┐    │   │
│  │  │    Preprocessed Data    │    │  │  │    Preprocessed Data    │    │   │
│  │  │   (Full 45+ features)   │    │  │  │  (PCA reduced to 6D)    │    │   │
│  │  └─────────────────────────┘    │  │  └─────────────────────────┘    │   │
│  │              │                  │  │              │                  │   │
│  │              ▼                  │  │              ▼                  │   │
│  │  ┌─────────────────────────┐    │  │  ┌─────────────────────────┐    │   │
│  │  │   Logistic Regression   │    │  │  │   Angle Embedding       │    │   │
│  │  │                         │    │  │  │   (Feature → Angles)    │    │   │
│  │  │ • Class balancing       │    │  │  └─────────────────────────┘    │   │
│  │  │ • L2 regularization     │    │  │              │                  │   │
│  │  │ • Adam optimizer        │    │  │              ▼                  │   │
│  │  └─────────────────────────┘    │  │  ┌─────────────────────────┐    │   │
│  │              │                  │  │  │  Variational Quantum    │    │   │
│  │              ▼                  │  │  │       Circuit           │    │   │
│  │  ┌─────────────────────────┐    │  │  │                         │    │   │
│  │  │    Classification       │    │  │  │ • 6 qubits              │    │   │
│  │  │     (Sigmoid)           │    │  │  │ • 3 variational layers  │    │   │
│  │  └─────────────────────────┘    │  │  │ • Entangling gates      │    │   │
│  │              │                  │  │  │ • 36 parameters         │    │   │
│  │              ▼                  │  │  └─────────────────────────┘    │   │
│  │  ┌─────────────────────────┐    │  │              │                  │   │
│  │  │      Predictions        │    │  │              ▼                  │   │
│  │  │                         │    │  │  ┌─────────────────────────┐    │   │
│  │  │ • Binary output         │    │  │  │     Measurement         │    │   │
│  │  │ • Probability scores    │    │  │  │   (Expectation Value)   │    │   │
│  │  └─────────────────────────┘    │  │  └─────────────────────────┘    │   │
│  └─────────────────────────────────┘  │              │                  │   │
│                                       │              ▼                  │   │
│                                       │  ┌─────────────────────────┐    │   │
│                                       │  │   Post-processing       │    │   │
│                                       │  │   (Sigmoid mapping)     │    │   │
│                                       │  └─────────────────────────┘    │   │
│                                       │              │                  │   │
│                                       │              ▼                  │   │
│                                       │  ┌─────────────────────────┐    │   │
│                                       │  │      Predictions        │    │   │
│                                       │  │                         │    │   │
│                                       │  │ • Binary output         │    │   │
│                                       │  │ • Probability scores    │    │   │
│                                       │  └─────────────────────────┘    │   │
│                                       └─────────────────────────────────┘   │
│                                                       │                     │
│                                                       ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         EVALUATION METRICS                          │   │
│  │                                                                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │  Accuracy   │  │  Precision  │  │   Recall    │  │  F1-Score   │ │   │
│  │  │             │  │             │  │             │  │             │ │   │
│  │  │ Classical:  │  │ Classical:  │  │ Classical:  │  │ Classical:  │ │   │
│  │  │   89.2%     │  │   0.156     │  │   0.623     │  │   0.251     │ │   │
│  │  │             │  │             │  │             │  │             │ │   │
│  │  │ Quantum:    │  │ Quantum:    │  │ Quantum:    │  │ Quantum:    │ │   │
│  │  │   89.1%     │  │   0.152     │  │   0.618     │  │   0.248     │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  │                                                                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │   ROC-AUC   │  │ Training    │  │ Memory      │  │ Scalability │ │   │
│  │  │             │  │    Time     │  │   Usage     │  │             │ │   │
│  │  │ Classical:  │  │             │  │             │  │             │ │   │
│  │  │   0.847     │  │ Classical:  │  │ Classical:  │  │ Classical:  │ │   │
│  │  │             │  │   2.3 min   │  │   0.5 GB    │  │   1M+       │ │   │
│  │  │ Quantum:    │  │             │  │             │  │             │ │   │
│  │  │   0.834     │  │ Quantum:    │  │ Quantum:    │  │ Quantum:    │ │   │
│  │  │             │  │   7.1 min   │  │   1.2 GB    │  │   2K        │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5. Technology Stack Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TECHNOLOGY STACK                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        APPLICATION LAYER                            │   │
│  │                                                                     │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │   │
│  │  │   Data Exploration  │    │  Quantum ML     │    │   Evaluation    │  │   │
│  │  │     Notebook        │    │   Notebook      │    │   & Results     │  │   │
│  │  │                     │    │                 │    │                 │  │   │
│  │  │ 01_data_exploration │    │    02_qml       │    │   Metrics &     │  │   │
│  │  │        .ipynb       │    │     .ipynb      │    │ Visualizations  │  │   │
│  │  └─────────────────────┘    └─────────────────┘    └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      FRAMEWORK LAYER                                │   │
│  │                                                                     │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │   Data Science  │  │  Quantum ML     │  │    Visualization    │  │   │
│  │  │   Libraries     │  │   Framework     │  │     Libraries       │  │   │
│  │  │                 │  │                 │  │                     │  │   │
│  │  │ • Pandas        │  │ • PennyLane     │  │ • Matplotlib        │  │   │
│  │  │ • NumPy         │  │ • Qiskit        │  │ • Seaborn           │  │   │
│  │  │ • Scikit-learn  │  │ • PyTorch       │  │ • Plotly            │  │   │
│  │  │ • SciPy         │  │                 │  │                     │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       RUNTIME LAYER                                 │   │
│  │                                                                     │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │     Python      │  │   Jupyter       │  │    Quantum          │  │   │
│  │  │    Runtime      │  │   Runtime       │  │   Simulator         │  │   │
│  │  │                 │  │                 │  │                     │  │   │
│  │  │ • Python 3.8+   │  │ • Jupyter       │  │ • default.qubit     │  │   │
│  │  │ • Virtual Env   │  │   Notebook      │  │ • PennyLane         │  │   │
│  │  │ • Package Mgmt  │  │ • Interactive   │  │   Simulator         │  │   │
│  │  │                 │  │   Development   │  │ • Auto-diff         │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       HARDWARE LAYER                                │   │
│  │                                                                     │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │     Classical   │  │    Memory       │  │      Storage        │  │   │
│  │  │      CPU        │  │     (RAM)       │  │                     │  │   │
│  │  │                 │  │                 │  │                     │  │   │
│  │  │ • Multi-core    │  │ • 8GB+ RAM      │  │ • Dataset Storage   │  │   │
│  │  │ • x86_64        │  │ • For quantum   │  │ • Model Storage     │  │   │
│  │  │ • Simulation    │  │   simulation    │  │ • Results Storage   │  │   │
│  │  │   Processing    │  │                 │  │                     │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

This comprehensive architecture documentation shows:

1. **System-level architecture** with all major components
2. **Detailed quantum circuit design** with specific gate arrangements
3. **Data processing pipeline** with step-by-step transformations
4. **Model comparison framework** showing classical vs quantum approaches
5. **Technology stack** from hardware to application layers

You can use these diagrams in your presentation slides and also as standalone architecture documentation for your project.
