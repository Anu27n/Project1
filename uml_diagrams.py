"""
UML Diagrams for Quantum Machine Learning Fraud Detection Project
Creates Use Case and Class Diagrams
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse
import numpy as np

def create_use_case_diagram():
    """Create Use Case Diagram for the system"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    
    # System boundary
    system_box = FancyBboxPatch((2.5, 1.5), 11, 9, boxstyle="round,pad=0.2", 
                                facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(system_box)
    ax.text(8, 10.8, 'Quantum ML Fraud Detection System', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    # Actors
    actors = [
        {'name': 'Data Scientist', 'pos': (1, 8.5)},
        {'name': 'Banking Analyst', 'pos': (1, 6.5)},
        {'name': 'System Admin', 'pos': (1, 4.5)},
        {'name': 'Fraud Investigator', 'pos': (15, 7)}
    ]
    
    for actor in actors:
        # Actor stick figure
        ax.plot(actor['pos'][0], actor['pos'][1], 'o', markersize=8, color='black')
        ax.plot([actor['pos'][0], actor['pos'][0]], [actor['pos'][1]-0.2, actor['pos'][1]-0.8], 'k-', linewidth=2)
        ax.plot([actor['pos'][0]-0.3, actor['pos'][0]+0.3], [actor['pos'][1]-0.4, actor['pos'][1]-0.4], 'k-', linewidth=2)
        ax.plot([actor['pos'][0], actor['pos'][0]-0.2], [actor['pos'][1]-0.8, actor['pos'][1]-1.2], 'k-', linewidth=2)
        ax.plot([actor['pos'][0], actor['pos'][0]+0.2], [actor['pos'][1]-0.8, actor['pos'][1]-1.2], 'k-', linewidth=2)
        ax.text(actor['pos'][0], actor['pos'][1]-1.5, actor['name'], ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Use Cases (ellipses)
    use_cases = [
        {'name': 'Load Dataset', 'pos': (4, 9), 'size': (1.4, 0.6)},
        {'name': 'Preprocess Data', 'pos': (6.5, 9), 'size': (1.6, 0.6)},
        {'name': 'Apply PCA', 'pos': (9, 9), 'size': (1.2, 0.6)},
        {'name': 'Train Classical\nModel', 'pos': (4, 7.5), 'size': (1.6, 0.8)},
        {'name': 'Train Quantum\nModel', 'pos': (7, 7.5), 'size': (1.6, 0.8)},
        {'name': 'Evaluate\nPerformance', 'pos': (10, 7.5), 'size': (1.6, 0.8)},
        {'name': 'Generate Reports', 'pos': (4, 6), 'size': (1.6, 0.6)},
        {'name': 'Compare Models', 'pos': (7, 6), 'size': (1.6, 0.6)},
        {'name': 'Visualize Results', 'pos': (10, 6), 'size': (1.6, 0.6)},
        {'name': 'Configure\nSystem', 'pos': (4, 4.5), 'size': (1.4, 0.8)},
        {'name': 'Monitor\nTraining', 'pos': (7, 4.5), 'size': (1.4, 0.8)},
        {'name': 'Export Models', 'pos': (10, 4.5), 'size': (1.4, 0.6)},
        {'name': 'Detect Fraud\nPatterns', 'pos': (7, 3), 'size': (1.6, 0.8)}
    ]
    
    for uc in use_cases:
        ellipse = Ellipse(uc['pos'], uc['size'][0], uc['size'][1], 
                         facecolor='lightyellow', edgecolor='black')
        ax.add_patch(ellipse)
        ax.text(uc['pos'][0], uc['pos'][1], uc['name'], ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    # Relationships (simplified)
    relationships = [
        (actors[0]['pos'], (4, 9)),  # Data Scientist -> Load Dataset
        (actors[0]['pos'], (6.5, 9)),  # Data Scientist -> Preprocess Data
        (actors[0]['pos'], (4, 7.5)),  # Data Scientist -> Train Classical Model
        (actors[0]['pos'], (7, 7.5)),  # Data Scientist -> Train Quantum Model
        (actors[1]['pos'], (10, 7.5)),  # Banking Analyst -> Evaluate Performance
        (actors[1]['pos'], (7, 6)),  # Banking Analyst -> Compare Models
        (actors[2]['pos'], (4, 4.5)),  # System Admin -> Configure System
        (actors[3]['pos'], (7, 3)),  # Fraud Investigator -> Detect Fraud Patterns
    ]
    
    for start, end in relationships:
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k--', alpha=0.6)
    
    ax.set_title('Use Case Diagram: Quantum ML Fraud Detection System', 
                fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/workspaces/Project1/use_case_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_class_diagram():
    """Create Class Diagram for the system"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    
    # Class definitions
    classes = [
        {
            'name': 'DataLoader',
            'pos': (2, 11),
            'size': (3, 2.5),
            'attributes': [
                '- file_path: str',
                '- data: DataFrame',
                '- target_column: str'
            ],
            'methods': [
                '+ load_csv()',
                '+ validate_data()',
                '+ get_data_info()'
            ]
        },
        {
            'name': 'DataPreprocessor',
            'pos': (6.5, 11),
            'size': (3.5, 2.5),
            'attributes': [
                '- scaler: StandardScaler',
                '- encoder: OneHotEncoder',
                '- pca: PCA'
            ],
            'methods': [
                '+ handle_missing_values()',
                '+ detect_outliers()',
                '+ encode_categorical()',
                '+ apply_scaling()',
                '+ reduce_dimensions()'
            ]
        },
        {
            'name': 'ClassicalClassifier',
            'pos': (2, 7.5),
            'size': (3.5, 2.5),
            'attributes': [
                '- model: LogisticRegression',
                '- parameters: dict',
                '- is_trained: bool'
            ],
            'methods': [
                '+ train(X, y)',
                '+ predict(X)',
                '+ cross_validate()',
                '+ get_feature_importance()'
            ]
        },
        {
            'name': 'QuantumClassifier',
            'pos': (7, 7.5),
            'size': (3.5, 2.5),
            'attributes': [
                '- circuit: qml.QNode',
                '- n_qubits: int',
                '- n_layers: int',
                '- parameters: np.array'
            ],
            'methods': [
                '+ create_circuit()',
                '+ angle_embedding()',
                '+ variational_layers()',
                '+ train_quantum()',
                '+ predict_quantum()'
            ]
        },
        {
            'name': 'ModelEvaluator',
            'pos': (12, 8.5),
            'size': (3.5, 3),
            'attributes': [
                '- metrics: dict',
                '- confusion_matrix: array',
                '- roc_curve: tuple'
            ],
            'methods': [
                '+ calculate_accuracy()',
                '+ calculate_precision()',
                '+ calculate_recall()',
                '+ calculate_f1_score()',
                '+ plot_roc_curve()',
                '+ generate_report()'
            ]
        },
        {
            'name': 'ExperimentManager',
            'pos': (2, 4),
            'size': (4, 2.5),
            'attributes': [
                '- config: dict',
                '- results: list',
                '- models: dict'
            ],
            'methods': [
                '+ run_classical_experiment()',
                '+ run_quantum_experiment()',
                '+ compare_models()',
                '+ save_results()'
            ]
        },
        {
            'name': 'VisualizationEngine',
            'pos': (8, 4),
            'size': (3.5, 2.5),
            'attributes': [
                '- plot_style: str',
                '- figure_size: tuple',
                '- colors: list'
            ],
            'methods': [
                '+ plot_training_curves()',
                '+ plot_confusion_matrix()',
                '+ plot_feature_importance()',
                '+ create_comparison_chart()'
            ]
        },
        {
            'name': 'QuantumCircuit',
            'pos': (13, 4.5),
            'size': (3.5, 2),
            'attributes': [
                '- n_qubits: int',
                '- depth: int',
                '- gates: list'
            ],
            'methods': [
                '+ add_rotation_gate()',
                '+ add_cnot_gate()',
                '+ measure()',
                '+ get_expectation()'
            ]
        },
        {
            'name': 'FraudDetectionSystem',
            'pos': (7, 1),
            'size': (4, 2),
            'attributes': [
                '- classical_model: ClassicalClassifier',
                '- quantum_model: QuantumClassifier',
                '- threshold: float'
            ],
            'methods': [
                '+ detect_fraud()',
                '+ ensemble_predict()',
                '+ update_models()'
            ]
        }
    ]
    
    # Draw classes
    for cls in classes:
        # Class rectangle
        rect = FancyBboxPatch(
            (cls['pos'][0] - cls['size'][0]/2, cls['pos'][1] - cls['size'][1]/2),
            cls['size'][0], cls['size'][1],
            boxstyle="round,pad=0.05",
            facecolor='lightcyan',
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Class name
        ax.text(cls['pos'][0], cls['pos'][1] + cls['size'][1]/2 - 0.2, cls['name'],
                ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Separator line
        ax.plot([cls['pos'][0] - cls['size'][0]/2 + 0.1, cls['pos'][0] + cls['size'][0]/2 - 0.1],
                [cls['pos'][1] + cls['size'][1]/2 - 0.4, cls['pos'][1] + cls['size'][1]/2 - 0.4],
                'k-', linewidth=1)
        
        # Attributes
        attr_y = cls['pos'][1] + cls['size'][1]/2 - 0.7
        for attr in cls['attributes']:
            ax.text(cls['pos'][0] - cls['size'][0]/2 + 0.1, attr_y, attr,
                    ha='left', va='center', fontsize=8)
            attr_y -= 0.25
        
        # Separator line
        ax.plot([cls['pos'][0] - cls['size'][0]/2 + 0.1, cls['pos'][0] + cls['size'][0]/2 - 0.1],
                [attr_y + 0.1, attr_y + 0.1], 'k-', linewidth=1)
        
        # Methods
        method_y = attr_y - 0.1
        for method in cls['methods']:
            ax.text(cls['pos'][0] - cls['size'][0]/2 + 0.1, method_y, method,
                    ha='left', va='center', fontsize=8)
            method_y -= 0.25
    
    # Relationships (simplified)
    relationships = [
        ((2, 11), (6.5, 11), 'uses'),  # DataLoader -> DataPreprocessor
        ((6.5, 11), (2, 7.5), 'feeds'),  # DataPreprocessor -> ClassicalClassifier
        ((6.5, 11), (7, 7.5), 'feeds'),  # DataPreprocessor -> QuantumClassifier
        ((7, 7.5), (13, 4.5), 'contains'),  # QuantumClassifier -> QuantumCircuit
        ((2, 7.5), (12, 8.5), 'evaluated by'),  # ClassicalClassifier -> ModelEvaluator
        ((7, 7.5), (12, 8.5), 'evaluated by'),  # QuantumClassifier -> ModelEvaluator
        ((2, 4), (8, 4), 'uses'),  # ExperimentManager -> VisualizationEngine
        ((7, 1), (2, 7.5), 'contains'),  # FraudDetectionSystem -> ClassicalClassifier
        ((7, 1), (7, 7.5), 'contains'),  # FraudDetectionSystem -> QuantumClassifier
    ]
    
    for start, end, label in relationships:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
    
    ax.set_title('Class Diagram: Quantum ML Fraud Detection System', 
                fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/workspaces/Project1/class_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Creating UML Diagrams for Quantum ML Fraud Detection System...")
    print("\n1. Creating Use Case Diagram...")
    create_use_case_diagram()
    
    print("\n2. Creating Class Diagram...")
    create_class_diagram()
    
    print("\nUML Diagrams created successfully!")
    print("Files saved:")
    print("- use_case_diagram.png")
    print("- class_diagram.png")
