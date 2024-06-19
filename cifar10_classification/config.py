
# config.py
import os

import sys

from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.getenv('PYTHONPATH'))
# Chemins vers les données
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'cifar-10-batches-py')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
INTERIM_DATA_DIR = os.path.join(BASE_DIR, 'data', 'interim')
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, 'data', 'external')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
# Hyperparamètres
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 500
SEED = 42
N_ESTIMATOR_RANDOM_FOREST = 5
KERNEL = 'linear'
LOSS = 'log_loss' #The 'loss' parameter of SGDClassifier must be a str among {'modified_huber', 'epsilon_insensitive', 'hinge', 'huber', 'squared_hinge', 'log_loss', 'squared_epsilon_insensitive', 'perceptron', 'squared_error'}
# Autres configurations
IMG_SIZE = (32, 32, 3)
NUM_CLASSES = 10

# Liste des noms de labels
LABEL_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

# Model types disponibles
MODEL_TYPES = ['logistic', 'random_forest', 'sgd', 'svm', 'knn', 'naive_bayes', 'linear_svm']

# Hyperparamètres spécifiques aux modèles
HYPERPARAMETERS = {
    'logistic': {
        'C': 0.1,
        'dual': False,
        'fit_intercept': True,
        'intercept_scaling': 1,
        'max_iter': 2000,
        #'multi_class': 'auto',
        'penalty': 'l2',
        'random_state': 101,
        'solver': 'lbfgs',
        'tol': 0.0001,
        'verbose': 0,
        'warm_start': False
    },
    'random_forest': {
        'bootstrap': True,
        'ccp_alpha': 0.0,
        'criterion': 'gini',
        'max_depth': 20,
        'max_features': 'sqrt',
        'min_impurity_decrease': 0.0,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'min_weight_fraction_leaf': 0.0,
        'n_estimators': 200,
        'oob_score': False,
        'random_state': 101,
        'verbose': 0,
        'warm_start': False
    },
    'sgd': {
        'alpha': 0.0001,
        'average': False,
        'early_stopping': False,
        'epsilon': 0.1,
        'eta0': 0.0,
        'fit_intercept': True,
        'l1_ratio': 0.15,
        'learning_rate': 'optimal',
        'loss': 'log_loss',
        'max_iter': 1000,
        'n_iter_no_change': 5,
        'penalty': 'l2',
        'power_t': 0.5,
        'random_state': 101,
        'shuffle': True,
        'tol': 0.0001,
        'validation_fraction': 0.1,
        'verbose': 0,
        'warm_start': False
    },
    'svm': {
        'C': 10,
        'break_ties': False,
        'cache_size': 200,
        'coef0': 0.0,
        'decision_function_shape': 'ovr',
        'degree': 3,
        'gamma': 'scale',
        'kernel': 'rbf',
        'max_iter': -1,
        'probability': False,
        'random_state': 101,
        'shrinking': True,
        'tol': 0.001,
        'verbose': False
    }
}