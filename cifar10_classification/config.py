
# config.py
import os

# Chemins vers les données
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'cifar-10-batches-py')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

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
MODEL_TYPES = ['logistic', 'random_forest', 'sgd', 'svm', 'knn', 'naive_bayes', 'perceptron', 'mlp', 'linear_svm']