# train.py
import os
import sys

from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.getenv('PYTHONPATH'))
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from cifar10_classification.dataset import prepare_data
from cifar10_classification.modeling.predict import evaluate_model, predict_model
from cifar10_classification.features import extract_hog_features, flatten_images, save_processed_data
from cifar10_classification.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, N_ESTIMATOR_RANDOM_FOREST, KERNEL, LOSS, MODEL_TYPES, HYPERPARAMETERS
from sklearn.linear_model import SGDClassifier
def train_logistic_regression(X_train, y_train, X_val, y_val):
    params = HYPERPARAMETERS['logistic']
    scaler = StandardScaler()
    model = LogisticRegression(**params)
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)
    val_predictions = predict_model(pipeline, X_val)
    #accuracy = accuracy_score(y_val, val_predictions)
    return model, val_predictions

def train_random_forest(X_train, y_train, X_val, y_val):
    params = HYPERPARAMETERS['random_forest']
    scaler = StandardScaler()
    model = RandomForestClassifier(**params)
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)
    val_predictions = predict_model(pipeline, X_val)
    #accuracy = accuracy_score(y_val, val_predictions)
    return model, val_predictions

def train_svm(X_train, y_train, X_val, y_val):
    params = HYPERPARAMETERS['svm']
    scaler = StandardScaler()
    model = SVC(**params)
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)
    val_predictions = predict_model(pipeline, X_val)
    #accuracy = accuracy_score(y_val, val_predictions)
    return model, val_predictions

def train_sgd_classifier(X_train, y_train, X_val, y_val, loss=LOSS): #The 'loss' parameter of SGDClassifier must be a str among {'modified_huber', 'epsilon_insensitive', 'hinge', 'huber', 'squared_hinge', 'log_loss', 'squared_epsilon_insensitive', 'perceptron', 'squared_error'}
    scaler = StandardScaler()
    params = HYPERPARAMETERS['sgd']
    model = SGDClassifier(**params)
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)
    val_predictions = predict_model(pipeline, X_val)
    #accuracy = accuracy_score(y_val, val_predictions)
    return model, val_predictions

def train_knn(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    model = KNeighborsClassifier(n_neighbors=N_ESTIMATOR_RANDOM_FOREST)
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)
    val_predictions = predict_model(pipeline, X_val)
    #accuracy = accuracy_score(y_val, val_predictions)
    return model, val_predictions

def train_linear_svm(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    model = LinearSVC(max_iter=2000)
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)
    val_predictions = predict_model(pipeline, X_val)
    #accuracy = accuracy_score(y_val, val_predictions)
    return model, val_predictions

def train_naive_bayes(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    model = GaussianNB()
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)
    val_predictions = predict_model(pipeline, X_val)
    #accuracy = accuracy_score(y_val, val_predictions)
    return model, val_predictions

def train_classifier(X_train, y_train, X_val, y_val, model_type=MODEL_TYPES[0]):
    if model_type == 'logistic':
        return train_logistic_regression(X_train, y_train, X_val, y_val)
    elif model_type == 'random_forest':
        return train_random_forest(X_train, y_train, X_val, y_val)
    elif model_type == 'svm':
        return train_svm(X_train, y_train, X_val, y_val)
    elif model_type == 'sgd':
        return train_sgd_classifier(X_train, y_train, X_val, y_val)
    elif model_type == 'knn':
        return train_knn(X_train, y_train, X_val, y_val)
    elif model_type == 'linear_svm':
        return train_linear_svm(X_train, y_train, X_val, y_val)
    elif model_type == 'naive_bayes':
        return train_naive_bayes(X_train, y_train, X_val, y_val)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
def main():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

    # Example with flattened images
    X_train_flat = flatten_images(X_train)
    X_val_flat = flatten_images(X_val)

    # Example with HOG features
    X_train_hog = extract_hog_features(X_train)
    X_val_hog = extract_hog_features(X_val)
    X_test_hog = extract_hog_features(X_test)
    save_processed_data(X_train_hog, X_val_hog, X_test_hog, 'hog')
    model, val_predictions = train_classifier(X_train_hog, y_train, X_val_hog, y_val, model_type=MODEL_TYPES[0])
    print(classification_report(y_val, val_predictions))

if __name__ == '__main__':
    main()

