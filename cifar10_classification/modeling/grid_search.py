# grid_search.py

import os
import sys

from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ajouter le répertoire parent au PYTHONPATH
#sys.path.append(os.getenv('PYTHONPATH'))
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from cifar10_classification.dataset import prepare_data
from cifar10_classification.features import extract_features
from cifar10_classification.config import LABEL_NAMES
import joblib
import os
import pandas as pd
import time
def save_model(model, feature_method, classifier):
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_filename = os.path.join(model_dir, f'{classifier}_{feature_method}_best_model.pkl')
    joblib.dump(model, model_filename)



from sklearn.svm import LinearSVC

def train_linear_svm(X_train, y_train, X_val, y_val, use_pca=True): # Paramétrique linéaire
    print("Starting train_linear_svm...")
    start_time = time.time()
    scaler = StandardScaler()
    model = LinearSVC(max_iter=2000)
    if use_pca:
        n_components = min(100, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_components)
        pipeline = make_pipeline(scaler, pca, model)
    else:
        pipeline = make_pipeline(scaler, model)
    param_grid = {
        'linearsvc__C': [0.1, 1.0, 10.0]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=2)
    grid_search.fit(X_train, y_train)
    val_predictions = grid_search.predict(X_val)
    end_time = time.time()
    exec_time = end_time - start_time
    print("Completed train_linear_svm in", round(exec_time/60, 2), "minutes.")
    return grid_search.best_estimator_, val_predictions, exec_time

def train_logistic_regression(X_train, y_train, X_val, y_val, use_pca=True): # Paramétrique linéaire
    print("Starting train_logistic_regression...")
    start_time = time.time()
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=2000)
    if use_pca == True:
        n_components = min(100, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_components)
        pipeline = make_pipeline(scaler, pca, model)
    else:
        pipeline = make_pipeline(scaler, model)
    param_grid = {
        'logisticregression__C': [0.1, 1.0, 10.0],
        'logisticregression__solver': ['lbfgs', 'liblinear']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=2)
    grid_search.fit(X_train, y_train)
    val_predictions = grid_search.predict(X_val)
    end_time = time.time()
    exec_time = end_time - start_time
    print("Completed train_logistic_regression in", round(exec_time/60, 2), "minutes.")
    return grid_search.best_estimator_, val_predictions, exec_time

def train_random_forest(X_train, y_train, X_val, y_val, use_pca=True): # Non paramétrique non linéaire
    print("Starting train_random_forest...")
    start_time = time.time()
    scaler = StandardScaler()
    model = RandomForestClassifier()
    if use_pca == True:
        n_components = min(100, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_components)
        pipeline = make_pipeline(scaler, pca, model)
    else:
        pipeline = make_pipeline(scaler, model)
    param_grid = {
        'randomforestclassifier__criterion': ['gini', 'entropy'],
        'randomforestclassifier__n_estimators': [50, 100, 200],
        'randomforestclassifier__max_depth': [None, 10, 20]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=2)
    grid_search.fit(X_train, y_train)
    val_predictions = grid_search.predict(X_val)
    end_time = time.time()
    exec_time = end_time - start_time
    print("Completed train_random_forest in", round(exec_time/60, 2), "minutes.")
    return grid_search.best_estimator_, val_predictions, exec_time

def train_svm(X_train, y_train, X_val, y_val, use_pca=True): # Paramétrique non linéaire
    print("Starting train_svm...")
    start_time = time.time()
    scaler = StandardScaler()
    model = SVC()
    # Adjust n_components based on available samples and features
    if use_pca == True:
        n_components = min(100, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_components)
        pipeline = make_pipeline(scaler, pca, model)
    else:
        pipeline = make_pipeline(scaler, model)
    param_grid = {
        'svc__C': [0.1, 1.0, 10.0],
        'svc__kernel': ['linear', 'rbf', 'poly']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=2)
    grid_search.fit(X_train, y_train)
    val_predictions = grid_search.predict(X_val)
    end_time = time.time()
    exec_time = end_time - start_time
    print("Completed train_svm in", round(exec_time/60, 2), "minutes.")
    return grid_search.best_estimator_, val_predictions, exec_time

def train_knn(X_train, y_train, X_val, y_val, use_pca=True): # Non paramétrique non linéaire
    print("Starting train_knn...")
    start_time = time.time()
    scaler = StandardScaler()
    model = KNeighborsClassifier()
    if use_pca == True:
        n_components = min(100, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_components)
        pipeline = make_pipeline(scaler, pca, model)
    else:
        pipeline = make_pipeline(scaler, model)
    param_grid = {
        'kneighborsclassifier__n_neighbors': [3, 5, 7],
        'kneighborsclassifier__weights': ['uniform', 'distance'],
        'kneighborsclassifier__algorithm': ['ball_tree', 'kd_tree', 'brute']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=2)
    grid_search.fit(X_train, y_train)
    val_predictions = grid_search.predict(X_val)
    end_time = time.time()
    exec_time = end_time - start_time
    print("Completed train_knn in", round(exec_time/60, 2), "minutes.")
    return grid_search.best_estimator_, val_predictions, exec_time

def train_sgd_classifier(X_train, y_train, X_val, y_val, use_pca=True): # Paramétrique linéaire (Stochastic Gradient Descent)
    print("Starting train_sgd_classifier...")
    start_time = time.time()
    scaler = StandardScaler()
    model = SGDClassifier(max_iter=2000, tol=1e-3)
    if use_pca == True:
        n_components = min(100, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_components)
        pipeline = make_pipeline(scaler, pca, model)
    else:
        pipeline = make_pipeline(scaler, model)
    param_grid = {
        'sgdclassifier__loss': ['log_loss', 'modified_huber', 'perceptron'],
        'sgdclassifier__alpha': [0.0001, 0.001, 0.01]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=2)
    grid_search.fit(X_train, y_train)
    val_predictions = grid_search.predict(X_val)
    end_time = time.time()
    exec_time = end_time - start_time
    # deux chiffres après la virgule
    print("Completed train_sgd_classifier in", round(exec_time/60, 2), "minutes.")
    return grid_search.best_estimator_, val_predictions, exec_time

from sklearn.naive_bayes import GaussianNB
def train_naive_bayes(X_train, y_train, X_val, y_val, use_pca=True): # Paramétrique linéaire
    print("Starting train_naive_bayes...")
    start_time = time.time()
    scaler = StandardScaler()
    model = GaussianNB()
    if use_pca == True:
        n_components = min(100, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_components)
        pipeline = make_pipeline(scaler, pca, model)
    else:
        pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    end_time = time.time()
    exec_time = end_time - start_time
    print("Completed train_naive_bayes in", round(exec_time/60, 2), "minutes.")
    return pipeline, val_predictions, exec_time

def train_classifier(X_train, y_train, X_val, y_val, model_type='logistic', use_pca=True):
    print(f"Training classifier of type: {model_type}")
    if model_type == 'logistic':
        return train_logistic_regression(X_train, y_train, X_val, y_val, use_pca=use_pca)
    elif model_type == 'random_forest':
       return train_random_forest(X_train, y_train, X_val, y_val, use_pca=use_pca)
    elif model_type == 'svm':
        return train_svm(X_train, y_train, X_val, y_val, use_pca=use_pca)
    elif model_type == 'sgd':
        return train_sgd_classifier(X_train, y_train, X_val, y_val, use_pca=use_pca)
    elif model_type == 'knn':
       return train_knn(X_train, y_train, X_val, y_val, use_pca=use_pca)
    elif model_type == 'linear_svm':
        return train_linear_svm(X_train, y_train, X_val, y_val, use_pca=use_pca)
    elif model_type == 'naive_bayes':
        return train_naive_bayes(X_train, y_train, X_val, y_val, use_pca=use_pca)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    try:
        print("Preparing data...")
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
        print("Data prepared.")

        feature_methods = ['hog', 'flatten'] # 'sift'
        classifiers = ['naive_bayes', 'sgd', 'logistic', 'linear_svm', 'random_forest', 'svm', 'knn'] 
        results = []
        
        for feature_method in feature_methods:
            print(f"Extracting features using method: {feature_method}")
            X_train_features = extract_features(X_train, method=feature_method).astype(np.float32)
            X_val_features = extract_features(X_val, method=feature_method).astype(np.float32)
            X_test_features = extract_features(X_test, method=feature_method).astype(np.float32)
            print(f"Features extracted using method: {feature_method}")
            use_pca = True
            if feature_method == 'sift':
                use_pca = True
            for classifier in classifiers:
                try:
                    print(f"Training {classifier} with {feature_method} features...")
                    model, val_predictions, execution_time = train_classifier(X_train_features, y_train, X_val_features, y_val, model_type=classifier, use_pca=use_pca)

                    # Evaluate on validation set
                    report = classification_report(y_val, val_predictions, target_names=LABEL_NAMES, output_dict=True)
                    accuracy = accuracy_score(y_val, val_predictions)
                    results.append({'feature_method': feature_method, 'classifier': classifier, 'report': report, 'accuracy': accuracy, 'train_time': execution_time})
                    print(results[-1]['report'])
                    print(f"Accuracy: {accuracy}")

                    # Save the model
                    save_model(model, feature_method, classifier)
                    print(f"Model {classifier} with {feature_method} features saved.")
                except Exception as e:
                    print(f"Error training {classifier} with {feature_method} features: {e}")

        # Save results to a DataFrame and then to an Excel file
        results_df = pd.DataFrame(results)
        results_df.to_excel('results_logitic_regression.xlsx', index=False)
        print("Results saved to 'results_logitic_regression.xlsx'")
    except Exception as e:
        print(f"An error occurred: {e}")







