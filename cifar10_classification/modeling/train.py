# train.py
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from cifar10_classification.dataset import prepare_data
from cifar10_classification.features import extract_hog_features, flatten_images
from cifar10_classification.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, N_ESTIMATOR_RANDOM_FOREST, KERNEL, LOSS, MODEL_TYPES
from sklearn.linear_model import SGDClassifier
def train_logistic_regression(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=EPOCHS)
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    #accuracy = accuracy_score(y_val, val_predictions)
    return model, val_predictions

def train_random_forest(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=N_ESTIMATOR_RANDOM_FOREST)
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    #accuracy = accuracy_score(y_val, val_predictions)
    return model, val_predictions

def train_svm(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    model = SVC(kernel=KERNEL, C=1)
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    #accuracy = accuracy_score(y_val, val_predictions)
    return model, val_predictions

def train_sgd_classifier(X_train, y_train, X_val, y_val, loss=LOSS): #The 'loss' parameter of SGDClassifier must be a str among {'modified_huber', 'epsilon_insensitive', 'hinge', 'huber', 'squared_hinge', 'log_loss', 'squared_epsilon_insensitive', 'perceptron', 'squared_error'}
    scaler = StandardScaler()
    model = SGDClassifier(loss=loss, max_iter=1000, tol=1e-3)
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
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
    np.save('/home/mkbrad7/afs_epita/ING2/ML_reconnaissance_de_forme/Projet/classifiaction_cifar/data/processed/X_train_hog.npy', X_train_hog)
    np.save('/home/mkbrad7/afs_epita/ING2/ML_reconnaissance_de_forme/Projet/classifiaction_cifar/data/processed/X_val_hog.npy', X_val_hog)
    model, val_predictions = train_classifier(X_train_hog, y_train, X_val_hog, y_val, model_type=MODEL_TYPES[0])
    print(classification_report(y_val, val_predictions))

if __name__ == '__main__':
    main()

