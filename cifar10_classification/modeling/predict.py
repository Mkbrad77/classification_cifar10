# predict.py
import numpy as np
import os
import sys

from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.getenv('PYTHONPATH'))
from cifar10_classification.dataset import prepare_data
from cifar10_classification.features import extract_hog_features, flatten_images
from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import roc_auc_score, confusion_matrix
# predict.py
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
from cifar10_classification.config import MODEL_DIR

# def load_model(model_name):
#     model_path = os.path.join(MODEL_DIR, model_name)
#     return joblib.load(model_path)
def load_model(path):
    model = joblib.load(path)
    return model
def predict_model(model, X_test):
    predictions = model.predict(X_test)
    return predictions

def evaluate_model(model, X_test, y_test):
    predictions = predict_model(model, X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=1)
    #roc_auc = roc_auc_score(y_test, predictions)
    return accuracy, report, cm

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    X_train_flat = flatten_images(X_train)
    X_val_flat = flatten_images(X_val)
    X_test_flat = flatten_images(X_test)
    X_test_hog = extract_hog_features(X_test)
    model = load_model(MODEL_DIR + '/svm_hog_best_model.pkl')
    print("resultat du model svm avec hog")
    accuracy, report, cm = evaluate_model(model, X_test_hog, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Classification report:\n{report}")

if __name__ == "__main__":
    main()
