# predict.py
import numpy as np
from cifar10_classification.dataset import prepare_data
from cifar10_classification.features import extract_hog_features, flatten_images
from cifar10_classification.modeling.train import train_logistic_regression, train_random_forest, train_svm
from sklearn.metrics import accuracy_score


# predict.py
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
from cifar10_classification.config import MODEL_DIR

def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    return joblib.load(model_path)

def predict_model(model, X_test):
    predictions = model.predict(X_test)
    return predictions

def evaluate_model(model, X_test, y_test):
    predictions = predict_model(model, X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    X_train_flat = flatten_images(X_train)
    X_val_flat = flatten_images(X_val)
    X_test_flat = flatten_images(X_test)
    model = load_model('logistic_regression.pkl')
    accuracy, report = evaluate_model(model, X_test_flat, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Classification report:\n{report}")

if __name__ == "__main__":
    main()