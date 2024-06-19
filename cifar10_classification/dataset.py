
# dataset.py
import os
import sys

from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.getenv('PYTHONPATH'))
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from cifar10_classification.config import DATA_DIR, SEED, PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, EXTERNAL_DATA_DIR

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10(data_dir=DATA_DIR):
    data_batches = [unpickle(os.path.join(data_dir, f"data_batch_{i}")) for i in range(1, 6)]
    test_batch = unpickle(os.path.join(data_dir, "test_batch"))
    
    X_train = np.vstack([batch[b'data'] for batch in data_batches])
    y_train = np.hstack([batch[b'labels'] for batch in data_batches])
    
    X_test = test_batch[b'data']
    y_test = test_batch[b'labels']
    
    return (X_train, y_train), (X_test, y_test)


def save_interim_data(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    np.save(os.path.join(INTERIM_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(INTERIM_DATA_DIR, 'Y_train.npy'), Y_train)
    np.save(os.path.join(INTERIM_DATA_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(INTERIM_DATA_DIR, 'Y_val.npy'), Y_val)
    np.save(os.path.join(INTERIM_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(INTERIM_DATA_DIR, 'Y_test.npy'), Y_test)

def load_interim_data():
    X_train = np.load(os.path.join(INTERIM_DATA_DIR, 'X_train.npy'))
    Y_train = np.load(os.path.join(INTERIM_DATA_DIR, 'Y_train.npy'))
    X_val = np.load(os.path.join(INTERIM_DATA_DIR, 'X_val.npy'))
    Y_val = np.load(os.path.join(INTERIM_DATA_DIR, 'Y_val.npy'))
    X_test = np.load(os.path.join(INTERIM_DATA_DIR, 'X_test.npy'))
    Y_test = np.load(os.path.join(INTERIM_DATA_DIR, 'Y_test.npy'))
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def save_external_data(X_train, X_val, X_test, extract_method):
    np.save(os.path.join(EXTERNAL_DATA_DIR, 'X_train_' + extract_method + '.npy'), X_train)
    np.save(os.path.join(EXTERNAL_DATA_DIR, 'X_val_' + extract_method + '.npy'), X_val)
    np.save(os.path.join(EXTERNAL_DATA_DIR, 'X_test_' + extract_method + '.npy'), X_test)

def load_external_data(extract_method):
    X_train = np.load(os.path.join(EXTERNAL_DATA_DIR, 'X_train_' + extract_method + '.npy'))
    X_val = np.load(os.path.join(EXTERNAL_DATA_DIR, 'X_val_' + extract_method + '.npy'))
    X_test = np.load(os.path.join(EXTERNAL_DATA_DIR, 'X_test_' + extract_method + '.npy'))
    
    return X_train, X_val, X_test

def prepare_data():
    (X_train, y_train), (X_test, y_test) = load_cifar10()
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
    save_interim_data(X_train, y_train, X_val, y_val, X_test, y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    #save_interim_data(X_train, y_train, X_val, y_val, X_test, y_test)