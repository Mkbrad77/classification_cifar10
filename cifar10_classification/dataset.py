
# dataset.py

import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from cifar10_classification.config import DATA_DIR, SEED

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

def prepare_data():
    (X_train, y_train), (X_test, y_test) = load_cifar10()
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
    np.save('/home/mkbrad7/afs_epita/ING2/ML_reconnaissance_de_forme/Projet/classifiaction_cifar/data/interim/X_train.npy', X_train)
    np.save('/home/mkbrad7/afs_epita/ING2/ML_reconnaissance_de_forme/Projet/classifiaction_cifar/data/interim/y_train.npy', y_train)
    np.save('/home/mkbrad7/afs_epita/ING2/ML_reconnaissance_de_forme/Projet/classifiaction_cifar/data/interim/X_val.npy', X_val)
    np.save('/home/mkbrad7/afs_epita/ING2/ML_reconnaissance_de_forme/Projet/classifiaction_cifar/data/interim/y_val.npy', y_val)
    np.save('/home/mkbrad7/afs_epita/ING2/ML_reconnaissance_de_forme/Projet/classifiaction_cifar/data/interim/X_test.npy', X_test)
    np.save('/home/mkbrad7/afs_epita/ING2/ML_reconnaissance_de_forme/Projet/classifiaction_cifar/data/interim/y_test.npy', y_test)

    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    
    # print(f"Train: {X_train.shape}, {y_train.shape}")
    # print(f"Validation: {X_val.shape}, {y_val.shape}")
    # print(f"Test: {X_test.shape}, {y_test.shape}")