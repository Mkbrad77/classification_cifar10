# features.py
import os
import sys

from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.getenv('PYTHONPATH'))
import numpy as np
from skimage.feature import hog
from skimage import color
import cv2
import cv2
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from cifar10_classification.dataset import prepare_data
from cifar10_classification.plots import display_hog_images
from sklearn.cluster import KMeans, MiniBatchKMeans
from cifar10_classification.config  import DATA_DIR, PROCESSED_DATA_DIR
# methode d'extraction bag of words  + histogramme descripteur sift 
def extract_sift_features(images, num_clusters=50):
    sift = cv2.SIFT_create()
    sift_features = []
    
    # Collect all descriptors
    all_descriptors = []
    for image in images:
        img = image.reshape(3, 32, 32).transpose(1, 2, 0)
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            all_descriptors.extend(descriptors.astype(np.float32))
    
    # Apply KMeans clustering to find num_clusters clusters in all descriptors
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(np.array(all_descriptors))
    cluster_centers = kmeans.cluster_centers_

    # Create histogram of visual words for each image
    for image in images:
        img = image.reshape(3, 32, 32).transpose(1, 2, 0)
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is None:
            descriptors = np.zeros((1, sift.descriptorSize()), np.float32)
        
        histogram = np.zeros(num_clusters, dtype=np.float32)
        if descriptors is not None:
            cluster_assignments = kmeans.predict(descriptors.astype(np.float32))
            for cluster_idx in cluster_assignments:
                histogram[cluster_idx] += 1
        
        sift_features.append(histogram)
    
    return np.array(sift_features, dtype=np.float32)

# def extract_hog_features(images):
#     hog_features = []
#     for image in images:
#         img = image.reshape(3, 32, 32).transpose(1, 2, 0)
#         gray_image = color.rgb2gray(img)
#         hog_feature = hog(gray_image, pixels_per_cell=(8, 8))
#         hog_features.append(hog_feature)
#     return np.array(hog_features)
def extract_hog_features(images, visualize=False):
    hog_features = []
    hog_images = []
    for image in images:
        img = image.reshape(3, 32, 32).transpose(1, 2, 0)
        gray_image = color.rgb2gray(img)
        if visualize:
            hog_feature, hog_image = hog(gray_image, pixels_per_cell=(8, 8), visualize=True)
            hog_images.append(hog_image)
        else:
            hog_feature = hog(gray_image, pixels_per_cell=(8, 8))
        hog_features.append(hog_feature)
    if visualize:
        return np.array(hog_features), np.array(hog_images)
    else:
        return np.array(hog_features)



def flatten_images(images):
    return images.reshape(images.shape[0], -1)


def extract_features(images, method='hog', visualize=False):
    if method == 'hog':
        return extract_hog_features(images, visualize=visualize)
    elif method == 'sift':
        return extract_sift_features(images)
    elif method == 'flatten':
        return flatten_images(images)
    else:
        raise ValueError(f"Unknown method: {method}")


def load_model(path):
    model = joblib.load(path)
    return model
def save_processed_data(X_train, X_val, X_test, extract_method):
    np.save(os.path.join(PROCESSED_DATA_DIR, f'X_train_{extract_method}.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, f'X_val_{extract_method}.npy'), X_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, f'X_test_{extract_method}.npy'), X_test)

def load_processed_data(extract_method):
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, f'X_train_{extract_method}.npy'))
    X_val = np.load(os.path.join(PROCESSED_DATA_DIR, f'X_val_{extract_method}.npy'))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, f'X_test_{extract_method}.npy'))
    
    return X_train, X_val, X_test
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    X_train_hog, train_hog_image = extract_features(X_train, method='hog', visualize=True)
    X_val_hog, val_hog_image = extract_features(X_val, method='hog', visualize=True)
    X_test_hog, test_hog_image = extract_features(X_test, method='hog', visualize=True)
    save_processed_data(X_train_hog, X_val_hog, X_test_hog, 'hog')
    display_hog_images(X_train, train_hog_image, 'train')
    #print(f"Extracted Features Shape: {features.shape}")

