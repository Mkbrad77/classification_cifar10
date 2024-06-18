# features.py
import numpy as np
from skimage.feature import hog
from skimage import color
import cv2
import cv2
import numpy as np
import joblib
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
from cifar10_classification.config  import DATA_DIR
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

def extract_hog_features(images):
    hog_features = []
    for image in images:
        img = image.reshape(3, 32, 32).transpose(1, 2, 0)
        gray_image = color.rgb2gray(img)
        hog_feature = hog(gray_image, pixels_per_cell=(8, 8))
        hog_features.append(hog_feature)
    return np.array(hog_features)

def flatten_images(images):
    return images.reshape(images.shape[0], -1)

# def extract_sift_features(images):
#     sift = cv2.SIFT_create()
#     sift_features = []
#     for image in images:
#         img = image.reshape(3, 32, 32).transpose(1, 2, 0)
#         gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         keypoints, descriptors = sift.detectAndCompute(gray_image, None)
#         if descriptors is None:
#             descriptors = np.zeros((1, sift.descriptorSize()), np.float32)
#         sift_features.append(descriptors.flatten())
#     return np.array(sift_features)

def extract_features(images, method='hog'):
    if method == 'hog':
        return extract_hog_features(images)
    elif method == 'sift':
        return extract_sift_features(images)
    elif method == 'flatten':
        return flatten_images(images)
    else:
        raise ValueError(f"Unknown method: {method}")

def save_features(features, method, data_split):
    output_dir = os.path.join(DATA_DIR, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'X_{data_split}_{method}.npy')
    np.save(output_path, features)

def load_model(path):
    model = joblib.load(path)
    return model

if __name__ == "__main__":
    from dataset import prepare_data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    features = extract_features(X_train, method='hog')
    print(f"Extracted Features Shape: {features.shape}")

