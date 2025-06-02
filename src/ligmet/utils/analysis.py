import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score
from scipy.spatial.distance import cdist

def compute_precision_recall(predicted_coords, true_coords, dist_threshold=2.0):
    pdb_precisions = []
    pdb_recalls = []

    if len(predicted_coords) == 0:
        return 0.0, 0.0 if len(true_coords) > 0 else (1.0, 1.0)
    
    tp = 0
    fp = 0
    fn = 0
    matched_true = set()
    for i, pred_coord in enumerate(predicted_coords):
        distances = np.linalg.norm(true_coords - pred_coord, axis=1)
        if np.any(distances < dist_threshold):
            tp += 1
            matched_true.add(np.argmin(distances))
        else:
            fp += 1
    fn = len(true_coords) - len(matched_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    pdb_precisions.append(precision)
    pdb_recalls.append(recall)
    
    mean_precision = np.mean(pdb_precisions)
    mean_recall = np.mean(pdb_recalls)
    
    return mean_precision, mean_recall



def dbscan_clustering(coords, preds, eps=2.0, min_samples=2, method='max'):
    """
    Perform DBSCAN clustering on the given coordinates and return cluster representatives.

    Parameters:
        coords (np.ndarray): Coordinates to cluster, shape (N, 3)
        preds (np.ndarray): Prediction scores corresponding to each coordinate, shape (N,)
        eps (float): DBSCAN eps parameter
        min_samples (int): DBSCAN min_samples parameter
        method (str): 'max' to return the coord with max pred per cluster,
                      'mean' to return the mean coord per cluster

    Returns:
        np.ndarray: Representative coordinates per cluster
        np.ndarray: Cluster labels for all coordinates (same length as `coords`)
    """
    if len(coords) == 0:
        return np.array([]), np.array([])

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    unique_labels = set(labels)
    representatives = []

    for label in unique_labels:
        if label == -1:
            continue  # skip noise
        indices = np.where(labels == label)[0]
        if method == 'max':
            best_idx = indices[np.argmax(preds[indices])]
            representatives.append(coords[best_idx])
        elif method == 'mean':
            cluster_mean = np.mean(coords[indices], axis=0)
            representatives.append(cluster_mean)
        else:
            raise ValueError("method must be 'max' or 'mean'")

    return np.array(representatives), labels
