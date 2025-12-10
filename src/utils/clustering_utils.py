"""
Clustering utilities for chart analysis.

Provides shared clustering algorithms used across different classifiers and handlers.
"""
import numpy as np
from typing import List

def cluster_1d_dbscan(values: List[float], tolerance: float) -> List[float]:
    """
    Cluster 1D values using DBSCAN for adaptive handling of uneven distributions.
    
    Args:
        values: List of scalar values to cluster.
        tolerance: Epsilon (eps) parameter for DBSCAN (distance threshold).
        
    Returns:
        Sorted list of cluster centers.
    """
    if not values:
        return []
    if len(values) == 1:
        return values
        
    from sklearn.cluster import DBSCAN
    
    arr = np.array(values).reshape(-1, 1)
    
    # Run DBSCAN
    clustering = DBSCAN(eps=tolerance, min_samples=1).fit(arr)
    labels = clustering.labels_
    
    unique_labels = set(labels)
    centers = []
    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = labels == lbl
        centers.append(np.mean(arr[mask]))
    
    return sorted(centers)
