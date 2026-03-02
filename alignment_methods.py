import numpy as np
from scipy.linalg import sqrtm

def euclidean_alignment(X):
    """
    Apply Euclidean Alignment (EA) to EEG epoch data.
    
    """
    # compute covariance matrix for each epoch
    cov_matrices = np.array([np.cov(epoch) for epoch in X])

    # compute mean covariance across all epochs
    mean_cov = np.mean(cov_matrices, axis=0)
    
    # compute inverse square root of mean covariance
    R_inv_sqrt = np.linalg.inv(sqrtm(mean_cov))
    
    # apply alignment to each epoch
    X_aligned = np.array([R_inv_sqrt @ epoch for epoch in X])
    
    return X_aligned