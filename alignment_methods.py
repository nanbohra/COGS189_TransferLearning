import numpy as np
from scipy.linalg import sqrtm
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm

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


def riemannian_alignment(X):
    """
    Apply Riemannian Alignment (RA) to EEG epoch data.

    Like Euclidean Alignment, this recenters each subject's epochs by applying
    R^(-1/2) @ epoch, but R is the Riemannian (geometric) mean of the epoch
    covariance matrices rather than the arithmetic mean.

    Covariance matrices are Symmetric Positive Definite (SPD) and live on a
    curved manifold -- the arithmetic mean can leave the manifold, while the
    Riemannian mean stays on it and is a more geometrically correct "center".
    This typically improves cross-subject generalization.

    Parameters
    ----------
    X : np.ndarray, shape (n_epochs, n_channels, n_times)

    Returns
    -------
    X_aligned : np.ndarray, shape (n_epochs, n_channels, n_times)
    """
    # compute covariance matrix for each epoch: shape (n_epochs, n_channels, n_channels)
    cov_matrices = np.array([np.cov(epoch) for epoch in X])

    # compute Riemannian (geometric) mean of covariance matrices
    # iterative algorithm (Karcher flow) -- stays on the SPD manifold
    R = mean_riemann(cov_matrices)

    # compute inverse square root of Riemannian mean
    R_inv_sqrt = invsqrtm(R)

    # apply alignment to each epoch
    X_aligned = np.array([R_inv_sqrt @ epoch for epoch in X])

    return X_aligned
