import numpy as np
from scipy.linalg import sqrtm, svd
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm


class SRM:
    """
    Lightweight SRM — mirrors brainiak.funcalign.srm.SRM API.
    """

    def __init__(self, n_iter=10, features=20):
        self.n_iter = n_iter
        self.features = features
        self.w_ = None  # list of (n_ch, n_features) orthogonal bases

    def fit(self, X_list):
        n_subjects = len(X_list)
        k = self.features

        # initialise W via SVD of each subject
        self.w_ = []
        for X in X_list:
            U, _, Vt = svd(X, full_matrices=False)
            self.w_.append(U[:, :k])

        for _ in range(self.n_iter):
            # shared response
            S = sum(W.T @ X for W, X in zip(self.w_, X_list)) / n_subjects
            # update W matrices
            self.w_ = []
            for X in X_list:
                M = X @ S.T
                U, _, Vt = svd(M, full_matrices=False)
                self.w_.append(U @ Vt)

        return self

    def transform(self, X_list):
        return [W.T @ X for W, X in zip(self.w_, X_list)]


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


def srm_alignment(X_list, n_features=20):
    """
    Apply Shared Response Modeling (SRM) to EEG data from multiple subjects.

    Parameters
    ----------
    X_list : list of np.ndarray, each shape (n_channels, n_times_total)
             Note: SRM requires subjects to have the same number of time points.
    n_features : int
                 The dimension of the shared feature space (latent space).

    Returns
    -------
    X_shared : list of np.ndarray, each shape (n_features, n_times_total)
    srm      : fitted SRM instance (needed to access srm.w_ for test projection)
    """
    srm = SRM(n_iter=10, features=n_features)
    srm.fit(X_list)
    X_shared = srm.transform(X_list)
    return X_shared, srm
