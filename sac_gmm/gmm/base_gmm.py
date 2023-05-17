import os
import sys
from pathlib import Path

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix())  # sac_gmm
sys.path.insert(0, root.as_posix())  # Root

import numpy as np


class BaseGMM(object):
    """Gaussian Mixture Model.

    Parameters
    ----------
    n_components : int
        Number of components that compose the GMM.

    priors : array-like, shape (n_components,), optional
        Weights of the components.

    means : array-like, shape (n_components, n_features), optional
        Means of the components.

    covariances : array-like, shape (n_components, n_features, n_features), optional
        Covariances of the components.

    """

    def __init__(self, n_components=3, priors=None, means=None, covariances=None, model_dir=None):
        # GMM
        self.n_components = n_components
        self.priors = priors
        self.means = means
        self.covariances = covariances
        self.model_dir = model_dir

        if self.priors is not None:
            self.priors = np.asarray(self.priors)
        if self.means is not None:
            self.means = np.asarray(self.means)
        if self.covariances is not None:
            self.covariances = np.asarray(self.covariances)

    def fit(self, dataset):
        """
        fits a GMM on demonstrations
        Args:
            dataset: skill demonstrations
        """
        raise NotImplementedError

    def predict(self, x):
        """Infers the remaining state at the partially defined point x.

        Args:
            x (np.array): Partial point to infer remaining state at.

        Returns:
            np.array: Inferred remaining state at x.
        """

        raise NotImplementedError

    def save_model(self):
        filename = self.model_dir + "/gmm_params.npy"
        np.save(
            filename,
            {
                "priors": self.priors,
                "mu": self.means,
                "sigma": self.covariances,
            },
        )
        self.logger.info(f"Saved GMM params at {filename}")

    def load_model(self):
        filename = self.model_dir + "/gmm_params.npy"
        self.logger.info(f"Loading GMM params from {filename}")

        gmm = np.load(filename, allow_pickle=True).item()

        self.priors = np.array(gmm["priors"])
        self.means = np.array(gmm["mu"])
        self.covariances = np.array(gmm["sigma"])
