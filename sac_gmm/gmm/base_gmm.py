import os
import sys
from pathlib import Path
from sac_gmm.gmm.utils.plot_utils import visualize_3d_gmm

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

    def __init__(self, n_components=3, priors=None, means=None, covariances=None, plot=None, model_dir=None):
        # GMM
        self.n_components = n_components
        self.priors = priors
        self.means = means
        self.covariances = covariances
        self.plot = plot
        self.model_dir = model_dir

        self.name = "GMM"

        # Data
        self.dataset = None
        self.state_type = None
        self.dim = None
        self.data = None

        if self.priors is not None:
            self.priors = np.asarray(self.priors)
        if self.means is not None:
            self.means = np.asarray(self.means)
        if self.covariances is not None:
            self.covariances = np.asarray(self.covariances)

    def preprocess_data(self, dataset, obj_type=True, normalize=False):
        # Stack position and velocity data
        demos_xdx = [np.hstack([dataset.X[i], dataset.dX[i]]) for i in range(dataset.X.shape[0])]
        # Stack demos
        demos = demos_xdx[0]
        for i in range(1, dataset.X.shape[0]):
            demos = np.vstack([demos, demos_xdx[i]])

        X = demos[:, : self.dim]
        Y = demos[:, self.dim :]

        data = np.hstack((X, Y))

        if obj_type:
            data = self.float_to_object(data)

        return data

    def float_to_object(self, param):
        data = np.empty((param.shape[0], 2), dtype=object)
        for n in range(param.shape[0]):
            data[n] = [param[n, : self.dim], param[n, self.dim :]]
        return data

    def set_data_params(self, dataset, obj_type=True):
        self.dataset = dataset
        self.state_type = self.dataset.state_type
        self.dim = self.dataset.X.numpy().shape[-1]
        self.data = self.preprocess_data(dataset, obj_type=obj_type, normalize=False)

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

    def plot_gmm(self, obj_type=True):
        if not obj_type:
            means = self.float_to_object(self.means)

        # Pick 15 random datapoints from X to plot
        rand_idx = np.random.choice(np.arange(1, len(self.dataset.X)), size=15, replace=False, p=None)
        plot_data = self.dataset.X[rand_idx[0]].numpy()
        for i in rand_idx[1:]:
            plot_data = np.vstack([plot_data, self.dataset.X[i].numpy()])

        plot_means = np.empty((self.n_components, 3))
        for i in range(plot_means.shape[0]):
            for j in range(plot_means.shape[1]):
                plot_means[i, j] = means[i, 0][j]

        temp = self.covariances[:, : self.dim, : self.dim]
        plot_covariances = np.empty((self.n_components, 3))
        for i in range(plot_covariances.shape[0]):
            for j in range(plot_covariances.shape[1]):
                plot_covariances[i, j] = temp[i][j, j]

        return visualize_3d_gmm(
            points=plot_data,
            w=self.priors,
            mu=plot_means.T,
            stdev=plot_covariances.T,
            skill=self.dataset.skill,
            export_dir=self.model_dir,
            export_type="gif",
        )
