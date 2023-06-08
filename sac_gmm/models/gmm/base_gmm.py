import logging
import os
import sys
from pathlib import Path
import gym
from sac_gmm.models.gmm.utils.plot_utils import visualize_3d_gmm
from sac_gmm.utils.posdef import isPD, nearestPD
from pytorch_lightning.utilities import rank_zero_only

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix())  # sac_gmm
sys.path.insert(0, root.as_posix())  # Root

import numpy as np


logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


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

    def __init__(
        self, n_components=3, priors=None, means=None, covariances=None, plot=None, model_dir=None, state_type=None
    ):
        # GMM
        self.n_components = n_components
        self.priors = priors
        self.means = means
        self.covariances = covariances
        self.plot = plot
        self.model_dir = model_dir
        self.state_type = state_type

        self.name = "GMM"

        # Data
        self.dataset = None
        self.data = None
        if self.state_type == "pos" or "ori":
            self.dim = 3
        elif self.state_type == "pos_ori":
            self.dim = 6
        else:
            raise NotImplementedError

        if self.priors is not None:
            self.priors = np.asarray(self.priors)
        if self.means is not None:
            self.means = np.asarray(self.means)
        if self.covariances is not None:
            self.covariances = np.asarray(self.covariances)

    def preprocess_data(self, dataset, normalize=False):
        if self.state_type == "pos_ori":
            # Stack position, velocity and quaternion data
            demos_xdx = [np.hstack([dataset.X[i], dataset.dX[i], dataset.Ori[i]]) for i in range(dataset.X.shape[0])]
        else:
            # Stack position and velocity data
            demos_xdx = [np.hstack([dataset.X[i], dataset.dX[i]]) for i in range(dataset.X.shape[0])]

        # Stack demos
        demos = demos_xdx[0]
        for i in range(1, dataset.X.shape[0]):
            demos = np.vstack([demos, demos_xdx[i]])

        X = demos[:, : self.dim]
        Y = demos[:, self.dim : self.dim + self.dim]

        if self.state_type == "pos_ori":
            Y_Ori = demos[:, self.dim + self.dim :]
            data = np.empty((X.shape[0], 3), dtype=object)
            for n in range(X.shape[0]):
                data[n] = [X[n], Y[n], Y_Ori[n]]
        else:
            data = np.hstack((X, Y))

        return data

    def set_data_params(self, dataset):
        self.dataset = dataset
        self.dim = self.dataset.X.numpy().shape[-1]
        self.data = self.preprocess_data(dataset, normalize=False)

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
        log_rank_0(f"Saved GMM params at {filename}")

    def load_model(self):
        filename = self.model_dir + "/gmm_params.npy"
        log_rank_0(f"Loading GMM params from {filename}")

        gmm = np.load(filename, allow_pickle=True).item()

        self.priors = np.array(gmm["priors"])
        self.means = np.array(gmm["mu"])
        self.covariances = np.array(gmm["sigma"])

    def copy_model(self, dynsys):
        """Copies GMM params to self from the input GMM class object

        Args:
            dynsys (BaseGMM|ManifoldGMM|BayesianGMM): GMM class object

        Returns:
            None
        """
        self.priors = np.copy(dynsys.priors)
        self.means = np.copy(dynsys.means)
        self.covariances = np.copy(dynsys.covariances)

    def update_model(self, delta):
        """Updates GMM parameters by given delta changes (i.e. SAC's output)

        Args:
            delta (dict): Changes given by the SAC agent to be made to the GMM parameters

        Returns:
            None
        """
        # Priors
        if "priors" in delta:
            delta_priors = delta["priors"].reshape(self.priors.shape)
            self.priors += delta_priors
            self.priors[self.priors < 0] = 0
            self.priors /= self.priors.sum()

        # Means
        if "mu" in delta:
            delta_means = delta["mu"].reshape(self.means.shape)
            self.means += delta_means

        # Covariances
        if "sigma" in delta:
            d_sigma = delta["sigma"]
            dim = self.means.shape[2] // 2
            num_gaussians = self.means.shape[0]

            # Create sigma_state symmetric matrix
            half_mat_size = int(dim * (dim + 1) / 2)
            for i in range(num_gaussians):
                d_sigma_state = d_sigma[half_mat_size * i : half_mat_size * (i + 1)]
                mat_d_sigma_state = np.zeros((dim, dim))
                mat_d_sigma_state[np.triu_indices(dim)] = d_sigma_state
                mat_d_sigma_state = mat_d_sigma_state + mat_d_sigma_state.T
                mat_d_sigma_state[np.diag_indices(dim)] = mat_d_sigma_state[np.diag_indices(dim)] / 2
                self.sigma[:dim, :dim, i] += mat_d_sigma_state
                if not isPD(self.sigma[:dim, :dim, i]):
                    self.sigma[:dim, :dim, i] = nearestPD(self.sigma[:dim, :dim, i])

            # Create sigma cross correlation matrix
            d_sigma_cc = np.array(d_sigma[half_mat_size * num_gaussians :])
            d_sigma_cc = d_sigma_cc.reshape((dim, dim, num_gaussians))
            self.covariances[dim : 2 * dim, 0:dim] += d_sigma_cc

    def model_params(self, cov=False):
        """Returns GMM priors and means as a flattened vector

        Args:
            None

        Returns:
            params (np.array): GMM params flattened
        """
        priors = self.priors
        means = self.means.flatten()
        params = np.concatenate((priors, means), axis=-1)

        return params

    def plot_gmm(self, obj_type=True):
        # if not obj_type:
        #     means = self.float_to_object(self.means)
        # else:
        #     means = self.means

        self.reshape_params(to="gmr-specific")

        # Pick 15 random datapoints from X to plot
        points = self.dataset.X.numpy()[:, :, :3]
        rand_idx = np.random.choice(np.arange(0, len(points)), size=15)
        points = np.vstack(points[rand_idx, :, :])
        means = np.vstack(self.means[:, 0])
        covariances = self.covariances[:, :3, :3]

        return visualize_3d_gmm(
            points=points,
            priors=self.priors,
            means=means,
            covariances=covariances,
            save_dir=self.model_dir,
        )

    def get_reshaped_means(self):
        """Reshape means from (n_components, 2) to (n_components, 2, state_size)"""
        new_means = np.empty((self.n_components, 2, self.dim))
        for i in range(new_means.shape[0]):
            for j in range(new_means.shape[1]):
                new_means[i, j, :] = self.means[i][j]
        return new_means

    def get_reshape_data(self):
        reshaped_data = np.empty((self.data.shape[0], 2), dtype=object)
        for n in range(self.data.shape[0]):
            reshaped_data[n] = [self.data[n, : self.dim], self.data[n, self.dim :]]
        return reshaped_data
