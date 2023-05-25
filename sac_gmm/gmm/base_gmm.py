import os
import sys
from pathlib import Path
import gym
from sac_gmm.gmm.utils.plot_utils import visualize_3d_gmm
from sac_gmm.utils.posdef import isPD, nearestPD

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

    def __init__(
        self, n_components=3, priors=None, means=None, covariances=None, plot=None, model_dir=None, state_size=None
    ):
        # GMM
        self.n_components = n_components
        self.priors = priors
        self.means = means
        self.covariances = covariances
        self.plot = plot
        self.model_dir = model_dir
        self.state_size = state_size

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

        if obj_type and (data.dtype != "O"):
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

    def copy_model(self, dynsys):
        """Copies GMM params to self from the input GMM class object

        Args:
            dynsys (BaseGMM|ManifoldGMM|BayesianGMM): GMM class object

        Returns:
            None
        """
        self.priors = dynsys.priors
        self.means = dynsys.means
        self.covariances = dynsys.covariances

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

    def get_update_range_parameter_space(self):
        """Returns GMM parameters range as a gym.spaces.Dict for the agent to predict

        Returns:
            param_space : gym.spaces.Dict
                Range of GMM parameters parameters
        """
        param_space = {}
        param_space["priors"] = gym.spaces.Box(low=-0.1, high=0.1, shape=(self.priors.size,))
        param_space["mu"] = gym.spaces.Box(low=-0.01, high=0.01, shape=(self.means.size,))
        param_space["sigma"] = gym.spaces.Box(low=-1e-6, high=1e-6, shape=(self.covariances.size,))

        dim = self.means.shape[2] // 2
        num_gaussians = self.means.shape[0]
        sigma_change_size = int(num_gaussians * dim * (dim + 1) / 2 + dim * dim * num_gaussians)
        param_space["sigma"] = gym.spaces.Box(low=-1e-6, high=1e-6, shape=(sigma_change_size,))
        return gym.spaces.Dict(param_space)

    def plot_gmm(self, obj_type=True):
        if not obj_type:
            means = self.float_to_object(self.means)
        else:
            means = self.means

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

    def sample_starts(self, size=1, scale=0.15):
        """Samples starting points from dataset's average starting points.
        At the moment, only works for position.
        """
        start = self.dataset.start
        sampled = np.hstack(
            (
                np.random.normal(loc=start[0], scale=scale, size=size).reshape(size, -1),
                np.random.normal(loc=start[1], scale=scale, size=size).reshape(size, -1),
                np.random.normal(loc=start[2], scale=scale, size=size).reshape(size, -1),
            )
        )
        return sampled
