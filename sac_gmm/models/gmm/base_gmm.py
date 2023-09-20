import logging
import os
import sys
from pathlib import Path
import gym
from sac_gmm.models.gmm.utils.plot_utils import visualize_3d_gmm
from sac_gmm.utils.posdef import isPD, nearestPD
from sac_gmm.models.gmm.utils.rotation_utils import compute_euler_difference
from pytorch_lightning.utilities import rank_zero_only
import pybullet

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

    Type 1: GMM with position as input and velocity as output
    Type 2: GMM with position as input and next position as output
    Type 3: GMM with position as input and next position and next orientation as output
    Type 4: One GMM with position as input and velocity as output and
            another GMM with position as input and orientation as output

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
        self, n_components=3, priors=None, means=None, covariances=None, plot=None, model_dir=None, gmm_type=None
    ):
        # GMM
        self.n_components = n_components
        self.priors = priors
        self.means = means
        self.covariances = covariances
        self.plot = plot
        self.model_dir = model_dir
        self.gmm_type = gmm_type
        self.pos_dt = 0.02
        self.ori_dt = 0.05
        self.goal = None
        self.start = None
        self.fixed_ori = None

        self.name = "GMM"

        # Data
        self.dataset = None
        self.data = None

        if self.priors is not None:
            self.priors = np.asarray(self.priors)
        if self.means is not None:
            self.means = np.asarray(self.means)
        if self.covariances is not None:
            self.covariances = np.asarray(self.covariances)

        # Useful for GMM type 4
        self.data2 = None
        self.priors2 = None
        self.means2 = None
        self.covariances2 = None

    def fit(self, dataset):
        """
        fits a GMM on demonstrations
        Args:
            dataset: skill demonstrations
        """
        raise NotImplementedError

    def predict1(self, x):
        """
        Predict function for GMM type 1
        """
        raise NotImplementedError

    def predict2(self, x):
        """
        Predict function for GMM type 2
        """
        raise NotImplementedError

    def predict3(self, x):
        """
        Predict function for GMM type 3
        """
        raise NotImplementedError

    def predict4(self, x):
        """
        Predict function for GMM type 4
        """
        raise NotImplementedError

    def set_skill_params(self, dataset):
        self.pos_dt = dataset.pos_dt
        self.ori_dt = dataset.ori_dt
        self.fixed_ori = dataset.fixed_ori
        self.goal = dataset.goal
        self.start = dataset.start

    def set_data_params(self, dataset):
        self.dataset = dataset
        self.data, self.data2 = self.preprocess_data()

    def preprocess_data(self):
        data, data2 = None, None
        # Data size
        data_size = self.dataset.X_pos.shape[0]
        # Horizontal stack demos
        if self.gmm_type in [1, 2, 4]:
            # Stack X_pos and dX_pos data
            demos_xdx = [
                np.hstack([self.dataset.X_pos[i], self.dataset.dX_pos[i]]) for i in range(self.dataset.X_pos.shape[0])
            ]
        else:
            # Stack X_pos, dX_pos and X_ori data
            demos_xdx = [
                np.hstack([self.dataset.X_pos[i], self.dataset.dX_pos[i], self.dataset.X_ori[i]])
                for i in range(data_size)
            ]
        # Vertical stack demos
        demos = demos_xdx[0]
        for i in range(1, data_size):
            demos = np.vstack([demos, demos_xdx[i]])

        X_pos = demos[:, :3]
        dX_pos = demos[:, 3:6]

        if self.gmm_type in [1, 2, 4]:
            data = np.hstack((X_pos, dX_pos))
        else:
            X_ori = demos[:, 6:]
            data = np.empty((X_pos.shape[0], 3), dtype=object)
            for n in range(X_pos.shape[0]):
                data[n] = [X_pos[n], dX_pos[n], X_ori[n]]

        # Data for the second GMM
        if self.gmm_type == 4:
            # Stack X_pos and X_ori data
            demos_xdx = [np.hstack([self.dataset.X_pos[i], self.dataset.X_ori[i]]) for i in range(data_size)]
            demos = demos_xdx[0]
            for i in range(1, data_size):
                demos = np.vstack([demos, demos_xdx[i]])
            X_pos = demos[:, :3]
            X_ori = demos[:, 3:]
            data2 = np.hstack((X_pos, X_ori))

        return data, data2

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
        if self.gmm_type == 4:
            filename = self.model_dir + "/gmm_params2.npy"
            np.save(
                filename,
                {
                    "priors": self.priors2,
                    "mu": self.means2,
                    "sigma": self.covariances2,
                },
            )
            log_rank_0(f"Saved second GMM params at {filename}")

    def load_model(self):
        filename = self.model_dir + "/gmm_params.npy"
        log_rank_0(f"Loading GMM params from {filename}")

        gmm = np.load(filename, allow_pickle=True).item()

        self.priors = np.array(gmm["priors"])
        self.means = np.array(gmm["mu"])
        self.covariances = np.array(gmm["sigma"])
        if self.gmm_type == 4:
            filename = self.model_dir + "/gmm_params2.npy"
            log_rank_0(f"Loading second GMM params from {filename}")

            gmm = np.load(filename, allow_pickle=True).item()

            self.priors2 = np.array(gmm["priors"])
            self.means2 = np.array(gmm["mu"])
            self.covariances2 = np.array(gmm["sigma"])

    def copy_model(self, gmm):
        """Copies GMM params to self from the input GMM class object

        Args:
            gmm (BaseGMM|ManifoldGMM|BayesianGMM): GMM class object

        Returns:
            None
        """
        self.priors = np.copy(gmm.priors)
        self.means = np.copy(gmm.means)
        self.covariances = np.copy(gmm.covariances)

        if gmm.gmm_type == 4:
            self.priors2 = np.copy(gmm.priors2)
            self.means2 = np.copy(gmm.means2)
            self.covariances2 = np.copy(gmm.covariances2)

    def model_params(self, cov=False):
        """Returns GMM priors and means as a flattened vector

        Args:
            None

        Returns:
            params (np.array): GMM params flattened
        """
        priors = self.priors
        means = self.means.flatten()
        # params = np.concatenate((priors, means), axis=-1)
        params = priors
        for x in means:
            params = np.append(params, x)

        return params

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
            if self.gmm_type in [1, 2]:
                delta_means = delta["mu"].reshape(self.means.shape)
                self.means += delta_means
            elif self.gmm_type == 3:
                # Only update position and next position means
                idx = 0
                for i in range(self.means.shape[0]):
                    self.means[i, 0] += delta["mu"][idx : idx + 3]
                    self.means[i, 1] += delta["mu"][idx : idx + 3]
                    idx += 3
        pass
        # # Covariances
        # if "sigma" in delta:
        #     d_sigma = delta["sigma"]
        #     dim = self.means.shape[2] // 2
        #     num_gaussians = self.means.shape[0]

        #     # Create sigma_state symmetric matrix
        #     half_mat_size = int(dim * (dim + 1) / 2)
        #     for i in range(num_gaussians):
        #         d_sigma_state = d_sigma[half_mat_size * i : half_mat_size * (i + 1)]
        #         mat_d_sigma_state = np.zeros((dim, dim))
        #         mat_d_sigma_state[np.triu_indices(dim)] = d_sigma_state
        #         mat_d_sigma_state = mat_d_sigma_state + mat_d_sigma_state.T
        #         mat_d_sigma_state[np.diag_indices(dim)] = mat_d_sigma_state[np.diag_indices(dim)] / 2
        #         self.sigma[:dim, :dim, i] += mat_d_sigma_state
        #         if not isPD(self.sigma[:dim, :dim, i]):
        #             self.sigma[:dim, :dim, i] = nearestPD(self.sigma[:dim, :dim, i])

        #     # Create sigma cross correlation matrix
        #     d_sigma_cc = np.array(d_sigma[half_mat_size * num_gaussians :])
        #     d_sigma_cc = d_sigma_cc.reshape((dim, dim, num_gaussians))
        #     self.covariances[dim : 2 * dim, 0:dim] += d_sigma_cc

    def plot_gmm(self, obj_type=True):
        points = self.dataset.X_pos.numpy()
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

    def reshape_params(self, to="generic"):
        """Reshapes model params to/from generic/gmr-specific shapes.
        E.g., For N GMM components, S state size, generic shapes are
        self.priors = (N,);
        self.means = (N, 2*S);
        self.covariances = (N, 2*S, 2*S)

        Gmr-specific: self.means = (N, 2, S)
        """
        if self.gmm_type in [1, 2]:
            # priors and covariances already match shape
            shape = None
            if to == "generic":
                shape = (self.n_components, 2 * 3)
            else:
                shape = (self.n_components, 2, 3)
            self.means = self.means.reshape(shape)

    def get_reshaped_means(self):
        """Reshape means from (n_components, 2) to (n_components, 2, state_size)"""
        if self.gmm_type in [1, 2]:
            new_means = np.empty((self.n_components, 2, 3))
            for i in range(new_means.shape[0]):
                for j in range(new_means.shape[1]):
                    new_means[i, j, :] = self.means[i][j]
            return new_means
        else:
            return np.copy(self.means)

    def get_reshaped_data(self):
        reshaped_data, reshaped_data2 = None, None
        if self.gmm_type in [1, 2, 4]:
            reshaped_data = np.empty((self.data.shape[0], 2), dtype=object)
            for n in range(self.data.shape[0]):
                reshaped_data[n] = [self.data[n, :3], self.data[n, 3:]]
            if self.gmm_type == 4:
                reshaped_data2 = np.empty((self.data2.shape[0], 2), dtype=object)
                for n in range(self.data.shape[0]):
                    reshaped_data2[n] = [self.data2[n, :3], self.data2[n, 3:]]
            return reshaped_data, reshaped_data2
        else:
            return np.copy(self.data), None

    def predict(self, x):
        if self.gmm_type == 1:
            dx_pos = self.predict1(x[:3])
            target_ori = self.fixed_ori
        elif self.gmm_type == 2:
            next_pos = self.predict2(x[:3])
            dx_pos = (next_pos + self.goal - x[:3]) / self.pos_dt
            target_ori = self.fixed_ori
        elif self.gmm_type == 3:
            out = self.predict3(x[:3])
            if np.isnan(out[0]):
                return np.zeros(3), np.zeros(3), True
            dx_pos = (out[:3] + self.goal - x[:3]) / self.pos_dt
            target_ori = pybullet.getEulerFromQuaternion(out[3:])
        else:
            dx_pos, x_ori = self.predict4(x[:3])
            target_ori = pybullet.getEulerFromQuaternion(x_ori)
        dx_ori = compute_euler_difference(target_ori, x[3:6])
        return dx_pos, dx_ori, False
