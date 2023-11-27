import logging
import os
import sys
import wandb
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only
import time
import copy

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix())  # sac_gmm
sys.path.insert(0, root.as_posix())  # Root

import numpy as np
import riepybdlib.manifold as rm
from riepybdlib.angular_representations import Quaternion
import riepybdlib.statistics as rs


from sac_gmm.gmm.utils.plot_utils import visualize_3d_gmm
from sac_gmm.gmm.base_gmm import BaseGMM


logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class RiepybdlibGMM(BaseGMM):
    def __init__(self, skill, plot, gmm_type):
        super(RiepybdlibGMM, self).__init__(
            n_components=skill.n_components, plot=plot, model_dir=skill.skills_dir, gmm_type=gmm_type
        )

        if gmm_type not in [1, 2, 3, 4, 5]:
            raise ValueError(f"RiepybdlibGMM only supports gmm_type 1 and 2, 3, 4 and 5 not {gmm_type}")

        self.name = "RiepybdlibGMM"
        self.skill = skill.skill

        self.manifold = None
        self.manifold2 = None  # Only applies to gmm type 4
        self.logger = None
        self.model_dir = os.path.join(Path(skill.skills_dir).expanduser(), skill.skill, self.name, f"type{gmm_type}")
        os.makedirs(self.model_dir, exist_ok=True)
        self.gmm = None
        self.gmm2 = None

    def make_manifold(self):
        manifold, manifold2 = None, None
        eucl3 = rm.get_euclidean_manifold(3, "3d-eucl")
        quat = rm.get_quaternion_manifold("orientation")
        if self.gmm_type in [1, 2]:
            manifold = eucl3 * eucl3
        elif self.gmm_type == 3:
            manifold = eucl3 * eucl3 * quat
        elif self.gmm_type in [4, 5]:
            manifold = eucl3 * eucl3
            manifold2 = eucl3 * quat
        return manifold, manifold2

    def preprocess_data(self, dataset):
        data_X_pos = dataset.X_pos.numpy().reshape(-1, 3)
        data_dX_pos = dataset.dX_pos.numpy().reshape(-1, 3)

        data = [(data_X_pos[i], data_dX_pos[i]) for i in range(data_X_pos.shape[0])]

        if self.gmm_type == 5:
            data_X_ori = dataset.X_ori.numpy().reshape(-1, 4)
            data2 = [
                (data_X_pos[i], Quaternion(data_X_ori[i, 3], data_X_ori[i, :3])) for i in range(data_X_pos.shape[0])
            ]
        elif self.gmm_type == 3:
            data_X_ori = dataset.X_ori.numpy().reshape(-1, 4)
            data = [
                (data_X_pos[i], data_dX_pos[i], Quaternion(data_X_ori[i, 3], data_X_ori[i, :3]))
                for i in range(data_X_pos.shape[0])
            ]
            data2 = None
        else:
            data2 = None
        return data, data2

    def fit(self, dataset):
        # Dataset
        self.manifold, self.manifold2 = self.make_manifold()
        # self.set_data_params(dataset)
        # self.data, self.data2 = self.get_reshaped_data()
        self.dataset = dataset
        data, data2 = self.preprocess_data(dataset)

        # K-Means
        self.gmm = rs.GMM(self.manifold, self.n_components)
        self.gmm.kmeans(data)

        # GMM
        log_rank_0(f"Type {self.gmm_type} Riepybdlib GMM with K-Means priors")
        start = time.time()
        lik, avglik = self.gmm.fit(data, reg_lambda=1e-2)
        log_rank_0(f"Type {self.gmm_type} First GMM train time: {time.time() - start} seconds")

        # Train another GMM if type 4 or 5
        if self.gmm_type in [4, 5] and self.manifold2 is not None and data2 is not None:
            self.gmm2 = rs.GMM(self.manifold2, self.n_components)
            self.gmm2.kmeans(data2)
            log_rank_0(f"Type {self.gmm_type} Second Riepybdlib GMM with K-Means priors")
            start = time.time()
            lik, avglik = self.gmm2.fit(data2, reg_lambda=1e-2)
            log_rank_0(f"Type {self.gmm_type} Second GMM train time: {time.time() - start} seconds")

        # Save GMM params
        # self.reshape_params(to="generic")  # Only applies to gmm type 1 and 2
        self.save_model()

        # Plot GMM
        if self.plot:
            outfile = self.plot_gmm()

        self.logger.log_table(key="fit", columns=["GMM"], data=[[wandb.Video(outfile)]])

    def save_model(self):
        self.gmm.save(f"{self.model_dir}/gmm")
        log_rank_0(f"Saved GMM params at {self.model_dir}/gmm")
        if self.gmm_type in [4, 5]:
            log_rank_0(f"Saved GMM params at {self.model_dir}/gmm2")
            self.gmm2.save(f"{self.model_dir}/gmm2")

    def load_model(self):
        self.manifold, self.manifold2 = self.make_manifold()
        log_rank_0(f"Loading GMM params from {self.model_dir}/gmm")
        self.gmm = rs.GMM(self.manifold, self.n_components).load(
            f"{self.model_dir}/gmm", self.n_components, self.manifold
        )
        if self.gmm_type in [4, 5]:
            log_rank_0(f"Loading second GMM params from {self.model_dir}/gmm2")
            self.gmm2 = rs.GMM(self.manifold2, self.n_components).load(
                f"{self.model_dir}/gmm2", self.n_components, self.manifold2
            )

    def update_model(self, delta):
        # Priors
        if "priors" in delta:
            delta_priors = delta["priors"].reshape(self.gmm.priors.shape)
            self.gmm.priors += delta_priors
            self.gmm.priors[self.gmm.priors < 0] = 0
            self.gmm.priors /= self.gmm.priors.sum()

        # Means
        if "mu" in delta:
            if self.gmm_type in [1, 4]:
                idx = 0
                for i in range(len(self.gmm.gaussians)):
                    self.gmm.gaussians[i].mu = (
                        self.gmm.gaussians[i].mu[0] + delta["mu"][idx : idx + 3],
                        self.gmm.gaussians[i].mu[1] + delta["mu"][idx + 3 : idx + 6],
                    )
                    idx += 6
            elif self.gmm_type in [2, 5]:
                # Update position and next position means with the same delta
                idx = 0
                for i in range(len(self.gmm.gaussians)):
                    self.gmm.gaussians[i].mu = (
                        self.gmm.gaussians[i].mu[0] + delta["mu"][idx : idx + 3],
                        self.gmm.gaussians[i].mu[1] + delta["mu"][idx : idx + 3],
                    )
                    idx += 3
            else:
                if "quat" not in delta:
                    # Only update position means for now
                    idx = 0
                    for i in range(len(self.gmm.gaussians)):
                        self.gmm.gaussians[i].mu = (
                            self.gmm.gaussians[i].mu[0] + delta["mu"][idx : idx + 3],
                            self.gmm.gaussians[i].mu[1] + delta["mu"][idx : idx + 3],
                            self.gmm.gaussians[i].mu[2],
                        )
                        idx += 3
                else:
                    # Update position and orientation means
                    idx = 0
                    quat_idx = 0
                    quat_manifold = rm.get_quaternion_manifold("orientation")
                    for i in range(len(self.gmm.gaussians)):
                        self.gmm.gaussians[i].mu = (
                            self.gmm.gaussians[i].mu[0] + delta["mu"][idx : idx + 3],
                            self.gmm.gaussians[i].mu[1] + delta["mu"][idx : idx + 3],
                            quat_manifold.exp(delta["quat"][quat_idx : quat_idx + 4], base=self.gmm.gaussians[i].mu[2]),
                        )
                        idx += 3
                        quat_idx += 4

    def copy_model(self, gmm_obj):
        self.gmm = copy.deepcopy(gmm_obj.gmm)

    def model_params(self, cov=False):
        priors = self.gmm.priors
        if self.gmm_type in [1, 2, 4, 5]:
            means = np.array([x.mu for x in self.gmm.gaussians]).flatten()
            params = np.concatenate((priors, means), axis=-1)
        else:
            means_pos = np.array([x.mu[:2] for x in self.gmm.gaussians]).flatten()
            means_ori = np.array([x.mu[2].to_nparray() for x in self.gmm.gaussians]).flatten()
            params = np.concatenate((priors, means_pos, means_ori), axis=-1)
        return params

    def get_params_size(self):
        """
        This function is used to set the refine actor's input size
        """
        if self.gmm_type == 3:
            # Only update position means for now
            means_pos_size = np.array([x.mu[:2] for x in self.gmm.gaussians]).flatten().size
            means_ori_size = np.array([x.mu[2].to_nparray() for x in self.gmm.gaussians]).flatten().size
            means_size = means_pos_size + means_ori_size
        else:
            means_size = np.array([x.mu for x in self.gmm.gaussians]).flatten().size
        cov_size = np.array([x.sigma for x in self.gmm.gaussians]).flatten().size
        return self.gmm.priors.size, means_size, cov_size

    def predict1(self, x):
        return self.gmm.gmr(x[:3], i_in=0, i_out=1)[0].mu

    def predict2(self, x):
        return self.predict1(x)

    def predict3(self, x):
        out = self.gmm.gmr(x[:3], i_in=0, i_out=[1, 2])[0].mu
        dx_ori = out[1].to_nparray()
        dx_ori = np.concatenate([dx_ori[1:], [dx_ori[0]]])
        return out[0], dx_ori

    def predict4(self, x):
        dx_ori = self.gmm2.gmr(x[:3], i_in=0, i_out=1)[0].mu.to_nparray()
        dx_ori = np.concatenate([dx_ori[1:], [dx_ori[0]]])  # Scalar last
        return self.predict1(x), dx_ori

    def predict5(self, x):
        return self.predict4(x)

    def plot_gmm(self):
        points = self.dataset.X_pos.numpy()
        rand_idx = np.random.choice(np.arange(0, len(points)), size=15)
        points = np.vstack(points[rand_idx, :, :])
        means = np.array([m.mu[0] for m in self.gmm.gaussians])
        covariances = np.array([m.sigma[:3, :3] for m in self.gmm.gaussians])
        priors = self.gmm.priors

        return visualize_3d_gmm(
            points=points,
            priors=priors,
            means=means,
            covariances=covariances,
            save_dir=self.model_dir,
        )
