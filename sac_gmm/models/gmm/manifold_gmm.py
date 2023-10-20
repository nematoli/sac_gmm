import logging
import os
import sys
import wandb
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only
import time

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix())  # sac_gmm
sys.path.insert(0, root.as_posix())  # Root

import numpy as np
from pymanopt.manifolds import Euclidean, Sphere, Product

from sac_gmm.models.gmm.base_gmm import BaseGMM
from sac_gmm.models.gmm.utils.manifold_clustering import manifold_k_means, manifold_gmm_em
from sac_gmm.models.gmm.utils.manifold_gmr import manifold_gmr

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class ManifoldGMM(BaseGMM):
    def __init__(self, skill, plot, gmm_type):
        super(ManifoldGMM, self).__init__(
            n_components=skill.n_components, plot=plot, model_dir=skill.skills_dir, gmm_type=gmm_type
        )

        self.name = "ManifoldGMM"
        self.skill = skill.skill

        self.manifold = None
        self.manifold2 = None  # Only applies to gmm type 4
        self.logger = None
        self.model_dir = os.path.join(Path(skill.skills_dir).expanduser(), skill.skill, self.name, f"type{gmm_type}")
        os.makedirs(self.model_dir, exist_ok=True)

    def make_manifold(self):
        manifold, manifold2 = None, None
        if self.gmm_type in [1, 2]:
            manifold = Product([Euclidean(3), Euclidean(3)])
        elif self.gmm_type == 3:
            manifold = Product([Euclidean(3), Euclidean(3), Sphere(4)])
        elif self.gmm_type in [4, 5]:
            manifold = Product([Euclidean(3), Euclidean(3)])
            manifold2 = Product([Euclidean(3), Sphere(4)])
        return manifold, manifold2

    def fit(self, dataset):
        # Dataset
        self.set_data_params(dataset)
        self.data, self.data2 = self.get_reshaped_data()
        self.manifold, self.manifold2 = self.make_manifold()

        if self.gmm_type in [1, 2, 4, 5]:
            total_dim = 3 + 3
        else:
            total_dim = 3 + 3 + 4

        # K-Means
        km_means, km_assignments = manifold_k_means(self.manifold, self.data, nb_clusters=self.n_components)

        # GMM
        log_rank_0(f"Type {self.gmm_type} Manifold GMM with K-Means priors")
        start = time.time()
        init_covariances = np.concatenate(self.n_components * [np.eye(total_dim)[None]], 0)
        init_priors = np.zeros(self.n_components)
        for k in range(self.n_components):
            init_priors[k] = np.sum(km_assignments == k) / len(km_assignments)
        self.means, self.covariances, self.priors, assignments = manifold_gmm_em(
            self.manifold,
            self.data,
            self.n_components,
            initial_means=km_means,
            initial_covariances=init_covariances,
            initial_priors=init_priors,
            logger=logger,
        )
        # Train another GMM if type 4
        if self.gmm_type in [4, 5] and self.manifold2 is not None:
            total_dim = 3 + 4
            km_means, km_assignments = manifold_k_means(self.manifold2, self.data2, nb_clusters=self.n_components)
            log_rank_0(f"Type {self.gmm_type} Second Manifold GMM with K-Means priors")
            init_covariances = np.concatenate(self.n_components * [np.eye(total_dim)[None]], 0)
            init_priors = np.zeros(self.n_components)
            for k in range(self.n_components):
                init_priors[k] = np.sum(km_assignments == k) / len(km_assignments)
            self.means2, self.covariances2, self.priors2, assignments = manifold_gmm_em(
                self.manifold2,
                self.data2,
                self.n_components,
                initial_means=km_means,
                initial_covariances=init_covariances,
                initial_priors=init_priors,
                logger=logger,
            )

        log_rank_0(f"Type {self.gmm_type} GMM train time: {time.time() - start} seconds")
        # Reshape means from (n_components, 2) to (n_components, 2, state_size)
        # Only applies to gmm type 1, 2, 4, 5
        self.means = self.get_reshaped_means()

        # Save GMM params
        # self.reshape_params(to="generic")  # Only applies to gmm type 1 and 2
        self.save_model()

        # Plot GMM
        if self.plot:
            outfile = self.plot_gmm()

        self.logger.log_table(key="fit", columns=["GMM"], data=[[wandb.Video(outfile)]])

    def predict1(self, x):
        dx, _, __ = manifold_gmr(
            x.reshape(1, -1),
            self.manifold,
            self.means,
            self.covariances,
            self.priors,
            in_manifold_idx=[0],
            out_manifold_idx=[1],
        )
        return dx[0]

    def predict2(self, x):
        return self.predict1(x)

    def predict3(self, x):
        dx, _, __ = manifold_gmr(
            x.reshape(1, -1),
            self.manifold,
            self.means,
            self.covariances,
            self.priors,
            in_manifold_idx=[0],
            out_manifold_idx=[1, 2],
        )
        return dx[0]

    def predict4(self, x):
        dx, _, __ = manifold_gmr(
            x.reshape(1, -1),
            self.manifold2,
            self.means2,
            self.covariances2,
            self.priors2,
            in_manifold_idx=[0],
            out_manifold_idx=[1],
        )
        return self.predict1(x), dx[0]

    def predict5(self, x):
        return self.predict4(x)
