import os
import sys
import wandb
from pathlib import Path

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix())  # sac_gmm
sys.path.insert(0, root.as_posix())  # Root

import numpy as np
from pymanopt.manifolds import Euclidean, Sphere, Product

from sac_gmm.gmm.base_gmm import BaseGMM
from sac_gmm.gmm.utils.manifold_clustering import manifold_k_means, manifold_gmm_em
from sac_gmm.gmm.utils.manifold_gmr import manifold_gmr

import logging


class ManifoldGMM(BaseGMM):
    def __init__(self, n_components, plot, model_dir, state_size):
        super(ManifoldGMM, self).__init__(
            n_components=n_components, plot=plot, model_dir=model_dir, state_size=state_size
        )

        self.name = "ManifoldGMM"

        self.manifold = None

        self.logger = logging.getLogger(f"{self.name}")

    def make_manifold(self, dim):
        if self.state_type in ["pos", "joint"]:
            in_manifold = Euclidean(dim)
            out_manifold = Euclidean(dim)
        elif self.state_type == "ori":
            in_manifold = Sphere(dim)
            out_manifold = Sphere(dim)
        elif self.state_type == "pos_ori":
            in_dim = 3
            out_dim_e = 3
            out_dim_q = 4
            in_manifold = Euclidean(in_dim)
            out_manifold1 = Euclidean(out_dim_e)
            out_manifold2 = Sphere(out_dim_q)
            manifold = Product([in_manifold, out_manifold1, out_manifold2])
            return manifold
        manifold = Product([in_manifold, out_manifold])
        return manifold

    def fit(self, dataset, wandb_flag=False):
        # Dataset
        self.set_data_params(dataset)
        self.manifold = self.make_manifold(self.dim)

        # K-Means
        km_means, km_assignments = manifold_k_means(self.manifold, self.data, nb_clusters=self.n_components)
        # GMM
        total_dim = self.dim + self.dim
        if self.state_type == "pos_ori":
            total_dim += 4
        self.logger.info("Manifold GMM with K-Means priors")
        init_covariances = np.concatenate(self.n_components * [np.eye(total_dim)[None]], 0)
        init_priors = np.zeros(self.n_components)
        for k in range(self.n_components):
            init_priors[k] = np.sum(km_assignments == k) / len(km_assignments)
        self.means, self.covariances, self.priors, self.assignments = manifold_gmm_em(
            self.manifold,
            self.data,
            self.n_components,
            initial_means=km_means,
            initial_covariances=init_covariances,
            initial_priors=init_priors,
            logger=self.logger,
        )
        # Reshape means from (n_components, 2) to (n_components, 2, state_size)
        self.means = self.get_reshaped_means()

        # Save GMM params
        self.reshape_params(to="generic")
        self.save_model()
        self.reshape_params(to="gmr-specific")

        # Plot GMM
        if self.plot:
            outfile = self.plot_gmm()

        if wandb_flag:
            config = {"n_comp": self.n_components}
            wandb.init(
                project="ds-training",
                entity="in-ac",
                name=f"{dataset.skill}_{dataset.state_type}_gmm",
                config=config,
            )
            wandb.log({"GMM-Viz": wandb.Video(outfile)})
            wandb.finish()

    def predict(self, x):
        if self.state_type == "pos_ori":
            out_manifold_idx = [1, 2]
        else:
            out_manifold_idx = [1]
        dx, _, __ = manifold_gmr(
            x.reshape(1, -1),
            self.manifold,
            self.means,
            self.covariances,
            self.priors,
            in_manifold_idx=[0],
            out_manifold_idx=out_manifold_idx,
        )
        return dx[0]

    def load_model(self):
        super().load_model()
        self.reshape_params(to="gmr-specific")

    def get_reshaped_means(self):
        """Reshape means from (n_components, 2) to (n_components, 2, state_size)"""
        new_means = np.empty((self.n_components, 2, self.state_size))
        for i in range(new_means.shape[0]):
            for j in range(new_means.shape[1]):
                new_means[i, j, :] = self.means[i][j]
        return new_means

    def reshape_params(self, to="generic"):
        """Reshapes model params to/from generic/gmr-specific shapes.
        E.g., For N GMM components, S state size, generic shapes are
        self.priors = (N,);
        self.means = (N, 2*S);
        self.covariances = (N, 2*S, 2*S)

        Gmr-specific: self.means = (N, 2, S)
        """
        # priors and covariances already match shape
        shape = None
        if to == "generic":
            shape = (self.n_components, 2 * self.state_size)
        else:
            shape = (self.n_components, 2, self.state_size)
        self.means = self.means.reshape(shape)
