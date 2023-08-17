import os
import sys
import wandb
from pathlib import Path
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from gmr import GMM
from sac_gmm.models.gmm.base_gmm import BaseGMM

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix())  # sac_gmm
sys.path.insert(0, root.as_posix())  # Root


class BayesianGMM(BaseGMM):
    def __init__(self, skill, max_iter, plot):
        super(BayesianGMM, self).__init__(
            n_components=skill.n_components, plot=plot, model_dir=skill.skills_dir, state_type=skill.state_type
        )

        self.name = "BayesianGMM"
        self.random_state = np.random.RandomState(0)

        self.max_iter = max_iter
        self.bgmm = BayesianGaussianMixture(n_components=self.n_components, max_iter=self.max_iter)
        self.gmm = None
        self.logger = None
        self.model_dir = os.path.join(Path(skill.skills_dir).expanduser(), skill.name, skill.state_type, self.name)
        os.makedirs(self.model_dir, exist_ok=True)

    def fit(self, dataset):
        self.set_data_params(dataset)
        self.bgmm = self.bgmm.fit(self.data)
        self.means, self.covariances, self.priors = self.bgmm.means_, self.bgmm.covariances_, self.bgmm.weights_

        self.gmm = GMM(
            n_components=self.n_components,
            priors=self.priors,
            means=self.means,
            covariances=self.covariances,
            random_state=self.random_state,
        )

        # Save GMM params
        self.save_model()

        # Plot GMM
        if self.plot:
            outfile = self.plot_gmm()

        self.logger.log_table(key="fit", columns=["GMM"], data=[[wandb.Video(outfile)]])

    def predict(self, x):
        cgmm = self.gmm.condition([0, 1, 2], x.reshape(1, -1))
        dx = cgmm.sample_confidence_region(1, alpha=0.7).reshape(-1)
        return dx

    def load_model(self):
        super().load_model()
        self.gmm = GMM(
            n_components=self.n_components,
            priors=self.priors,
            means=self.means,
            covariances=self.covariances,
            random_state=self.random_state,
        )

    def update_model(self, delta):
        super().update_model(delta)
        self.gmm.means = self.means
        self.gmm.priors = self.priors

    def copy_model(self, gmm):
        super().copy_model(gmm)
        self.gmm.means = self.means
        self.gmm.priors = self.priors
