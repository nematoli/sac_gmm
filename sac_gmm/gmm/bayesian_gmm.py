from sklearn.mixture import BayesianGaussianMixture
from gmr import GMM
from sac_gmm.gmm.base_gmm import BaseGMM
import numpy as np
import logging
from pathlib import Path
import os


class BayesianGMM(BaseGMM):
    def __init__(self, skill, max_iter, plot, gmm_type):
        super(BayesianGMM, self).__init__(
            n_components=skill.n_components, plot=plot, model_dir=skill.skills_dir, gmm_type=gmm_type
        )

        if gmm_type not in [
            1,
            2,
        ]:
            raise ValueError(f"BayesianGMM only supports gmm_type 1 and 2, not {gmm_type}")

        self.name = "BayesianGMM"
        self.skill = skill.skill
        self.random_state = np.random.RandomState(0)
        self.max_iter = max_iter
        self.bgmm = BayesianGaussianMixture(n_components=self.n_components, max_iter=self.max_iter)
        self.gmm = None

        self.logger = logging.getLogger(f"{self.name}")

        self.model_dir = os.path.join(Path(skill.skills_dir).expanduser(), skill.skill, self.name, f"type{gmm_type}")
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
            outfile = self.plot_gmm(obj_type=False)

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

    def predict1(self, x):
        cgmm = self.gmm.condition([0, 1, 2], x[:3].reshape(1, -1))
        dx = cgmm.sample_confidence_region(1, alpha=0.7).reshape(-1)
        return dx

    def predict2(self, x):
        return self.predict1(x)

    def predict3(self, x):
        """
        Predict function for GMM type 3
        """
        raise ValueError("GMM type 4 not supported for BayesianGMM")

    def predict4(self, x):
        """
        Predict function for GMM type 4
        """
        raise ValueError("GMM type 4 not supported for BayesianGMM")

    def predict5(self, x):
        """
        Predict function for GMM type 5
        """
        raise ValueError("GMM type 5 not supported for BayesianGMM")
