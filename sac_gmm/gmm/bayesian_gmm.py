from sklearn.mixture import BayesianGaussianMixture
from gmr import GMM
from sac_gmm.gmm.base_gmm import BaseGMM
import numpy as np
import wandb


class BayesianGMM(BaseGMM):
    def __init__(self, n_components, max_iter, plot, model_dir, state_size):
        super(BayesianGMM, self).__init__(
            n_components=n_components, plot=plot, model_dir=model_dir, state_size=state_size
        )

        self.name = "BayesianGMM"
        self.random_state = np.random.RandomState(0)

        self.max_iter = max_iter
        self.bgmm = BayesianGaussianMixture(n_components=self.n_components, max_iter=self.max_iter)
        self.gmm = None
        self.logger = None

    def fit(self, dataset):
        self.set_data_params(dataset, obj_type=False)
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
