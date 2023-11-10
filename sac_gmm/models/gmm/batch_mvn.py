import numpy as np
from .utils.gmr.gmr.utils import check_random_state
import scipy as sp
from scipy.stats import chi2, norm
import torch


def invert_indices(n_features, indices):
    inv = np.ones(n_features, dtype=bool)
    inv[indices] = False
    (inv,) = np.where(inv)
    return inv


class BatchMVN(object):
    """
    Batch Multivariate normal distribution.
    """

    def __init__(self, mean=None, covariance=None, verbose=0, random_state=None):
        if torch.is_tensor(mean):
            self.mean = mean
        else:
            self.mean = torch.from_numpy(mean)
        if torch.is_tensor(covariance):
            self.covariance = covariance
        else:
            self.covariance = torch.from_numpy(covariance)
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.norm = None

        if self.mean is not None:
            self.mean = torch.asarray(self.mean)
        if self.covariance is not None:
            self.covariance = torch.asarray(self.covariance)

    def _check_initialized(self):
        if self.mean is None:
            raise ValueError("Mean has not been initialized")
        if self.covariance is None:
            raise ValueError("Covariance has not been initialized")

    def condition(self, indices, x):
        """Conditional distribution over given indices.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to condition.

        x : torch tensor, shape (n_new_features,)
            Values of the features that we know.

        Returns
        -------
        conditional : MVN
            Conditional MVN distribution p(Y | X=x).
        """
        self._check_initialized()
        mean, covariance = batch_condition(
            self.mean,
            self.covariance,
            invert_indices(self.mean.shape[1], indices),
            indices,
            x,
        )
        return BatchMVN(mean=mean, covariance=covariance, random_state=self.random_state)

    def marginalize(self, indices):
        """Marginalize over everything except the given indices.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to keep.

        Returns
        -------
        marginal : MVN
            Marginal MVN distribution.
        """
        self._check_initialized()
        return BatchMVN(
            mean=self.mean[:, indices],
            covariance=self.covariance[:, indices[0] : indices[-1] + 1, indices[0] : indices[-1] + 1],
            random_state=self.random_state,
        )

    def to_norm_factor_and_exponents(self, X):
        """Compute normalization factor and exponents of Gaussian.

        These values can be used to compute the probability density function
        of this Gaussian: p(x) = norm_factor * np.exp(exponents).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        norm_factor : float
            Normalization factor: constant term outside of exponential
            function in probability density function of this Gaussian.

        exponents : array, shape (n_samples,)
            Exponents to compute probability density function.
        """
        self._check_initialized()

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)

        X = torch.atleast_2d(X)
        n_features = X.shape[1]

        try:
            L = torch.linalg.cholesky(self.covariance, upper=False)
        except RuntimeError:
            # Degenerated covariance, try to add regularization
            L = torch.linalg.cholesky(self.covariance + 1e-3 * torch.eye(n_features), upper=False)

        X_minus_mean = X - self.mean

        if self.norm is None:
            # Suppress a determinant of 0 to avoid numerical problems
            L_det = torch.linalg.det(L)
            ge_zero_flag = L_det > torch.finfo(L.dtype).eps
            L_det = torch.where(ge_zero_flag, L_det, torch.finfo(L.dtype).eps)
            self.norm = 0.5 / torch.tensor(np.pi) ** (0.5 * n_features) / L_det

        # Solve L x = (X - mean)^T for x with triangular L
        # (LL^T = Sigma), that is, x = L^T^-1 (X - mean)^T.
        # We can avoid covariance inversion when computing
        # (X - mean) Sigma^-1 (X - mean)^T  with this trick,
        # since Sigma^-1 = L^T^-1 L^-1.
        X_normalized = torch.transpose(torch.linalg.solve_triangular(L, X_minus_mean.unsqueeze(-1), upper=False), 2, 1)

        exponent = -0.5 * torch.sum(X_normalized**2, axis=2)

        return self.norm.squeeze(), exponent.squeeze()


def batch_regression_coefficients(batch_covariance, i_out, i_in, batch_cov_12=None):
    """Compute regression coefficients to predict conditional distribution.

    Parameters
    ----------
    covariance : torch array, shape (batch_size, n_features, n_features)
        Covariance of MVN

    i_out : array, shape (n_features_out,)
        Output feature indices

    i_in : array, shape (n_features_in,)
        Input feature indices

    cov_12 : torch array, shape (batch_size, n_features_out, n_features_in), optional (default: None)
        Precomputed block of the covariance matrix between input features and
        output features

    Returns
    -------
    regression_coeffs : torch array, shape (batch_size, n_features_out, n_features_in)
        Regression coefficients. These can be used to compute the mean of the
        conditional distribution as
        mean[i1] + regression_coeffs.dot((X - mean[i2]).T).T
    """
    if batch_cov_12 is None:
        batch_cov_12 = batch_covariance[:, i_out[0] : i_out[-1] + 1, i_in[0] : i_in[-1] + 1]
    batch_cov_22 = batch_covariance[:, i_in[0] : i_in[-1] + 1, i_in[0] : i_in[-1] + 1]
    batch_prec_22 = torch.linalg.pinv(batch_cov_22, hermitian=True)
    return torch.bmm(batch_cov_12, batch_prec_22)


# %%
def batch_condition(batch_mean, batch_covariance, i_out, i_in, batch_X):
    """Compute conditional mean and covariance.

    Parameters
    ----------
    mean : array, shape (batch_size, n_features,)
        Mean of MVN

    covariance : array, shape (n_features, n_features)
        Covariance of MVN

    i_out : array, shape (n_features_out,)
        Output feature indices

    i_in : array, shape (n_features_in,)
        Input feature indices

    X : array, shape (n_samples, n_features_out)
        Inputs

    Returns
    -------
    mean : array, shape (n_features_out,)
        Mean of the conditional distribution

    covariance : array, shape (n_features_out, n_features_out)
        Covariance of the conditional distribution
    """
    batch_cov_12 = batch_covariance[:, i_out[0] : i_out[-1] + 1, i_in[0] : i_in[-1] + 1]
    batch_cov_11 = batch_covariance[:, i_out[0] : i_out[-1] + 1, i_out[0] : i_out[-1] + 1]
    regression_coeffs = batch_regression_coefficients(batch_covariance, i_out, i_in, batch_cov_12=batch_cov_12)

    diff = batch_X - batch_mean[:, i_in]
    batch_mean = (
        batch_mean[:, i_out] + torch.transpose(torch.bmm(regression_coeffs, diff.unsqueeze(-1)), 1, 2).squeeze()
    )

    batch_covariance = batch_cov_11 - torch.bmm(regression_coeffs, torch.transpose(batch_cov_12, 1, 2))
    return batch_mean, batch_covariance
