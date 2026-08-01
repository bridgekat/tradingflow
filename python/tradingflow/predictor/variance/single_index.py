from ._base import VariancePredictor, covariance_predictor
from ._common import single_index_covariance


def build(**kwargs) -> VariancePredictor:
    r"""Constructs a single-index factor-model covariance predictor.

    Fits \(r_i(t) = \alpha_i + \beta_i f(t) + \epsilon_i(t)\) stock by stock
    against the equal-weighted cross-sectional mean return as the market proxy,
    and returns \(\Sigma = \sigma_f^2 \beta\beta^T +
    \mathrm{diag}(\sigma_\epsilon^2)\) — the *SI* estimator of Pantaleo et al.
    (2010).
    """
    return covariance_predictor(single_index_covariance, **kwargs)
