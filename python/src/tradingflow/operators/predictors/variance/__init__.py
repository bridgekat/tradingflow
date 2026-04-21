r"""Concrete variance-predictor implementations.

The estimators in this module closely follow Pantaleo, Tumminello,
Lillo, and Mantegna, *When do improved covariance matrix estimators
enhance portfolio optimization? An empirical comparative study of nine
estimators* (arXiv:1004.4272, 2010).  Together with the Markowitz
baseline they form the four canonical families surveyed in the paper —
sample, spectral, hierarchical-clustering, and shrinkage.

All estimators share the same interface: they subclass
[`VariancePredictor`][tradingflow.operators.predictors.variance_predictor.VariancePredictor],
ignore the `features_series` input, and emit an `(N, N)` covariance
matrix on every rebalance tick for the stocks currently in the universe.

**Baseline**

- [`Sample`][tradingflow.operators.predictors.variance.sample.Sample] — sample
  covariance of historical returns (Markowitz baseline).

**Spectral estimators**

- [`SingleIndex`][tradingflow.operators.predictors.variance.single_index.SingleIndex]
  — single-factor covariance
  \(\sigma_f^{2} \beta \beta^T + \mathrm{diag}(\sigma_\epsilon^{2})\)
  using the equal-weighted cross-sectional mean return as the market
  proxy.
- [`RMT0`][tradingflow.operators.predictors.variance.rmt.RMT0] — zeros
  eigenvalues below the Laloux-corrected Marchenko-Pastur bound
  (Rosenow et al., 2002).
- [`RMTM`][tradingflow.operators.predictors.variance.rmt.RMTM] — replaces
  sub-threshold eigenvalues with their mean (Potters et al., 2005).

**Hierarchical-clustering estimators**

- [`UPGMA`][tradingflow.operators.predictors.variance.hierarchical.UPGMA] — size-weighted
  arithmetic-mean linkage on the sample correlation.
- [`WPGMA`][tradingflow.operators.predictors.variance.hierarchical.WPGMA] — unweighted
  arithmetic-mean linkage on the sample correlation.
- [`Hausdorff`][tradingflow.operators.predictors.variance.hierarchical.Hausdorff] —
  Hausdorff-style linkage using the original pairwise similarities.

**Shrinkage estimators**

- [`Shrinkage`][tradingflow.operators.predictors.variance.shrinkage.Shrinkage] —
  Ledoit-Wolf linear shrinkage with a pluggable `target` parameter
  (a [`Target`][tradingflow.operators.predictors.variance.shrinkage.Target] enum
  member) selecting one of the three targets surveyed in the paper:
  `Target.COMMON_COVARIANCE` (default),
  `Target.CONSTANT_CORRELATION`, or `Target.SINGLE_INDEX`.
"""

from .hierarchical import UPGMA, WPGMA, Hausdorff
from .rmt import RMT0, RMTM
from .sample import Sample
from .shrinkage import Shrinkage, Target
from .single_index import SingleIndex

__all__ = [
    "Sample",
    "Shrinkage",
    "SingleIndex",
    "Target",
    "RMT0",
    "RMTM",
    "UPGMA",
    "WPGMA",
    "Hausdorff",
]
