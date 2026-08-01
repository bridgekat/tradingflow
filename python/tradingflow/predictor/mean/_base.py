"""Windowed mean-return predictor harness."""

from dataclasses import dataclass

import numpy as np

from .._panel import PanelPredictor, PanelState


def pool_and_subsample(
    x: np.ndarray, y: np.ndarray, max_samples: int | None = None, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Flattens the `(T, N, F)` / `(T, N)` training window to pooled rows, drops
    rows with any non-finite feature or target, and — when `max_samples` is set
    and exceeded — keeps a uniform random subset of that size.

    The draw is i.i.d. over the valid pooled rows, so the pooled regression
    stays unbiased and distribution-preserving; it only caps the dominant
    `O(m F²)` factorization cost (the read/standardize term is the cheaper,
    bandwidth-bound `O(m F)`, and is left uncapped). `seed` makes the subsample
    reproducible — a fresh `Generator` per call, independent of global state,
    so it is deterministic and thread-safe under the work-stealing pool.

    The subsample is taken on the row *indices*, so only the kept rows are ever
    copied (a single gather) rather than the whole valid block followed by a
    second subsampling copy.
    """
    t, n, f = x.shape
    x = x.reshape(t * n, f)
    y = y.reshape(t * n)
    keep = np.flatnonzero(np.isfinite(x).all(axis=1) & np.isfinite(y))
    if max_samples is not None and keep.size > int(max_samples):
        keep = keep[np.random.default_rng(seed).choice(keep.size, int(max_samples), replace=False)]
    return x[keep], y[keep]


def standardize(x: np.ndarray, y: np.ndarray):
    """Pool-standardizes features and target symmetrically.

    Pooled mean and population standard deviation (`ddof=0`), with a
    constant-column / constant-target fallback of `1` so the corresponding
    standardized values are zero and contribute nothing to the solve — the
    predictor then produces zero coefficients and constant predictions at the
    target mean.
    """
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=0)
    x_std = np.where(x_std > 0, x_std, 1.0)

    y_mean = float(y.mean())
    y_std = float(y.std(ddof=0))
    y_std = y_std if y_std > 0 else 1.0

    return (x - x_mean) / x_std, (y - y_mean) / y_std, x_mean, x_std, y_mean, y_std


class LinearParams:
    """Fitted coefficients with the pooled standardization scaler."""

    __slots__ = ("beta", "intercept", "x_mean", "x_std")

    def __init__(self, beta, intercept, x_mean, x_std):
        self.beta = beta
        self.intercept = intercept
        self.x_mean = x_mean
        self.x_std = x_std


def linear_predict(x: np.ndarray, params: LinearParams) -> np.ndarray:
    """Applies the fitted standardization scaler, then the linear model."""
    return (x - params.x_mean) / params.x_std @ params.beta + params.intercept


@dataclass(slots=True)
class MeanPanelState(PanelState):
    """A mean predictor emits one expected return per stock."""

    @staticmethod
    def empty(n: int) -> np.ndarray:
        return np.full(n, np.nan)

    @staticmethod
    def scatter(out: np.ndarray, mask: np.ndarray, values: np.ndarray) -> None:
        out[mask] = values


class MeanPredictor(PanelPredictor):
    """A mean predictor fits a panel regression, so it retains the window by
    default. A model that fits from the target alone — `sample` — can turn
    `retain_features` off and stop paying for history it never reads."""

    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[bool, np.ndarray]
    type Context = int
    type State = MeanPanelState

    state_type = MeanPanelState
