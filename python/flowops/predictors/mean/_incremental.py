"""Incremental (recursive-least-squares) mean predictors.

The pooled OLS / Ridge fit depends on the training window only through its
**sufficient statistics** -- per-stock moments ``(count, Sx, Sxx, Sy, Sxy, Syy)``.
Those are additive, so this base maintains them incrementally as the window
slides: each rebalance folds in only the ticks added since the last one (and, for
a rolling window, removes the ticks that aged out) instead of re-reading and
re-factorizing the whole window. The per-rebalance cost is then **independent of
how far back history reaches** -- the exact fix for the expanding-window blow-up
that the random subsample (`max_samples`) only approximated.

Summing the *active* stocks' moments each rebalance and solving reproduces the
original active-masked pooled fit (to solver tolerance) with no subsampling -- the
fit is over exactly the same samples; only the linear algebra differs (normal
equations on the reconstructed standardized Gram vs. QR on the raw design).

Subclass hooks (all operate on an opaque per-predictor ``running`` state):

* ``init_running_fn(num_stocks, num_features) -> running``
* ``add_sample_fn(running, x_t, y_t)`` -- fold tick t's ``(N, F)`` / ``(N,)``
  features/target into ``running``.
* ``remove_sample_fn(running, x_t, y_t)`` -- undo a tick; **only called for a
  rolling window**, and may raise if the predictor cannot down-date.
* ``counts_fn(running) -> (N,)`` -- per-stock valid-sample counts (for ``min_periods``).
* ``fit_fn(running, mask) -> params`` -- solve over the active subset ``mask``.
* ``predict_fn(state, features, params)`` -- same contract as `MeanPredictor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(slots=True)
class IncrementalMeanPredictorState:
    """Mutable state for `IncrementalMeanPredictor`."""

    num_stocks: int
    num_features: int
    universe_size: int
    target_offset: int
    refit_every: int
    window: int | None
    min_periods: int | None
    add_sample_fn: Callable
    remove_sample_fn: Callable
    counts_fn: Callable
    fit_fn: Callable
    predict_fn: Callable
    running: object
    n_added: int = 0
    n_removed: int = 0
    cached_params: object = None
    fitted: bool = False
    rebalance_count: int = 0


class IncrementalMeanPredictor:
    r"""Abstract incremental cross-sectional mean predictor (see module docs).

    Input ports: ``(universe, features_series, target_series)`` -- same as
    `MeanPredictor`. ``window`` is the rolling training length in ticks (``None``
    = expanding window, the original default); a rolling window additionally calls
    ``remove_sample_fn`` as ticks age out.
    """

    def __init__(
        self,
        *,
        num_stocks: int,
        num_features: int,
        universe_size: int,
        target_offset: int,
        init_running_fn: Callable,
        add_sample_fn: Callable,
        remove_sample_fn: Callable,
        counts_fn: Callable,
        fit_fn: Callable,
        predict_fn: Callable,
        refit_every: int = 1,
        window: int | None = None,
        min_periods: int | None = None,
    ) -> None:
        assert target_offset >= 0
        assert refit_every >= 1
        assert window is None or window >= 1
        self._num_stocks = int(num_stocks)
        self._num_features = int(num_features)
        self._universe_size = int(universe_size)
        self._target_offset = int(target_offset)
        self._refit_every = int(refit_every)
        self._window = None if window is None else int(window)
        self._min_periods = min_periods
        self._init_running_fn = init_running_fn
        self._add_sample_fn = add_sample_fn
        self._remove_sample_fn = remove_sample_fn
        self._counts_fn = counts_fn
        self._fit_fn = fit_fn
        self._predict_fn = predict_fn

    def init(self, inputs, timestamp: int) -> IncrementalMeanPredictorState:
        return IncrementalMeanPredictorState(
            num_stocks=self._num_stocks,
            num_features=self._num_features,
            universe_size=self._universe_size,
            target_offset=self._target_offset,
            refit_every=self._refit_every,
            window=self._window,
            min_periods=self._min_periods,
            add_sample_fn=self._add_sample_fn,
            remove_sample_fn=self._remove_sample_fn,
            counts_fn=self._counts_fn,
            fit_fn=self._fit_fn,
            predict_fn=self._predict_fn,
            running=self._init_running_fn(self._num_stocks, self._num_features),
        )

    @staticmethod
    def compute(
        state: IncrementalMeanPredictorState,
        inputs,
        output,
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        universe_view, features_series_view, target_series_view = inputs
        if not produced[0]:  # rebalance trigger
            return False

        n_features = len(features_series_view)
        n_target = len(target_series_view)
        assert n_features == n_target, (
            f"incremental_mean_predictor: features/target lengths differ "
            f"({n_features} vs {n_target})"
        )

        # Pairs: features[i] with target[i + target_offset], i in 0..n_pair.
        n_pair = max(0, n_target - state.target_offset)

        # Fold in the ticks added since the last rebalance (each seen exactly once).
        for i in range(state.n_added, n_pair):
            state.add_sample_fn(
                state.running, features_series_view[i], target_series_view[i + state.target_offset]
            )
        state.n_added = n_pair

        # Rolling window: drop the ticks that have aged out of [n_pair - window, n_pair).
        if state.window is not None:
            window_start = max(0, n_pair - state.window)
            for i in range(state.n_removed, window_start):
                state.remove_sample_fn(
                    state.running, features_series_view[i], target_series_view[i + state.target_offset]
                )
            state.n_removed = window_start

        should_refit = (not state.fitted) or (state.rebalance_count % state.refit_every == 0)
        state.rebalance_count += 1

        features = features_series_view[-1]
        mask = universe_view.to_numpy() > 0
        assert int(mask.sum()) <= state.universe_size, (
            f"incremental_mean_predictor: universe has {int(mask.sum())} nonzero entries, "
            f"exceeds universe_size={state.universe_size}"
        )
        if state.min_periods is not None:
            mask = mask & (state.counts_fn(state.running) >= state.min_periods)
        mask = mask & np.isfinite(features).all(axis=1)

        if should_refit and mask.any():
            state.cached_params = state.fit_fn(state.running, mask)
            state.fitted = True

        mu = np.full((state.num_stocks,), np.nan, dtype=np.float64)
        if state.fitted and mask.any():
            mu[mask] = state.predict_fn(state, features[mask], state.cached_params)
        output.write(mu)
        return True


# ---------------------------------------------------------------------------
# Recursive-least-squares implementation of the hooks (OLS / Ridge)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Moments:
    """Per-stock sufficient statistics over the (sliding) training window."""

    n: np.ndarray  # (N,)   valid-sample counts
    sx: np.ndarray  # (N, F)    Sum x
    sxx: np.ndarray  # (N, F, F) Sum x x^T
    sy: np.ndarray  # (N,)   Sum y
    sxy: np.ndarray  # (N, F)    Sum x y
    syy: np.ndarray  # (N,)   Sum y^2


@dataclass(slots=True)
class IncrementalLinearParams:
    """Fitted coefficients + the pooled standardization scaler (predict contract)."""

    beta: np.ndarray
    intercept: float
    x_mean: np.ndarray
    x_std: np.ndarray


def rls_init_running(num_stocks: int, num_features: int) -> _Moments:
    n_, f_ = int(num_stocks), int(num_features)
    return _Moments(
        n=np.zeros(n_),
        sx=np.zeros((n_, f_)),
        sxx=np.zeros((n_, f_, f_)),
        sy=np.zeros(n_),
        sxy=np.zeros((n_, f_)),
        syy=np.zeros(n_),
    )


def _accumulate(m: _Moments, x_t: np.ndarray, y_t: np.ndarray, sign: float) -> None:
    """Add (`sign`=+1) or remove (`sign`=-1) tick t's per-stock contributions."""
    valid = np.isfinite(x_t).all(axis=1) & np.isfinite(y_t)  # (N,)
    xv = np.where(valid[:, None], x_t, 0.0)  # (N, F), zero for invalid stocks
    yv = np.where(valid, y_t, 0.0)  # (N,)
    m.n += sign * valid
    m.sx += sign * xv
    m.sxx += sign * (xv[:, :, None] * xv[:, None, :])
    m.sy += sign * yv
    m.sxy += sign * (xv * yv[:, None])
    m.syy += sign * (yv * yv)


def rls_add_sample(m: _Moments, x_t: np.ndarray, y_t: np.ndarray) -> None:
    _accumulate(m, x_t, y_t, 1.0)


def rls_remove_sample(m: _Moments, x_t: np.ndarray, y_t: np.ndarray) -> None:
    _accumulate(m, x_t, y_t, -1.0)


def rls_counts(m: _Moments) -> np.ndarray:
    return m.n


def rls_fit(m: _Moments, mask: np.ndarray, alpha: float) -> IncrementalLinearParams:
    """Solve pooled standardized OLS (``alpha=0``) / Ridge from the active moments.

    Reconstructs the same pooled mean/std and standardized Gram the original fit
    builds, then solves the normal equations ``(Z'Z + alpha*n*I) beta = Z'y`` --
    equivalent to the original QR fit on the same active samples.
    """
    f_ = m.sx.shape[1]
    n = float(m.n[mask].sum())
    if n <= 0:
        return IncrementalLinearParams(np.zeros(f_), 0.0, np.zeros(f_), np.ones(f_))

    sx = m.sx[mask].sum(axis=0)
    sxx = m.sxx[mask].sum(axis=0)
    sy = float(m.sy[mask].sum())
    sxy = m.sxy[mask].sum(axis=0)
    syy = float(m.syy[mask].sum())

    # Pooled mean / population std (ddof=0), with constant-column fallback std=1.
    x_mean = sx / n
    x_std = np.sqrt(np.maximum(np.diag(sxx) / n - x_mean**2, 0.0))
    x_std = np.where(x_std > 0, x_std, 1.0)
    y_mean = sy / n
    y_std = float(np.sqrt(max(syy / n - y_mean**2, 0.0)))
    y_std = y_std if y_std > 0 else 1.0

    fallback = IncrementalLinearParams(np.zeros(f_), y_mean, x_mean, x_std)

    # Standardized Gram from raw moments: Z'Z = (Sxx - n mu mu^T) / (sx sx^T);
    # Z'y_norm = (Sxy - n mu y_mean) / (sx * sy).
    ztz = (sxx - n * np.outer(x_mean, x_mean)) / np.outer(x_std, x_std)
    zty = (sxy - n * x_mean * y_mean) / (x_std * y_std)
    a = ztz + alpha * n * np.eye(f_)
    try:
        beta_tilde = np.linalg.solve(a, zty)
    except np.linalg.LinAlgError:
        return fallback
    if not np.all(np.isfinite(beta_tilde)):
        return fallback
    # Recover coefficients in the original-y scale.
    return IncrementalLinearParams(y_std * beta_tilde, y_mean, x_mean, x_std)


def rls_predict(state, features: np.ndarray, params: IncrementalLinearParams) -> np.ndarray:
    z = (features - params.x_mean) / params.x_std
    return z @ params.beta + params.intercept


def unsupported_remove(running, x_t, y_t) -> None:
    """A `remove_sample_fn` for predictors that cannot down-date (rolling window)."""
    raise NotImplementedError(
        "this incremental predictor does not support a rolling window (no remove_sample_fn)"
    )
