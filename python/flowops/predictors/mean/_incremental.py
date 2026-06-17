"""Incremental (recursive-least-squares) mean predictors.

A `MeanPredictor` re-reads the whole training window and re-factorizes it every
rebalance. The incremental version instead keeps a **sample pool** of sufficient
statistics that is updated one trading day at a time, so the per-rebalance cost is
independent of how far back history reaches.

The design splits cleanly into a generic *harness* and a model *pool*:

* :class:`IncrementalMeanPredictor` -- the harness. Each rebalance it folds the
  trading days added since the last one into the pool (``add_samples``, keyed by a
  monotone day index), drops the days that aged out of a rolling window
  (``remove_samples``), and asks the pool to ``predict``. It owns only the day
  counter, the ``target_offset`` pairing and the refit cadence.
* :class:`RlsMeanPool` -- the model. It maintains the per-stock OLS/Ridge moments,
  the **active aggregate** Gram, and the active mask, and does the fit + predict.

The pool exposes exactly three operations, so a concrete predictor (Ridge, OLS) is
just a ``build()`` that wires a pool factory -- no per-hook plumbing:

* ``add_samples(index, x, y)`` -- fold day ``index``'s ``(N, F)`` features and the
  paired ``(N,)`` forward returns into the pool.
* ``remove_samples(index)`` -- drop a previously added day (rolling window only).
* ``predict(universe, features, refit_due) -> (N,)`` -- refresh the active mask,
  fit if due, and return the per-stock prediction.

**Fit on all samples.** The pool fits on **every stock with valid features** each
rebalance -- the universe only selects which predictions are *emitted*, not which
samples are fitted (future portfolio construction is whole-market: it down-weights
out-of-universe names in the optimizer rather than dropping them). Fitting on the
whole panel needs only a single aggregate Gram ``G = sum over all valid samples of
(x x^T, x y, ...)``, folded one trading day at a time via ``X^T X`` (a BLAS GEMM,
``O(N*F)`` memory traffic, no ``(N, F, F)`` array). Each refit is then the
``O(F^3)`` standardized-Gram solve. Per-stock valid-sample counts are kept (cheap,
``(N,)``) only for the ``min_periods`` prediction filter; the rolling window
down-dates aged days from the same aggregate (the day's raw vectors are retained
for the window span). This is both the simplest and the fastest option -- the
per-rebalance fit is independent of stock count, and there is no per-stock
``(N, F, F)`` moment array to build and re-read every day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(slots=True)
class IncrementalLinearParams:
    """Fitted coefficients + the pooled standardization scaler (predict contract)."""

    beta: np.ndarray
    intercept: float
    x_mean: np.ndarray
    x_std: np.ndarray


def _solve_standardized(
    n: float,
    sx: np.ndarray,
    sxx: np.ndarray,
    sy: float,
    sxy: np.ndarray,
    syy: float,
    alpha: float,
) -> IncrementalLinearParams:
    """Solve pooled standardized OLS (``alpha=0``) / Ridge from raw moments.

    Reconstructs the pooled mean/std and the standardized Gram, then solves the
    normal equations ``(Z'Z + alpha*n*I) beta = Z'y`` -- equivalent to a QR fit on
    the same samples. The penalty matches `ridge`: the sample-size-invariant
    ``(1/n)||...||^2 + alpha||b||^2`` reduces to ``Z'Z + alpha*n*I`` on the
    pool-standardized design.
    """
    f_ = sx.shape[0]
    if n <= 0:
        return IncrementalLinearParams(np.zeros(f_), 0.0, np.zeros(f_), np.ones(f_))

    # Pooled mean / population std (ddof=0), with constant-column fallback std=1.
    x_mean = sx / n
    x_std = np.sqrt(np.maximum(np.diag(sxx) / n - x_mean**2, 0.0))
    x_std = np.where(x_std > 0, x_std, 1.0)
    y_mean = sy / n
    y_std = float(np.sqrt(max(syy / n - y_mean**2, 0.0)))
    y_std = y_std if y_std > 0 else 1.0

    fallback = IncrementalLinearParams(np.zeros(f_), y_mean, x_mean, x_std)

    # Standardized Gram from raw moments.
    ztz = (sxx - n * np.outer(x_mean, x_mean)) / np.outer(x_std, x_std)
    zty = (sxy - n * x_mean * y_mean) / (x_std * y_std)
    a = ztz + alpha * n * np.eye(f_)
    try:
        beta_tilde = np.linalg.solve(a, zty)
    except np.linalg.LinAlgError:
        return fallback
    if not np.all(np.isfinite(beta_tilde)):
        return fallback
    return IncrementalLinearParams(y_std * beta_tilde, y_mean, x_mean, x_std)


class RlsMeanPool:
    """All-sample pooled OLS / Ridge sample pool.

    Fits on every stock with valid features (the universe only selects which
    predictions are emitted), so it keeps a single aggregate Gram folded one
    trading day at a time via ``X^T X`` -- ``O(N*F)`` memory traffic, no
    ``(N, F, F)`` per-stock array. Per-stock valid-sample counts are kept (cheap,
    ``(N,)``) only for the ``min_periods`` prediction filter; the rolling window
    down-dates aged days from the aggregate (their raw vectors are retained).
    """

    def __init__(
        self,
        *,
        num_stocks: int,
        num_features: int,
        universe_size: int,
        min_periods: int | None,
        window: int | None,
        alpha: float,
    ) -> None:
        n_, f_ = int(num_stocks), int(num_features)
        self.num_stocks = n_
        self.num_features = f_
        self.universe_size = int(universe_size)
        self.min_periods = min_periods
        self.alpha = float(alpha)
        # Single aggregate over all valid samples.
        self.g_n = 0.0
        self.g_sx = np.zeros(f_)
        self.g_sxx = np.zeros((f_, f_))
        self.g_sy = 0.0
        self.g_sxy = np.zeros(f_)
        self.g_syy = 0.0
        # Per-stock valid-sample counts, for the `min_periods` prediction filter.
        self.counts = np.zeros(n_)
        # Raw days retained for down-dating, keyed by index (rolling window only).
        self._days: dict[int, tuple[np.ndarray, np.ndarray]] | None = {} if window is not None else None
        self.cached: IncrementalLinearParams | None = None
        self.fitted = False

    # -- sample pool --------------------------------------------------------
    def add_samples(self, index: int, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
        self.counts += valid
        self._fold(x, y, valid, 1.0)
        if self._days is not None:
            self._days[index] = (x.copy(), y.copy())

    def remove_samples(self, index: int) -> None:
        x, y = self._days.pop(index)
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
        self.counts -= valid
        self._fold(x, y, valid, -1.0)

    def predict(self, universe: np.ndarray, features: np.ndarray, refit_due: bool) -> np.ndarray:
        if (refit_due or not self.fitted) and self.g_n > 0:
            self.cached = _solve_standardized(
                self.g_n, self.g_sx, self.g_sxx, self.g_sy, self.g_sxy, self.g_syy, self.alpha
            )
            self.fitted = True

        mask = np.asarray(universe) > 0
        assert int(mask.sum()) <= self.universe_size, (
            f"incremental_mean_predictor: universe has {int(mask.sum())} nonzero entries, "
            f"exceeds universe_size={self.universe_size}"
        )
        if self.min_periods is not None:
            mask = mask & (self.counts >= self.min_periods)
        mask = mask & np.isfinite(features).all(axis=1)

        mu = np.full((self.num_stocks,), np.nan, dtype=np.float64)
        if self.fitted and mask.any():
            p = self.cached
            z = (features[mask] - p.x_mean) / p.x_std
            mu[mask] = z @ p.beta + p.intercept
        return mu

    # -- internals ----------------------------------------------------------
    def _fold(self, x: np.ndarray, y: np.ndarray, valid: np.ndarray, sign: float) -> None:
        """Fold (sign=+1) or unfold (sign=-1) a day's valid samples into the
        single aggregate via ``X^T X`` over the valid rows."""
        xr = x[valid]
        yr = y[valid]
        if xr.shape[0] == 0:
            return
        self.g_sxx += sign * (xr.T @ xr)
        self.g_sx += sign * xr.sum(axis=0)
        self.g_sxy += sign * (xr.T @ yr)
        self.g_sy += sign * float(yr.sum())
        self.g_syy += sign * float(yr @ yr)
        self.g_n += sign * float(xr.shape[0])


def rls_pool_factory(
    *,
    num_stocks: int,
    num_features: int,
    universe_size: int,
    min_periods: int | None,
    window: int | None,
    alpha: float,
) -> Callable[[], RlsMeanPool]:
    """A zero-arg factory building an `RlsMeanPool` with the bound config -- what a
    concrete predictor passes to the harness."""
    return lambda: RlsMeanPool(
        num_stocks=num_stocks,
        num_features=num_features,
        universe_size=universe_size,
        min_periods=min_periods,
        window=window,
        alpha=alpha,
    )


@dataclass(slots=True)
class _HarnessState:
    pool: object
    target_offset: int
    window: int | None
    refit_every: int
    n_added: int = 0
    n_removed: int = 0
    rebalance_count: int = 0


class IncrementalMeanPredictor:
    r"""Generic incremental panel-regression harness (see the module docstring).

    Input ports: ``(universe, features_series, target_series)``. The model lives
    entirely in ``pool``; the harness only schedules ``add_samples`` /
    ``remove_samples`` by monotone day index and the refit cadence, then calls
    ``pool.predict``.
    """

    def __init__(
        self,
        *,
        pool_factory: Callable[[], object],
        target_offset: int,
        refit_every: int = 1,
        window: int | None = None,
    ) -> None:
        assert target_offset >= 0
        assert refit_every >= 1
        assert window is None or window >= 1
        self._pool_factory = pool_factory
        self._target_offset = int(target_offset)
        self._refit_every = int(refit_every)
        self._window = None if window is None else int(window)

    def init(self, inputs, timestamp: int) -> _HarnessState:
        return _HarnessState(
            pool=self._pool_factory(),
            target_offset=self._target_offset,
            window=self._window,
            refit_every=self._refit_every,
        )

    @staticmethod
    def compute(
        state: _HarnessState,
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
        pool = state.pool
        off = state.target_offset

        # Fold in the days added since the last rebalance (each seen once).
        for i in range(state.n_added, n_pair):
            pool.add_samples(i, features_series_view[i], target_series_view[i + off])
        state.n_added = n_pair

        # Rolling window: drop the days that aged out of [n_pair - window, n_pair).
        if state.window is not None:
            window_start = max(0, n_pair - state.window)
            for i in range(state.n_removed, window_start):
                pool.remove_samples(i)
            state.n_removed = window_start

        refit_due = (state.rebalance_count % state.refit_every) == 0
        state.rebalance_count += 1

        mu = pool.predict(universe_view.to_numpy(), features_series_view[-1], refit_due)
        output.write(mu)
        return True
