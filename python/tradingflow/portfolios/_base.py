"""Abstract portfolio operator bases (ported to the pyhost contract).

Ported from `tradingflow.operators.portfolios.{mean,variance,mean_variance}_portfolio`.
The `init`/`compute` bodies are verbatim; only the handle-taking `__init__` and the
`tradingflow` base class are dropped.  Leaf modules subclass one of these and expose
`build(**kwargs)`.

Each base takes the per-stock dimension (`num_stocks`) and the `logarithmic` flag as
plain config; `num_stocks` may be left `None` and is then read from the input views at
`init` time (the universe / predicted-returns view shape).
"""

from __future__ import annotations

from typing import Callable
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Mean portfolio
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MeanPortfolioState:
    """Mutable state for `MeanPortfolio` subclasses."""

    num_stocks: int
    logarithmic: bool
    positions_fn: Callable[["MeanPortfolioState", np.ndarray], np.ndarray]


class MeanPortfolio:
    """Abstract portfolio constructor from per-stock predictions.

    Inputs: (rebalance_signal, universe, predicted_returns), the arrays both
    of shape ``(num_stocks,)``.
    Output: position weights, shape ``(num_stocks,)``.
    """

    def __init__(
        self,
        *,
        positions_fn: Callable[[MeanPortfolioState, np.ndarray], np.ndarray],
        num_stocks: int | None = None,
        logarithmic: bool = True,
    ) -> None:
        self._num_stocks = num_stocks
        self._logarithmic = logarithmic
        self._positions_fn = positions_fn

    def init(self, inputs) -> MeanPortfolioState:
        num_stocks = self._num_stocks
        if num_stocks is None:
            # inputs = (rebalance_signal, universe, mu): read the universe.
            num_stocks = int(inputs[1].shape[0])
        return MeanPortfolioState(
            num_stocks=num_stocks,
            logarithmic=self._logarithmic,
            positions_fn=self._positions_fn,
        )

    @staticmethod
    def compute(
        inputs,
        state: MeanPortfolioState,
        timestamp: int,
    ) -> np.ndarray | None:
        # Trigger on the leading rebalance signal; the predictor's mu is
        # read as the last prediction even when it did not produce this
        # cycle.
        if not inputs[0]:
            return None

        universe = inputs[1].value()
        mu = inputs[2].value()

        mask = (universe > 0) & np.isfinite(mu)
        sub_mu = mu[mask]
        # Lognormal conversion (zero-covariance specialisation):
        #   mu_lin[i] = exp(mu_log[i]) - 1
        if state.logarithmic:
            sub_mu = np.expm1(sub_mu)

        positions = np.zeros_like(universe, dtype=np.float64)
        if mask.any():
            positions[mask] = state.positions_fn(state, sub_mu)

        return positions


# ---------------------------------------------------------------------------
# Warm-startable solving portfolios
# ---------------------------------------------------------------------------
#
# The `Variance` / `MeanVariance` bases drive iterative solvers, so they share a
# warm-start contract that the closed-form `Mean` leaves above do not need:
#
#   * `init_fn(max_universe_size)` (optional) builds a *fixed-size* solver once at
#     `init` time and returns an opaque handle stored in `state.solver`.  The
#     solving leaves size their CVXPY `Parameter`s / `Variable` to
#     `max_universe_size` so each rebalance re-solves the same (DPP) problem with
#     new parameter values rather than rebuilding it.  Every solve therefore
#     receives active arrays of length `<= max_universe_size`.
#   * the base carries the previous rebalance's full-universe weights in
#     `state.previous_positions` and hands each solve `sub_previous_positions =
#     previous_positions[mask]` — the prior weights permuted into the *new* active
#     indexing (names that left the universe are dropped; new names seed at 0) —
#     so the leaf can seed the solver's initial point.
#
# Unified solve signature for both bases:
#   ``positions_fn(state, active_indices, sub_universe, sub_previous_positions,
#                  sub_mu, sub_sigma)``
# `active_indices` are the global indices of the active stocks (mask order), for a
# persistent slot allocator that keeps continuing names in fixed solver slots so
# the warm start stays aligned.  `sub_mu is None` for the variance-only base.  All
# active arrays are length `n = mask.sum() (<= max_universe_size)`.


@dataclass(slots=True)
class VariancePortfolioState:
    """Mutable state for `VariancePortfolio` subclasses."""

    num_stocks: int
    max_universe_size: int
    logarithmic: bool
    positions_fn: Callable[..., np.ndarray]
    solver: object
    previous_positions: np.ndarray


class VariancePortfolio:
    """Abstract portfolio constructor from covariance alone (no expected returns).

    Inputs: (rebalance_signal, universe ``(num_stocks,)``,
    predicted_covariances ``(num_stocks, num_stocks)``).
    Output: position weights, shape ``(num_stocks,)``.

    `positions_fn` is ``(state, sub_universe, sub_previous_positions, None, sub_sigma)``
    over the active subset (`sub_mu` is `None`); `sub_universe` is renormalised to
    sum to 1 and `sub_previous_positions` is the prior solution in the new indexing.
    """

    def __init__(
        self,
        *,
        positions_fn: Callable[..., np.ndarray],
        init_fn: Callable[[int], object] | None = None,
        num_stocks: int | None = None,
        max_universe_size: int | None = None,
        logarithmic: bool = True,
    ) -> None:
        self._num_stocks = num_stocks
        self._max_universe_size = max_universe_size
        self._logarithmic = logarithmic
        self._positions_fn = positions_fn
        self._init_fn = init_fn

    def init(self, inputs) -> VariancePortfolioState:
        num_stocks = self._num_stocks
        if num_stocks is None:
            # inputs = (rebalance_signal, universe, sigma): read the universe.
            num_stocks = int(inputs[1].shape[0])
        max_universe_size = self._max_universe_size if self._max_universe_size is not None else num_stocks
        solver = self._init_fn(max_universe_size) if self._init_fn is not None else None
        return VariancePortfolioState(
            num_stocks=num_stocks,
            max_universe_size=max_universe_size,
            logarithmic=self._logarithmic,
            positions_fn=self._positions_fn,
            solver=solver,
            previous_positions=np.zeros(num_stocks, dtype=np.float64),
        )

    @staticmethod
    def compute(
        inputs,
        state: VariancePortfolioState,
        timestamp: int,
    ) -> np.ndarray | None:
        # Trigger on the leading rebalance signal; the predictor's sigma is
        # read as the last prediction even when it did not produce this
        # cycle.
        if not inputs[0]:
            return None

        universe = inputs[1].value()
        sigma = inputs[2].value()

        mask = (universe > 0) & np.isfinite(np.diag(sigma))
        sub_sigma = sigma[np.ix_(mask, mask)]
        if not np.all(np.isfinite(sub_sigma)):
            raise ValueError("sub-covariance matrix contains non-finite entries")
        # Lognormal conversion (zero-mean specialisation):
        #   mu_lin[i]       = exp(0.5 * Sigma_log[i, i]) - 1
        #   Sigma_lin[i, j] = (1 + mu_lin[i])(1 + mu_lin[j])
        #                     * (exp(Sigma_log[i, j]) - 1)
        if state.logarithmic:
            sub_mu = np.expm1(0.5 * np.diag(sub_sigma))
            factor = 1.0 + sub_mu
            sub_sigma = np.outer(factor, factor) * np.expm1(sub_sigma)

        positions = np.zeros_like(universe, dtype=np.float64)
        if mask.any():
            k = int(mask.sum())
            if k > state.max_universe_size:
                raise ValueError(f"active universe size {k} exceeds max_universe_size " f"{state.max_universe_size}")
            sub_universe = universe[mask]
            active_indices = np.nonzero(mask)[0]
            sub_previous_positions = state.previous_positions[mask]
            positions[mask] = state.positions_fn(
                state, active_indices, sub_universe, sub_previous_positions, None, sub_sigma
            )

        state.previous_positions = positions
        return positions


@dataclass(slots=True)
class MeanVariancePortfolioState:
    """Mutable state for `MeanVariancePortfolio` subclasses."""

    num_stocks: int
    max_universe_size: int
    logarithmic: bool
    positions_fn: Callable[..., np.ndarray]
    solver: object
    previous_positions: np.ndarray


class MeanVariancePortfolio:
    """Abstract portfolio constructor from predicted returns and covariance.

    Inputs: (rebalance_signal, universe ``(num_stocks,)``, predicted_returns
    ``(num_stocks,)``, predicted_covariances ``(num_stocks, num_stocks)``).
    Output: position weights, shape ``(num_stocks,)``.

    `positions_fn` is ``(state, sub_universe, sub_previous_positions, sub_mu,
    sub_sigma) -> weights`` over the active subset, with both moments in
    linear-return units, `sub_universe` the benchmark subset (renormalised to sum
    to 1), and `sub_previous_positions` the prior solution permuted into the new
    indexing (for warm-starting the solver).
    """

    def __init__(
        self,
        *,
        positions_fn: Callable[..., np.ndarray],
        init_fn: Callable[[int], object] | None = None,
        num_stocks: int | None = None,
        max_universe_size: int | None = None,
        logarithmic: bool = True,
    ) -> None:
        self._num_stocks = num_stocks
        self._max_universe_size = max_universe_size
        self._logarithmic = logarithmic
        self._positions_fn = positions_fn
        self._init_fn = init_fn

    def init(self, inputs) -> MeanVariancePortfolioState:
        num_stocks = self._num_stocks
        if num_stocks is None:
            # inputs = (rebalance_signal, universe, mu, sigma): read the
            # universe.
            num_stocks = int(inputs[1].shape[0])
        max_universe_size = self._max_universe_size if self._max_universe_size is not None else num_stocks
        solver = self._init_fn(max_universe_size) if self._init_fn is not None else None
        return MeanVariancePortfolioState(
            num_stocks=num_stocks,
            max_universe_size=max_universe_size,
            logarithmic=self._logarithmic,
            positions_fn=self._positions_fn,
            solver=solver,
            previous_positions=np.zeros(num_stocks, dtype=np.float64),
        )

    @staticmethod
    def compute(
        inputs,
        state: MeanVariancePortfolioState,
        timestamp: int,
    ) -> np.ndarray | None:
        # Trigger on the leading rebalance signal; both mu and sigma are
        # read as the last predictions even when they did not produce this
        # cycle.
        if not inputs[0]:
            return None

        universe = inputs[1].value()
        mu = inputs[2].value()
        sigma = inputs[3].value()

        mask = (universe > 0) & np.isfinite(mu) & np.isfinite(np.diag(sigma))
        sub_mu = mu[mask]
        sub_sigma = sigma[np.ix_(mask, mask)]
        if not np.all(np.isfinite(sub_sigma)):
            raise ValueError("sub-covariance matrix contains non-finite entries")
        # Full lognormal → linear-return moment map:
        #   mu_lin[i]       = exp(mu_log[i] + 0.5 * Sigma_log[i, i]) - 1
        #   Sigma_lin[i, j] = (1 + mu_lin[i])(1 + mu_lin[j])
        #                     * (exp(Sigma_log[i, j]) - 1)
        if state.logarithmic:
            sub_mu = np.expm1(sub_mu + 0.5 * np.diag(sub_sigma))
            factor = 1.0 + sub_mu
            sub_sigma = np.outer(factor, factor) * np.expm1(sub_sigma)

        positions = np.zeros_like(universe, dtype=np.float64)
        if mask.any():
            k = int(mask.sum())
            if k > state.max_universe_size:
                raise ValueError(f"active universe size {k} exceeds max_universe_size " f"{state.max_universe_size}")
            sub_universe = universe[mask]
            active_indices = np.nonzero(mask)[0]
            sub_previous_positions = state.previous_positions[mask]
            positions[mask] = state.positions_fn(
                state, active_indices, sub_universe, sub_previous_positions, sub_mu, sub_sigma
            )

        state.previous_positions = positions
        return positions
