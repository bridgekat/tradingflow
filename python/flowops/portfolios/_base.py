"""Abstract portfolio operator bases (ported to the flowops host contract).

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

    Inputs: (universe, predicted_returns), both shape ``(num_stocks,)``.
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

    def init(self, inputs, timestamp: int) -> MeanPortfolioState:
        num_stocks = self._num_stocks
        if num_stocks is None:
            num_stocks = int(inputs[1].shape[0])
        return MeanPortfolioState(
            num_stocks=num_stocks,
            logarithmic=self._logarithmic,
            positions_fn=self._positions_fn,
        )

    @staticmethod
    def compute(
        state: MeanPortfolioState,
        inputs,
        output,
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        # Trigger on universe updates: the universe is the canonical
        # rebalance signal, and the predictor's mu is stored as the last
        # prediction even when it did not produce this cycle.
        if not produced[0]:
            return False

        universe = inputs[0].value()
        mu = inputs[1].value()

        mask = (universe > 0) & np.isfinite(mu)
        sub_mu = mu[mask]
        # Lognormal conversion (zero-covariance specialisation):
        #   mu_lin[i] = exp(mu_log[i]) - 1
        if state.logarithmic:
            sub_mu = np.expm1(sub_mu)

        positions = np.zeros_like(universe, dtype=np.float64)
        if mask.any():
            positions[mask] = state.positions_fn(state, sub_mu)

        output.write(positions)
        return True


# ---------------------------------------------------------------------------
# Variance portfolio
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VariancePortfolioState:
    """Mutable state for `VariancePortfolio` subclasses."""

    num_stocks: int
    logarithmic: bool
    positions_fn: Callable[["VariancePortfolioState", np.ndarray], np.ndarray]


class VariancePortfolio:
    """Abstract portfolio constructor from covariance alone (no expected returns).

    Inputs: (universe ``(num_stocks,)``, predicted_covariances ``(num_stocks, num_stocks)``).
    Output: position weights, shape ``(num_stocks,)``.
    """

    def __init__(
        self,
        *,
        positions_fn: Callable[[VariancePortfolioState, np.ndarray], np.ndarray],
        num_stocks: int | None = None,
        logarithmic: bool = True,
    ) -> None:
        self._num_stocks = num_stocks
        self._logarithmic = logarithmic
        self._positions_fn = positions_fn

    def init(self, inputs, timestamp: int) -> VariancePortfolioState:
        num_stocks = self._num_stocks
        if num_stocks is None:
            num_stocks = int(inputs[0].shape[0])
        return VariancePortfolioState(
            num_stocks=num_stocks,
            logarithmic=self._logarithmic,
            positions_fn=self._positions_fn,
        )

    @staticmethod
    def compute(
        state: VariancePortfolioState,
        inputs,
        output,
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        # Trigger on universe updates: the universe is the canonical
        # rebalance signal, and the predictor's sigma is stored as the
        # last prediction even when it did not produce this cycle.
        if not produced[0]:
            return False

        universe = inputs[0].value()
        sigma = inputs[1].value()

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
            positions[mask] = state.positions_fn(state, sub_sigma)

        output.write(positions)
        return True


# ---------------------------------------------------------------------------
# Mean-variance portfolio
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MeanVariancePortfolioState:
    """Mutable state for `MeanVariancePortfolio` subclasses."""

    num_stocks: int
    logarithmic: bool
    positions_fn: Callable[
        ["MeanVariancePortfolioState", np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]


class MeanVariancePortfolio:
    """Abstract portfolio constructor from predicted returns and covariance.

    Inputs: (universe ``(num_stocks,)``, predicted_returns ``(num_stocks,)``,
    predicted_covariances ``(num_stocks, num_stocks)``).
    Output: position weights, shape ``(num_stocks,)``.

    `positions_fn` is ``(state, mu, Sigma, x_bm) -> weights`` over the active subset,
    with both moments in linear-return units and ``x_bm`` the benchmark subset
    (renormalised to sum to 1).
    """

    def __init__(
        self,
        *,
        positions_fn: Callable[
            [MeanVariancePortfolioState, np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ],
        num_stocks: int | None = None,
        logarithmic: bool = True,
    ) -> None:
        self._num_stocks = num_stocks
        self._logarithmic = logarithmic
        self._positions_fn = positions_fn

    def init(self, inputs, timestamp: int) -> MeanVariancePortfolioState:
        num_stocks = self._num_stocks
        if num_stocks is None:
            num_stocks = int(inputs[1].shape[0])
        return MeanVariancePortfolioState(
            num_stocks=num_stocks,
            logarithmic=self._logarithmic,
            positions_fn=self._positions_fn,
        )

    @staticmethod
    def compute(
        state: MeanVariancePortfolioState,
        inputs,
        output,
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        # Trigger on universe updates: the universe is the canonical
        # rebalance signal, and both mu and sigma are stored as the last
        # predictions even when they did not produce this cycle.
        if not produced[0]:
            return False

        universe = inputs[0].value()
        mu = inputs[1].value()
        sigma = inputs[2].value()

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
            # Benchmark subset, renormalised to sum to 1 over the active
            # subset.  When some universe>0 stocks are masked out (NaN in
            # mu or sigma diagonal), their cap weight is redistributed
            # proportionally over the kept names so that x_bm still
            # represents a valid full-position benchmark for the same
            # subset that x will be optimised over.  Falls back to equal
            # weights if the kept benchmark mass is non-positive.
            sub_universe = universe[mask]
            s = float(sub_universe.sum())
            sub_universe = (
                sub_universe / s
                if s > 0
                else np.full(sub_universe.shape, 1.0 / sub_universe.size)
            )
            positions[mask] = state.positions_fn(state, sub_mu, sub_sigma, sub_universe)

        output.write(positions)
        return True
