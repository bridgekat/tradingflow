"""Harness shared by every portfolio operator.

All three input shapes — mean only, variance only, both — do the same thing
around the solve: gate on the rebalance signal, mask down to the stocks worth
optimizing over, map lognormal moments to linear ones, hand the active
sub-problem to a solver, scatter the answer back and retain it. Only the
moments read differ, so that is the one hook, and it lives on the *state* so
that `reset` and `compute` need no `self`.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(slots=True)
class PortfolioState:
    """Everything a portfolio carries between generations.

    This mirrors the Rust `Operator::State`: `init` moves the build
    configuration in here, so `reset` and `compute` are free functions of
    `(inputs, state)` and nothing varies on the operator instance — which
    matters because a module binding `__op__` shares one instance across every
    node that loads it. `slots=True` makes a stray attribute an
    `AttributeError` rather than a silent cross-node alias.
    """

    #: `(state, active, universe, previous, mu, sigma) -> weights` over the
    #: active subset.
    solve: Callable
    logarithmic: bool
    max_universe_size: int
    #: The retained weights, re-emitted on every non-rebalance tick and handed
    #: to the next solve as its warm start.
    weights: np.ndarray
    #: Whatever `init_solver` built, or `None` for the closed-form portfolios.
    solver: object = None

    @staticmethod
    def moments(inputs):
        """Unpacks `(universe, mu, sigma)`, with `None` for what is absent."""
        raise NotImplementedError


def to_linear(mu: np.ndarray | None, sigma: np.ndarray | None):
    r"""Maps lognormal moments to linear-return moments.

    For log returns `r` with mean `m` and covariance `S`, the linear return
    `e^r - 1` has

        mu_lin[i]    = exp(m[i] + S[i, i] / 2) - 1
        Sigma_lin[i, j] = (1 + mu_lin[i]) (1 + mu_lin[j]) (exp(S[i, j]) - 1)

    with the specialisations that fall out when one moment is absent: no
    covariance means the variance term drops from the drift, and no mean means
    the drift is the variance term alone.
    """
    if sigma is None:
        return np.expm1(mu), None

    drift = 0.5 * np.diag(sigma)
    mu_lin = np.expm1(drift if mu is None else mu + drift)
    factor = 1.0 + mu_lin
    sigma_lin = np.outer(factor, factor) * np.expm1(sigma)
    return (None if mu is None else mu_lin), sigma_lin


class Portfolio:
    """Gate, mask, convert, solve, scatter, retain."""

    #: The [`PortfolioState`] subclass this portfolio builds, carrying which
    #: moments it reads.
    state_type: type[PortfolioState] = PortfolioState

    def __init__(
        self,
        *,
        solve,
        init_solver=None,
        max_universe_size: int | None = None,
        logarithmic: bool = True,
    ) -> None:
        assert max_universe_size is None or max_universe_size >= 1
        self.solve = solve
        self.init_solver = init_solver
        self.max_universe_size = max_universe_size
        self.logarithmic = logarithmic

    def init(self, inputs) -> PortfolioState:
        universe = inputs[1]
        n = universe.shape[0]
        m = n if self.max_universe_size is None else self.max_universe_size
        return self.state_type(
            solve=self.solve,
            logarithmic=self.logarithmic,
            max_universe_size=m,
            weights=np.zeros(n),
            solver=None if self.init_solver is None else self.init_solver(m),
        )

    @staticmethod
    def reset(_, state: PortfolioState):
        return (False, state.weights)

    @staticmethod
    def compute(inputs, state: PortfolioState, _):
        if not inputs[0]:
            return (False, state.weights)

        universe, mu, sigma = state.moments(inputs)

        # A stock is worth optimizing over when it is in the universe and every
        # moment the portfolio depends on is finite for it. A NaN prediction is
        # the predictor saying it has no opinion, not a zero one.
        mask = universe > 0
        if mu is not None:
            mask = mask & np.isfinite(mu)
        if sigma is not None:
            mask = mask & np.isfinite(np.diag(sigma))

        weights = np.zeros(universe.shape[0])
        if mask.any():
            active = int(mask.sum())
            if active > state.max_universe_size:
                raise ValueError(
                    f"active universe size {active} exceeds " f"max_universe_size {state.max_universe_size}"
                )

            sub_mu = None if mu is None else mu[mask]
            sub_sigma = None if sigma is None else sigma[np.ix_(mask, mask)]
            if sub_sigma is not None and not np.all(np.isfinite(sub_sigma)):
                # A finite diagonal is not enough: an off-diagonal NaN would
                # silently poison the whole solve.
                raise ValueError("active covariance block contains non-finite entries")

            if state.logarithmic:
                sub_mu, sub_sigma = to_linear(sub_mu, sub_sigma)

            weights[mask] = state.solve(
                state,
                np.nonzero(mask)[0],
                universe[mask],
                state.weights[mask],
                sub_mu,
                sub_sigma,
            )

        state.weights = weights
        return (True, state.weights)


@dataclass(slots=True)
class MeanPortfolioState(PortfolioState):
    @staticmethod
    def moments(inputs):
        _, universe, mu = inputs
        return universe, mu, None


@dataclass(slots=True)
class VariancePortfolioState(PortfolioState):
    @staticmethod
    def moments(inputs):
        _, universe, sigma = inputs
        return universe, None, sigma


@dataclass(slots=True)
class MeanVariancePortfolioState(PortfolioState):
    @staticmethod
    def moments(inputs):
        _, universe, mu, sigma = inputs
        return universe, mu, sigma


class MeanPortfolio(Portfolio):
    """Weights from predicted returns alone."""

    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[bool, np.ndarray]
    type Context = int
    type State = MeanPortfolioState

    state_type = MeanPortfolioState


class VariancePortfolio(Portfolio):
    """Weights from a covariance matrix alone — risk with no view on return."""

    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[bool, np.ndarray]
    type Context = int
    type State = VariancePortfolioState

    state_type = VariancePortfolioState


class MeanVariancePortfolio(Portfolio):
    """Weights from both predicted returns and covariance."""

    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[bool, np.ndarray]
    type Context = int
    type State = MeanVariancePortfolioState

    state_type = MeanVariancePortfolioState


def mean_portfolio(weights_of, **kwargs) -> MeanPortfolio:
    """Wraps a `predicted_returns -> weights` function into a mean portfolio."""
    return MeanPortfolio(
        solve=lambda state, active, universe, previous, mu, sigma: weights_of(mu),
        **kwargs,
    )
