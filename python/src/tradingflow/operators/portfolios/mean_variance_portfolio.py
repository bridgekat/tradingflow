"""Abstract mean-variance portfolio operator."""

from __future__ import annotations

from typing import Callable
from dataclasses import dataclass

import numpy as np

from ... import ArrayView, Handle, NodeKind, Operator


@dataclass(slots=True)
class MeanVariancePortfolioState:
    """Mutable state for [`MeanVariancePortfolio`][tradingflow.operators.portfolios.mean_variance_portfolio.MeanVariancePortfolio] subclasses."""

    num_stocks: int
    logarithmic: bool
    positions_fn: Callable[["MeanVariancePortfolioState", np.ndarray, np.ndarray], np.ndarray]


class MeanVariancePortfolio(
    Operator[
        ArrayView[np.float64],
        ArrayView[np.float64],
        ArrayView[np.float64],
        ArrayView[np.float64],
        MeanVariancePortfolioState,
    ]
):
    r"""Abstract portfolio constructor from predicted returns and covariance.

    Triggered by `universe` updates — the universe is the canonical
    rebalance signal.  On each trigger, delegates to `positions_fn` to
    compute position weights.  Only stocks with positive universe
    weights, finite predicted returns, and finite diagonal covariance
    entries are passed to `positions_fn`; the result is scattered back
    to the full dimension with zeros elsewhere.

    The rebalance cadence is inherited from upstream: `universe` is
    typically clocked by a rebalance clock (e.g. via
    [`Clocked`][tradingflow.operators.clocked.Clocked]), so this operator runs
    at that cadence.  `predicted_returns` and `predicted_covariances`
    are read as the latest stored predictions at the trigger — neither
    need produce on the same cycle.

    ## Expected prediction semantics

    Controlled by `logarithmic`:

    - `logarithmic=True` (default): both `predicted_returns` and
      `predicted_covariances` are in **log-return** units.  The base
      class maps them to linear-return moments via the full lognormal
      moment map before calling `positions_fn`:

          mu_lin[i]       = exp(mu_log[i] + ½ Sigma_log[i, i]) - 1
          Sigma_lin[i, j] = (1 + mu_lin[i]) (1 + mu_lin[j])
                            · (exp(Sigma_log[i, j]) - 1)

    - `logarithmic=False`: both inputs are already in **linear-return**
      units and are forwarded to `positions_fn` unchanged.

    Either way, `positions_fn` always sees matched linear-return
    mean/covariance pairs — exactly the units Markowitz-style
    objectives \(\mu^T w - \frac{\gamma}{2} w^T \Sigma w\) expect.
    Training the upstream predictors on log returns (more symmetric,
    closer to Gaussian) while letting this boundary handle the
    conversion tends to give tighter estimates than directly modelling
    linear returns; the two upstream predictors must of course agree
    on the target series so the units on the way in are consistent.

    ## NaN behavior

    `predicted_returns` and `predicted_covariances` are allowed to
    contain `NaN` entries — per the
    [`MeanPredictor`][tradingflow.operators.predictors.mean_predictor.MeanPredictor]
    and
    [`VariancePredictor`][tradingflow.operators.predictors.variance_predictor.VariancePredictor]
    contracts, these mark stocks with insufficient data.  The base
    class subsets to `(universe > 0) & np.isfinite(mu) &
    np.isfinite(np.diag(Sigma))` before calling `positions_fn`, so
    subclasses never see `NaN` inputs.  The sub-covariance-matrix of
    the remaining stocks must not contain non-finite off-diagonal
    entries (the base class raises `ValueError` if it does).  The
    emitted position vector is zero for stocks outside the valid
    subset.

    Parameters
    ----------
    universe
        Handle to universe weights, shape `(num_stocks,)`.
        Stocks with positive values are included in the optimization.
    predicted_returns
        Handle to predicted returns array, shape `(num_stocks,)`.  Log
        or linear depending on `logarithmic`.
    predicted_covariances
        Handle to predicted covariance matrix, shape
        `(num_stocks, num_stocks)`.  Log or linear depending on
        `logarithmic`.
    positions_fn
        `(state, mu, Sigma) -> weights`.  Receives only the
        universe-active subset, with both moments in linear-return units.
    logarithmic
        If `True` (default), the two inputs are interpreted as
        log-return moments and converted via the full lognormal moment
        map before the inner call; if `False`, they are interpreted as
        linear-return moments and passed through unchanged.
    """

    def __init__(
        self,
        universe: Handle,
        predicted_returns: Handle,
        predicted_covariances: Handle,
        *,
        positions_fn: Callable[[MeanVariancePortfolioState, np.ndarray, np.ndarray], np.ndarray],
        logarithmic: bool = True,
    ) -> None:
        assert len(universe.shape) == 1
        assert len(predicted_returns.shape) == 1
        assert len(predicted_covariances.shape) == 2
        assert predicted_returns.shape[0] == universe.shape[0]
        assert predicted_covariances.shape[0] == universe.shape[0]
        assert predicted_covariances.shape[1] == universe.shape[0]

        self._num_stocks = predicted_returns.shape[0]
        self._logarithmic = logarithmic
        self._positions_fn = positions_fn

        inputs = (universe, predicted_returns, predicted_covariances)
        super().__init__(
            inputs=inputs,
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(self._num_stocks,),
            name=type(self).__name__,
        )

    def init(
        self,
        inputs: tuple[
            ArrayView[np.float64],
            ArrayView[np.float64],
            ArrayView[np.float64],
        ],
        timestamp: int,
    ) -> MeanVariancePortfolioState:
        return MeanVariancePortfolioState(
            num_stocks=self._num_stocks,
            logarithmic=self._logarithmic,
            positions_fn=self._positions_fn,
        )

    @staticmethod
    def compute(
        state: MeanVariancePortfolioState,
        inputs: tuple[
            ArrayView[np.float64],
            ArrayView[np.float64],
            ArrayView[np.float64],
        ],
        output: ArrayView[np.float64],
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
            positions[mask] = state.positions_fn(state, sub_mu, sub_sigma)

        output.write(positions)
        return True
