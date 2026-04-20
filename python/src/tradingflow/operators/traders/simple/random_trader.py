"""Random trader — randomly selects stocks weighted by soft positions."""

from __future__ import annotations

import numpy as np

from ..simple_trader import SimpleTrader, SimpleTraderState


class RandomTrader(SimpleTrader):
    """Randomly select stocks with probability proportional to soft weights.

    On each rebalance, picks `portfolio_size` stocks (without
    replacement) using the soft position weights as sampling
    probabilities, then allocates equal market-value weight
    `1 / portfolio_size` to each chosen stock.

    Parameters
    ----------
    soft_positions
        Soft position weights, shape `(num_stocks,)`.
    prices
        Stacked OHLCV prices, shape `(num_stocks, 5)`.
    adjusts
        Stacked forward adjustment factors, shape `(num_stocks,)`.
    portfolio_size
        Number of stocks to hold.
    **kwargs
        Forwarded to [`SimpleTrader`][tradingflow.operators.traders.SimpleTrader].
    """

    def __init__(
        self,
        soft_positions,
        prices,
        adjusts,
        *,
        portfolio_size: int = 20,
        **kwargs,
    ) -> None:
        rng = np.random.default_rng()
        super().__init__(
            soft_positions,
            prices,
            adjusts,
            trade_fn=lambda state, sp: _trade_fn(state, sp, portfolio_size, rng),
            **kwargs,
        )


def _trade_fn(
    state: SimpleTraderState,
    soft_positions: np.ndarray,
    portfolio_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    N = state.num_stocks
    current_value = state._current_value
    exec_price = state._exec_price

    # Normalize soft weights into sampling probabilities.
    weights = np.maximum(soft_positions, 0.0)
    s = weights.sum()
    if s <= 0:
        return np.zeros(N, dtype=np.float64)
    weights /= s

    # Randomly select portfolio_size stocks (or fewer if not enough candidates).
    n_candidates = (weights > 0).sum()
    n_select = min(portfolio_size, n_candidates)
    if n_select == 0:
        return np.zeros(N, dtype=np.float64)

    chosen = rng.choice(N, n_select, replace=False, p=weights)

    # Equal-weight hard positions for chosen stocks.
    hard_positions = np.zeros(N, dtype=np.float64)
    hard_positions[chosen] = 1.0 / portfolio_size * s

    # Convert to lot counts.
    lots = np.zeros(N, dtype=np.float64)
    for i in range(N):
        p = exec_price[i]
        if not np.isfinite(p) or p <= 0:
            continue
        target_value = hard_positions[i] * current_value
        target_shares = target_value / p
        lots[i] = np.round((target_shares - state.shares[i]) / state.lot_size)

    return lots
