"""Validation tests for ported trader operators (no pytest).

Run with:
    PYTHONPATH=python .venv-ft/Scripts/python.exe python/flowops/tests/test_traders.py
"""

from __future__ import annotations

import numpy as np

from flowops._testkit import FakeArrayView
from flowops.traders import benchmark as benchmark_mod
from flowops.traders import simple_trader as simple_trader_mod
from flowops.traders import random_trader as random_trader_mod


def _make_inputs(soft, close, adjust, upper, lower):
    """Build the 5-tuple of read-only array views for one tick."""
    return (
        FakeArrayView(soft),
        FakeArrayView(close),
        FakeArrayView(adjust),
        FakeArrayView(upper),
        FakeArrayView(lower),
    )


def _drive(op, ticks):
    """Run init + a compute() per tick. `ticks` is a list of dicts.

    Each tick dict has keys soft/close/adjust/upper/lower (arrays) and
    `sp_produced` (bool: was the soft-positions input updated this tick).
    Returns list of (holdings_value, cash) outputs.
    """
    first = ticks[0]
    inputs0 = _make_inputs(
        first["soft"], first["close"], first["adjust"], first["upper"], first["lower"]
    )
    state = op.init(inputs0, timestamp=0)

    outputs = []
    for k, tick in enumerate(ticks):
        inputs = _make_inputs(
            tick["soft"], tick["close"], tick["adjust"], tick["upper"], tick["lower"]
        )
        output = FakeArrayView(np.zeros(2), writable=True)
        produced = (bool(tick["sp_produced"]), True, True, True, True)
        ret = op.compute(state, inputs, output, timestamp=k * 86_400_000_000_000, produced=produced)
        assert ret is True, "compute() must return True"
        assert output.written is not None, "compute() must write output"
        assert output.written.shape == (2,), f"output shape must be (2,), got {output.written.shape}"
        assert np.all(np.isfinite(output.written)), f"non-finite output at tick {k}: {output.written}"
        outputs.append(tuple(output.written.tolist()))
    return outputs


def _scenario(N, T, rng, with_nan=True, with_limits=True):
    """Generate a T-tick scenario for N stocks: trending prices, an
    occasional rebalance signal, a suspended stock, and price limits."""
    base = rng.uniform(10.0, 100.0, size=N)
    ticks = []
    prev_close = base.copy()
    for t in range(T):
        # Random-walk closes.
        close = prev_close * (1.0 + rng.normal(0.0, 0.02, size=N))
        close = np.maximum(close, 0.5)

        # Suspend stock 0 on a couple of ticks (NaN close).
        if with_nan and t in (3, 4):
            close = close.copy()
            close[0] = np.nan

        adjust = np.ones(N)  # flat adjustment (no dividends) for determinism
        # +/-10% daily price limits relative to previous close.
        if with_limits:
            upper = prev_close * 1.10
            lower = prev_close * 0.90
        else:
            upper = np.full(N, np.nan)
            lower = np.full(N, np.nan)

        # Soft positions: long-only weights summing to ~1, refreshed every 5 ticks.
        sp_produced = (t % 5 == 0)
        w = rng.uniform(0.0, 1.0, size=N)
        w /= w.sum()
        soft = w

        ticks.append(
            dict(soft=soft, close=close, adjust=adjust, upper=upper, lower=lower, sp_produced=sp_produced)
        )
        prev_close = np.where(np.isfinite(close), close, prev_close)
    return ticks


def test_benchmark():
    rng = np.random.default_rng(0)
    N, T = 8, 30
    op = benchmark_mod.build(num_stocks=N, initial_cash=1_000_000.0, use_adjusts=True)
    ticks = _scenario(N, T, rng)
    outs = _drive(op, ticks)

    # Tick 0 signal is pending; nothing executed yet -> all cash.
    hv0, cash0 = outs[0]
    assert abs(hv0) < 1e-9, f"benchmark should hold nothing on tick 0, got hv={hv0}"
    assert abs(cash0 - 1_000_000.0) < 1e-6, f"benchmark cash on tick 0 should equal initial, got {cash0}"

    # After the first execution the portfolio should be mostly invested.
    hv_later, cash_later = outs[6]
    nav_later = hv_later + cash_later
    assert nav_later > 0, "NAV must stay positive"
    assert hv_later > 0, "benchmark should be invested after a rebalance"

    # num_stocks inferred from the input view when omitted.
    op2 = benchmark_mod.build(initial_cash=500_000.0, use_adjusts=False)
    outs2 = _drive(op2, _scenario(N, T, np.random.default_rng(1)))
    assert all(np.isfinite(hv) and np.isfinite(c) for hv, c in outs2)
    print(f"  benchmark: {T} ticks ok; final NAV={outs[-1][0] + outs[-1][1]:.2f}")


def test_simple_trader_default():
    rng = np.random.default_rng(2)
    N, T = 6, 30
    op = simple_trader_mod.build(
        num_stocks=N,
        initial_cash=1_000_000.0,
        lot_size=100.0,
        fee_base=5.0,
        fee_rate=0.0003,
    )
    ticks = _scenario(N, T, rng)
    outs = _drive(op, ticks)

    hv0, cash0 = outs[0]
    assert abs(hv0) < 1e-9, "trader holds nothing on tick 0 (pending signal)"
    assert abs(cash0 - 1_000_000.0) < 1e-6, "trader cash equals initial on tick 0"

    nav_final = outs[-1][0] + outs[-1][1]
    assert nav_final > 0, "NAV stays positive"
    # With fees + lot rounding the trader holds something after rebalances.
    assert any(hv > 0 for hv, _ in outs), "trader should be invested at some point"
    print(f"  simple_trader: {T} ticks ok; final NAV={nav_final:.2f}")


def test_simple_trader_custom_trade_fn():
    """Pass a custom trade_fn through build (the framework's extension point)."""
    N, T = 4, 12
    rng = np.random.default_rng(3)

    def all_in_first(state, sp):
        # Put everything into stock 0 (in lots) when tradable.
        lots = np.zeros(state.num_stocks)
        p = state._exec_price[0]
        if np.isfinite(p) and p > 0:
            target_shares = state._current_value / p
            lots[0] = np.round((target_shares - state.shares[0]) / state.lot_size)
        return lots

    op = simple_trader_mod.build(
        num_stocks=N, initial_cash=200_000.0, lot_size=100.0,
        fee_base=5.0, fee_rate=0.0003, trade_fn=all_in_first,
    )
    outs = _drive(op, _scenario(N, T, rng, with_nan=False))
    assert all(np.isfinite(hv) and np.isfinite(c) for hv, c in outs)
    print(f"  simple_trader(custom trade_fn): {T} ticks ok; final NAV={outs[-1][0] + outs[-1][1]:.2f}")


def test_random_trader():
    rng = np.random.default_rng(4)
    N, T = 12, 40
    op = random_trader_mod.build(
        num_stocks=N,
        portfolio_size=5,
        initial_cash=1_000_000.0,
        lot_size=100.0,
        fee_base=5.0,
        fee_rate=0.0003,
        seed=123,
    )
    ticks = _scenario(N, T, rng)
    outs = _drive(op, ticks)

    nav_final = outs[-1][0] + outs[-1][1]
    assert nav_final > 0, "NAV stays positive"
    assert any(hv > 0 for hv, _ in outs), "random trader should be invested at some point"

    # Reproducibility: same seed -> same outputs.
    op_a = random_trader_mod.build(num_stocks=N, portfolio_size=5, seed=999)
    op_b = random_trader_mod.build(num_stocks=N, portfolio_size=5, seed=999)
    sc = _scenario(N, T, np.random.default_rng(7))
    outs_a = _drive(op_a, sc)
    outs_b = _drive(op_b, sc)
    assert outs_a == outs_b, "same seed must give identical results"
    print(f"  random_trader: {T} ticks ok; final NAV={nav_final:.2f}; seed-reproducible")


def test_bankruptcy_absorbing():
    """Drive NAV non-positive (huge fees on tiny capital) and confirm the
    wiped-out state outputs (0, 0) thereafter."""
    N, T = 3, 8
    rng = np.random.default_rng(5)
    # Tiny capital, enormous flat fee per trade => trading bleeds to zero.
    op = simple_trader_mod.build(
        num_stocks=N, initial_cash=300.0, lot_size=1.0,
        fee_base=1_000.0, fee_rate=0.0,
    )
    ticks = _scenario(N, T, rng, with_nan=False, with_limits=False)
    # Force a strong long signal each tick so it tries to trade.
    for tk in ticks:
        tk["soft"] = np.full(N, 1.0 / N)
        tk["sp_produced"] = True
    outs = _drive(op, ticks)
    # Once wiped out, output should be exactly (0, 0).
    assert outs[-1] == (0.0, 0.0), f"expected absorbing (0,0) at end, got {outs[-1]}"
    print(f"  bankruptcy absorbing: end output={outs[-1]}")


def main():
    print("Running flowops traders tests...")
    test_benchmark()
    test_simple_trader_default()
    test_simple_trader_custom_trade_fn()
    test_random_trader()
    test_bankruptcy_absorbing()
    print("ALL TRADERS TESTS PASSED")


if __name__ == "__main__":
    main()
