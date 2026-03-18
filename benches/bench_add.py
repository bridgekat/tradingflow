"""Benchmark: element-wise addition via raw NumPy vs. Rust/Python operators.

Run::

    python benches/bench_add.py

Optional environment variable `BENCH_N` overrides the default series length.
"""

from __future__ import annotations

import asyncio
import os
import timeit

import numpy as np

from tradingflow import Observable, Scenario, Series
from tradingflow.operators import add as py_add
from tradingflow.operators.apply import Apply
from tradingflow.sources import ArrayBundleSource

from tradingflow._native import bench_add_compute as _rust_add_compute
from tradingflow._native import bench_add_compute_obs as _rust_add_compute_obs
from tradingflow._native import bench_add_loop as _rust_add_loop
from tradingflow._native import bench_add_loop_interleaved as _rust_add_loop_interleaved
from tradingflow._native import bench_add_loop_fnptr as _rust_add_loop_fnptr
from tradingflow._native import bench_scenario_compute as _rust_scenario_compute
from tradingflow._native import bench_scenario_compute_obs as _rust_scenario_compute_obs
from tradingflow._native import bench_scenario_chain as _rust_scenario_chain
from tradingflow._native import bench_scenario_chain_obs as _rust_scenario_chain_obs
from tradingflow._native import bench_scenario_sparse as _rust_scenario_sparse
from tradingflow._native import bench_scenario_sparse_obs as _rust_scenario_sparse_obs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stats(samples: list[float]) -> tuple[float, float, float]:
    mn = min(samples)
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / len(samples)
    return mn, mean, var**0.5


def _make_arrays(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = np.ascontiguousarray(rng.standard_normal(n), dtype=np.float64)
    b = np.ascontiguousarray(rng.standard_normal(n), dtype=np.float64)
    ts = np.arange(n, dtype=np.int64)
    return a, b, ts


def _print_row(label: str, samples: list[float], min_np: float, col: int = 28) -> None:
    mn, mean, std = _stats(samples)
    ratio = mn / min_np
    print(f"{label:<{col}}  {mn:>10.2f}  {mean:>10.2f}  {std:>10.2f}  {ratio:>7.2f}x")


# ---------------------------------------------------------------------------
# Python compute benchmark (no Scenario — tight loop)
# ---------------------------------------------------------------------------


def _py_add_compute(raw_a: np.ndarray, raw_b: np.ndarray, timestamps_ns: np.ndarray) -> Series:
    """Python equivalent of bench_add_compute: Observable + Apply in a tight loop."""
    n = len(raw_a)
    obs_a = Observable((), np.float64)
    obs_b = Observable((), np.float64)
    obs_a.write(np.float64(0.0))
    obs_b.write(np.float64(0.0))

    def _add_fn(args):
        return args[0] + args[1]

    op = Apply((obs_a, obs_b), (), np.float64, _add_fn)
    state = op.init_state()
    out = Series((), np.float64)

    for i in range(n):
        obs_a.write(np.float64(raw_a[i]))
        obs_b.write(np.float64(raw_b[i]))
        ts = timestamps_ns[i].view("datetime64[ns]")
        value, state = op.compute(ts, (obs_a, obs_b), state)
        if value is not None:
            out.append_unchecked(ts, value)

    return out


def _py_add_compute_obs(raw_a: np.ndarray, raw_b: np.ndarray, timestamps_ns: np.ndarray) -> float:
    """Python equivalent of bench_add_compute_obs: Observable only, no Series output."""
    n = len(raw_a)
    obs_a = Observable((), np.float64)
    obs_b = Observable((), np.float64)
    obs_out = Observable((), np.float64)
    obs_a.write(np.float64(0.0))
    obs_b.write(np.float64(0.0))
    obs_out.write(np.float64(0.0))

    def _add_fn(args):
        return args[0] + args[1]

    op = Apply((obs_a, obs_b), (), np.float64, _add_fn)
    state = op.init_state()

    for i in range(n):
        obs_a.write(np.float64(raw_a[i]))
        obs_b.write(np.float64(raw_b[i]))
        ts = timestamps_ns[i].view("datetime64[ns]")
        value, state = op.compute(ts, (obs_a, obs_b), state)
        if value is not None:
            obs_out.write(value)

    return float(obs_out.last)


# ---------------------------------------------------------------------------
# Python Scenario benchmark (full async runtime)
# ---------------------------------------------------------------------------


def _py_scenario_compute(raw_a: np.ndarray, raw_b: np.ndarray, timestamps_ns: np.ndarray) -> Series:
    """Python equivalent of bench_scenario_compute: Scenario + add operator."""
    ts_dt = timestamps_ns.astype("datetime64[ns]")

    src_a = ArrayBundleSource.from_arrays(timestamps=ts_dt, values=raw_a, initial=0.0)
    src_b = ArrayBundleSource.from_arrays(timestamps=ts_dt, values=raw_b, initial=0.0)

    scenario = Scenario()
    a = scenario.add_source(src_a)
    b = scenario.add_source(src_b)
    result_obs = scenario.add_operator(py_add(a, b))
    result = scenario.materialize(result_obs)
    asyncio.run(scenario.run())
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run(n: int, repeats: int = 5, number: int = 200) -> None:
    print(f"Series length N = {n:,}   (repeats={repeats}, number={number})\n")

    raw_a, raw_b, timestamps_ns = _make_arrays(n)

    # NumPy baseline: copy both inputs + add (3 arrays, fair comparison)
    def bench_numpy():
        ca = raw_a.copy()
        cb = raw_b.copy()
        return ca + cb

    # Warm up + validate
    np_result = bench_numpy()
    rust_loop_result = _rust_add_loop(raw_a, raw_b, timestamps_ns)
    rust_loop_il_result = _rust_add_loop_interleaved(raw_a, raw_b, timestamps_ns)
    rust_loop_fnptr_result = _rust_add_loop_fnptr(raw_a, raw_b, timestamps_ns)
    rust_result = _rust_add_compute(raw_a, raw_b, timestamps_ns)
    rust_obs_result = _rust_add_compute_obs(raw_a, raw_b, timestamps_ns)
    scenario_result = _rust_scenario_compute(raw_a, raw_b, timestamps_ns)
    scenario_obs_result = _rust_scenario_compute_obs(raw_a, raw_b, timestamps_ns)
    # py_result = _py_add_compute(raw_a, raw_b, timestamps_ns)
    # py_obs_result = _py_add_compute_obs(raw_a, raw_b, timestamps_ns)
    # py_scenario_result = _py_scenario_compute(raw_a, raw_b, timestamps_ns)

    np.testing.assert_allclose(rust_loop_result.values_array(), np_result)
    np.testing.assert_allclose(rust_loop_il_result.values_array(), np_result)
    np.testing.assert_allclose(rust_loop_fnptr_result.values_array(), np_result)
    np.testing.assert_allclose(rust_result.values_array(), np_result)
    np.testing.assert_allclose(scenario_result.values_array(), np_result)
    np.testing.assert_allclose(rust_obs_result.values_array(), [np_result[-1]])
    np.testing.assert_allclose(scenario_obs_result.values_array(), [np_result[-1]])
    # np.testing.assert_allclose(py_result.values, np_result)
    # np.testing.assert_allclose(py_obs_result, np_result[-1])
    # np.testing.assert_allclose(py_scenario_result.values, np_result)

    # -- Reduced iteration count for slow Python benchmarks --
    # py_number = max(1, number // 20)  # noqa: F841

    # Timings
    raw_numpy = timeit.repeat(bench_numpy, number=number, repeat=repeats)
    raw_rust_loop = timeit.repeat(lambda: _rust_add_loop(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats)
    raw_rust_loop_il = timeit.repeat(
        lambda: _rust_add_loop_interleaved(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats
    )
    raw_rust_loop_fnptr = timeit.repeat(
        lambda: _rust_add_loop_fnptr(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats
    )
    raw_rust = timeit.repeat(lambda: _rust_add_compute(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats)
    raw_rust_obs = timeit.repeat(
        lambda: _rust_add_compute_obs(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats
    )
    raw_scenario = timeit.repeat(
        lambda: _rust_scenario_compute(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats
    )
    raw_scenario_obs = timeit.repeat(
        lambda: _rust_scenario_compute_obs(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats
    )
    # raw_py = timeit.repeat(
    #     lambda: _py_add_compute(raw_a, raw_b, timestamps_ns), number=py_number, repeat=repeats
    # )
    # raw_py_obs = timeit.repeat(
    #     lambda: _py_add_compute_obs(raw_a, raw_b, timestamps_ns), number=py_number, repeat=repeats
    # )
    # raw_py_scenario = timeit.repeat(
    #     lambda: _py_scenario_compute(raw_a, raw_b, timestamps_ns), number=py_number, repeat=repeats
    # )

    to_us = lambda raw, num=number: [t / num * 1e6 for t in raw]
    min_np, _, _ = _stats(to_us(raw_numpy))

    col = 28
    print(f"{'approach':<{col}}  {'min (us)':>10}  {'mean (us)':>10}  {'stdev (us)':>10}  {'ratio':>8}")
    print("-" * (col + 50))

    rows = [
        ("numpy", to_us(raw_numpy)),
        ("rust (3x sep)", to_us(raw_rust_loop)),
        ("rust (3x interleaved)", to_us(raw_rust_loop_il)),
        ("rust (3x il+fnptr)", to_us(raw_rust_loop_fnptr)),
        ("rust (compute+series)", to_us(raw_rust)),
        ("rust (compute+obs)", to_us(raw_rust_obs)),
        ("rust (scenario+series)", to_us(raw_scenario)),
        ("rust (scenario+obs)", to_us(raw_scenario_obs)),
        # ("python (compute+series)", to_us(raw_py, py_number)),
        # ("python (compute+obs)", to_us(raw_py_obs, py_number)),
        # ("python (scenario+series)", to_us(raw_py_scenario, py_number)),
    ]
    for label, samples in rows:
        _print_row(label, samples, min_np, col)

    # Chain benchmarks — materialized vs observable-only
    for depth in [5, 20, 100]:
        chain_result = _rust_scenario_chain(raw_a, raw_b, timestamps_ns, depth)
        chain_obs_result = _rust_scenario_chain_obs(raw_a, raw_b, timestamps_ns, depth)
        assert len(chain_result) == n
        np.testing.assert_allclose(chain_obs_result.values_array(), [chain_result.values_array()[-1]])

        raw_chain = timeit.repeat(
            lambda d=depth: _rust_scenario_chain(raw_a, raw_b, timestamps_ns, d),
            number=number,
            repeat=repeats,
        )
        raw_chain_obs = timeit.repeat(
            lambda d=depth: _rust_scenario_chain_obs(raw_a, raw_b, timestamps_ns, d),
            number=number,
            repeat=repeats,
        )
        _print_row(f"rust (chain d={depth} +series)", to_us(raw_chain), min_np, col)
        _print_row(f"rust (chain d={depth} +obs)", to_us(raw_chain_obs), min_np, col)

    # Sparse benchmarks — materialized vs observable-only
    for total, active in [(100, 5), (1000, 5), (1000, 50)]:
        sparse_result = _rust_scenario_sparse(raw_a, raw_b, timestamps_ns, total, active)
        sparse_obs_result = _rust_scenario_sparse_obs(raw_a, raw_b, timestamps_ns, total, active)
        assert len(sparse_result) == n
        np.testing.assert_allclose(sparse_obs_result.values_array(), [sparse_result.values_array()[-1]])

        raw_sparse = timeit.repeat(
            lambda t=total, a=active: _rust_scenario_sparse(raw_a, raw_b, timestamps_ns, t, a),
            number=number,
            repeat=repeats,
        )
        raw_sparse_obs = timeit.repeat(
            lambda t=total, a=active: _rust_scenario_sparse_obs(raw_a, raw_b, timestamps_ns, t, a),
            number=number,
            repeat=repeats,
        )
        _print_row(f"rust (sparse {active}/{total} +series)", to_us(raw_sparse), min_np, col)
        _print_row(f"rust (sparse {active}/{total} +obs)", to_us(raw_sparse_obs), min_np, col)

    print()


if __name__ == "__main__":
    n = int(os.environ.get("BENCH_N", 10_000))
    _run(n)
