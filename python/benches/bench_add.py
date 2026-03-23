"""Benchmark: element-wise addition — Rust pipeline layers vs NumPy.

Mirrors the test cases in ``benches/bench_add.rs`` exactly, plus a NumPy
vectorised baseline for cross-language comparison.

Run::

    python benches/bench_add.py

Optional environment variable ``BENCH_N`` overrides the default series length.
"""

from __future__ import annotations

import os
import timeit

import numpy as np

from tradingflow._native import bench_baseline_add as _rust_baseline_add
from tradingflow._native import bench_baseline_add_series as _rust_baseline_add_series
from tradingflow._native import bench_store_add as _rust_store_add
from tradingflow._native import bench_store_add_series as _rust_store_add_series
from tradingflow._native import bench_store_compute as _rust_store_compute
from tradingflow._native import bench_store_compute_series as _rust_store_compute_series
from tradingflow._native import bench_scenario_operator as _rust_scenario_operator
from tradingflow._native import bench_scenario_operator_series as _rust_scenario_operator_series
from tradingflow._native import bench_scenario_chain as _rust_scenario_chain
from tradingflow._native import bench_scenario_sparse as _rust_scenario_sparse


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


def _print_row(label: str, samples: list[float], min_np: float, col: int = 36) -> None:
    mn, mean, std = _stats(samples)
    ratio = mn / min_np
    print(f"{label:<{col}}  {mn:>10.2f}  {mean:>10.2f}  {std:>10.2f}  {ratio:>7.2f}x")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run(n: int, repeats: int = 5, number: int = 200) -> None:
    print(f"Series length N = {n:,}   (repeats={repeats}, number={number})\n")

    raw_a, raw_b, timestamps_ns = _make_arrays(n)
    np_result = raw_a + raw_b

    # -- Warm up + validate --------------------------------------------------

    np.testing.assert_allclose(_rust_baseline_add(raw_a, raw_b, timestamps_ns).values_array(), np_result)
    np.testing.assert_allclose(_rust_baseline_add_series(raw_a, raw_b, timestamps_ns).values_array(), np_result)
    np.testing.assert_allclose(_rust_store_add(raw_a, raw_b, timestamps_ns).values_array(), [np_result[-1]])
    np.testing.assert_allclose(_rust_store_add_series(raw_a, raw_b, timestamps_ns).values_array(), np_result)
    np.testing.assert_allclose(_rust_store_compute(raw_a, raw_b, timestamps_ns).values_array(), [np_result[-1]])
    np.testing.assert_allclose(_rust_store_compute_series(raw_a, raw_b, timestamps_ns).values_array(), np_result)
    np.testing.assert_allclose(_rust_scenario_operator(raw_a, raw_b, timestamps_ns).values_array(), [np_result[-1]])
    np.testing.assert_allclose(_rust_scenario_operator_series(raw_a, raw_b, timestamps_ns).values_array(), np_result)

    # -- NumPy baseline ------------------------------------------------------

    def bench_numpy():
        return raw_a + raw_b

    # -- Timings -------------------------------------------------------------

    raw_numpy = timeit.repeat(bench_numpy, number=number, repeat=repeats)

    benches = [
        ("numpy", raw_numpy),
        (
            "baseline_add",
            timeit.repeat(lambda: _rust_baseline_add(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats),
        ),
        (
            "baseline_add_series",
            timeit.repeat(
                lambda: _rust_baseline_add_series(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats
            ),
        ),
        (
            "store_add",
            timeit.repeat(lambda: _rust_store_add(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats),
        ),
        (
            "store_add_series",
            timeit.repeat(lambda: _rust_store_add_series(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats),
        ),
        (
            "store_compute",
            timeit.repeat(lambda: _rust_store_compute(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats),
        ),
        (
            "store_compute_series",
            timeit.repeat(
                lambda: _rust_store_compute_series(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats
            ),
        ),
        (
            "scenario_operator",
            timeit.repeat(lambda: _rust_scenario_operator(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats),
        ),
        (
            "scenario_operator_series",
            timeit.repeat(
                lambda: _rust_scenario_operator_series(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats
            ),
        ),
    ]

    to_us = lambda raw, num=number: [t / num * 1e6 for t in raw]
    min_np, _, _ = _stats(to_us(raw_numpy))

    col = 36
    print(f"{'approach':<{col}}  {'min (us)':>10}  {'mean (us)':>10}  {'stdev (us)':>10}  {'ratio':>8}")
    print("-" * (col + 50))

    for label, raw in benches:
        _print_row(label, to_us(raw), min_np, col)

    # -- Chain benchmarks ----------------------------------------------------

    for depth in [1, 5, 10]:
        chain_result = _rust_scenario_chain(raw_a, raw_b, timestamps_ns, depth)
        raw_chain = timeit.repeat(
            lambda d=depth: _rust_scenario_chain(raw_a, raw_b, timestamps_ns, d),
            number=number,
            repeat=repeats,
        )
        _print_row(f"scenario_chain_depth{depth}", to_us(raw_chain), min_np, col)

    # -- Sparse benchmarks ---------------------------------------------------

    for total, active in [(100, 5), (1000, 5)]:
        sparse_result = _rust_scenario_sparse(raw_a, raw_b, timestamps_ns, total, active)
        raw_sparse = timeit.repeat(
            lambda t=total, a=active: _rust_scenario_sparse(raw_a, raw_b, timestamps_ns, t, a),
            number=number,
            repeat=repeats,
        )
        _print_row(f"scenario_sparse_{total}total_{active}active", to_us(raw_sparse), min_np, col)

    print()


if __name__ == "__main__":
    n = int(os.environ.get("BENCH_N", 10_000))
    _run(n)
