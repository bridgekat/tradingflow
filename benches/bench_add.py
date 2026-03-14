"""Benchmark: element-wise addition via raw NumPy vs. Rust native operators.

Run::

    python benches/bench_add.py

Optional environment variable ``BENCH_N`` overrides the default series length.
"""

from __future__ import annotations

import os
import timeit

import numpy as np

from tradingflow_native import bench_add_compute as _rust_add_compute
from tradingflow_native import bench_add_loop as _rust_add_loop
from tradingflow_native import bench_add_loop_interleaved as _rust_add_loop_interleaved
from tradingflow_native import bench_add_loop_fnptr as _rust_add_loop_fnptr
from tradingflow_native import bench_scenario_compute as _rust_scenario_compute
from tradingflow_native import bench_scenario_chain as _rust_scenario_chain
from tradingflow_native import bench_scenario_sparse as _rust_scenario_sparse


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
    scenario_result = _rust_scenario_compute(raw_a, raw_b, timestamps_ns)
    np.testing.assert_allclose(rust_loop_result.values_array(), np_result)
    np.testing.assert_allclose(rust_loop_il_result.values_array(), np_result)
    np.testing.assert_allclose(rust_loop_fnptr_result.values_array(), np_result)
    np.testing.assert_allclose(rust_result.values_array(), np_result)
    np.testing.assert_allclose(scenario_result.values_array(), np_result)

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
    raw_scenario = timeit.repeat(
        lambda: _rust_scenario_compute(raw_a, raw_b, timestamps_ns), number=number, repeat=repeats
    )

    to_us = lambda raw: [t / number * 1e6 for t in raw]
    min_np, _, _ = _stats(to_us(raw_numpy))

    col = 22
    print(f"{'approach':<{col}}  {'min (us)':>10}  {'mean (us)':>10}  {'stdev (us)':>10}  {'ratio':>8}")
    print("-" * (col + 44))

    rows = [
        ("numpy", to_us(raw_numpy)),
        ("rust (3×sep)", to_us(raw_rust_loop)),
        ("rust (3×interleaved)", to_us(raw_rust_loop_il)),
        ("rust (3×il+fnptr)", to_us(raw_rust_loop_fnptr)),
        ("rust (compute)", to_us(raw_rust)),
        ("rust (scenario)", to_us(raw_scenario)),
    ]
    for label, samples in rows:
        mn, mean, std = _stats(samples)
        ratio = mn / min_np
        print(f"{label:<{col}}  {mn:>10.2f}  {mean:>10.2f}  {std:>10.2f}  {ratio:>7.2f}x")

    # Chain & sparse benchmarks — same N, amortises setup over many ticks
    for depth in [5, 20, 100]:
        chain_result = _rust_scenario_chain(raw_a, raw_b, timestamps_ns, depth)
        assert len(chain_result) == n
        raw_chain = timeit.repeat(
            lambda d=depth: _rust_scenario_chain(raw_a, raw_b, timestamps_ns, d),
            number=number,
            repeat=repeats,
        )
        mn, mean, std = _stats(to_us(raw_chain))
        ratio = mn / min_np
        label = f"rust (chain d={depth})"
        print(f"{label:<{col}}  {mn:>10.2f}  {mean:>10.2f}  {std:>10.2f}  {ratio:>7.2f}x")

    for total, active in [(100, 5), (1000, 5), (1000, 50)]:
        sparse_result = _rust_scenario_sparse(raw_a, raw_b, timestamps_ns, total, active)
        assert len(sparse_result) == n
        raw_sparse = timeit.repeat(
            lambda t=total, a=active: _rust_scenario_sparse(raw_a, raw_b, timestamps_ns, t, a),
            number=number,
            repeat=repeats,
        )
        mn, mean, std = _stats(to_us(raw_sparse))
        ratio = mn / min_np
        label = f"rust (sparse {active}/{total})"
        print(f"{label:<{col}}  {mn:>10.2f}  {mean:>10.2f}  {std:>10.2f}  {ratio:>7.2f}x")

    print()


if __name__ == "__main__":
    n = int(os.environ.get("BENCH_N", 10_000))
    _run(n)
