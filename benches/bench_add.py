"""Benchmark: element-wise addition via raw NumPy vs. the ``add`` Apply operator.

Compares three approaches for summing two scalar float64 series of length N:

* **numpy** – ``series_a.values + series_b.values`` (batch vectorised NumPy).
* **loop** – plain Python ``for`` loop reading each element pair and summing
  with ``+``.
* **compute** – :func:`~tradingflow.operators.add` operator's ``compute()``
  called directly on incrementally-built series (mirrors the Scenario's
  ``_flush_queue`` path — no ``Series.to()`` binary search).

Run::

    python benchmarks/bench_add.py

Optional environment variable ``BENCH_N`` overrides the default series length.
"""

from __future__ import annotations

import os
import timeit
from collections.abc import Sequence
from typing import Any

import numpy as np

from tradingflow import Series
from tradingflow.operators import add


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_series(n: int) -> Series[tuple[()], np.float64]:
    """Return a scalar float64 series of length *n* with nanosecond timestamps."""
    s: Series[tuple[()], np.float64] = Series((), np.dtype(np.float64))
    rng = np.random.default_rng(0)
    values = rng.standard_normal(n)
    for i in range(n):
        s.append(np.datetime64(i, "ns"), np.array(values[i], dtype=np.float64))
    return s


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------


def bench_numpy(
    a: Series[tuple[()], np.float64],
    b: Series[tuple[()], np.float64],
) -> np.ndarray[Any, Any]:
    """Batch addition: add the full value arrays in one NumPy operation."""
    return a.values + b.values


def bench_loop(
    a: Series[tuple[()], np.float64],
    b: Series[tuple[()], np.float64],
) -> np.ndarray[Any, Any]:
    """Plain Python loop: iterate element-by-element and sum with ``+``."""
    a_vals = a.values
    b_vals = b.values
    return np.array([a_vals[i] + b_vals[i] for i in range(len(a_vals))])


def bench_compute(
    raw_a: np.ndarray[Any, Any],
    raw_b: np.ndarray[Any, Any],
    timestamps: Sequence[np.datetime64],
) -> Series[Any, Any]:
    """Scenario-like path: build series incrementally, call compute() directly.

    Mirrors ``_flush_queue`` which appends to source series then calls
    ``op.compute(time, op.inputs, state)`` — no ``Series.to()`` slicing.
    """
    a: Series[tuple[()], np.float64] = Series((), np.dtype(np.float64))
    b: Series[tuple[()], np.float64] = Series((), np.dtype(np.float64))
    op = add(a, b)
    state = op.init_state()
    output: Series[Any, Any] = Series(op.shape, op.dtype)
    out_dtype = output.dtype
    for i, t in enumerate(timestamps):
        a.append_unchecked(t, raw_a[i])
        b.append_unchecked(t, raw_b[i])
        value, state = op.compute(t, op.inputs, state)
        if value is not None:
            output.append_unchecked(t, np.asarray(value, dtype=out_dtype))
    return output


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run(n: int, repeats: int = 5, number: int = 200) -> None:
    print(f"Series length N = {n:,}   (repeats={repeats}, number={number})\n")

    a = _make_series(n)
    b = _make_series(n)
    timestamps = [np.datetime64(i, "ns") for i in range(n)]

    raw_a = a.values.copy()
    raw_b = b.values.copy()

    # Warm up (also validates correctness).
    np_result = bench_numpy(a, b)
    loop_result = bench_loop(a, b)
    compute_result = bench_compute(raw_a, raw_b, timestamps)
    assert len(compute_result) == n, "compute produced wrong number of outputs"
    np.testing.assert_allclose(loop_result, np_result, err_msg="loop and numpy results differ")
    np.testing.assert_allclose(compute_result.values, np_result, err_msg="compute and numpy results differ")

    # Timings — repeat the measurement *repeats* times, each calling the
    # function *number* times.  Report per-call min / mean / stdev across
    # repeats (min is the most reproducible metric per Python timeit docs).
    raw_numpy = timeit.repeat(lambda: bench_numpy(a, b), number=number, repeat=repeats)
    raw_loop = timeit.repeat(lambda: bench_loop(a, b), number=number, repeat=repeats)
    raw_compute = timeit.repeat(lambda: bench_compute(raw_a, raw_b, timestamps), number=number, repeat=repeats)

    per_call_numpy = [t / number * 1e6 for t in raw_numpy]  # us per call
    per_call_loop = [t / number * 1e6 for t in raw_loop]
    per_call_compute = [t / number * 1e6 for t in raw_compute]

    def _stats(samples: list[float]) -> tuple[float, float, float]:
        mn = min(samples)
        mean = sum(samples) / len(samples)
        var = sum((s - mean) ** 2 for s in samples) / len(samples)
        return mn, mean, var**0.5

    min_np, mean_np, std_np = _stats(per_call_numpy)
    min_lp, mean_lp, std_lp = _stats(per_call_loop)
    min_cm, mean_cm, std_cm = _stats(per_call_compute)

    col = 18
    print(f"{'approach':<{col}}  {'min (us)':>10}  {'mean (us)':>10}  {'stdev (us)':>10}  {'ratio':>8}")
    print("-" * (col + 44))
    print(f"{'numpy':<{col}}  {min_np:>10.2f}  {mean_np:>10.2f}  {std_np:>10.2f}  {'1.00x':>8}")
    ratio_lp = min_lp / min_np
    print(f"{'loop':<{col}}  {min_lp:>10.2f}  {mean_lp:>10.2f}  {std_lp:>10.2f}  {ratio_lp:>7.2f}x")
    ratio_cm = min_cm / min_np
    print(f"{'compute':<{col}}  {min_cm:>10.2f}  {mean_cm:>10.2f}  {std_cm:>10.2f}  {ratio_cm:>7.2f}x")
    print()


if __name__ == "__main__":
    n = int(os.environ.get("BENCH_N", 1_000))
    _run(n)
