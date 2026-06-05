#!/usr/bin/env python
"""Measure a cvxpy solver's GIL-release / solve-parallelism at scale.

On a standard (GIL) CPython, multiple ``Problem.solve()`` calls run in parallel
only if the solver **releases the GIL** during its native (C/Rust) solve. The
flow engine runs portfolio-optimizer operators on a work-stealing pool, so this
property determines whether independent solves (parameter sweeps, multi-account,
cross-sectional) overlap.

This script builds ``threads`` independent QPs (one per worker thread), warms
them once (so canonicalization is cached and only the native ``solve`` is timed),
then compares ``threads * solves`` solves run serially vs. on ``threads`` OS
threads. ``speedup > 1`` ⇒ the solver releases the GIL; ``speedup -> min(threads,
cores)`` ⇒ near-ideal solve-parallelism.

Covariance models:
  * ``dense``  — Sigma = A Aᵀ/n + εI, via ``quad_form(w, psd_wrap(Sigma))``.
                 NOTE: conic solvers (SCS, CLARABEL) reformulate a dense
                 quadratic to a second-order cone, which needs an O(n³) matrix
                 square root at canonicalization (warmup) — heavy for large n.
                 Use a QP-native solver (OSQP, MOSEK, GUROBI) or the factor
                 model for large dense problems.
  * ``factor`` — Sigma = F Fᵀ + diag(d) with F: (n, k). Canonicalizes to a small
                 (k)-dim cone, so it scales to n = 5000+ cheaply. Realistic for
                 factor-risk models.

Tuning
------
For clean *cross-problem* parallelism, pin each solve's *internal* BLAS/OpenMP
threads to 1 — otherwise N concurrent solves each spawn cores-many BLAS threads
and oversubscribe the machine. Set before launching::

    set OMP_NUM_THREADS=1        (also MKL_NUM_THREADS / OPENBLAS_NUM_THREADS)

Large dense conic solves are memory-bandwidth bound, so per-problem speedup
saturates below the thread count regardless of the GIL; many smaller solves
parallelize closer to ideal.

Examples
--------
    python cvxpy_solve_parallelism.py --solver SCS --n 1000 2000 --threads 8
    python cvxpy_solve_parallelism.py --solver SCS --n 5000 --model factor --factors 64
    python cvxpy_solve_parallelism.py --solver CLARABEL --n 2000 --threads 16
    python cvxpy_solve_parallelism.py --solver MOSEK --n 3000 --threads 16   # if licensed
"""

from __future__ import annotations

import argparse
import os
import threading
import time

import cvxpy as cp
import numpy as np


def make_problem(seed: int, n: int, model: str, factors: int) -> cp.Problem:
    """A long-only mean-variance QP: maximize muᵀw − ½ risk, s.t. 1ᵀw=1, w≥0."""
    rng = np.random.default_rng(seed)
    w = cp.Variable(n)
    mu = rng.standard_normal(n)
    if model == "dense":
        a = rng.standard_normal((n, n))
        sigma = a @ a.T / n + np.eye(n) * 1e-2
        risk = cp.quad_form(w, cp.psd_wrap(sigma))
    else:  # factor model: tractable at large n
        f = rng.standard_normal((n, factors)) / np.sqrt(factors)
        d = rng.uniform(0.5, 1.5, n)
        risk = cp.sum_squares(f.T @ w) + cp.sum(cp.multiply(d, cp.square(w)))
    return cp.Problem(cp.Maximize(mu @ w - 0.5 * risk), [cp.sum(w) == 1, w >= 0])


def bench(solver: str, n: int, threads: int, solves: int, model: str,
          factors: int, opts: dict) -> None:
    t = time.perf_counter()
    probs = [make_problem(s, n, model, factors) for s in range(threads)]
    build = time.perf_counter() - t

    t = time.perf_counter()  # warm up: caches canonicalization, primes caches
    for p in probs:
        p.solve(solver=solver, **opts)
    warm = time.perf_counter() - t

    t = time.perf_counter()  # serial baseline
    for _ in range(solves):
        for p in probs:
            p.solve(solver=solver, **opts)
    serial = time.perf_counter() - t

    def worker(p: cp.Problem) -> None:
        for _ in range(solves):
            p.solve(solver=solver, **opts)

    t = time.perf_counter()  # one OS thread per problem
    ths = [threading.Thread(target=worker, args=(p,)) for p in probs]
    for x in ths:
        x.start()
    for x in ths:
        x.join()
    par = time.perf_counter() - t

    print(
        f"  n={n:>5} model={model:<6} threads={threads:<3} solves/thr={solves}  "
        f"build={build:6.2f}s warm={warm:6.2f}s ~solve={warm / threads * 1000:7.1f}ms  "
        f"serial={serial:7.2f}s parallel={par:7.2f}s  speedup={serial / par:5.2f}x"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--solver", default="SCS", help="cvxpy solver name")
    ap.add_argument("--n", type=int, nargs="+", default=[1000], help="problem size(s)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--solves", type=int, default=3, help="solves per thread")
    ap.add_argument("--model", choices=["dense", "factor"], default="dense")
    ap.add_argument("--factors", type=int, default=64, help="factor count (factor model)")
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VAL",
                    help="extra solver option, e.g. --opt eps=1e-3 (repeatable)")
    args = ap.parse_args()

    opts: dict = {}
    for kv in args.opt:
        k, v = kv.split("=", 1)
        try:
            opts[k] = int(v)
        except ValueError:
            try:
                opts[k] = float(v)
            except ValueError:
                opts[k] = v

    print(f"cvxpy {cp.__version__}  installed_solvers={cp.installed_solvers()}  cores={os.cpu_count()}")
    print(f"solver={args.solver} opts={opts}")
    for n in args.n:
        bench(args.solver, n, args.threads, args.solves, args.model, args.factors, opts)


if __name__ == "__main__":
    main()
