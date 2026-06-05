"""Self-contained validation tests for the ported `flowops.portfolios` operators.

Run with:
    PYTHONPATH=python .venv-ft/Scripts/python.exe python/flowops/tests/test_portfolios.py

NumPy/SciPy-only operators (mean leaves, markowitz_admm, the solver libraries)
run on the free-threaded venv.  The cvxpy-backed operators (minimum_variance,
markowitz, benchmark_relative) are exercised only when cvxpy is importable
(it is NOT on `.venv-ft`); otherwise they are reported as skipped blockers.
"""

import importlib.util

import numpy as np

from flowops._testkit import FakeArrayView

from flowops.portfolios.mean.proportional import build as build_proportional
from flowops.portfolios.mean.rank_equal import build as build_rank_equal
from flowops.portfolios.mean.rank_linear import build as build_rank_linear
from flowops.portfolios.mean.rank_bucket import build as build_rank_bucket
from flowops.portfolios.mean.softmax import build as build_softmax
from flowops.portfolios.mean_variance.markowitz_admm import build as build_markowitz_admm
from flowops.portfolios.mean_variance._modes import Mode

_HAS_CVXPY = importlib.util.find_spec("cvxpy") is not None


def _drive(op, state, inputs, output, ts, produced):
    return op.compute(state, tuple(inputs), output, ts, tuple(produced))


def _spd_cov(rng, n, scale=0.02):
    """Random symmetric positive-definite covariance, small (daily) scale."""
    a = rng.standard_normal((n, n))
    cov = a @ a.T / n + np.eye(n)
    return cov * (scale ** 2)


# ---------------------------------------------------------------------------
# Mean-only leaves (universe, predicted_returns) -> weights
# ---------------------------------------------------------------------------


def _run_mean_op(op, universe, mu, *, produced=(True, True)):
    n = universe.shape[0]
    state = op.init((FakeArrayView(universe), FakeArrayView(mu)), 0)
    out = FakeArrayView(np.zeros(n), writable=True)
    emitted = _drive(op, state, [FakeArrayView(universe), FakeArrayView(mu)], out, 0, list(produced))
    return emitted, out


def test_proportional():
    n = 6
    universe = np.ones(n)
    # use logarithmic=False so values are linear; mixed signs
    mu = np.array([0.05, -0.02, 0.01, 0.0, 0.03, -0.10])
    op = build_proportional(num_stocks=n, logarithmic=False)
    emitted, out = _run_mean_op(op, universe, mu)
    assert emitted is True
    w = out.written
    assert w.shape == (n,)
    assert np.all(np.isfinite(w))
    assert np.all(w >= 0.0)
    assert np.isclose(w.sum(), 1.0)
    # negative-return stocks get zero
    assert w[1] == 0.0 and w[5] == 0.0 and w[3] == 0.0
    # largest positive return gets largest weight
    assert np.argmax(w) == 0
    # not produced => no emit
    op2 = build_proportional(num_stocks=n, logarithmic=False)
    emitted2, _ = _run_mean_op(op2, universe, mu, produced=(False, True))
    assert emitted2 is False
    print("test_proportional OK")


def test_rank_equal():
    n = 10
    universe = np.ones(n)
    rng = np.random.default_rng(1)
    mu = rng.standard_normal(n) * 0.05
    op = build_rank_equal(num_stocks=n, top_fraction=0.3, logarithmic=True)
    emitted, out = _run_mean_op(op, universe, mu)
    assert emitted is True
    w = out.written
    assert np.all(np.isfinite(w)) and np.all(w >= 0)
    assert np.isclose(w.sum(), 1.0)
    nz = np.flatnonzero(w)
    # equal weights on the selected names
    assert np.allclose(w[nz], w[nz][0])
    print("test_rank_equal OK")


def test_rank_linear():
    n = 10
    universe = np.ones(n)
    rng = np.random.default_rng(2)
    mu = rng.standard_normal(n) * 0.05
    op = build_rank_linear(num_stocks=n, top_fraction=0.5, logarithmic=True)
    emitted, out = _run_mean_op(op, universe, mu)
    assert emitted is True
    w = out.written
    assert np.all(np.isfinite(w)) and np.all(w >= 0)
    assert np.isclose(w.sum(), 1.0)
    # linear schedule => distinct positive weights, monotone with prediction rank
    nz = np.flatnonzero(w)
    assert nz.size >= 2
    # highest predicted positive-return stock should have the largest weight
    pos = np.maximum(np.expm1(mu), 0.0)
    if (pos > 0).any():
        top_idx = int(np.argmax(pos))
        assert np.argmax(w) == top_idx
    print("test_rank_linear OK")


def test_rank_bucket():
    n = 10
    universe = np.ones(n)
    feature = np.arange(n, dtype=float)  # strictly increasing
    # top decile [0.9, 1.0) -> only the largest-feature stock
    op = build_rank_bucket(num_stocks=n, low=0.9, high=1.0)
    emitted, out = _run_mean_op(op, universe, feature)
    assert emitted is True
    w = out.written
    assert np.isclose(w.sum(), 1.0)
    assert np.argmax(w) == n - 1 and (w > 0).sum() == 1
    # bottom decile [0.0, 0.1)
    op2 = build_rank_bucket(num_stocks=n, low=0.0, high=0.1)
    _, out2 = _run_mean_op(op2, universe, feature)
    w2 = out2.written
    assert np.isclose(w2.sum(), 1.0)
    assert np.argmax(w2) == 0 and (w2 > 0).sum() == 1
    print("test_rank_bucket OK")


def test_softmax():
    n = 8
    universe = np.ones(n)
    rng = np.random.default_rng(3)
    mu = rng.standard_normal(n) * 0.05
    op = build_softmax(num_stocks=n, temperature=0.5, logarithmic=False)
    emitted, out = _run_mean_op(op, universe, mu)
    assert emitted is True
    w = out.written
    assert np.all(np.isfinite(w)) and np.all(w >= 0)
    assert np.isclose(w.sum(), 1.0)
    print("test_softmax OK")


def test_mean_nan_and_universe_subset():
    """Masked-out stocks (universe<=0 or NaN mu) get exactly zero weight."""
    n = 6
    universe = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    mu = np.array([0.05, np.nan, 0.04, 0.03, 0.02, 0.01])
    op = build_proportional(num_stocks=n, logarithmic=False)
    _, out = _run_mean_op(op, universe, mu)
    w = out.written
    assert w[2] == 0.0 and w[5] == 0.0, "universe<=0 -> zero"
    assert w[1] == 0.0, "NaN mu -> zero"
    assert np.isclose(w.sum(), 1.0)
    print("test_mean_nan_and_universe_subset OK")


# ---------------------------------------------------------------------------
# mean_variance: MarkowitzADMM (NumPy/SciPy only)
# ---------------------------------------------------------------------------


def _run_mv_op(op, universe, mu, sigma):
    n = universe.shape[0]
    inputs = (FakeArrayView(universe), FakeArrayView(mu), FakeArrayView(sigma))
    state = op.init(inputs, 0)
    out = FakeArrayView(np.zeros(n), writable=True)
    emitted = _drive(op, state, list(inputs), out, 0, [True, True, True])
    return emitted, out


def test_markowitz_admm_long_only():
    n = 8
    rng = np.random.default_rng(10)
    universe = np.ones(n)
    mu = rng.standard_normal(n) * 0.01
    sigma = _spd_cov(rng, n)
    op = build_markowitz_admm(
        num_stocks=n,
        mode=Mode.MIN_MEAN_VARIANCE,
        bound=5.0,
        long_only=True,
        full_position=True,
        logarithmic=False,
    )
    emitted, out = _run_mv_op(op, universe, mu, sigma)
    assert emitted is True
    w = out.written
    assert w.shape == (n,)
    assert np.all(np.isfinite(w))
    assert np.all(w >= -1e-8), w.min()
    assert np.isclose(w.sum(), 1.0, atol=1e-2), w.sum()
    print(f"test_markowitz_admm_long_only OK (sum={w.sum():.4f}, min={w.min():.4e})")


def test_markowitz_admm_int_mode_and_underinvest():
    """Mode passed as int (host config) and full_position=False (cash slack)."""
    n = 6
    rng = np.random.default_rng(11)
    universe = np.ones(n)
    mu = rng.standard_normal(n) * 0.01
    sigma = _spd_cov(rng, n)
    op = build_markowitz_admm(
        num_stocks=n,
        mode=3,  # Mode.MIN_MEAN_VARIANCE as a plain int
        bound=10.0,
        long_only=True,
        full_position=False,
        logarithmic=False,
    )
    emitted, out = _run_mv_op(op, universe, mu, sigma)
    assert emitted is True
    w = out.written
    assert np.all(np.isfinite(w)) and np.all(w >= -1e-8)
    assert w.sum() <= 1.0 + 1e-2, w.sum()
    print(f"test_markowitz_admm_underinvest OK (sum={w.sum():.4f})")


def test_markowitz_admm_rejects_other_modes():
    raised = False
    try:
        build_markowitz_admm(num_stocks=4, mode=Mode.MAX_RETURN_GIVEN_STD_DEV, bound=0.1)
    except ValueError:
        raised = True
    assert raised, "MarkowitzADMM must reject non-MIN_MEAN_VARIANCE modes"
    print("test_markowitz_admm_rejects_other_modes OK")


# ---------------------------------------------------------------------------
# Solver libraries (pure NumPy/SciPy) - smoke tests
# ---------------------------------------------------------------------------


def test_admm_mnr_solver_library():
    from flowops.portfolios.mean_variance.admm_mnr import (
        solve_portfolio,
        FactorLSOperator,
        BlockDiagLSOperator,
        admm_mnr,
    )

    rng = np.random.default_rng(20)
    n, k = 12, 3
    F = rng.standard_normal((n, k)) * 0.1
    D = np.full(n, 1e-3)
    mu = rng.standard_normal(n) * 0.01

    res = solve_portfolio(
        factor_loadings=F,
        specific_var=D,
        expected_returns=mu,
        delta=1.0,
        budget=1.0,
        lower=0.0,
        outer_tol=1e-3,
        max_outer=200,
        completion=False,
    )
    w = res["weights"]
    assert w.shape == (n,)
    assert np.all(np.isfinite(w))
    assert np.all(w >= -1e-6)
    assert np.isclose(w.sum(), 1.0, atol=1e-2), w.sum()

    # block-diagonal (2 identical accounts) matvec consistency
    base = FactorLSOperator(F, D, 1.0)
    bd = BlockDiagLSOperator(base, 2)
    x = rng.standard_normal(2 * n)
    y = bd.matvec(x)
    assert y.shape[0] == bd.shape[0] and np.all(np.isfinite(y))
    print(f"test_admm_mnr_solver_library OK (sum={w.sum():.4f}, status={res['metrics']['status']})")


def test_augmented_saddle_admm_library():
    from flowops.portfolios.mean_variance.augmented_saddle_admm import (
        setup_sector_constrained,
        solve_admm,
    )

    rng = np.random.default_rng(21)
    n = 9
    a = rng.standard_normal((n, n))
    sigma = a @ a.T / n + np.eye(n) * 0.5
    mu = rng.standard_normal(n) * 0.01
    sectors = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    targets = [1.0 / 3, 1.0 / 3, 1.0 / 3]

    K_times, rhs_builder, info = setup_sector_constrained(
        sigma, mu, delta=1.0, sector_membership=sectors, sector_targets=targets
    )
    result = solve_admm(
        K_times,
        rhs_builder,
        n_residual_block=info["n_residual_block"],
        n_x_block=info["n_x_block"],
        n_lambda_block=info["n_lambda_block"],
        project_fn=lambda z: np.maximum(z, 0.0),
        inner_rtol=1e-6,
        outer_tol=1e-5,
        max_outer=500,
    )
    x = result.x
    assert x.shape == (n,)
    assert np.all(np.isfinite(x)) and np.all(x >= -1e-6)
    # each sector hits its target weight
    for idxs, t in zip(sectors, targets):
        assert np.isclose(x[idxs].sum(), t, atol=1e-2), (x[idxs].sum(), t)
    print(f"test_augmented_saddle_admm_library OK (sum={x.sum():.4f})")


# ---------------------------------------------------------------------------
# cvxpy-backed operators (blockers on ft) - only run if cvxpy is present
# ---------------------------------------------------------------------------


def test_cvxpy_operators():
    if not _HAS_CVXPY:
        print("test_cvxpy_operators SKIPPED (cvxpy unavailable - blocker on .venv-ft)")
        return

    from flowops.portfolios.variance.minimum_variance import build as build_minvar
    from flowops.portfolios.mean_variance.markowitz import build as build_markowitz
    from flowops.portfolios.mean_variance.benchmark_relative import build as build_bench

    rng = np.random.default_rng(30)
    n = 8
    universe = np.ones(n)
    mu = rng.standard_normal(n) * 0.01
    sigma = _spd_cov(rng, n)

    # MinimumVariance (universe, covariance)
    mv = build_minvar(num_stocks=n, long_only=True, logarithmic=False)
    inputs = (FakeArrayView(universe), FakeArrayView(sigma))
    state = mv.init(inputs, 0)
    out = FakeArrayView(np.zeros(n), writable=True)
    assert _drive(mv, state, list(inputs), out, 0, [True, True])
    w = out.written
    assert np.all(np.isfinite(w)) and np.all(w >= -1e-6) and np.isclose(w.sum(), 1.0, atol=1e-2)

    # Markowitz MIN_MEAN_VARIANCE
    mk = build_markowitz(num_stocks=n, mode=Mode.MIN_MEAN_VARIANCE, bound=5.0, logarithmic=False)
    inputs = (FakeArrayView(universe), FakeArrayView(mu), FakeArrayView(sigma))
    state = mk.init(inputs, 0)
    out = FakeArrayView(np.zeros(n), writable=True)
    assert _drive(mk, state, list(inputs), out, 0, [True, True, True])
    w = out.written
    assert np.all(np.isfinite(w)) and np.all(w >= -1e-6)

    # BenchmarkRelative
    br = build_bench(num_stocks=n, bound=0.05, logarithmic=False)
    inputs = (FakeArrayView(universe), FakeArrayView(mu), FakeArrayView(sigma))
    state = br.init(inputs, 0)
    out = FakeArrayView(np.zeros(n), writable=True)
    assert _drive(br, state, list(inputs), out, 0, [True, True, True])
    w = out.written
    assert np.all(np.isfinite(w)) and np.isclose(w.sum(), 1.0, atol=1e-2)

    print("test_cvxpy_operators OK")


def _main():
    test_proportional()
    test_rank_equal()
    test_rank_linear()
    test_rank_bucket()
    test_softmax()
    test_mean_nan_and_universe_subset()
    test_markowitz_admm_long_only()
    test_markowitz_admm_int_mode_and_underinvest()
    test_markowitz_admm_rejects_other_modes()
    test_admm_mnr_solver_library()
    test_augmented_saddle_admm_library()
    test_cvxpy_operators()
    print("\nALL PORTFOLIO TESTS PASSED")


if __name__ == "__main__":
    _main()
