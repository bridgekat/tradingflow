"""Self-contained validation tests for the ported `flowops.predictors` group.

Drives every ported mean/variance predictor through init + several compute()
calls with synthetic numpy data of realistic shapes, asserting they run and
produce finite/expected output.  No pytest: plain asserts + a __main__ runner.

Run with:
    PYTHONPATH=python .venv-ft/Scripts/python.exe python/flowops/tests/test_predictors.py
"""

import importlib.util

import numpy as np

from flowops._testkit import FakeArrayView, FakeSeriesView

_HAS_CVXPY = importlib.util.find_spec("cvxpy") is not None

from flowops.predictors.mean import sample as mean_sample
from flowops.predictors.mean import single_feature as mean_single_feature
from flowops.predictors.mean import linear_regression as mean_lr
from flowops.predictors.mean import ridge as mean_ridge
from flowops.predictors.mean import lasso as mean_lasso

from flowops.predictors.variance import sample as var_sample
from flowops.predictors.variance import single_index as var_single_index
from flowops.predictors.variance import shrinkage as var_shrinkage
from flowops.predictors.variance import rmt as var_rmt
from flowops.predictors.variance import hierarchical as var_hier


N = 6  # num_stocks
F = 3  # num_features
T = 40  # series length


def _make_panel(seed: int = 0):
    """Build a (universe, features_series, target_series) fixture.

    features_series: list of (N, F) arrays; target_series: list of (N,)
    arrays.  Target is a noisy linear function of the features so the
    regression predictors have real signal to fit.
    """
    rng = np.random.default_rng(seed)
    true_beta = rng.normal(size=F)

    feats = []
    targs = []
    for _ in range(T):
        x = rng.normal(size=(N, F))
        y = x @ true_beta + 0.1 * rng.normal(size=N)
        feats.append(x.astype(np.float64))
        targs.append(y.astype(np.float64))

    universe = FakeArrayView(np.ones(N), writable=False)
    features_series = FakeSeriesView(feats, elem_shape=(N, F))
    target_series = FakeSeriesView(targs, elem_shape=(N,))
    return universe, features_series, target_series


def _drive(op, mean_output: bool, *, n_calls: int = 3, seed: int = 0):
    """Run init + n_calls of compute(); return list of written outputs.

    `mean_output` selects the output shape: (N,) for mean predictors,
    (N, N) for variance predictors.  The universe always "produces"
    (rebalance tick).  We also fire one non-produced tick to verify it is
    ignored.
    """
    universe, features_series, target_series = _make_panel(seed)
    inputs = (universe, features_series, target_series)
    out_shape = (N,) if mean_output else (N, N)
    output = FakeArrayView(np.zeros(out_shape), writable=True)

    state = op.init(inputs, timestamp=0)

    # Non-rebalance tick: universe did not produce -> compute returns False.
    produced_false = (False, True, True)
    emitted = op.compute(state, inputs, output, timestamp=0, produced=produced_false)
    assert emitted is False, "non-rebalance tick must not emit"

    outputs = []
    for k in range(n_calls):
        produced = (True, True, True)
        emitted = op.compute(state, inputs, output, timestamp=10 + k, produced=produced)
        assert emitted is True, "rebalance tick must emit"
        assert output.written is not None
        assert output.written.shape == out_shape
        outputs.append(output.written.copy())
    return outputs


def _assert_all_finite_full_universe(outputs, square: bool):
    """With a full all-finite universe every output entry must be finite."""
    for o in outputs:
        assert np.all(np.isfinite(o)), f"expected all-finite output, got {o}"
        if square:
            assert np.allclose(o, o.T, atol=1e-8), "covariance must be symmetric"


# ---------------------------------------------------------------------------
# Mean predictors
# ---------------------------------------------------------------------------

def test_mean_sample():
    op = mean_sample.build(num_stocks=N, num_features=F, universe_size=N, target_offset=1)
    outputs = _drive(op, mean_output=True)
    _assert_all_finite_full_universe(outputs, square=False)
    # Sample mean predictor ignores features: output is the per-stock mean
    # of the target window, independent of the current features.
    print("  mean.sample OK:", np.round(outputs[-1], 4))


def test_mean_single_feature():
    op = mean_single_feature.build(
        num_stocks=N, num_features=F, universe_size=N, target_offset=0, feature_index=1
    )
    outputs = _drive(op, mean_output=True)
    _assert_all_finite_full_universe(outputs, square=False)
    print("  mean.single_feature OK:", np.round(outputs[-1], 4))


def test_mean_linear_regression():
    op = mean_lr.build(num_stocks=N, num_features=F, universe_size=N, target_offset=1)
    outputs = _drive(op, mean_output=True)
    _assert_all_finite_full_universe(outputs, square=False)
    print("  mean.linear_regression OK:", np.round(outputs[-1], 4))


def test_mean_ridge():
    op = mean_ridge.build(
        num_stocks=N, num_features=F, universe_size=N, target_offset=1, alpha=0.5
    )
    outputs = _drive(op, mean_output=True)
    _assert_all_finite_full_universe(outputs, square=False)

    # Ridge with alpha=0 should match plain OLS (LinearRegression) closely.
    op0 = mean_ridge.build(
        num_stocks=N, num_features=F, universe_size=N, target_offset=1, alpha=0.0
    )
    op_ols = mean_lr.build(num_stocks=N, num_features=F, universe_size=N, target_offset=1)
    r0 = _drive(op0, mean_output=True)[-1]
    ols = _drive(op_ols, mean_output=True)[-1]
    assert np.allclose(r0, ols, atol=1e-6), "ridge(alpha=0) must equal OLS"
    print("  mean.ridge OK (alpha=0 == OLS):", np.round(outputs[-1], 4))


def test_mean_lasso():
    # Lasso fits via cvxpy. On the standard `.venv` (cvxpy present) the fit must
    # run and produce finite output; on a no-cvxpy interpreter the deferred import
    # must raise on the first rebalance. Either way the op constructs and inits.
    op = mean_lasso.build(
        num_stocks=N, num_features=F, universe_size=N, target_offset=1, alpha=0.1
    )
    universe, features_series, target_series = _make_panel()
    inputs = (universe, features_series, target_series)
    output = FakeArrayView(np.zeros(N), writable=True)
    state = op.init(inputs, timestamp=0)

    if not _HAS_CVXPY:
        raised = False
        try:
            op.compute(state, inputs, output, timestamp=1, produced=(True, True, True))
        except ImportError:
            raised = True
        assert raised, "lasso fit must raise ImportError without cvxpy"
        print("  mean.lasso OK (constructs/inits; fit blocked by missing cvxpy)")
        return

    assert op.compute(state, inputs, output, timestamp=1, produced=(True, True, True))
    assert np.all(np.isfinite(output.written)), "lasso must produce finite predictions"
    print("  mean.lasso OK (cvxpy fit, finite output)")


# ---------------------------------------------------------------------------
# Variance predictors
# ---------------------------------------------------------------------------

def test_var_sample():
    op = var_sample.build(num_stocks=N, num_features=F, universe_size=N, target_offset=1)
    outputs = _drive(op, mean_output=False)
    _assert_all_finite_full_universe(outputs, square=True)
    # Diagonal (variances) must be non-negative.
    for o in outputs:
        assert np.all(np.diag(o) >= -1e-12)
    print("  variance.sample OK; diag:", np.round(np.diag(outputs[-1]), 4))


def test_var_single_index():
    op = var_single_index.build(num_stocks=N, num_features=F, universe_size=N, target_offset=1)
    outputs = _drive(op, mean_output=False)
    _assert_all_finite_full_universe(outputs, square=True)
    for o in outputs:
        assert np.all(np.diag(o) >= -1e-12)
    print("  variance.single_index OK; diag:", np.round(np.diag(outputs[-1]), 4))


def test_var_shrinkage():
    for tgt in (1, 2, 3):
        op = var_shrinkage.build(
            num_stocks=N, num_features=F, universe_size=N, target_offset=1, target=tgt
        )
        outputs = _drive(op, mean_output=False)
        _assert_all_finite_full_universe(outputs, square=True)
        for o in outputs:
            assert np.all(np.diag(o) >= -1e-12)
        print(f"  variance.shrinkage[target={tgt}] OK; diag:", np.round(np.diag(outputs[-1]), 4))


def test_var_rmt():
    for mode in ("zero", "mean"):
        op = var_rmt.build(
            mode=mode, num_stocks=N, num_features=F, universe_size=N, target_offset=1
        )
        outputs = _drive(op, mean_output=False)
        _assert_all_finite_full_universe(outputs, square=True)
        for o in outputs:
            assert np.all(np.diag(o) >= -1e-12)
        print(f"  variance.rmt[{mode}] OK; diag:", np.round(np.diag(outputs[-1]), 4))
    # Direct per-class builds also work.
    assert var_rmt.build_rmt0(num_stocks=N, num_features=F, universe_size=N, target_offset=1) is not None
    assert var_rmt.build_rmtm(num_stocks=N, num_features=F, universe_size=N, target_offset=1) is not None


def test_var_hierarchical():
    for method in ("upgma", "wpgma", "hausdorff"):
        op = var_hier.build(
            method=method, num_stocks=N, num_features=F, universe_size=N, target_offset=1
        )
        outputs = _drive(op, mean_output=False)
        _assert_all_finite_full_universe(outputs, square=True)
        for o in outputs:
            assert np.all(np.diag(o) >= -1e-12)
        print(f"  variance.hierarchical[{method}] OK; diag:", np.round(np.diag(outputs[-1]), 4))


# ---------------------------------------------------------------------------
# Extra coverage: partial universe + NaN handling (mean & variance bases)
# ---------------------------------------------------------------------------

def test_partial_universe_and_nan():
    """A stock out of universe / with NaN features yields NaN in the output."""
    rng = np.random.default_rng(7)
    feats = [rng.normal(size=(N, F)).astype(np.float64) for _ in range(T)]
    targs = [rng.normal(size=N).astype(np.float64) for _ in range(T)]

    # Drop stock 0 from the universe; corrupt the latest features of stock 1.
    uni = np.ones(N)
    uni[0] = 0.0
    feats[-1][1, :] = np.nan

    universe = FakeArrayView(uni, writable=False)
    features_series = FakeSeriesView(feats, elem_shape=(N, F))
    target_series = FakeSeriesView(targs, elem_shape=(N,))
    inputs = (universe, features_series, target_series)

    # Mean predictor: out-of-universe and NaN-feature stocks -> NaN.
    op = mean_sample.build(num_stocks=N, num_features=F, universe_size=N, target_offset=1)
    out = FakeArrayView(np.zeros(N), writable=True)
    st = op.init(inputs, 0)
    op.compute(st, inputs, out, 1, (True, True, True))
    mu = out.written
    assert np.isnan(mu[0]), "out-of-universe stock must be NaN"
    assert np.isnan(mu[1]), "NaN-feature stock must be NaN"
    assert np.all(np.isfinite(mu[2:])), "in-universe finite stocks must be finite"

    # Variance predictor: NaN rows/cols for dropped stocks.
    opv = var_sample.build(num_stocks=N, num_features=F, universe_size=N, target_offset=1)
    outv = FakeArrayView(np.zeros((N, N)), writable=True)
    stv = opv.init(inputs, 0)
    opv.compute(stv, inputs, outv, 1, (True, True, True))
    sig = outv.written
    assert np.all(np.isnan(sig[0, :])) and np.all(np.isnan(sig[:, 0]))
    assert np.all(np.isnan(sig[1, :])) and np.all(np.isnan(sig[:, 1]))
    sub = sig[2:, 2:]
    assert np.all(np.isfinite(sub)), "remaining submatrix must be finite"
    print("  partial-universe / NaN handling OK")


def test_refit_every_caching():
    """refit_every>1 reuses cached params between cadence ticks (still emits)."""
    op = mean_ridge.build(
        num_stocks=N, num_features=F, universe_size=N, target_offset=1,
        alpha=0.3, refit_every=3,
    )
    outputs = _drive(op, mean_output=True, n_calls=5)
    _assert_all_finite_full_universe(outputs, square=False)
    print("  refit_every caching OK")


def test_mean_subsample():
    """`max_samples` caps the pooled training rows: it drops NaN rows, bounds the
    count deterministically, and is a no-op above the available count."""
    from flowops.predictors.mean._base import pool_and_subsample

    rng = np.random.default_rng(3)
    t2, n2, f2 = 50, 20, 3
    x = rng.normal(size=(t2, n2, f2))
    y = rng.normal(size=(t2, n2))

    # No cap: all (finite) pooled rows.
    xa, ya = pool_and_subsample(x, y, None, 0)
    assert xa.shape == (t2 * n2, f2) and ya.shape == (t2 * n2,)

    # Cap below the count: exactly max_samples rows, deterministic for a fixed seed.
    xb, yb = pool_and_subsample(x, y, 100, 7)
    xb2, yb2 = pool_and_subsample(x, y, 100, 7)
    assert xb.shape == (100, f2) and yb.shape == (100,)
    assert np.array_equal(xb, xb2) and np.array_equal(yb, yb2)

    # NaN rows dropped before sampling.
    x[0, 0, 0] = np.nan
    xc, yc = pool_and_subsample(x, y, None, 0)
    assert len(yc) == t2 * n2 - 1 and np.all(np.isfinite(xc))

    # End-to-end: a cap above the sample count is a no-op vs the uncapped fit.
    op_full = mean_lr.build(num_stocks=N, num_features=F, universe_size=N, target_offset=1)
    op_cap = mean_lr.build(
        num_stocks=N, num_features=F, universe_size=N, target_offset=1, max_samples=10**9
    )
    out_full = _drive(op_full, mean_output=True)
    out_cap = _drive(op_cap, mean_output=True)
    assert np.allclose(out_full[-1], out_cap[-1]), "max_samples above the count must be a no-op"
    print("  mean subsample OK (cap bounds + deterministic + large-cap == full)")


def test_incremental_equivalence():
    """Incremental RLS OLS/Ridge reproduce the original full-window fit (expanding
    and rolling), to solver tolerance, with no subsampling.

    Drives the original and the incremental operator over the *same* series as it
    grows one tick at a time -- the original re-reads the whole window each
    rebalance, the incremental folds in only the new tick -- and asserts the
    per-rebalance predictions match.
    """
    from flowops.predictors.mean import linear_regression as o_lr
    from flowops.predictors.mean import ridge as o_ridge
    from flowops.predictors.mean import incremental_linear_regression as i_lr
    from flowops.predictors.mean import incremental_ridge as i_ridge

    rng = np.random.default_rng(2)
    true_beta = rng.normal(size=F)
    feats = [rng.normal(size=(N, F)) for _ in range(T)]
    targs = [feats[t] @ true_beta + 0.1 * rng.normal(size=N) for t in range(T)]
    universe = FakeArrayView(np.ones(N), writable=False)
    cfg = dict(num_stocks=N, num_features=F, universe_size=N, target_offset=1, min_periods=3)

    cases = [
        ("OLS expanding", o_lr.build(**cfg), i_lr.build(**cfg)),
        ("Ridge expanding", o_ridge.build(alpha=0.7, **cfg), i_ridge.build(alpha=0.7, **cfg)),
        ("Ridge rolling(10)", o_ridge.build(alpha=0.7, max_periods=10, **cfg), i_ridge.build(alpha=0.7, window=10, **cfg)),
    ]
    for name, op_o, op_i in cases:
        full = (universe, FakeSeriesView(feats, elem_shape=(N, F)), FakeSeriesView(targs, elem_shape=(N,)))
        so = op_o.init(full, 0)
        si = op_i.init(full, 0)
        for t in range(2, T + 1):
            fv = FakeSeriesView(feats[:t], elem_shape=(N, F))
            tv = FakeSeriesView(targs[:t], elem_shape=(N,))
            oo = FakeArrayView(np.zeros(N), writable=True)
            oi = FakeArrayView(np.zeros(N), writable=True)
            op_o.compute(so, (universe, fv, tv), oo, t, (True, True, True))
            op_i.compute(si, (universe, fv, tv), oi, t, (True, True, True))
            assert np.allclose(oo.written, oi.written, atol=1e-6, equal_nan=True), (
                f"{name}: mismatch at t={t}, max|d|={np.nanmax(np.abs(oo.written - oi.written)):.3e}"
            )
    print("  incremental equivalence OK (OLS/Ridge, expanding + rolling)")


def main():
    tests = [
        test_mean_sample,
        test_mean_single_feature,
        test_mean_linear_regression,
        test_mean_ridge,
        test_mean_lasso,
        test_mean_subsample,
        test_incremental_equivalence,
        test_var_sample,
        test_var_single_index,
        test_var_shrinkage,
        test_var_rmt,
        test_var_hierarchical,
        test_partial_universe_and_nan,
        test_refit_every_caching,
    ]
    passed = 0
    for t in tests:
        print(f"[run] {t.__name__}")
        t()
        passed += 1
    print(f"\nAll {passed}/{len(tests)} predictor tests passed.")


if __name__ == "__main__":
    main()
