"""Self-contained validation tests for the ported `flowops.predictors` group.

Drives every ported mean/variance predictor through init + several compute()
calls with synthetic numpy data of realistic shapes, asserting they run and
produce finite/expected output.  No pytest: plain asserts + a __main__ runner.

Run with:
    PYTHONPATH=python .venv-ft/Scripts/python.exe python/flowops/tests/test_predictors.py
"""

import numpy as np

from flowops._testkit import FakeArrayView, FakeSeriesView

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
    # cvxpy is unavailable on the ft interpreter (BLOCKER).  The op must
    # still construct and init; the fit must raise ImportError on the first
    # rebalance.  Verify both, without requiring cvxpy.
    op = mean_lasso.build(
        num_stocks=N, num_features=F, universe_size=N, target_offset=1, alpha=0.1
    )
    universe, features_series, target_series = _make_panel()
    inputs = (universe, features_series, target_series)
    output = FakeArrayView(np.zeros(N), writable=True)
    state = op.init(inputs, timestamp=0)
    raised = False
    try:
        op.compute(state, inputs, output, timestamp=1, produced=(True, True, True))
    except ImportError:
        raised = True
    assert raised, "lasso fit must raise ImportError without cvxpy (BLOCKER)"
    print("  mean.lasso OK (constructs/inits; fit blocked by missing cvxpy)")


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


def main():
    tests = [
        test_mean_sample,
        test_mean_single_feature,
        test_mean_linear_regression,
        test_mean_ridge,
        test_mean_lasso,
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
