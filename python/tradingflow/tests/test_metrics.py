"""Self-contained validation tests for the ported `tradingflow.metrics` operators.

Run with:
    PYTHONPATH=python .venv-ft/Scripts/python.exe python/tradingflow/tests/test_metrics.py
"""

import numpy as np

from tradingflow._testkit import FakeArrayView, FakeSeriesView
from tradingflow.metrics.mean.information_coefficient import build as build_ic
from tradingflow.metrics.mean.regression_coefficients import build as build_regcoef
from tradingflow.metrics.variance.log_likelihood import build as build_ll
from tradingflow.metrics.variance.minimum_variance import build as build_mv


def _drive(op, state, inputs, ts):
    """Run one compute() call, returning its output batch (or `None`).

    `inputs` is the unified tuple — one entry per wired leaf in wiring
    order, signal pulses as plain bools.
    """
    return op.compute(tuple(inputs), state, ts)


def _scalar(emitted):
    """Extract a Python float from an emitted scalar batch (0-d or (1,))."""
    return float(np.asarray(emitted).ravel()[0])


def test_information_coefficient():
    n = 8
    op = build_ic(num_stocks=n)
    state = op.init((False, FakeArrayView(np.zeros(n)), False, FakeArrayView(np.zeros(n))))

    rng = np.random.default_rng(0)
    pred0 = rng.standard_normal(n)

    # First prediction: warmup, stores scores, no emit.
    emitted = _drive(op, state, [True, FakeArrayView(pred0), False, FakeArrayView(np.zeros(n))], 0)
    assert emitted is None

    # A few target ticks: correlated with pred0 so IC is meaningful.
    ics = []
    for k in range(3):
        tgt = pred0 + 0.3 * rng.standard_normal(n)
        emitted = _drive(op, state, [False, FakeArrayView(pred0), True, FakeArrayView(tgt)], 10 + k)
        assert emitted is None, "target-only tick must not emit"
        valid = np.isfinite(pred0) & np.isfinite(tgt)
        ics.append(float(np.corrcoef(pred0[valid], tgt[valid])[0, 1]))

    # Second prediction: emit mean daily IC.
    pred1 = rng.standard_normal(n)
    emitted = _drive(op, state, [True, FakeArrayView(pred1), False, FakeArrayView(np.zeros(n))], 20)
    assert emitted is not None
    expected = float(np.mean(ics))
    assert np.isclose(_scalar(emitted), expected), (emitted, expected)
    assert -1.0 <= _scalar(emitted) <= 1.0
    print("test_information_coefficient OK")


def test_regression_coefficients():
    f = 3
    op = build_regcoef(num_features=f, max_periods=None, min_periods=4)

    # Build aligned target / baseline series. True model: y = X beta + alpha.
    rng = np.random.default_rng(1)
    t = 30
    x = rng.standard_normal((t, f))
    beta = np.array([1.5, -2.0, 0.5])
    alpha = 0.25
    y = x @ beta + alpha + 1e-9 * rng.standard_normal(t)

    target = FakeSeriesView([np.asarray(v) for v in y], elem_shape=())
    baseline = FakeSeriesView([x[i] for i in range(t)], elem_shape=(f,))

    state = op.init((False, target, baseline))

    # Signal false -> no refit.
    emitted = _drive(op, state, [False, target, baseline], 0)
    assert emitted is None

    # Signal true -> fit and emit (F+1,) with intercept last.
    coef = _drive(op, state, [True, target, baseline], 1)
    assert coef is not None
    assert coef.shape == (f + 1,)
    assert np.all(np.isfinite(coef))
    assert np.allclose(coef[:f], beta, atol=1e-4), coef[:f]
    assert np.isclose(coef[f], alpha, atol=1e-4), coef[f]

    # min_periods gate: tiny series below threshold -> no emit.
    op2 = build_regcoef(num_features=f, min_periods=10)
    small_t = FakeSeriesView([np.asarray(v) for v in y[:3]], elem_shape=())
    small_b = FakeSeriesView([x[i] for i in range(3)], elem_shape=(f,))
    state2 = op2.init((False, small_t, small_b))
    emitted = _drive(op2, state2, [True, small_t, small_b], 1)
    assert emitted is None, "min_periods gate should suppress emit"

    # Rank deficiency -> NaN out (collinear baseline columns).
    op3 = build_regcoef(num_features=f)
    xc = x.copy()
    xc[:, 1] = 2.0 * xc[:, 0]  # collinear
    yc = xc @ beta + alpha
    tc = FakeSeriesView([np.asarray(v) for v in yc], elem_shape=())
    bc = FakeSeriesView([xc[i] for i in range(t)], elem_shape=(f,))
    state3 = op3.init((False, tc, bc))
    emitted = _drive(op3, state3, [True, tc, bc], 1)
    assert emitted is not None
    assert np.all(np.isnan(emitted)), "rank-deficient fit must emit NaN"
    print("test_regression_coefficients OK")


def _spd_cov(rng, n, scale=0.01):
    a = rng.standard_normal((n, n))
    cov = a @ a.T / n
    cov += np.eye(n) * 1e-3
    return cov * scale


def test_log_likelihood():
    n = 6
    op = build_ll(num_stocks=n)
    state = op.init((False, FakeArrayView(np.zeros((n, n))), False, FakeArrayView(np.zeros(n))))

    rng = np.random.default_rng(2)
    sigma0 = _spd_cov(rng, n)

    # First prediction: warmup, caches Sigma+/log|Sigma|, no emit.
    emitted = _drive(op, state, [True, FakeArrayView(sigma0), False, FakeArrayView(np.zeros(n))], 0)
    assert emitted is None
    assert state.sigma_inv is not None
    assert np.isfinite(state.log_det)

    # Accumulate target ticks; one carries a NaN (treated as zero).
    quads = []
    sinv = state.sigma_inv
    ldet = state.log_det
    for k in range(4):
        r = rng.standard_normal(n) * 0.1
        if k == 1:
            r[2] = np.nan
        emitted = _drive(op, state, [False, FakeArrayView(sigma0), True, FakeArrayView(r)], 10 + k)
        assert emitted is None
        rz = np.where(np.isfinite(r), r, 0.0)
        quads.append(float(rz @ sinv @ rz))

    # Second prediction: emit period-averaged NLL.
    sigma1 = _spd_cov(rng, n)
    emitted = _drive(op, state, [True, FakeArrayView(sigma1), False, FakeArrayView(np.zeros(n))], 20)
    assert emitted is not None
    expected = ldet + float(np.mean(quads))
    assert np.isclose(_scalar(emitted), expected), (emitted, expected)
    assert np.isfinite(_scalar(emitted))

    # Excluded stock (NaN on diagonal) must not break the pseudo-inverse.
    sigma_excl = sigma1.copy()
    sigma_excl[0, 0] = np.nan
    op2 = build_ll(num_stocks=n)
    s2 = op2.init((False, FakeArrayView(sigma_excl), False, FakeArrayView(np.zeros(n))))
    _drive(op2, s2, [True, FakeArrayView(sigma_excl), False, FakeArrayView(np.zeros(n))], 0)
    assert np.allclose(s2.sigma_inv[0, :], 0.0) and np.allclose(s2.sigma_inv[:, 0], 0.0)
    print("test_log_likelihood OK")


def test_minimum_variance():
    n = 6
    op = build_mv(num_stocks=n)
    state = op.init((False, FakeArrayView(np.zeros((n, n))), False, FakeArrayView(np.zeros(n))))

    rng = np.random.default_rng(3)
    sigma0 = _spd_cov(rng, n)  # log-return covariance

    emitted = _drive(op, state, [True, FakeArrayView(sigma0), False, FakeArrayView(np.zeros(n))], 0)
    assert emitted is None
    w = state.weights
    assert w is not None and w.shape == (n,)
    assert np.isclose(float(np.sum(w)), 1.0), w  # GMV weights sum to 1

    # Accumulate a few log-return target ticks (NaN treated as zero).
    rps = []
    for k in range(5):
        r_log = rng.standard_normal(n) * 0.05
        if k == 2:
            r_log[1] = np.nan
        emitted = _drive(op, state, [False, FakeArrayView(sigma0), True, FakeArrayView(r_log)], 10 + k)
        assert emitted is None
        rz = np.where(np.isfinite(r_log), r_log, 0.0)
        rps.append(float(w @ np.expm1(rz)))

    # Second prediction: emit realized variance of accumulated returns.
    sigma1 = _spd_cov(rng, n)
    emitted = _drive(op, state, [True, FakeArrayView(sigma1), False, FakeArrayView(np.zeros(n))], 20)
    assert emitted is not None
    rps_arr = np.array(rps)
    expected = float(np.mean(rps_arr**2) - np.mean(rps_arr) ** 2)
    assert np.isclose(_scalar(emitted), expected), (emitted, expected)
    assert np.isfinite(_scalar(emitted))
    assert _scalar(emitted) >= -1e-12  # variance non-negative
    print("test_minimum_variance OK")


def main():
    test_information_coefficient()
    test_regression_coefficients()
    test_log_likelihood()
    test_minimum_variance()
    print("\nAll tradingflow.metrics tests passed.")


if __name__ == "__main__":
    main()
