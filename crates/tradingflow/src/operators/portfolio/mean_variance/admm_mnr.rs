use pyo3::types::PyDictMethods;

use super::MeanVariancePortfolio;
use crate::operators::portfolio::Config;
use crate::python::py_segment_module;

/// Mean-variance optimization by matrix-free ADMM, without CVXPY.
///
/// Solves the same problem as [`markowitz`](super::markowitz) in
/// [`Mode::MinMeanVariance`](super::Mode::MinMeanVariance) — maximize
/// `μᵀx - delta · xᵀΣx` subject to a budget and a box — but reaches it through
/// an ADMM outer loop over MINRES solves of the saddle system, so the
/// covariance enters only as the operator `x ↦ B(Bᵀx) + d²x`. Nothing is ever
/// factorized inside the solve, each iteration costs `O(N k)` rather than
/// `O(N²)`, and there is no canonicalization to amortize.
///
/// The practical difference from `markowitz`: no CVXPY dependency, so this runs
/// wherever NumPy and SciPy do, and cost scales with the factor rank rather
/// than the cross-section. The cost is an iterative solve with tolerances to
/// tune rather than a solver that either converges or reports why not.
///
/// `delta` is the risk-aversion coefficient, `cap` an optional per-name upper
/// bound, and `factor_rank` the rank of the covariance approximation the
/// operator is built from — `None` uses the exact covariance, at `O(N³)` per
/// rebalance.
///
/// See [module-level docs](super::super) for inputs and outputs.
///
/// # Panics
///
/// If `delta` is not positive.
pub fn admm_mnr(
    config: Config,
    delta: f64,
    long_only: bool,
    cap: Option<f64>,
    factor_rank: Option<usize>,
) -> impl MeanVariancePortfolio {
    assert!(delta > 0.0, "admm_mnr delta must be positive, got {delta}");
    py_segment_module(
        "tradingflow.portfolio.mean_variance.admm_mnr",
        config.params(|d| {
            d.set_item("delta", delta)?;
            d.set_item("long_only", long_only)?;
            d.set_item("cap", cap)?;
            d.set_item("factor_rank", factor_rank)
        }),
    )
}
