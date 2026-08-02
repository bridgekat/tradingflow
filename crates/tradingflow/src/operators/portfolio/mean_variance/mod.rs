//! Portfolios trading predicted return against predicted risk.
//!
//! [`markowitz`] is the classical formulation, with [`Mode`] choosing which of
//! the four equivalent ways to trade return against risk is stated as the
//! objective and which as the constraint. [`benchmark_relative`] optimizes
//! against the universe as a benchmark instead of in absolute terms, which is
//! what an index-relative mandate actually needs. [`admm_mnr`] solves the same
//! problem as `markowitz` in [`Mode::MinMeanVariance`], but without CVXPY.
//!
//! # Why the covariance is approximated
//!
//! The CVXPY-backed portfolios cannot take the covariance as a solver
//! parameter directly: the DPP canonicalization map for an `[N, N]` parameter
//! has shape `N x N²`, which exhausts memory around a thousand names. Passing
//! it as a non-parameter instead re-canonicalizes the whole problem every
//! solve, which defeats the warm start.
//!
//! So they approximate `Σ ≈ B Bᵀ + diag(d²)` from its top `factor_rank`
//! eigenpairs, with the diagonal matched exactly so per-name variances are
//! preserved and only the off-diagonal residual beyond those factors is
//! dropped. The risk term becomes `‖Bᵀx‖² + ‖d ⊙ x‖²`, which *is* DPP, so the
//! problem is built once and each rebalance only sets parameters. Raising
//! `factor_rank` buys accuracy at linear cost in the parameter size.
//!
//! [`admm_mnr`] avoids the question: it never forms the covariance, only
//! multiplies by it.
//!
//! See [module-level docs](super) for inputs and outputs.

mod admm_mnr;
mod benchmark_relative;
mod markowitz;

pub use admm_mnr::admm_mnr;
pub use benchmark_relative::benchmark_relative;
pub use markowitz::{Mode, markowitz};

use crate::data::Instant;
use crate::ports::{ArrayPort, SignalPort};

/// The wiring every mean-variance portfolio shares:
/// `(rebalance_signal, universe, predicted_returns, covariance)`.
pub type Inputs = (
    SignalPort<0>,
    ArrayPort<f64, 1>,
    ArrayPort<f64, 1>,
    ArrayPort<f64, 2>,
);

/// A rebalance pulse and the `[N]` book.
pub type Outputs = (SignalPort<0>, ArrayPort<f64, 1>);

/// What every constructor in this module returns.
pub trait MeanVariancePortfolio:
    crate::graph::Operator<Inputs = Inputs, Outputs = Outputs, Context = Instant>
{
}

impl<S> MeanVariancePortfolio for S where
    S: crate::graph::Operator<Inputs = Inputs, Outputs = Outputs, Context = Instant>
{
}
