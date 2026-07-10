//! Typed constructors for the `flowops` Python predictors, portfolios and metrics.
//!
//! Each `flowops` operator is reached through [`PyClassOperator`], whose three
//! moving parts must agree: the **input port tuple** (a turbofish), the **module
//! path** (a string), and the **`PyParams`** (stringly-keyed). Spelling them out
//! at every call site is what made swapping a predictor or an optimizer a
//! four-place edit — and it is how the covariance rank in
//! `benchmark_relative_strategy` silently disagreed with `flowops`' documented
//! `(N, N)` input (a rank-1 `PyClassOperator` flattens `[n, n]` to `[n*n]`, and
//! the Python side's `np.diag` then *builds* an `n²×n²` matrix).
//!
//! Here each model is one function: its parameters are typed and named, its port
//! shape is fixed by the return type, and the module path appears exactly once.
//! Swapping an optimizer is swapping a call.

use flowgraph::typed::RefPort;

use tradingflow::operators::{Clock, PyClassOperator, PyParams};
use tradingflow::{Array, Series};

// ===========================================================================
// Port shapes
// ===========================================================================

/// A predictor's inputs: `(universe, feature panel history, target history)`.
pub type PredictorIn = (
    RefPort<Array<f64, 1>>,
    RefPort<Series<f64, 2>>,
    RefPort<Series<f64, 1>>,
);

/// A mean predictor: `(N,)` expected returns.
pub type MeanPredictor = PyClassOperator<PredictorIn, 1>;

/// A covariance predictor: an `(N, N)` matrix. The rank-2 parameter is what
/// keeps the matrix a matrix across the Python boundary.
pub type CovPredictor = PyClassOperator<PredictorIn, 2>;

/// A mean-only portfolio's inputs: `(universe, predicted returns)`.
pub type MeanPortfolio = PyClassOperator<(RefPort<Array<f64, 1>>, RefPort<Array<f64, 1>>), 1>;

/// A mean-variance portfolio's inputs: `(universe, predicted returns, covariance)`.
pub type MeanVarPortfolio = PyClassOperator<
    (
        RefPort<Array<f64, 1>>,
        RefPort<Array<f64, 1>>,
        RefPort<Array<f64, 2>>,
    ),
    1,
>;

/// The rolling market beta/alpha metric: `(clock, strategy log-returns, index log-returns)`.
pub type RegressionCoefficients = PyClassOperator<
    (
        RefPort<()>,
        RefPort<Series<f64, 0>>,
        RefPort<Series<f64, 1>>,
    ),
    1,
>;

/// The GMV realized-variance metric: `(covariance, raw returns)` → scalar.
pub type MinimumVariance =
    PyClassOperator<(RefPort<Array<f64, 2>>, RefPort<Array<f64, 1>>), 0>;

/// The information-coefficient metric: `(factor, target)` → scalar correlation.
pub type InformationCoefficient =
    PyClassOperator<(RefPort<Array<f64, 1>>, RefPort<Array<f64, 1>>), 0>;

// ===========================================================================
// Shared shape
// ===========================================================================

/// The panel dimensions every predictor needs.
#[derive(Clone, Copy)]
pub struct Dims {
    /// Rows of the cross-section (all loaded symbols).
    pub num_stocks: usize,
    /// Feature-panel columns.
    pub num_features: usize,
    /// In-universe stocks per rebalance.
    pub universe_size: usize,
    /// Forward-return offset pairing `features[i]` with `target[i + offset]`.
    pub target_offset: i64,
}

impl Dims {
    fn params(&self) -> PyParams {
        PyParams::new()
            .int("num_stocks", self.num_stocks as i64)
            .int("num_features", self.num_features as i64)
            .int("universe_size", self.universe_size as i64)
            .int("target_offset", self.target_offset)
    }
}

/// `Mode` in `flowops.portfolios.mean_variance._modes`.
#[derive(Clone, Copy)]
pub enum Mode {
    MinMeanVariance = 3,
}

// ===========================================================================
// Mean predictors
// ===========================================================================

/// Pooled Ridge regression on the factor panel. `alpha` is the L2 penalty that
/// regularises a collinear panel (the 145-factor set leans on it heavily).
pub fn ridge_mean(d: Dims, min_periods: i64, alpha: f64, clk: &Clock) -> MeanPredictor {
    PyClassOperator::from_module(
        "flowops.predictors.mean.incremental_ridge",
        d.params().int("min_periods", min_periods).float("alpha", alpha),
        vec![d.num_stocks],
        clk.clone(),
    )
}

/// Unregularised pooled linear regression on the factor panel.
pub fn linear_regression_mean(d: Dims, min_periods: i64, clk: &Clock) -> MeanPredictor {
    PyClassOperator::from_module(
        "flowops.predictors.mean.incremental_linear_regression",
        d.params().int("min_periods", min_periods),
        vec![d.num_stocks],
        clk.clone(),
    )
}

// ===========================================================================
// Covariance predictors
// ===========================================================================

/// A covariance estimator: a `flowops.predictors.variance.*` module plus the
/// estimator-specific knobs (shrinkage `target`, RMT `mode`). Construct with
/// [`shrinkage`](CovEstimator::shrinkage) / [`sample`](CovEstimator::sample) /
/// [`rmt`](CovEstimator::rmt) / [`single_index`](CovEstimator::single_index).
#[derive(Clone, Copy)]
pub struct CovEstimator {
    pub name: &'static str,
    module: &'static str,
    target: Option<i64>,
    mode: Option<&'static str>,
}

impl CovEstimator {
    pub const fn sample(name: &'static str) -> Self {
        Self { name, module: "flowops.predictors.variance.sample", target: None, mode: None }
    }
    /// `target`: 1 = common covariance, 2 = constant correlation, 3 = single index.
    pub const fn shrinkage(name: &'static str, target: i64) -> Self {
        Self {
            name,
            module: "flowops.predictors.variance.shrinkage",
            target: Some(target),
            mode: None,
        }
    }
    /// `mode`: `"zero"` or `"mean"`.
    pub const fn rmt(name: &'static str, mode: &'static str) -> Self {
        Self { name, module: "flowops.predictors.variance.rmt", target: None, mode: Some(mode) }
    }
    pub const fn single_index(name: &'static str) -> Self {
        Self { name, module: "flowops.predictors.variance.single_index", target: None, mode: None }
    }

    /// Build the rank-2 covariance operator. `min_periods` is the per-stock
    /// coverage filter; pass `None` to leave the estimator's own default.
    pub fn build(
        &self,
        d: Dims,
        max_periods: i64,
        min_periods: Option<i64>,
        clk: &Clock,
    ) -> CovPredictor {
        let mut p = d.params().int("max_periods", max_periods);
        if let Some(m) = min_periods {
            p = p.int("min_periods", m);
        }
        if let Some(t) = self.target {
            p = p.int("target", t);
        }
        if let Some(m) = self.mode {
            p = p.str("mode", m);
        }
        PyClassOperator::from_module(self.module, p, vec![d.num_stocks, d.num_stocks], clk.clone())
    }
}

/// Ledoit-Wolf shrinkage toward the common-covariance target — the default
/// covariance for the mean-variance strategies.
pub fn shrinkage_cov(
    d: Dims,
    max_periods: i64,
    min_periods: i64,
    clk: &Clock,
) -> CovPredictor {
    CovEstimator::shrinkage("shrinkage_comm_cov", 1).build(d, max_periods, Some(min_periods), clk)
}

// ===========================================================================
// Portfolios
// ===========================================================================

/// Rank-linear weights over the predicted returns: no covariance, no solver.
pub fn rank_linear(num_stocks: usize, top_fraction: f64, clk: &Clock) -> MeanPortfolio {
    PyClassOperator::from_module(
        "flowops.portfolios.mean.rank_linear",
        PyParams::new()
            .int("num_stocks", num_stocks as i64)
            .float("top_fraction", top_fraction),
        vec![num_stocks],
        clk.clone(),
    )
}

/// Markowitz mean-variance via cvxpy. `bound` is the risk-aversion δ.
pub fn markowitz(
    num_stocks: usize,
    universe_size: usize,
    mode: Mode,
    bound: f64,
    long_only: bool,
    clk: &Clock,
) -> MeanVarPortfolio {
    PyClassOperator::from_module(
        "flowops.portfolios.mean_variance.markowitz",
        PyParams::new()
            .int("num_stocks", num_stocks as i64)
            .int("max_universe_size", universe_size as i64)
            .int("mode", mode as i64)
            .float("bound", bound)
            .bool("long_only", long_only),
        vec![num_stocks],
        clk.clone(),
    )
}

/// Benchmark-relative (enhanced-index) Markowitz via cvxpy: maximise predicted
/// return subject to a tracking-error budget. `bound` is the **daily** γ_TE.
pub fn benchmark_relative(
    num_stocks: usize,
    universe_size: usize,
    bound: f64,
    long_only: bool,
    full_position: bool,
    clk: &Clock,
) -> MeanVarPortfolio {
    PyClassOperator::from_module(
        "flowops.portfolios.mean_variance.benchmark_relative",
        PyParams::new()
            .int("num_stocks", num_stocks as i64)
            .int("max_universe_size", universe_size as i64)
            .float("bound", bound)
            .bool("long_only", long_only)
            .bool("full_position", full_position),
        vec![num_stocks],
        clk.clone(),
    )
}

// ===========================================================================
// Metrics
// ===========================================================================

/// Rolling market beta / alpha of the strategy's log-returns against the index's
/// (the regressor adds the intercept, so the output is `[beta, alpha]`).
pub fn regression_coefficients(
    num_features: i64,
    max_periods: i64,
    min_periods: i64,
    clk: &Clock,
) -> RegressionCoefficients {
    PyClassOperator::from_module(
        "flowops.metrics.mean.regression_coefficients",
        PyParams::new()
            .int("num_features", num_features)
            .int("max_periods", max_periods)
            .int("min_periods", min_periods),
        vec![2],
        clk.clone(),
    )
}

/// Realized variance of the GMV portfolio implied by a covariance estimate — a
/// pure diagnostic of covariance quality.
pub fn minimum_variance(num_stocks: usize, clk: &Clock) -> MinimumVariance {
    PyClassOperator::from_module(
        "flowops.metrics.variance.minimum_variance",
        PyParams::new().int("num_stocks", num_stocks as i64),
        vec![],
        clk.clone(),
    )
}

/// The cross-sectional correlation of a factor with a target, per rebalance.
/// Feeding it two *ranked* vectors makes it the Spearman (rank) IC; feeding raw
/// values makes it the Pearson IC.
pub fn information_coefficient(num_stocks: usize, clk: &Clock) -> InformationCoefficient {
    PyClassOperator::from_module(
        "flowops.metrics.mean.information_coefficient",
        PyParams::new().int("num_stocks", num_stocks as i64),
        vec![],
        clk.clone(),
    )
}
