//! Mean-variance strategy: shrinkage covariance + Markowitz, risk-aversion sweep.
//!
//! Markowitz mean-variance strategies over a sweep of risk-aversion deltas,
//! sharing one **Ridge** mean predictor and one **Shrinkage** covariance
//! predictor over a configurable factor panel (`--feature-set`, default `all` =
//! the 145 CICC handbook factors, feeding both predictors). Each delta drives a
//! cvxpy **Markowitz** portfolio (`Mode.MIN_MEAN_VARIANCE`, long-only), traded
//! frictionlessly via `Benchmark`, versus the cap-weighted index. On the
//! whole-market top-800 universe the 145-factor panel lifts the best variant's
//! Sharpe from 0.19 (canonical) to 0.35 while cutting max drawdown below the
//! index's (−39% vs −48%).
//!
//! This is the first example that solves a **cvxpy** optimizer *inside the
//! engine*: the Markowitz portfolio releases the GIL during its SCS solve, so
//! the work-stealing pool overlaps the per-delta solves. Needs `--features
//! python` and a **GIL** venv with cvxpy installed.
//!
//! ```text
//! cargo run --example mean_variance_strategy --features python -- --index-size 1000
//! python examples/plot_strategy.py target/mean_variance_strategy.csv
//! ```

#[path = "common/mod.rs"]
mod common;

use clap::Parser;

use tradingflow::{Retention, Scenario, WallClock};

use common::models::{Mode, markowitz, ridge_mean, shrinkage_cov};
use common::strategy::{Market, NavTable};
use common::{FeatureSet, TARGET_OFFSET};

const DELTAS: [f64; 8] = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0];
/// Shrinkage covariance training window (most-recent pairs fed to the fit).
const COV_MAX_PERIODS: i64 = 200;
const MIN_PERIODS: i64 = 100;
const RIDGE_ALPHA: f64 = 0.01;

/// Mean-variance strategy: shrinkage covariance + Markowitz, risk-aversion sweep.
#[derive(Parser)]
struct Args {
    #[command(flatten)]
    common: common::CommonArgs,
    /// Rolling feature window in trading days (momentum / volatility / turnover MAs).
    /// Only used by `--feature-set canonical`.
    #[arg(long)]
    window: usize,
    /// Feature panel: `all` (145 CICC handbook factors, default — feeds both the
    /// Ridge mean predictor and the shrinkage covariance factor model, with Ridge
    /// regularising the collinearity), `cicc` (curated ~24), or `canonical` (the
    /// legacy 7-factor panel, uses `--window`).
    #[arg(long = "feature-set", default_value = "all")]
    feature_set: String,
}

#[tokio::main]
async fn main() {
    let Args {
        common: args,
        window,
        feature_set,
    } = Args::parse();
    let fset = FeatureSet::parse(&feature_set);
    let symbols = common::load_symbols(&args.data_dir);
    eprintln!(
        "loaded {} symbols; index_size={}; deltas={DELTAS:?}",
        symbols.len(),
        args.index_size
    );

    let mut sc = Scenario::new(WallClock);

    // The shared panel / target feed the shrinkage covariance predictor too,
    // which fits over its last `COV_MAX_PERIODS` pairs — so the records must
    // retain that window (the mean predictor's single-pair need is subsumed).
    let panel_ret =
        Retention::count((COV_MAX_PERIODS + TARGET_OFFSET) as usize + common::RETAIN_MARGIN);
    let m = Market::build(&mut sc, &symbols, &args, window, fset, panel_ret);
    eprintln!(
        "feature set `{feature_set}`: {} features",
        m.dims.num_features
    );

    // Mean predictor (demeaned target) and covariance predictor (raw target).
    let predicted_returns = sc.push(
        ridge_mean(m.dims, MIN_PERIODS, RIDGE_ALPHA),
        (m.universe_ref, m.features.series, m.demeaned_series),
    );
    let predicted_cov = sc.push(
        shrinkage_cov(m.dims, COV_MAX_PERIODS, MIN_PERIODS),
        (m.universe_ref, m.features.series, m.target_series),
    );

    let h_index = m.index_nav(&mut sc);

    // One Markowitz portfolio per delta — the optimizer swap point.
    let variants: Vec<_> = DELTAS
        .iter()
        .map(|&delta| {
            let soft = sc.push(
                markowitz(m.n, args.index_size, Mode::MinMeanVariance, delta, true),
                (m.universe_ref, predicted_returns, predicted_cov),
            );
            (delta, m.record_nav(&mut sc, soft))
        })
        .collect();

    let session = common::run(sc, &args).await;

    // Extract + report.
    let begin = args.begin().as_nanos();
    let mut table = NavTable::default();
    let s = table.add(&session, "index", begin, h_index);
    println!(
        "index:       final={:.0} CNY  cagr={:.2}% sharpe={:.3} mdd={:.2}%",
        s.final_value,
        s.cagr * 100.0,
        s.sharpe,
        s.mdd * 100.0
    );
    for (delta, h) in variants {
        let s = table.add(&session, format!("delta_{delta}"), begin, h);
        println!(
            "delta={delta:>5}: final={:.0} CNY  cagr={:.2}% sharpe={:.3} mdd={:.2}%",
            s.final_value,
            s.cagr * 100.0,
            s.sharpe,
            s.mdd * 100.0,
        );
    }
    table.write("target/mean_variance_strategy.csv");
}
