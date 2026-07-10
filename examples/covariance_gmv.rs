//! Variance-estimator comparison via GMV portfolio realized variance.
//!
//! Compares covariance estimators inside a Markowitz mean-variance strategy.
//! All seven estimators (sample, common-covariance /
//! constant-correlation / single-index shrinkage, RMT-0, RMT-M, single-index)
//! each feed (a) a `MinimumVariance` realized-variance
//! metric — a pure diagnostic of covariance quality — and (b) a cvxpy
//! **Markowitz** portfolio (long-only and long-short), all sharing one
//! **LinearRegression** mean predictor so differences are attributable to the
//! covariance estimator alone.
//!
//! Solves cvxpy optimizers in the engine → needs `--features python` and a
//! **GIL** venv with cvxpy.
//!
//! ```text
//! cargo run --example covariance_gmv --features python -- --index-size 1000
//! python examples/plot_strategy.py target/covariance_gmv.csv
//! ```

#[path = "common/mod.rs"]
mod common;

use clap::Parser;

use tradingflow::operators::{as_view, diff, own, record};
use tradingflow::{Retention, Scenario, WallClock};

use common::models::{linear_regression_mean, markowitz, minimum_variance, CovEstimator, Mode};
use common::strategy::{Market, NavH, NavTable};
use common::FeatureSet;

const RISK_AVERSION: f64 = 1.0;
const MIN_PERIODS: i64 = 100;
const TRADING_DAYS: f64 = 252.0;

/// The estimators under comparison, in report order.
const ESTIMATORS: [CovEstimator; 7] = [
    CovEstimator::sample("sample"),
    CovEstimator::shrinkage("shrinkage_comm_cov", 1),
    CovEstimator::shrinkage("shrinkage_const_corr", 2),
    CovEstimator::shrinkage("shrinkage_single_index", 3),
    CovEstimator::rmt("rmt_0", "zero"),
    CovEstimator::rmt("rmt_m", "mean"),
    CovEstimator::single_index("single_index"),
];

/// Variance-estimator comparison via GMV portfolio realized variance.
#[derive(Parser)]
struct Args {
    #[command(flatten)]
    common: common::CommonArgs,
    /// Rolling feature window in trading days (momentum / volatility / turnover MAs).
    #[arg(long)]
    window: usize,
}

/// Per-estimator records: long / long-short NAV and the GMV realized variance.
struct Rec {
    name: &'static str,
    long: NavH,
    ls: NavH,
    mv: NavH,
}

#[tokio::main]
async fn main() {
    let Args {
        common: args,
        window,
    } = Args::parse();
    let symbols = common::load_symbols(&args.data_dir);
    eprintln!(
        "loaded {} symbols; index_size={}",
        symbols.len(),
        args.index_size
    );

    let mut sc = Scenario::new(WallClock);
    let clk = sc.time();

    let m = Market::build(
        &mut sc,
        &symbols,
        &args,
        window,
        FeatureSet::Canonical,
        Retention::UNBOUNDED,
    );
    // Raw daily log returns for the realized-variance metric, as a whole-array
    // `RefPort` (the metric is a Python operator).
    let log_returns = sc.push(diff(), m.log_adj);
    let log_returns_ref = sc.push(own(), log_returns);

    let predicted_returns = sc.push(
        linear_regression_mean(m.dims, MIN_PERIODS, &clk),
        (m.universe_ref, m.features.series, m.demeaned_series),
    );

    let h_index = m.index_nav(&mut sc);

    let recs: Vec<Rec> = ESTIMATORS
        .iter()
        .map(|e| {
            // Covariance window = rebalance period, and there is no per-stock
            // `min_periods` filter on the covariance estimators (the mean
            // `LinearRegression` above keeps its own `min_periods`).
            let cov = sc.push(
                e.build(m.dims, args.rebalance_days, None, &clk),
                (m.universe_ref, m.features.series, m.target_series),
            );

            // GMV realized-variance metric (diagnostic; fed cov + raw returns).
            let mv = sc.push(minimum_variance(m.n, &clk), (cov, log_returns_ref));
            let mv_v = sc.push(as_view(), mv);

            // Long-only and long-short Markowitz portfolios.
            let nav: Vec<NavH> = [true, false]
                .into_iter()
                .map(|long_only| {
                    let soft = sc.push(
                        markowitz(
                            m.n,
                            args.index_size,
                            Mode::MinMeanVariance,
                            RISK_AVERSION,
                            long_only,
                            &clk,
                        ),
                        (m.universe_ref, predicted_returns, cov),
                    );
                    m.record_nav(&mut sc, soft)
                })
                .collect();

            Rec {
                name: e.name,
                long: nav[0],
                ls: nav[1],
                mv: sc.push(record(&clk), mv_v),
            }
        })
        .collect();

    let session = common::run(sc, &args).await;

    let begin = args.begin().as_nanos();
    let mut table = NavTable::default();
    let index = table.add(&session, "index", begin, h_index);
    println!("index: final={:.0} CNY", index.final_finite);

    for r in &recs {
        let long = table.add(&session, format!("{}_long", r.name), begin, r.long);
        let ls = table.add(&session, format!("{}_ls", r.name), begin, r.ls);
        // Mean GMV realized variance → annualized realized vol.
        let mvv = common::read_scalar_series(&session, r.mv).1;
        let finite: Vec<f64> = mvv
            .into_iter()
            .filter(|x| x.is_finite() && *x >= 0.0)
            .collect();
        let ann_vol = if finite.is_empty() {
            f64::NAN
        } else {
            (finite.iter().sum::<f64>() / finite.len() as f64 * TRADING_DAYS).sqrt()
        };
        println!(
            "{:>13}: GMV ann_vol={:.4}  long_final={:.0}  ls_final={:.0} CNY",
            r.name, ann_vol, long.final_finite, ls.final_finite,
        );
    }

    table.write("target/covariance_gmv.csv");
}
