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

use tradingflow::clock::UnixClock;
use tradingflow::data::Retention;
use tradingflow::graph::Builder;
use tradingflow::operators::rolling::diff;
use tradingflow::operators::series::record_all;
use tradingflow::ports::SeriesPortHandle;

use common::models::{CovEstimator, Mode, linear_regression_mean, markowitz, minimum_variance};
use common::strategy::{Market, NavTable, TRADING_DAYS};

const RISK_AVERSION: f64 = 1.0;
const MIN_PERIODS: i64 = 100;

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
}

/// Per-estimator records: long / long-short NAV and the GMV realized variance.
struct Rec {
    name: &'static str,
    long: SeriesPortHandle<f64, 0>,
    ls: SeriesPortHandle<f64, 0>,
    mv: SeriesPortHandle<f64, 0>,
}

#[tokio::main]
async fn main() {
    let Args { common: args } = Args::parse();
    let symbols = common::load_symbols(&args.data_dir);
    eprintln!(
        "loaded {} symbols; index_size={}",
        symbols.len(),
        args.index_size
    );

    let mut sc = Builder::new(UnixClock);

    let m = Market::build(&mut sc, &symbols, &args, Retention::unbounded());
    // Raw daily log returns for the realized-variance metric (an ordinary
    // `ArrayPort` view — the Python metric consumes it directly).
    let log_returns = sc.segment(diff(1), m.log_adj);

    let predicted_returns = sc.segment(
        linear_regression_mean(m.dims, MIN_PERIODS),
        (m.universe, m.features.series, m.demeaned_series),
    );

    let h_index = m.index_nav(&mut sc);

    let recs: Vec<Rec> = ESTIMATORS
        .iter()
        .map(|e| {
            // Covariance window = rebalance period, and there is no per-stock
            // `min_periods` filter on the covariance estimators (the mean
            // `LinearRegression` above keeps its own `min_periods`).
            let cov = sc.segment(
                e.build(m.dims, args.rebalance_days, None),
                (m.universe, m.features.series, m.target_series),
            );

            // GMV realized-variance metric (diagnostic; fed cov + raw returns).
            let mv = sc.segment(minimum_variance(m.n), (cov, log_returns));

            // Long-only and long-short Markowitz portfolios.
            let nav: Vec<SeriesPortHandle<f64, 0>> = [true, false]
                .into_iter()
                .map(|long_only| {
                    let soft = sc.segment(
                        markowitz(
                            m.n,
                            args.index_size,
                            Mode::MinMeanVariance,
                            RISK_AVERSION,
                            long_only,
                        ),
                        (m.universe, predicted_returns, cov),
                    );
                    m.record_nav(&mut sc, soft)
                })
                .collect();

            Rec {
                name: e.name,
                long: nav[0],
                ls: nav[1],
                mv: sc.segment(record_all(), mv),
            }
        })
        .collect();

    let session = common::run(sc, &args).await;

    let begin = args.begin().as_offset().as_nanos();
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
