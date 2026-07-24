//! Benchmark-relative mean-variance strategy on the A-shares panel.
//!
//! Enhanced-index (benchmark-relative) strategies over a sweep of *annualised*
//! tracking-error budgets γ. Each γ drives a cvxpy **BenchmarkRelative**
//! portfolio — maximise predicted return subject to a second-order-cone
//! tracking-error constraint against the cap-weighted benchmark `x_bm` (the
//! universe weights) — sharing one **Ridge** mean predictor and one
//! **Shrinkage** covariance predictor. Budgets are converted to daily units
//! (γ_daily = γ_ann / √252) to match the 1-day moments.
//!
//! Like `mean_variance_strategy`, this solves a **cvxpy** optimizer inside the
//! engine, so it needs `--features python` and a **GIL** venv with cvxpy.
//!
//! ```text
//! cargo run --example benchmark_relative_strategy --features python -- --index-size 1000
//! python examples/plot_strategy.py target/benchmark_relative_strategy.csv
//! ```

#[path = "common/mod.rs"]
mod common;

use clap::Parser;

use tradingflow::clock::UnixClock;
use tradingflow::data::Retention;
use tradingflow::graph::Builder;

use common::models::{benchmark_relative, ridge_mean, shrinkage_cov};
use common::strategy::{Market, NavTable, TRADING_DAYS};

const TRACKING_ERRORS_ANN: [f64; 3] = [0.02, 0.05, 0.10];
const COV_MAX_PERIODS: i64 = 200;
const MIN_PERIODS: i64 = 100;
const RIDGE_ALPHA: f64 = 1.0;

/// Benchmark-relative mean-variance strategy on the A-shares panel.
#[derive(Parser)]
struct Args {
    #[command(flatten)]
    common: common::CommonArgs,
}

#[tokio::main]
async fn main() {
    let Args { common: args } = Args::parse();
    let symbols = common::load_symbols(&args.data_dir);
    eprintln!(
        "loaded {} symbols; index_size={}; gammas_ann={TRACKING_ERRORS_ANN:?}",
        symbols.len(),
        args.index_size
    );

    let mut sc = Builder::new(UnixClock);

    let m = Market::build(&mut sc, &symbols, &args, Retention::unbounded());

    let predicted_returns = sc.segment(
        ridge_mean(m.dims, MIN_PERIODS, RIDGE_ALPHA),
        (m.universe, m.features.series, m.demeaned_series),
    );
    let predicted_cov = sc.segment(
        shrinkage_cov(m.dims, COV_MAX_PERIODS, MIN_PERIODS),
        (m.universe, m.features.series, m.target_series),
    );

    let h_index = m.index_nav(&mut sc);

    // One BenchmarkRelative portfolio per (daily) tracking-error budget.
    let variants: Vec<_> = TRACKING_ERRORS_ANN
        .iter()
        .map(|&gamma_ann| {
            let gamma_daily = gamma_ann / TRADING_DAYS.sqrt();
            let soft = sc.segment(
                benchmark_relative(m.n, args.index_size, gamma_daily, true, true),
                (m.universe, predicted_returns, predicted_cov),
            );
            (gamma_ann, m.record_nav(&mut sc, soft))
        })
        .collect();

    let session = common::run(sc, &args).await;

    let begin = args.begin().as_offset().as_nanos();
    let mut table = NavTable::default();
    let s = table.add(&session, "index", begin, h_index);
    println!(
        "index:          final={:.0} CNY  cagr={:.2}% sharpe={:.3} mdd={:.2}%",
        s.final_value,
        s.cagr * 100.0,
        s.sharpe,
        s.mdd * 100.0
    );
    for (gamma_ann, h) in variants {
        let s = table.add(&session, format!("gamma_{gamma_ann}"), begin, h);
        println!(
            "gamma_ann={:>5.0e}: final={:.0} CNY  cagr={:.2}% sharpe={:.3} mdd={:.2}%",
            gamma_ann,
            s.final_value,
            s.cagr * 100.0,
            s.sharpe,
            s.mdd * 100.0,
        );
    }
    table.write("target/benchmark_relative_strategy.csv");
}
