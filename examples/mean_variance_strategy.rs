//! Mean-variance strategy: shrinkage covariance + Markowitz, risk-aversion sweep.
//!
//! Markowitz mean-variance strategies over a sweep of risk-aversion deltas,
//! sharing one **Ridge** mean predictor and one **Shrinkage** covariance
//! predictor over the 145-factor CICC handbook panel (feeding both predictors).
//! Each delta drives a cvxpy **Markowitz** portfolio
//! (`Mode.MIN_MEAN_VARIANCE`, long-only), traded frictionlessly via
//! `Benchmark`, versus the cap-weighted index. On the whole-market top-800
//! universe the 145-factor panel gives the best variant a Sharpe of 0.35,
//! while cutting max drawdown below the
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

use tradingflow::graph::Builder;
use tradingflow::operators::portfolio::mean_variance::{Mode, markowitz};
use tradingflow::operators::predictor::mean::ridge_incr;
use tradingflow::operators::predictor::variance::{Target, shrinkage};
use tradingflow::operators::{portfolio, predictor};
use tradingflow::time::UnixTime;

use common::strategy::{Market, NavTable};

const DELTAS: [f64; 8] = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0];
/// Shrinkage covariance training window (most-recent pairs fed to the fit).
const COV_MAX_PERIODS: usize = 200;
const MIN_PERIODS: usize = 100;
const RIDGE_ALPHA: f64 = 0.01;
/// Rank of the covariance approximation the Markowitz solve optimizes against.
const FACTOR_RANK: usize = 20;

/// Mean-variance strategy: shrinkage covariance + Markowitz, risk-aversion sweep.
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
        "loaded {} symbols; index_size={}; deltas={DELTAS:?}",
        symbols.len(),
        args.index_size
    );

    let mut sc = Builder::new(UnixTime);

    let m = Market::build(&mut sc, &symbols, &args);
    eprintln!("{} features", m.features.names.len());

    // Each predictor keeps its own training window: the incremental Ridge only
    // needs the pair being folded, while the shrinkage covariance holds
    // `COV_MAX_PERIODS` cross-sections to re-estimate from.
    let mean_config = predictor::Config {
        target_offset: 1,
        min_periods: Some(MIN_PERIODS),
        universe_size: Some(args.index_size),
        ..predictor::Config::default()
    };
    let cov_config = predictor::Config {
        max_periods: Some(COV_MAX_PERIODS),
        ..mean_config
    };

    // Mean predictor (demeaned target) and covariance predictor (raw target).
    let predicted_returns = sc.segment(
        ridge_incr(mean_config, RIDGE_ALPHA),
        m.predictor_inputs(m.demeaned),
    );
    let predicted_cov = sc.segment(
        shrinkage(cov_config, Target::CommonCovariance),
        m.predictor_inputs(m.target),
    );

    let portfolio_config = portfolio::Config {
        max_universe_size: Some(args.index_size),
        ..portfolio::Config::default()
    };

    let h_index = m.index_nav(&mut sc);

    // One Markowitz portfolio per delta — the optimizer swap point.
    let variants: Vec<_> = DELTAS
        .iter()
        .map(|&delta| {
            let soft = sc.segment(
                markowitz(
                    portfolio_config,
                    Mode::MinMeanVariance,
                    delta,
                    true,
                    true,
                    FACTOR_RANK,
                ),
                (
                    m.rebalance_signal,
                    m.universe,
                    predicted_returns.1,
                    predicted_cov.1,
                ),
            );
            (delta, m.record_nav(&mut sc, soft))
        })
        .collect();

    let session = common::run(sc, &args).await;

    // Extract + report.
    let begin = args.begin().as_offset().as_nanos();
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
