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

use tradingflow::data::Instant;
use tradingflow::graph::Builder;
use tradingflow::operators::metric::predictor::variance::minimum_variance;
use tradingflow::operators::portfolio::mean_variance::{Mode, markowitz};
use tradingflow::operators::predictor::mean::linear_regression_incr;
use tradingflow::operators::predictor::variance::{
    Replacement, Target, rmt, sample, shrinkage, single_index,
};
use tradingflow::operators::rolling::diff;
use tradingflow::operators::series::record_all;
use tradingflow::operators::{portfolio, predictor};
use tradingflow::ports::{ArrayPortHandle, SeriesPortHandle, SignalPortHandle};
use tradingflow::time::UnixTime;

use common::strategy::{Market, NavTable, TRADING_DAYS};

const RISK_AVERSION: f64 = 1.0;
const MIN_PERIODS: usize = 100;
const FACTOR_RANK: usize = 20;

/// A covariance estimator under comparison.
///
/// The constructors return distinct opaque types, and a `Segment` cannot be
/// boxed into a trait object (its `State` is an associated type), so the
/// variants are named here and wired in a `match` — each arm produces the same
/// pair of handles, which is what unifies them.
#[derive(Clone, Copy)]
enum Kind {
    Sample,
    Shrinkage(Target),
    Rmt(Replacement),
    SingleIndex,
}

struct Estimator {
    name: &'static str,
    kind: Kind,
}

/// The estimators under comparison, in report order — every structured
/// alternative to the raw sample covariance the crate offers.
const ESTIMATORS: [Estimator; 7] = [
    Estimator {
        name: "sample",
        kind: Kind::Sample,
    },
    Estimator {
        name: "shrinkage_comm_cov",
        kind: Kind::Shrinkage(Target::CommonCovariance),
    },
    Estimator {
        name: "shrinkage_const_corr",
        kind: Kind::Shrinkage(Target::ConstantCorrelation),
    },
    Estimator {
        name: "shrinkage_single_index",
        kind: Kind::Shrinkage(Target::SingleIndex),
    },
    Estimator {
        name: "rmt_0",
        kind: Kind::Rmt(Replacement::Zero),
    },
    Estimator {
        name: "rmt_m",
        kind: Kind::Rmt(Replacement::Mean),
    },
    Estimator {
        name: "single_index",
        kind: Kind::SingleIndex,
    },
];

impl Kind {
    /// Wire this estimator into the graph, returning its
    /// `(signal, covariance)` output stream.
    fn segment(
        self,
        sc: &mut Builder<Instant, UnixTime>,
        config: predictor::Config,
        inputs: PredictorInputs,
    ) -> (SignalPortHandle<0>, ArrayPortHandle<f64, 2>) {
        match self {
            Kind::Sample => sc.segment(sample(config), inputs),
            Kind::Shrinkage(target) => sc.segment(shrinkage(config, target), inputs),
            Kind::Rmt(replacement) => sc.segment(rmt(config, replacement), inputs),
            Kind::SingleIndex => sc.segment(single_index(config), inputs),
        }
    }
}

/// What [`Market::predictor_inputs`](common::strategy::Market::predictor_inputs)
/// hands a predictor.
type PredictorInputs = (
    SignalPortHandle<0>,
    ArrayPortHandle<f64, 2>,
    ArrayPortHandle<f64, 1>,
    SignalPortHandle<0>,
    ArrayPortHandle<f64, 1>,
);

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

    let mut sc = Builder::new(UnixTime);

    let m = Market::build(&mut sc, &symbols, &args);
    // Raw daily log returns for the realized-variance metric.
    let log_returns = sc.segment(diff(1), (m.daily, m.log_adj));

    let mean_config = predictor::Config {
        target_offset: 1,
        min_periods: Some(MIN_PERIODS),
        universe_size: Some(args.index_size),
        ..predictor::Config::default()
    };
    // Covariance window = one rebalance period, and no per-stock coverage
    // filter (the mean predictor above keeps its own).
    let cov_config = predictor::Config {
        target_offset: 1,
        max_periods: Some(args.rebalance_days as usize),
        universe_size: Some(args.index_size),
        ..predictor::Config::default()
    };
    let portfolio_config = portfolio::Config {
        max_universe_size: Some(args.index_size),
        ..portfolio::Config::default()
    };

    let predicted_returns = sc.segment(
        linear_regression_incr(mean_config),
        m.predictor_inputs(m.demeaned),
    );

    let h_index = m.index_nav(&mut sc);

    let recs: Vec<Rec> = ESTIMATORS
        .iter()
        .map(|e| {
            let cov = e
                .kind
                .segment(&mut sc, cov_config, m.predictor_inputs(m.target));

            // GMV realized-variance metric (diagnostic; fed the covariance and
            // the raw returns on their own signals).
            let mv = sc.segment(minimum_variance(), (cov.0, cov.1, m.daily, log_returns));

            // Long-only and long-short Markowitz portfolios.
            let nav: Vec<SeriesPortHandle<f64, 0>> = [true, false]
                .into_iter()
                .map(|long_only| {
                    let soft = sc.segment(
                        markowitz(
                            portfolio_config,
                            Mode::MinMeanVariance,
                            RISK_AVERSION,
                            long_only,
                            true,
                            FACTOR_RANK,
                        ),
                        (m.rebalance_signal, m.universe, predicted_returns.1, cov.1),
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
