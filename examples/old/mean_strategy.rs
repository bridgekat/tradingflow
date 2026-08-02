//! Mean-only strategy: periodic linear regression + rank-linear portfolio.
//!
//! A cross-sectional linear-regression strategy on a bounded A-shares
//! universe: a pooled **Ridge** mean predictor on the 145-factor CICC handbook
//! panel (the pooled Ridge regularises their collinearity), a **RankLinear**
//! portfolio, and two traders — a frictionless `benchmark` and a lot/fee-aware
//! `random` — versus the cap-weighted index. On the whole-market top-800
//! universe the 145-factor panel yields an actual Sharpe of 0.41.
//! Performance metrics (Sharpe / compound return /
//! drawdown, and rolling market beta/alpha via `RegressionCoefficients`) are
//! computed natively, signal-gated on the rebalance schedule.
//!
//! The predictor, portfolio, and beta-alpha operators are Python (`tradingflow`)
//! operators on the shared interpreter (the two traders are native Rust), so
//! this needs `--features python` and a venv with NumPy (a standard GIL venv
//! is fine).
//!
//! ```text
//! cargo run --example mean_strategy --features python -- --index-size 1000
//! python examples/plot_strategy.py target/mean_strategy.csv
//! ```

#[path = "common/mod.rs"]
mod common;

use clap::Parser;

use tradingflow::graph::Builder;
use tradingflow::operators::elem::ln;
use tradingflow::operators::metric::performance::{comp_return, drawdown, return_sharpe};
use tradingflow::operators::portfolio::mean::rank_linear;
use tradingflow::operators::predictor::mean::ridge_incr;
use tradingflow::operators::rolling::diff;
use tradingflow::operators::series::record_all;
use tradingflow::operators::trader::fixed::{benchmark, random};
use tradingflow::operators::{portfolio, predictor};
use tradingflow::time::UnixTime;

use common::ic::trailing_beta_alpha;
use common::strategy::{INITIAL_CASH, Market, trim_scale};

const MIN_PERIODS: usize = 100;
const RIDGE_ALPHA: f64 = 0.01;
/// Rolling window (trading days) of the market beta/alpha regression.
const BETA_MAX_PERIODS: usize = 252;
const BETA_MIN_PERIODS: usize = 20;

/// Mean-only strategy: periodic linear regression + rank-linear portfolio.
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
        "loaded {} symbols; index_size={}",
        symbols.len(),
        args.index_size
    );

    let mut sc = Builder::new(UnixTime);

    // ---- Data + features ------------------------------------------------
    let m = Market::build(&mut sc, &symbols, &args);
    eprintln!("{} features", m.features.names.len());

    // ---- Predictor + portfolio ------------------------------------------
    // The incremental Ridge folds one (feature, target) pair per trading day,
    // so it never holds more than the pair it is forming.
    let predicted_returns = sc.segment(
        ridge_incr(
            predictor::Config {
                target_offset: 1,
                min_periods: Some(MIN_PERIODS),
                universe_size: Some(args.index_size),
                ..predictor::Config::default()
            },
            RIDGE_ALPHA,
        ),
        m.predictor_inputs(m.demeaned),
    );
    // The portfolio emits a `(signal, positions)` stream — the native traders
    // consume it directly.
    let soft_positions = sc.segment(
        rank_linear(
            portfolio::Config {
                max_universe_size: Some(args.index_size),
                ..portfolio::Config::default()
            },
            1.0,
        ),
        (m.rebalance_signal, m.universe, predicted_returns.1),
    );

    // ---- Traders (the cost-model swap point) ----------------------------
    let index_positions = (m.rebalance_signal, m.universe);
    let index_value = m.simulate(&mut sc, benchmark(true, 1.0), index_positions);
    let frictionless_value = m.simulate(&mut sc, benchmark(true, 1.0), soft_positions);
    let actual_value = m.simulate(
        &mut sc,
        random(true, INITIAL_CASH, 5.0, 0.001, 100.0, 20, 0),
        soft_positions,
    );

    // ---- Metrics (signal-gated, since inception) -------------------------
    let sharpe = sc.segment(return_sharpe(), (m.rebalance_signal, actual_value));
    let compound = sc.segment(comp_return(), (m.rebalance_signal, actual_value));
    let drawdown = sc.segment(drawdown(), (m.daily, actual_value));

    // Daily log returns of the strategy and of the index, recorded for the
    // market beta / alpha regression. That regression runs once, on the final
    // window, so it is computed from the records after the run rather than
    // rebuilt in-graph at every rebalance.
    let log_actual = sc.segment(ln(), actual_value);
    let strat_logret = sc.segment(diff(1), (m.daily, log_actual));
    let log_index = sc.segment(ln(), index_value);
    let index_logret = sc.segment(diff(1), (m.daily, log_index));
    let h_strat_logret = sc.segment(record_all(), (m.daily, strat_logret));
    let h_index_logret = sc.segment(record_all(), (m.daily, index_logret));

    // ---- Records --------------------------------------------------------
    let record_daily =
        |sc: &mut tradingflow::graph::Builder<_, _>, h| sc.segment(record_all(), (m.daily, h));
    let h_index = record_daily(&mut sc, index_value);
    let h_fric = record_daily(&mut sc, frictionless_value);
    let h_actual = record_daily(&mut sc, actual_value);
    let h_sharpe = record_daily(&mut sc, sharpe);
    let h_compound = record_daily(&mut sc, compound);
    let h_drawdown = record_daily(&mut sc, drawdown);

    let session = common::run(sc, &args).await;

    // ---- Extract + report ----------------------------------------------
    let begin = args.begin().as_offset().as_nanos();
    // The index / frictionless baselines are unit-cash: scale to the actual
    // capital. The actual trader already runs on `INITIAL_CASH`.
    let (it, iv) = common::read_scalar_series(&session, h_index);
    let (it, iv) = trim_scale(begin, it, iv);
    let (ft, fv) = common::read_scalar_series(&session, h_fric);
    let (ft, fv) = trim_scale(begin, ft, fv);
    let (at, av) = common::read_scalar_series(&session, h_actual);
    let (at, av): (Vec<i64>, Vec<f64>) =
        at.into_iter().zip(av).filter(|(t, _)| *t >= begin).unzip();

    if av.is_empty() {
        eprintln!("no data produced");
        std::process::exit(1);
    }

    let ppy = 365.0 / args.rebalance_days as f64;
    let last = |h| {
        common::read_scalar_series(&session, h)
            .1
            .into_iter()
            .last()
            .unwrap_or(f64::NAN)
    };
    let car = (last(h_compound) + 1.0).powf(ppy) - 1.0;
    let sr = last(h_sharpe) * ppy.sqrt();
    let mdd = common::read_scalar_series(&session, h_drawdown)
        .1
        .into_iter()
        .filter(|x| x.is_finite())
        .fold(0.0_f64, f64::min);
    let strat_logret = common::read_scalar_series(&session, h_strat_logret).1;
    let index_logret = common::read_scalar_series(&session, h_index_logret).1;
    let (beta, alpha) = trailing_beta_alpha(
        &strat_logret,
        &index_logret,
        BETA_MAX_PERIODS,
        BETA_MIN_PERIODS,
    );
    let alpha = alpha * 252.0;

    println!(
        "actual: {:.0} -> {:.0} CNY | annual={:.2}% sharpe={:.3} mdd={:.2}% beta={:.3} alpha_ann={:.2}%",
        av.first().copied().unwrap(),
        av.last().copied().unwrap(),
        car * 100.0,
        sr,
        mdd * 100.0,
        beta,
        alpha * 100.0,
    );

    let path = "target/mean_strategy.csv";
    common::write_wide_csv(
        path,
        &[
            ("index".into(), it, iv),
            ("frictionless".into(), ft, fv),
            ("actual".into(), at, av),
        ],
    );
    println!("wrote {path}\nplot with:  python examples/plot_strategy.py {path}");
}
