//! Mean-only strategy: periodic linear regression + rank-linear portfolio.
//!
//! A cross-sectional linear-regression strategy on a bounded A-shares
//! universe: a pooled **Ridge** mean predictor on the 145-factor CICC handbook
//! panel (the pooled Ridge regularises their collinearity), a **RankLinear**
//! portfolio, and two traders — a frictionless `Benchmark` and a lot/fee-aware
//! `RandomTrader` — versus the cap-weighted index. On the whole-market top-800
//! universe the 145-factor panel yields an actual Sharpe of 0.41.
//! Performance metrics (Sharpe / compound return /
//! drawdown, and rolling market beta/alpha via `RegressionCoefficients`) are
//! computed natively, clock-gated on the rebalance schedule.
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

use tradingflow::data::{Retention, SeriesView};
use tradingflow::operators::metrics::{compound_return, drawdown, sharpe_ratio};
use tradingflow::operators::num::diff;
use tradingflow::operators::num::log;
use tradingflow::operators::structural::record;
use tradingflow::operators::structural::stack;
use tradingflow::operators::traders::{benchmark, random_trader};
use tradingflow::{Scenario, WallClock};

use common::models::{rank_linear, regression_coefficients, ridge_mean};
use common::strategy::{INITIAL_CASH, Market, TARGET_OFFSET, trim_scale};

const MIN_PERIODS: i64 = 100;
const RIDGE_ALPHA: f64 = 0.01;
/// Rolling window (trading days) of the market beta/alpha regression.
const BETA_MAX_PERIODS: i64 = 252;
const BETA_MIN_PERIODS: i64 = 20;

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

    let mut sc = Scenario::new(WallClock);

    // ---- Data + features ------------------------------------------------
    // The incremental mean predictor folds one (feature, target) pair per tick,
    // so the recorded panel / target only need the last `TARGET_OFFSET + 1` rows.
    let feat_ret = Retention::count(TARGET_OFFSET as usize + common::RETAIN_MARGIN);
    let m = Market::build(&mut sc, &symbols, &args, feat_ret);
    eprintln!("{} features", m.dims.num_features);

    // ---- Predictor + portfolio ------------------------------------------
    let predicted_returns = sc.segment(
        ridge_mean(m.dims, MIN_PERIODS, RIDGE_ALPHA),
        (m.universe, m.features.series, m.demeaned_series),
    );
    // The Python portfolio emits ordinary `ArrayPort` position views — the
    // native traders consume them directly.
    let soft_positions = sc.segment(rank_linear(m.n, 1.0), (m.universe, predicted_returns));

    // ---- Traders (the cost-model swap point) ----------------------------
    let index_value = m.simulate(&mut sc, benchmark(m.n, 1.0, true), m.universe);
    let frictionless_value = m.simulate(&mut sc, benchmark(m.n, 1.0, true), soft_positions);
    let actual_value = m.simulate(
        &mut sc,
        random_trader(m.n, 20, INITIAL_CASH, 100.0, 5.0, 0.001, 0),
        soft_positions,
    );

    // ---- Metrics (clock-gated, since inception) -------------------------
    let sharpe = sc.segment(sharpe_ratio(), (m.rebalance_clock, actual_value));
    let compound = sc.segment(compound_return(), (m.rebalance_clock, actual_value));
    let drawdown = sc.segment(drawdown(), actual_value);

    // Rolling market beta / alpha vs the cap-weighted index, on daily log
    // returns of total value (regressor adds the intercept → output [beta, alpha]).
    let log_actual = sc.segment(log(), actual_value);
    let strat_logret = sc.segment(diff(), log_actual);
    let log_index = sc.segment(log(), index_value);
    let index_logret = sc.segment(diff(), log_index);
    let strat_logret_series = sc.segment(record(), strat_logret);
    // scalar -> (1,): stack the rank-0 view handle into a 1-vector.
    let index_logret_vec = sc.segment(stack(0), &[index_logret][..]);
    let index_logret_series = sc.segment(record(), index_logret_vec);
    let beta_alpha = sc.segment(
        regression_coefficients(1, BETA_MAX_PERIODS, BETA_MIN_PERIODS),
        (m.rebalance_clock, strat_logret_series, index_logret_series),
    );

    // ---- Records --------------------------------------------------------
    let h_index = sc.segment(record(), index_value);
    let h_fric = sc.segment(record(), frictionless_value);
    let h_actual = sc.segment(record(), actual_value);
    let h_sharpe = sc.segment(record(), sharpe);
    let h_compound = sc.segment(record(), compound);
    let h_drawdown = sc.segment(record(), drawdown);
    let h_beta_alpha = sc.segment(record(), beta_alpha);

    let session = common::run(sc, &args).await;

    // ---- Extract + report ----------------------------------------------
    let begin = args.begin().as_nanos();
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
    let ba: SeriesView<f64, 1> = session.view(h_beta_alpha);
    let (beta, alpha) = ba
        .data()
        .rchunks(2)
        .next()
        .map(|c| (c[0], c[1] * 252.0))
        .unwrap_or((f64::NAN, f64::NAN));

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
