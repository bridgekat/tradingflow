//! Mean-only strategy: periodic linear regression + rank-linear portfolio.
//!
//! A cross-sectional linear-regression strategy on a bounded A-shares
//! universe: a pooled **Ridge** mean predictor on a configurable factor panel
//! (`--feature-set`, default `all` = the 145 CICC handbook factors; the pooled
//! Ridge regularises their collinearity), a **RankLinear** portfolio, and two
//! traders — a frictionless `Benchmark` and a lot/fee-aware `RandomTrader` —
//! versus the cap-weighted index. On the whole-market top-800 universe the
//! 145-factor panel lifts the actual Sharpe from 0.38 (canonical 7-factor) to
//! 0.41. Performance metrics (Sharpe / compound return /
//! drawdown, and rolling market beta/alpha via `RegressionCoefficients`) are
//! computed natively, clock-gated on the rebalance schedule.
//!
//! The predictor, portfolio, and beta-alpha operators are Python (`flowops`)
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

use tradingflow::operators::{
    as_view, benchmark, compound_return, diff, drawdown, log, random_trader, record,
    ref_array_view, sharpe_ratio, stack,
};
use tradingflow::{Retention, Scenario, Series, WallClock};

use common::FeatureSet;
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
    /// Rolling feature window in trading days (momentum / volatility / turnover MAs).
    /// Only used by `--feature-set canonical`.
    #[arg(long)]
    window: usize,
    /// Feature panel: `all` (145 CICC handbook factors, default — the pooled
    /// Ridge regularises the heavy collinearity), `cicc` (curated ~24), or
    /// `canonical` (the legacy 7-factor panel, uses `--window`).
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
        "loaded {} symbols; index_size={}",
        symbols.len(),
        args.index_size
    );

    let mut sc = Scenario::new(WallClock);
    let clk = sc.time();

    // ---- Data + features ------------------------------------------------
    // The incremental mean predictor folds one (feature, target) pair per tick,
    // so the recorded panel / target only need the last `TARGET_OFFSET + 1` rows.
    let feat_ret = Retention::count(TARGET_OFFSET as usize + common::RETAIN_MARGIN);
    let m = Market::build(&mut sc, &symbols, &args, window, fset, feat_ret);
    eprintln!(
        "feature set `{feature_set}`: {} features",
        m.dims.num_features
    );

    // ---- Predictor + portfolio ------------------------------------------
    let predicted_returns = sc.push(
        ridge_mean(m.dims, MIN_PERIODS, RIDGE_ALPHA, &clk),
        (m.universe_ref, m.features.series, m.demeaned_series),
    );
    let soft_positions = sc.push(
        rank_linear(m.n, 1.0, &clk),
        (m.universe_ref, predicted_returns),
    );
    // Bridge the Python `RefPort` positions back into the view currency the
    // native traders speak.
    let soft_positions_v = sc.push(as_view(), soft_positions);

    // ---- Traders (the cost-model swap point) ----------------------------
    let index_value = m.simulate(&mut sc, benchmark(m.n, 1.0, true), m.universe);
    let frictionless_value = m.simulate(&mut sc, benchmark(m.n, 1.0, true), soft_positions_v);
    let actual_value = m.simulate(
        &mut sc,
        random_trader(m.n, 20, INITIAL_CASH, 100.0, 5.0, 0.001, 0),
        soft_positions_v,
    );

    // ---- Metrics (clock-gated, since inception) -------------------------
    let sharpe = sc.push(sharpe_ratio(), (m.rebalance_clock, actual_value));
    let compound = sc.push(compound_return(), (m.rebalance_clock, actual_value));
    let drawdown = sc.push(drawdown(), actual_value);

    // Rolling market beta / alpha vs the cap-weighted index, on daily log
    // returns of total value (regressor adds the intercept → output [beta, alpha]).
    let log_actual = sc.push(log(), actual_value);
    let strat_logret = sc.push(diff(), log_actual);
    let log_index = sc.push(log(), index_value);
    let index_logret = sc.push(diff(), log_index);
    let strat_logret_series = sc.push(record(&clk), strat_logret);
    // scalar -> (1,): bridge the rank-0 view into the `RefViewPort` slice `Stack`
    // consumes, then stack into a 1-vector.
    let index_logret_ref = sc.push(ref_array_view(), index_logret);
    let index_logret_vec = sc.push(stack(0), &[index_logret_ref][..]);
    let index_logret_series = sc.push(record(&clk), index_logret_vec);
    let beta_alpha = sc.push(
        regression_coefficients(1, BETA_MAX_PERIODS, BETA_MIN_PERIODS, &clk),
        (m.rebalance_clock, strat_logret_series, index_logret_series),
    );

    // ---- Records --------------------------------------------------------
    let h_index = sc.push(record(&clk), index_value);
    let h_fric = sc.push(record(&clk), frictionless_value);
    let h_actual = sc.push(record(&clk), actual_value);
    let h_sharpe = sc.push(record(&clk), sharpe);
    let h_compound = sc.push(record(&clk), compound);
    let h_drawdown = sc.push(record(&clk), drawdown);
    // `beta_alpha` is a Python `RefPort<Array>` output; bridge into the view
    // currency for recording.
    let beta_alpha_v = sc.push(as_view(), beta_alpha);
    let h_beta_alpha = sc.push(record(&clk), beta_alpha_v);

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
    let ba = session.ref_view(h_beta_alpha) as &Series<f64, 1>;
    let (beta, alpha) = ba
        .values()
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
