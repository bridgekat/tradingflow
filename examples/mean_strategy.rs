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

use flowgraph::typed::{Handle, RefPort};

use tradingflow::Scenario;
use tradingflow::operators::{
    ArrayValue, Benchmark, CompoundReturn, Diff, Drawdown, Map, PyClassOperator, PyParams,
    RandomTrader, SharpeRatio, Stack, log, multiply,
};
use tradingflow::sources::clock;
use tradingflow::{Array, ArrayView, Retention, Series, ViewPort};

use common::{own, ref_view, AvH};

const INITIAL_CASH: f64 = 1_000_000.0;
/// Forward-return offset pairing `features[i]` with `target[i+1]`.
const TARGET_OFFSET: i64 = 1;

/// A scalar view handle (a rank-0 `ArrayView` port).
type ScH = Handle<ViewPort<ArrayValue<f64, 0>>>;

/// Map a trader's `(holdings_value, cash)` view to its scalar total value.
fn total_value(sc: &mut Scenario, h: AvH) -> ScH {
    sc.add_operator(
        Map::new(|a: ArrayView<f64, 1>| Array::scalar(a.to_contiguous().iter().sum::<f64>())),
        h,
    )
}

use clap::Parser;

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
    let fset = common::FeatureSet::parse(&feature_set);
    let symbols = common::load_symbols(&args.data_dir);
    let n = symbols.len();
    let n_i = n as i64;
    let idx = args.index_size as i64;
    eprintln!("loaded {n} symbols; index_size={}", args.index_size);

    let mut sc = Scenario::new();
    let clk = sc.clock();

    // ---- Data + features ------------------------------------------------
    // The incremental mean predictor folds one (feature, target) pair per tick,
    // so the recorded panel / target only need the last `TARGET_OFFSET + 1` rows.
    let feat_ret = Retention::count(TARGET_OFFSET as usize + common::RETAIN_MARGIN);
    let st = common::build_stacked(&mut sc, &symbols, &args);
    let features = common::build_strategy_features(&mut sc, &st, window, fset, feat_ret);
    let num_features = features.names.len() as i64;
    eprintln!("feature set `{feature_set}`: {} features", num_features);
    let circ_market_cap = sc.add_operator(multiply::<f64, 1>(), (st.close, st.circ_shares));
    let log_adj = sc.add_operator(log::<f64, 1>(), st.adjusted_close);
    let (_target, _target_series, demeaned_series) =
        common::build_log_return_target(&mut sc, log_adj, feat_ret);
    let (upper, lower) = common::build_price_limits(&mut sc, st.close, 0.10);

    // ---- Universe + predictor + portfolio -------------------------------
    let rebalance_clock = sc.add_source(clock(args.rebalance_instants()), ());
    let universe = common::build_cap_weighted_universe(
        &mut sc,
        circ_market_cap,
        rebalance_clock,
        args.index_size,
    );
    // The Python predictor/portfolio operators consume whole-array `RefPort`s;
    // materialize the universe view once via `own`.
    let universe_ref = own(&mut sc, universe);

    let predicted_returns = sc.add_operator(
        PyClassOperator::<(
            RefPort<Array<f64, 1>>,
            RefPort<Series<f64>>,
            RefPort<Series<f64>>,
        )>::from_module(
            "flowops.predictors.mean.incremental_ridge",
            PyParams::new()
                .int("num_stocks", n_i)
                .int("num_features", num_features)
                .int("universe_size", idx)
                .int("target_offset", TARGET_OFFSET)
                .int("min_periods", 100)
                .float("alpha", 0.01),
            vec![n],
            clk.clone(),
        ),
        (universe_ref, features.series, demeaned_series),
    );

    let soft_positions = sc.add_operator(
        PyClassOperator::<(RefPort<Array<f64, 1>>, RefPort<Array<f64, 1>>)>::from_module(
            "flowops.portfolios.mean.rank_linear",
            PyParams::new()
                .int("num_stocks", n_i)
                .float("top_fraction", 1.0),
            vec![n],
            clk.clone(),
        ),
        (universe_ref, predicted_returns),
    );
    // Bridge the Python `RefPort` positions back into the view currency the
    // native traders speak.
    let soft_positions_v = sc.as_view(soft_positions);

    // ---- Traders --------------------------------------------------------
    let index = sc.add_operator(
        Benchmark::new(n, 1.0, true),
        (universe, st.close, st.adjusts, upper, lower),
    );
    let strategy_frictionless = sc.add_operator(
        Benchmark::new(n, 1.0, true),
        (soft_positions_v, st.close, st.adjusts, upper, lower),
    );
    let strategy_actual = sc.add_operator(
        RandomTrader::new(n, 20, INITIAL_CASH, 100.0, 5.0, 0.001, 0),
        (soft_positions_v, st.close, st.adjusts, upper, lower),
    );

    // ---- Values + metrics (clock-gated, since inception) ----------------
    let index_value = total_value(&mut sc, index);
    let frictionless_value = total_value(&mut sc, strategy_frictionless);
    let actual_value = total_value(&mut sc, strategy_actual);

    let sharpe = sc.add_operator(SharpeRatio::<f64, 0>::new(), (actual_value, rebalance_clock));
    let compound = sc.add_operator(
        CompoundReturn::<f64, 0>::new(),
        (actual_value, rebalance_clock),
    );
    let drawdown = sc.add_operator(Drawdown::<f64, 0>::new(), actual_value);

    // Rolling market beta / alpha vs the cap-weighted index, on daily log
    // returns of total value (regressor adds the intercept → output [beta, alpha]).
    let log_actual = sc.add_operator(log::<f64, 0>(), actual_value);
    let strat_logret = sc.add_operator(Diff::<f64, 0>::new(), log_actual);
    let log_index = sc.add_operator(log::<f64, 0>(), index_value);
    let index_logret = sc.add_operator(Diff::<f64, 0>::new(), log_index);
    let strat_logret_series = sc.add_record(strat_logret);
    // scalar -> (1,): bridge the rank-0 view into the `RefViewPort` slice `Stack`
    // consumes, then stack into a 1-vector.
    let index_logret_ref = ref_view::<0>(&mut sc, index_logret);
    let index_logret_vec = sc.add_operator(Stack::<f64, 0, 1>::new(0), &[index_logret_ref][..]);
    let index_logret_series = sc.add_record(index_logret_vec);
    let beta_alpha = sc.add_operator(
        PyClassOperator::<(RefPort<()>, RefPort<Series<f64>>, RefPort<Series<f64>>)>::from_module(
            "flowops.metrics.mean.regression_coefficients",
            PyParams::new()
                .int("num_features", 1)
                .int("max_periods", 252)
                .int("min_periods", 20),
            vec![2],
            clk.clone(),
        ),
        (rebalance_clock, strat_logret_series, index_logret_series),
    );

    // ---- Records --------------------------------------------------------
    let h_index = sc.add_record(index_value);
    let h_fric = sc.add_record(frictionless_value);
    let h_actual = sc.add_record(actual_value);
    let h_sharpe = sc.add_record(sharpe);
    let h_compound = sc.add_record(compound);
    let h_drawdown = sc.add_record(drawdown);
    // `beta_alpha` is a Python `RefPort<Array>` output; bridge into the view
    // currency for recording.
    let beta_alpha_v = sc.as_view(beta_alpha);
    let h_beta_alpha = sc.add_record(beta_alpha_v);

    let mut session = sc.build_with_threads(args.threads);
    let total = session.estimated_event_count();
    session.run(common::progress(total, args.begin())).await;
    eprintln!();

    // ---- Extract + report ----------------------------------------------
    let begin = args.begin().as_nanos();
    let (it, iv) = common::read_scalar_series(&session, h_index);
    let (ft, fv) = common::read_scalar_series(&session, h_fric);
    let (at, av) = common::read_scalar_series(&session, h_actual);
    // Unit-cash baselines (index / frictionless) scaled to the actual capital.
    let scale = |ts: Vec<i64>, v: Vec<f64>, k: f64| -> (Vec<i64>, Vec<f64>) {
        let v = v.into_iter().map(|x| x * k).collect();
        (ts, v)
    };
    let trim = |ts: Vec<i64>, v: Vec<f64>| -> (Vec<i64>, Vec<f64>) {
        ts.into_iter().zip(v).filter(|(t, _)| *t >= begin).unzip()
    };
    let (it, iv) = trim(it, iv);
    let (ft, fv) = trim(ft, fv);
    let (at, av) = trim(at, av);
    let (it, iv) = scale(it, iv, INITIAL_CASH);
    let (ft, fv) = scale(ft, fv, INITIAL_CASH);

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
    let ba = session.value(h_beta_alpha) as &Series<f64>;
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
