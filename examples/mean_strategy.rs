//! Port of `python/examples/mean_strategy.py` to the Rust flow engine.
//!
//! A cross-sectional linear-regression strategy on a bounded A-shares
//! universe: a pooled **Ridge** mean predictor on the canonical 7-factor panel
//! (`common::build_features`), a **RankLinear** portfolio, and two traders — a
//! frictionless `Benchmark` and a lot/fee-aware `RandomTrader` — versus the
//! cap-weighted index. Performance metrics (Sharpe / compound return /
//! drawdown, and rolling market beta/alpha via `RegressionCoefficients`) are
//! computed natively, clock-gated on the rebalance schedule.
//!
//! The predictor / portfolio / trader / beta-alpha operators are Python
//! (`flowops`) operators on the shared interpreter, so this needs
//! `--features pyflow` and a venv with NumPy (a standard GIL venv is fine).
//!
//! ```text
//! cargo run --example mean_strategy --features pyflow -- --index-size 1000
//! python examples/plot_strategy.py target/mean_strategy.csv
//! ```

#[path = "common/mod.rs"]
mod common;

use flowgraph::typed::{Handle, Port};

use tradingflow::flow::{
    CompoundReturn, Diff, Drawdown, Log, Map, Multiply, PyClassOperator, PyParams, Scenario,
    SharpeRatio, Stack,
};
use tradingflow::sources::clock;
use tradingflow::{Array, Series};

const INITIAL_CASH: f64 = 1_000_000.0;
const NUM_FEATURES: i64 = 7;

/// Map a trader's `(holdings_value, cash)` output to its scalar total value.
fn total_value(sc: &mut Scenario, h: Handle<Array<f64>>) -> Handle<Array<f64>> {
    sc.add_operator(
        Map::new(|a: &Array<f64>| Array::scalar(a.as_slice().iter().sum::<f64>())),
        h,
    )
}

#[tokio::main]
async fn main() {
    let args = common::Args::from_env();
    let symbols = common::load_symbols(&args.data_dir);
    let n = symbols.len();
    let n_i = n as i64;
    let idx = args.index_size as i64;
    eprintln!("loaded {n} symbols; index_size={}", args.index_size);

    let mut sc = Scenario::new();
    let clk = sc.clock();

    // ---- Data + features ------------------------------------------------
    let st = common::build_stacked(&mut sc, &symbols, &args);
    let features = common::build_features(&mut sc, &st, &args);
    let circ_market_cap = sc.add_operator(Multiply::<f64>::new(), (st.close, st.circ_shares));
    let log_adj = sc.add_operator(Log::<f64>::new(), st.adjusted_close);
    let (_target, _target_series, demeaned_series) =
        common::build_log_return_target(&mut sc, log_adj);
    let (upper, lower) = common::build_price_limits(&mut sc, st.close, 0.10);

    // ---- Universe + predictor + portfolio -------------------------------
    let rebalance_clock = sc.add_source(clock(args.rebalance_instants()), ());
    let universe = common::build_cap_weighted_universe(
        &mut sc,
        circ_market_cap,
        rebalance_clock,
        args.index_size,
    );

    let predicted_returns = sc.add_operator(
        PyClassOperator::<(Port<Array<f64>>, Port<Series<f64>>, Port<Series<f64>>)>::from_module(
            "flowops.predictors.mean.ridge",
            PyParams::new()
                .int("num_stocks", n_i)
                .int("num_features", NUM_FEATURES)
                .int("universe_size", idx)
                .int("target_offset", 1)
                .int("min_periods", 100)
                .float("alpha", 1.0),
            vec![n],
            clk.clone(),
        ),
        (universe, features.series, demeaned_series),
    );

    let soft_positions = sc.add_operator(
        PyClassOperator::<(Port<Array<f64>>, Port<Array<f64>>)>::from_module(
            "flowops.portfolios.mean.rank_linear",
            PyParams::new().int("num_stocks", n_i).float("top_fraction", 1.0),
            vec![n],
            clk.clone(),
        ),
        (universe, predicted_returns),
    );

    // ---- Traders --------------------------------------------------------
    let index = sc.add_operator(
        PyClassOperator::<[Port<Array<f64>>]>::from_module(
            "flowops.traders.benchmark",
            PyParams::new().int("num_stocks", n_i).float("initial_cash", 1.0).bool("use_adjusts", true),
            vec![2],
            clk.clone(),
        ),
        &[universe, st.close, st.adjusts, upper, lower][..],
    );
    let strategy_frictionless = sc.add_operator(
        PyClassOperator::<[Port<Array<f64>>]>::from_module(
            "flowops.traders.benchmark",
            PyParams::new().int("num_stocks", n_i).float("initial_cash", 1.0).bool("use_adjusts", true),
            vec![2],
            clk.clone(),
        ),
        &[soft_positions, st.close, st.adjusts, upper, lower][..],
    );
    let strategy_actual = sc.add_operator(
        PyClassOperator::<[Port<Array<f64>>]>::from_module(
            "flowops.traders.random_trader",
            PyParams::new()
                .int("num_stocks", n_i)
                .int("portfolio_size", 20)
                .int("seed", 0)
                .float("initial_cash", INITIAL_CASH)
                .float("lot_size", 100.0)
                .float("fee_base", 5.0)
                .float("fee_rate", 0.001),
            vec![2],
            clk.clone(),
        ),
        &[soft_positions, st.close, st.adjusts, upper, lower][..],
    );

    // ---- Values + metrics (clock-gated, since inception) ----------------
    let index_value = total_value(&mut sc, index);
    let frictionless_value = total_value(&mut sc, strategy_frictionless);
    let actual_value = total_value(&mut sc, strategy_actual);

    let sharpe = sc.add_operator(SharpeRatio::<f64>::new(), (actual_value, rebalance_clock));
    let compound = sc.add_operator(CompoundReturn::<f64>::new(), (actual_value, rebalance_clock));
    let drawdown = sc.add_operator(Drawdown::<f64>::new(), actual_value);

    // Rolling market beta / alpha vs the cap-weighted index, on daily log
    // returns of total value (regressor adds the intercept → output [beta, alpha]).
    let log_actual = sc.add_operator(Log::<f64>::new(), actual_value);
    let strat_logret = sc.add_operator(Diff::<f64>::new(), log_actual);
    let log_index = sc.add_operator(Log::<f64>::new(), index_value);
    let index_logret = sc.add_operator(Diff::<f64>::new(), log_index);
    let strat_logret_series = sc.add_record(strat_logret);
    let index_logret_vec = sc.add_operator(Stack::<f64>::new(0), &[index_logret][..]); // scalar -> (1,)
    let index_logret_series = sc.add_record(index_logret_vec);
    let beta_alpha = sc.add_operator(
        PyClassOperator::<(Port<()>, Port<Series<f64>>, Port<Series<f64>>)>::from_module(
            "flowops.metrics.mean.regression_coefficients",
            PyParams::new().int("num_features", 1).int("max_periods", 252).int("min_periods", 20),
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
    let h_beta_alpha = sc.add_record(beta_alpha);

    let mut session = sc.build_with_threads(args.threads);
    session.run(|_, _| {}).await;

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
    let last = |h| common::read_scalar_series(&session, h).1.into_iter().last().unwrap_or(f64::NAN);
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
