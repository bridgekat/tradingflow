//! Port of `python/examples/mean_variance_strategy.py` to the Rust flow engine.
//!
//! Markowitz mean-variance strategies over a sweep of risk-aversion deltas,
//! sharing one **Ridge** mean predictor and one **Shrinkage** covariance
//! predictor. Each delta drives a cvxpy **Markowitz** portfolio
//! (`Mode.MIN_MEAN_VARIANCE`, long-only), traded frictionlessly via
//! `Benchmark`, versus the cap-weighted index.
//!
//! This is the first example that solves a **cvxpy** optimizer *inside the
//! engine*: the Markowitz portfolio releases the GIL during its SCS solve, so
//! the work-stealing pool overlaps the per-delta solves. Needs `--features
//! pyflow` and a **GIL** venv with cvxpy installed (see the cvxpy notes in
//! `python/benchmarks/cvxpy_solve_parallelism.py`).
//!
//! ```text
//! cargo run --example mean_variance_strategy --features pyflow -- --index-size 1000
//! python examples/plot_strategy.py target/mean_variance_strategy.csv
//! ```

#[path = "common/mod.rs"]
mod common;

use flowgraph::typed::{Handle, Port};

use tradingflow::flow::{Log, Map, Multiply, PyClassOperator, PyParams, Scenario};
use tradingflow::sources::clock;
use tradingflow::{Array, Series};

const INITIAL_CASH: f64 = 1_000_000.0;
const NUM_FEATURES: i64 = 7;
const DELTAS: [f64; 8] = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0];
/// Mode.MIN_MEAN_VARIANCE in `flowops.portfolios.mean_variance._modes`.
const MODE_MIN_MEAN_VARIANCE: i64 = 3;

fn total_value(sc: &mut Scenario, h: Handle<Array<f64>>) -> Handle<Array<f64>> {
    sc.add_operator(Map::new(|a: &Array<f64>| Array::scalar(a.as_slice().iter().sum::<f64>())), h)
}

/// (cagr, annualized Sharpe, max drawdown) from a daily NAV series.
fn nav_stats(v: &[f64]) -> (f64, f64, f64) {
    let s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite() && *x > 0.0).collect();
    if s.len() < 10 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let years = s.len() as f64 / 252.0;
    let cagr = (s[s.len() - 1] / s[0]).powf(1.0 / years) - 1.0;
    let rets: Vec<f64> = s.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
    let mean = rets.iter().sum::<f64>() / rets.len() as f64;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64;
    let sd = var.sqrt();
    let sharpe = if sd > 0.0 { mean * 252.0 / (sd * 252.0_f64.sqrt()) } else { f64::NAN };
    let mut peak = f64::MIN;
    let mut mdd = 0.0;
    for &x in &s {
        if x > peak {
            peak = x;
        }
        let dd = x / peak - 1.0;
        if dd < mdd {
            mdd = dd;
        }
    }
    (cagr, sharpe, mdd)
}

#[tokio::main]
async fn main() {
    let args = common::Args::from_env();
    let symbols = common::load_symbols(&args.data_dir);
    let n = symbols.len();
    let n_i = n as i64;
    let idx = args.index_size as i64;
    eprintln!("loaded {n} symbols; index_size={}; deltas={DELTAS:?}", args.index_size);

    let mut sc = Scenario::new();
    let clk = sc.clock();

    let st = common::build_stacked(&mut sc, &symbols, &args);
    let features = common::build_features(&mut sc, &st, &args);
    let circ_market_cap = sc.add_operator(Multiply::<f64>::new(), (st.close, st.circ_shares));
    let log_adj = sc.add_operator(Log::<f64>::new(), st.adjusted_close);
    let (_target, target_series, demeaned_series) =
        common::build_log_return_target(&mut sc, log_adj);
    let (upper, lower) = common::build_price_limits(&mut sc, st.close, 0.10);

    let rebalance_clock = sc.add_source(clock(args.rebalance_instants()), ());
    let universe =
        common::build_cap_weighted_universe(&mut sc, circ_market_cap, rebalance_clock, args.index_size);

    // Mean predictor (demeaned target) and covariance predictor (raw target).
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
    let predicted_cov = sc.add_operator(
        PyClassOperator::<(Port<Array<f64>>, Port<Series<f64>>, Port<Series<f64>>)>::from_module(
            "flowops.predictors.variance.shrinkage",
            PyParams::new()
                .int("num_stocks", n_i)
                .int("num_features", NUM_FEATURES)
                .int("universe_size", idx)
                .int("target_offset", 1)
                .int("max_periods", 200)
                .int("min_periods", 100)
                .int("target", 1), // Target.COMMON_COVARIANCE
            vec![n, n],
            clk.clone(),
        ),
        (universe, features.series, target_series),
    );

    // Index baseline.
    let index = sc.add_operator(
        PyClassOperator::<[Port<Array<f64>>]>::from_module(
            "flowops.traders.benchmark",
            PyParams::new().int("num_stocks", n_i).float("initial_cash", 1.0).bool("use_adjusts", true),
            vec![2],
            clk.clone(),
        ),
        &[universe, st.close, st.adjusts, upper, lower][..],
    );
    let index_value = total_value(&mut sc, index);
    let h_index = sc.add_record(index_value);

    // One Markowitz portfolio per delta.
    let mut variant_handles: Vec<(f64, Handle<Series<f64>>)> = Vec::new();
    for &delta in &DELTAS {
        let soft = sc.add_operator(
            PyClassOperator::<(Port<Array<f64>>, Port<Array<f64>>, Port<Array<f64>>)>::from_module(
                "flowops.portfolios.mean_variance.markowitz",
                PyParams::new()
                    .int("num_stocks", n_i)
                    .int("mode", MODE_MIN_MEAN_VARIANCE)
                    .float("bound", delta)
                    .bool("long_only", true),
                vec![n],
                clk.clone(),
            ),
            (universe, predicted_returns, predicted_cov),
        );
        let fric = sc.add_operator(
            PyClassOperator::<[Port<Array<f64>>]>::from_module(
                "flowops.traders.benchmark",
                PyParams::new().int("num_stocks", n_i).float("initial_cash", 1.0).bool("use_adjusts", true),
                vec![2],
                clk.clone(),
            ),
            &[soft, st.close, st.adjusts, upper, lower][..],
        );
        let value = total_value(&mut sc, fric);
        variant_handles.push((delta, sc.add_record(value)));
    }

    let mut session = sc.build_with_threads(args.threads);
    let total = session.estimated_event_count();
    let counter = session.progress_counter();
    session.run(common::progress(total, args.begin(), counter)).await;
    eprintln!();

    // Extract + report.
    let begin = args.begin().as_nanos();
    let trim_scale = |ts: Vec<i64>, v: Vec<f64>| -> (Vec<i64>, Vec<f64>) {
        ts.into_iter().zip(v).filter(|(t, _)| *t >= begin).map(|(t, x)| (t, x * INITIAL_CASH)).unzip()
    };

    let (it, iv) = common::read_scalar_series(&session, h_index);
    let (it, iv) = trim_scale(it, iv);
    let (ic, is, im) = nav_stats(&iv);
    println!("index:       final={:.0} CNY  cagr={:.2}% sharpe={:.3} mdd={:.2}%", iv.last().copied().unwrap_or(f64::NAN), ic * 100.0, is, im * 100.0);

    let mut cols: Vec<(String, Vec<i64>, Vec<f64>)> = vec![("index".into(), it, iv)];
    for (delta, h) in &variant_handles {
        let (t, v) = common::read_scalar_series(&session, *h);
        let (t, v) = trim_scale(t, v);
        let (cagr, sharpe, mdd) = nav_stats(&v);
        println!(
            "delta={delta:>5}: final={:.0} CNY  cagr={:.2}% sharpe={:.3} mdd={:.2}%",
            v.last().copied().unwrap_or(f64::NAN),
            cagr * 100.0,
            sharpe,
            mdd * 100.0,
        );
        cols.push((format!("delta_{delta}"), t, v));
    }

    let path = "target/mean_variance_strategy.csv";
    common::write_wide_csv(path, &cols);
    println!("wrote {path}\nplot with:  python examples/plot_strategy.py {path}");
}
