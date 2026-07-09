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

use flowgraph::typed::{Handle, RefPort};

use tradingflow::operators::{
    log, multiply, ArrayValue, Benchmark, Map, PyClassOperator, PyParams,
};
use tradingflow::Scenario;
use tradingflow::sources::clock;
use tradingflow::{Array, ArrayView, Series, ViewPort};

use common::{own, AvH};

const INITIAL_CASH: f64 = 1_000_000.0;
const NUM_FEATURES: i64 = 7;
const TRACKING_ERRORS_ANN: [f64; 3] = [0.02, 0.05, 0.10];
const TRADING_DAYS: f64 = 252.0;

/// A scalar view handle (a rank-0 `ArrayView` port).
type ScH = Handle<ViewPort<ArrayValue<f64, 0>>>;

fn total_value(sc: &mut Scenario, h: AvH) -> ScH {
    sc.add_operator(
        Map::new(|a: ArrayView<f64, 1>| Array::scalar(a.to_contiguous().iter().sum::<f64>())),
        h,
    )
}

fn nav_stats(v: &[f64]) -> (f64, f64, f64) {
    let s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite() && *x > 0.0).collect();
    if s.len() < 10 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let years = s.len() as f64 / TRADING_DAYS;
    let cagr = (s[s.len() - 1] / s[0]).powf(1.0 / years) - 1.0;
    let rets: Vec<f64> = s.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
    let mean = rets.iter().sum::<f64>() / rets.len() as f64;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64;
    let sd = var.sqrt();
    let sharpe = if sd > 0.0 { mean * TRADING_DAYS / (sd * TRADING_DAYS.sqrt()) } else { f64::NAN };
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

use clap::Parser;

/// Benchmark-relative mean-variance strategy on the A-shares panel.
#[derive(Parser)]
struct Args {
    #[command(flatten)]
    common: common::CommonArgs,
    /// Rolling feature window in trading days (momentum / volatility / turnover MAs).
    #[arg(long)]
    window: usize,
}

#[tokio::main]
async fn main() {
    let Args { common: args, window } = Args::parse();
    let symbols = common::load_symbols(&args.data_dir);
    let n = symbols.len();
    let n_i = n as i64;
    let idx = args.index_size as i64;
    eprintln!("loaded {n} symbols; index_size={}; gammas_ann={TRACKING_ERRORS_ANN:?}", args.index_size);

    let mut sc = Scenario::new();
    let clk = sc.clock();

    let st = common::build_stacked(&mut sc, &symbols, &args);
    let features = common::build_features(&mut sc, &st, window, tradingflow::Retention::UNBOUNDED);
    let circ_market_cap = sc.add_operator(multiply::<f64, 1>(), (st.close, st.circ_shares));
    let log_adj = sc.add_operator(log::<f64, 1>(), st.adjusted_close);
    let (_target, target_series, demeaned_series) =
        common::build_log_return_target(&mut sc, log_adj, tradingflow::Retention::UNBOUNDED);
    let (upper, lower) = common::build_price_limits(&mut sc, st.close, 0.10);

    let rebalance_clock = sc.add_source(clock(args.rebalance_instants()));
    let universe =
        common::build_cap_weighted_universe(&mut sc, circ_market_cap, rebalance_clock, args.index_size);
    let universe_ref = own(&mut sc, universe);

    let predicted_returns = sc.add_operator(
        PyClassOperator::<(RefPort<Array<f64, 1>>, RefPort<Series<f64, 2>>, RefPort<Series<f64, 1>>)>::from_module(
            "flowops.predictors.mean.incremental_ridge",
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
        (universe_ref, features.series, demeaned_series),
    );
    let predicted_cov = sc.add_operator(
        PyClassOperator::<(RefPort<Array<f64, 1>>, RefPort<Series<f64, 2>>, RefPort<Series<f64, 1>>)>::from_module(
            "flowops.predictors.variance.shrinkage",
            PyParams::new()
                .int("num_stocks", n_i)
                .int("num_features", NUM_FEATURES)
                .int("universe_size", idx)
                .int("target_offset", 1)
                .int("max_periods", 200)
                .int("min_periods", 100)
                .int("target", 1),
            vec![n, n],
            clk.clone(),
        ),
        (universe_ref, features.series, target_series),
    );

    let index = sc.add_operator(
        Benchmark::new(n, 1.0, true),
        (universe, st.close, st.adjusts, upper, lower),
    );
    let index_value = total_value(&mut sc, index);
    let h_index = sc.add_record(index_value);

    // One BenchmarkRelative portfolio per (daily) tracking-error budget.
    let mut variant_handles: Vec<(f64, Handle<RefPort<Series<f64, 0>>>)> = Vec::new();
    for &gamma_ann in &TRACKING_ERRORS_ANN {
        let gamma_daily = gamma_ann / TRADING_DAYS.sqrt();
        let soft = sc.add_operator(
            PyClassOperator::<(RefPort<Array<f64, 1>>, RefPort<Array<f64, 1>>, RefPort<Array<f64, 1>>)>::from_module(
                "flowops.portfolios.mean_variance.benchmark_relative",
                PyParams::new()
                    .int("num_stocks", n_i)
                    .int("max_universe_size", idx)
                    .float("bound", gamma_daily)
                    .bool("long_only", true)
                    .bool("full_position", true),
                vec![n],
                clk.clone(),
            ),
            (universe_ref, predicted_returns, predicted_cov),
        );
        let soft_v = sc.as_view(soft);
        let fric = sc.add_operator(
            Benchmark::new(n, 1.0, true),
            (soft_v, st.close, st.adjusts, upper, lower),
        );
        let value = total_value(&mut sc, fric);
        variant_handles.push((gamma_ann, sc.add_record(value)));
    }

    let mut session = sc.build_with_threads(args.threads);
    let total = session.estimated_event_count();
    session.run(common::progress(total, args.begin())).await;
    eprintln!();

    let begin = args.begin().as_nanos();
    let trim_scale = |ts: Vec<i64>, v: Vec<f64>| -> (Vec<i64>, Vec<f64>) {
        ts.into_iter().zip(v).filter(|(t, _)| *t >= begin).map(|(t, x)| (t, x * INITIAL_CASH)).unzip()
    };

    let (it, iv) = common::read_scalar_series(&session, h_index);
    let (it, iv) = trim_scale(it, iv);
    let (ic, is, im) = nav_stats(&iv);
    println!("index:          final={:.0} CNY  cagr={:.2}% sharpe={:.3} mdd={:.2}%", iv.last().copied().unwrap_or(f64::NAN), ic * 100.0, is, im * 100.0);

    let mut cols: Vec<(String, Vec<i64>, Vec<f64>)> = vec![("index".into(), it, iv)];
    for (gamma_ann, h) in &variant_handles {
        let (t, v) = common::read_scalar_series(&session, *h);
        let (t, v) = trim_scale(t, v);
        let (cagr, sharpe, mdd) = nav_stats(&v);
        println!(
            "gamma_ann={:>5.0e}: final={:.0} CNY  cagr={:.2}% sharpe={:.3} mdd={:.2}%",
            gamma_ann,
            v.last().copied().unwrap_or(f64::NAN),
            cagr * 100.0,
            sharpe,
            mdd * 100.0,
        );
        cols.push((format!("gamma_{gamma_ann}"), t, v));
    }

    let path = "target/benchmark_relative_strategy.csv";
    common::write_wide_csv(path, &cols);
    println!("wrote {path}\nplot with:  python examples/plot_strategy.py {path}");
}
