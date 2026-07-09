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

use flowgraph::typed::{Handle, RefPort};

use tradingflow::operators::{
    ArrayValue, Benchmark, Diff, Map, PyClassOperator, PyParams, log, multiply,
};
use tradingflow::Scenario;
use tradingflow::sources::clock;
use tradingflow::{Array, ArrayView, Series, ViewPort};

use common::{own, AvH};

const INITIAL_CASH: f64 = 1_000_000.0;
const NUM_FEATURES: i64 = 7;
const RISK_AVERSION: f64 = 1.0;
const MODE_MIN_MEAN_VARIANCE: i64 = 3;
const TRADING_DAYS: f64 = 252.0;

/// A covariance estimator: module + optional shrinkage `target` / RMT `mode`.
struct Est {
    name: &'static str,
    module: &'static str,
    target: Option<i64>,
    mode: Option<&'static str>,
}

/// A scalar view handle (a rank-0 `ArrayView` port).
type ScH = Handle<ViewPort<ArrayValue<f64, 0>>>;

fn total_value(sc: &mut Scenario, h: AvH) -> ScH {
    sc.add_operator(
        Map::new(|a: ArrayView<f64, 1>| Array::scalar(a.to_contiguous().iter().sum::<f64>())),
        h,
    )
}

fn nav_final(v: &[f64]) -> f64 {
    v.iter().rev().copied().find(|x| x.is_finite()).unwrap_or(f64::NAN)
}

use clap::Parser;

/// Variance-estimator comparison via GMV portfolio realized variance.
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
    eprintln!("loaded {n} symbols; index_size={}", args.index_size);

    let mut sc = Scenario::new();
    let clk = sc.clock();

    let st = common::build_stacked(&mut sc, &symbols, &args);
    let features = common::build_features(&mut sc, &st, window, tradingflow::Retention::UNBOUNDED);
    let circ_market_cap = sc.add_operator(multiply::<f64, 1>(), (st.close, st.circ_shares));
    let log_adj = sc.add_operator(log::<f64, 1>(), st.adjusted_close);
    let (_target, target_series, demeaned_series) =
        common::build_log_return_target(&mut sc, log_adj, tradingflow::Retention::UNBOUNDED);
    // Raw daily log returns for the realized-variance metric.
    let log_returns = sc.add_operator(Diff::<f64, 1>::new(), log_adj);
    let (upper, lower) = common::build_price_limits(&mut sc, st.close, 0.10);

    let rebalance_clock = sc.add_source(clock(args.rebalance_instants()));
    let universe =
        common::build_cap_weighted_universe(&mut sc, circ_market_cap, rebalance_clock, args.index_size);
    // The Python predictor/portfolio operators consume whole-array `RefPort`s;
    // materialize the universe view once via `own`.
    let universe_ref = own(&mut sc, universe);
    // The realized-variance metric consumes raw returns as a whole-array
    // `RefPort`; bridge the `Diff` view once.
    let log_returns_ref = own(&mut sc, log_returns);

    let predicted_returns = sc.add_operator(
        PyClassOperator::<(RefPort<Array<f64, 1>>, RefPort<Series<f64, 2>>, RefPort<Series<f64, 1>>)>::from_module(
            "flowops.predictors.mean.incremental_linear_regression",
            PyParams::new()
                .int("num_stocks", n_i)
                .int("num_features", NUM_FEATURES)
                .int("universe_size", idx)
                .int("target_offset", 1)
                .int("min_periods", 100),
            vec![n],
            clk.clone(),
        ),
        (universe_ref, features.series, demeaned_series),
    );

    let index = sc.add_operator(
        Benchmark::new(n, 1.0, true),
        (universe, st.close, st.adjusts, upper, lower),
    );
    let index_value = total_value(&mut sc, index);
    let h_index = sc.add_record(index_value);

    let ests = [
        Est { name: "sample", module: "flowops.predictors.variance.sample", target: None, mode: None },
        Est { name: "shrinkage_comm_cov", module: "flowops.predictors.variance.shrinkage", target: Some(1), mode: None },
        Est { name: "shrinkage_const_corr", module: "flowops.predictors.variance.shrinkage", target: Some(2), mode: None },
        Est { name: "shrinkage_single_index", module: "flowops.predictors.variance.shrinkage", target: Some(3), mode: None },
        Est { name: "rmt_0", module: "flowops.predictors.variance.rmt", target: None, mode: Some("zero") },
        Est { name: "rmt_m", module: "flowops.predictors.variance.rmt", target: None, mode: Some("mean") },
        Est { name: "single_index", module: "flowops.predictors.variance.single_index", target: None, mode: None },
    ];

    // Per-estimator records: (name, long NAV, long-short NAV, GMV realized variance).
    struct Rec {
        name: &'static str,
        long: Handle<RefPort<Series<f64, 0>>>,
        ls: Handle<RefPort<Series<f64, 0>>>,
        mv: Handle<RefPort<Series<f64, 0>>>,
    }
    let mut recs: Vec<Rec> = Vec::new();

    for e in &ests {
        // Covariance predictor.
        // Covariance window = rebalance period, and there is no per-stock
        // `min_periods` filter on the covariance estimators (the
        // mean `LinearRegression` above keeps its own `min_periods=100`).
        let mut p = PyParams::new()
            .int("num_stocks", n_i)
            .int("num_features", NUM_FEATURES)
            .int("universe_size", idx)
            .int("target_offset", 1)
            .int("max_periods", args.rebalance_days);
        if let Some(t) = e.target {
            p = p.int("target", t);
        }
        if let Some(m) = e.mode {
            p = p.str("mode", m);
        }
        let cov = sc.add_operator(
            PyClassOperator::<
                (
                    RefPort<Array<f64, 1>>,
                    RefPort<Series<f64, 2>>,
                    RefPort<Series<f64, 1>>,
                ),
                2,
            >::from_module(
                e.module,
                p,
                vec![n, n],
                clk.clone(),
            ),
            (universe_ref, features.series, target_series),
        );

        // GMV realized-variance metric (diagnostic; fed cov + raw returns).
        let mv = sc.add_operator(
            PyClassOperator::<(RefPort<Array<f64, 2>>, RefPort<Array<f64, 1>>), 0>::from_module(
                "flowops.metrics.variance.minimum_variance",
                PyParams::new().int("num_stocks", n_i),
                vec![],
                clk.clone(),
            ),
            (cov, log_returns_ref),
        );

        // Long-only and long-short Markowitz portfolios.
        let mut nav: Vec<Handle<RefPort<Series<f64, 0>>>> = Vec::with_capacity(2);
        for long_only in [true, false] {
            let soft = sc.add_operator(
                PyClassOperator::<(
                    RefPort<Array<f64, 1>>,
                    RefPort<Array<f64, 1>>,
                    RefPort<Array<f64, 2>>,
                )>::from_module(
                    "flowops.portfolios.mean_variance.markowitz",
                    PyParams::new()
                        .int("num_stocks", n_i)
                        .int("max_universe_size", idx)
                        .int("mode", MODE_MIN_MEAN_VARIANCE)
                        .float("bound", RISK_AVERSION)
                        .bool("long_only", long_only),
                    vec![n],
                    clk.clone(),
                ),
                (universe_ref, predicted_returns, cov),
            );
            // Bridge the Python `RefPort` positions into the view currency the
            // native trader speaks.
            let soft_v = sc.as_view(soft);
            let fric = sc.add_operator(
                Benchmark::new(n, 1.0, true),
                (soft_v, st.close, st.adjusts, upper, lower),
            );
            let value = total_value(&mut sc, fric);
            nav.push(sc.add_record(value));
        }

        // Bridge the Python `RefPort` metric output into the view currency for
        // recording.
        let mv_v = sc.as_view(mv);
        recs.push(Rec { name: e.name, long: nav[0], ls: nav[1], mv: sc.add_record(mv_v) });
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
    println!("index: final={:.0} CNY", nav_final(&iv));

    let mut cols: Vec<(String, Vec<i64>, Vec<f64>)> = vec![("index".into(), it, iv)];
    for r in &recs {
        let (lt, lv) = common::read_scalar_series(&session, r.long);
        let (lt, lv) = trim_scale(lt, lv);
        let (st_, sv) = common::read_scalar_series(&session, r.ls);
        let (st_, sv) = trim_scale(st_, sv);
        // Mean GMV realized variance → annualized realized vol.
        let mvv = common::read_scalar_series(&session, r.mv).1;
        let finite: Vec<f64> = mvv.into_iter().filter(|x| x.is_finite() && *x >= 0.0).collect();
        let ann_vol = if finite.is_empty() {
            f64::NAN
        } else {
            (finite.iter().sum::<f64>() / finite.len() as f64 * TRADING_DAYS).sqrt()
        };
        println!(
            "{:>13}: GMV ann_vol={:.4}  long_final={:.0}  ls_final={:.0} CNY",
            r.name,
            ann_vol,
            nav_final(&lv),
            nav_final(&sv),
        );
        cols.push((format!("{}_long", r.name), lt, lv));
        cols.push((format!("{}_ls", r.name), st_, sv));
    }

    let path = "target/covariance_gmv.csv";
    common::write_wide_csv(path, &cols);
    println!("wrote {path}\nplot with:  python examples/plot_strategy.py {path}");
}
