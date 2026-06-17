//! Factor IC evaluation on the A-shares cross-sectional panel.
//!
//! Evaluates the information coefficient (IC) of each factor in the canonical
//! 7-factor panel (`common::build_features`) on a bounded A-shares universe.
//! Each feature is lagged one trading day, resampled onto the rebalance clock,
//! NaN-masked to the cap-weighted top-`index_size` universe, and paired with
//! the winsorized realized log-return target via the `InformationCoefficient`
//! metric. Per-feature IC series are written long-format for cumulative-IC
//! plotting; mean / std / IR are printed.
//!
//! The IC metric is a Python (`flowops`) operator (no cvxpy), so this needs
//! `--features python` and a venv with NumPy (a standard GIL venv is fine).
//!
//! ```text
//! cargo run --example factor_ic --features python -- --index-size 1000
//! python examples/plot_factor_ic.py target/factor_ic.csv
//! ```

#[path = "common/mod.rs"]
mod common;

use flowgraph::typed::RefPort;

use tradingflow::operators::{Apply, Lag, Log, Multiply, PyClassOperator, PyParams, Resample};
use tradingflow::Scenario;
use tradingflow::sources::clock;
use tradingflow::Array;

use clap::Parser;

/// Factor IC evaluation on the A-shares cross-sectional panel.
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
    eprintln!("loaded {n} symbols; index_size={}", args.index_size);

    let mut sc = Scenario::new();
    let clk = sc.clock();

    let st = common::build_stacked(&mut sc, &symbols, &args);
    let features = common::build_features(&mut sc, &st, window, tradingflow::Retention::UNBOUNDED);
    let circ_market_cap = sc.add_operator(Multiply::<f64>::new(), (st.close, st.circ_shares));
    let log_adj = sc.add_operator(Log::<f64>::new(), st.adjusted_close);
    let (target, _target_series, _demeaned) =
        common::build_log_return_target(&mut sc, log_adj, tradingflow::Retention::UNBOUNDED);

    let rebalance_clock = sc.add_source(clock(args.rebalance_instants()), ());
    let universe =
        common::build_cap_weighted_universe(&mut sc, circ_market_cap, rebalance_clock, args.index_size);

    let names = features.names.clone();
    let mut ic_handles = Vec::new();
    for &feature in &features.handles {
        // Lag one trading day, resample onto the rebalance clock.
        let feature_series = sc.add_record(feature);
        let lagged = sc.add_operator(Lag::<f64>::new(1, f64::NAN), feature_series);
        let aligned = sc.add_operator(
            Resample::<Array<f64>, ()>::new(),
            (rebalance_clock, lagged),
        );
        // NaN-mask out-of-universe stocks so they don't dilute the rank corr.
        let masked = sc.add_operator(
            Apply::<(RefPort<Array<f64>>, RefPort<Array<f64>>), Array<f64>, _>::new(
                |(f, u): (&Array<f64>, &Array<f64>)| {
                    let (fs, us) = (f.as_slice(), u.as_slice());
                    Array::from_vec(
                        f.shape(),
                        (0..fs.len()).map(|i| if us[i] > 0.0 { fs[i] } else { f64::NAN }).collect(),
                    )
                },
            ),
            (aligned, universe),
        );
        let metric = sc.add_operator(
            PyClassOperator::<(RefPort<Array<f64>>, RefPort<Array<f64>>)>::from_module(
                "flowops.metrics.mean.information_coefficient",
                PyParams::new().int("num_stocks", n_i),
                vec![],
                clk.clone(),
            ),
            (masked, target),
        );
        ic_handles.push(sc.add_record(metric));
    }

    let mut session = sc.build_with_threads(args.threads);
    let total = session.estimated_event_count();
    session.run(common::progress(total, args.begin())).await;
    eprintln!();

    // Per-feature IC stats + long-format CSV.
    let mut cols: Vec<(String, Vec<i64>, Vec<f64>)> = Vec::new();
    for (name, h) in names.iter().zip(ic_handles.iter()) {
        let (ts, v) = common::read_scalar_series(&session, *h);
        let finite: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
        if finite.is_empty() {
            println!("{name:>18}: no valid values");
        } else {
            let mean = finite.iter().sum::<f64>() / finite.len() as f64;
            let var = finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / finite.len() as f64;
            let std = var.sqrt();
            let ir = if std > 0.0 { mean / std } else { f64::NAN };
            println!(
                "{name:>18}: mean={mean:+.4} std={std:.4} IR={ir:+.4} ({} periods)",
                finite.len()
            );
        }
        cols.push((name.clone(), ts, v));
    }

    let path = "target/factor_ic.csv";
    common::write_long_csv(path, &cols);
    println!("wrote {path}\nplot with:  python examples/plot_factor_ic.py {path}");
}
