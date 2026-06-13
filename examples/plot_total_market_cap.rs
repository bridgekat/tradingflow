//! Cap-weighted A-shares index: total circulating market cap + NAV plot.
//!
//! Tracks a cap-weighted A-shares index: at every rebalance the top
//! `--index-size` stocks by circulating market cap form the universe (weights
//! ∝ circulating cap, renormalised). Two views are written to CSV:
//!
//! 1. the summed circulating market cap of the current constituents (daily), and
//! 2. the index total-return NAV from a frictionless `flowops` `Benchmark`
//!    (unit cash, dividend reinvestment via adjust factors).
//!
//! The Benchmark trader is a Python (`flowops`) operator on the shared
//! interpreter, so this example needs a venv with NumPy (a standard GIL venv is
//! fine — see env.ps1, and `examples/flowops_demo.rs`).
//!
//! ```text
//! cargo run --example plot_total_market_cap -- --index-size 1000
//! python examples/plot_total_market_cap.py target/plot_total_market_cap.csv
//! ```

#[path = "common/mod.rs"]
mod common;

use flowgraph::typed::{RefPort, RefPorts};

use tradingflow::operators::{Apply, Map, Multiply, PyClassOperator, PyParams, Resample};
use tradingflow::Scenario;
use tradingflow::sources::clock;
use tradingflow::Array;

#[tokio::main]
async fn main() {
    let args = common::Args::from_env();
    let symbols = common::load_symbols(&args.data_dir);
    let n = symbols.len();
    eprintln!("loaded {n} symbols; index_size={}", args.index_size);

    let mut sc = Scenario::new();
    let clk = sc.clock();

    let st = common::build_stacked(&mut sc, &symbols, &args);
    let circ_market_cap = sc.add_operator(Multiply::<f64>::new(), (st.close, st.circ_shares));

    let rebalance_clock = sc.add_source(clock(args.rebalance_instants()), ());
    let universe =
        common::build_cap_weighted_universe(&mut sc, circ_market_cap, rebalance_clock, args.index_size);

    // Hold the rebalance-day universe fixed between rebalances by re-emitting it
    // on the daily close pulse.
    let daily_universe = sc.add_operator(
        Resample::<Array<f64>, Array<f64>>::new(),
        (st.close, universe),
    );

    // Summed circulating market cap of the current constituents.
    let index_circ_market_cap = sc.add_operator(
        Apply::<(RefPort<Array<f64>>, RefPort<Array<f64>>), Array<f64>, _>::new(
            |(u, c): (&Array<f64>, &Array<f64>)| {
                let (us, cs) = (u.as_slice(), c.as_slice());
                let mut s = 0.0;
                for i in 0..us.len() {
                    if us[i] > 0.0 && cs[i].is_finite() {
                        s += cs[i];
                    }
                }
                Array::scalar(s)
            },
        ),
        (daily_universe, circ_market_cap),
    );

    // Frictionless cap-weighted index NAV via the flowops Benchmark trader.
    let (upper, lower) = common::build_price_limits(&mut sc, st.close, 0.10);
    let index = sc.add_operator(
        PyClassOperator::<RefPorts<Array<f64>>>::from_module(
            "flowops.traders.benchmark",
            PyParams::new()
                .int("num_stocks", n as i64)
                .float("initial_cash", 1.0)
                .bool("use_adjusts", true),
            vec![2],
            clk,
        ),
        &[universe, st.close, st.adjusts, upper, lower][..],
    );
    let index_value = sc.add_operator(
        Map::new(|a: &Array<f64>| Array::scalar(a.as_slice().iter().sum::<f64>())),
        index,
    );

    let h_mc = sc.add_record(index_circ_market_cap);
    let h_nav = sc.add_record(index_value);

    let mut session = sc.build_with_threads(args.threads);
    // Trim warmup output before `begin` so only the live index window is shown.
    let begin = args.begin();
    let total = session.estimated_event_count();
    session.run(common::progress(total, begin)).await;
    eprintln!();

    let (mc_ts, mc_v) = common::read_scalar_series(&session, h_mc);
    let (nav_ts, nav_v) = common::read_scalar_series(&session, h_nav);
    let keep = |ts: &[i64], v: &[f64]| -> (Vec<i64>, Vec<f64>) {
        ts.iter()
            .zip(v.iter())
            .filter(|(t, _)| **t >= begin.as_nanos())
            .map(|(t, x)| (*t, *x))
            .unzip()
    };
    let (mc_ts, mc_v) = keep(&mc_ts, &mc_v);
    let (nav_ts, nav_v) = keep(&nav_ts, &nav_v);

    if mc_ts.is_empty() {
        eprintln!("no data produced");
        std::process::exit(1);
    }
    println!(
        "index circ market cap: {:.2} -> {:.2} CNY (trillion); NAV: {:.4} -> {:.4}  ({} days)",
        mc_v.first().copied().unwrap_or(f64::NAN) / 1e12,
        mc_v.last().copied().unwrap_or(f64::NAN) / 1e12,
        nav_v.first().copied().unwrap_or(f64::NAN),
        nav_v.last().copied().unwrap_or(f64::NAN),
        mc_ts.len(),
    );

    let path = "target/plot_total_market_cap.csv";
    common::write_wide_csv(
        path,
        &[
            ("index_circ_market_cap".into(), mc_ts, mc_v),
            ("index_value".into(), nav_ts, nav_v),
        ],
    );
    println!("wrote {path}\nplot with:  python examples/plot_total_market_cap.py {path}");
}
