//! Cap-weighted A-shares index: total circulating market cap + NAV plot.
//!
//! Tracks a cap-weighted A-shares index: at every rebalance the top
//! `--index-size` stocks by circulating market cap form the universe (weights
//! ∝ circulating cap, renormalised). Two views are written to CSV:
//!
//! 1. the summed circulating market cap of the current constituents (daily), and
//! 2. the index total-return NAV from a frictionless native `Benchmark` trader
//!    (unit cash, dividend reinvestment via adjust factors).
//!
//! Every operator here is native Rust, so this example is a **pure-Rust
//! backtest** — no Python, no `--features python`, no interpreter setup.
//!
//! ```text
//! cargo run --example plot_total_market_cap -- --index-size 1000
//! python examples/plot_total_market_cap.py target/plot_total_market_cap.csv
//! ```

#[path = "common/mod.rs"]
mod common;

use tradingflow::operators::{Apply, ArrayValue, Benchmark, Map, multiply, record};
use tradingflow::{Scenario, WallClock};
use tradingflow::sources::clock;
use flowgraph::typed::ViewPort;
use tradingflow::{Array, ArrayView};

use tradingflow::operators::ResampleView;

use clap::Parser;

/// Cap-weighted A-shares index: total circulating market cap + NAV plot.
#[derive(Parser)]
struct Args {
    #[command(flatten)]
    common: common::CommonArgs,
}

#[tokio::main]
async fn main() {
    let Args { common: args } = Args::parse();
    let symbols = common::load_symbols(&args.data_dir);
    let n = symbols.len();
    eprintln!("loaded {n} symbols; index_size={}", args.index_size);

    let mut sc = Scenario::new(WallClock);
    let clk = sc.time();

    let st = common::build_stacked(&mut sc, &symbols, &args);
    let circ_market_cap = sc.push(multiply::<f64, 1>(), (st.close, st.circ_shares));

    let rebalance_clock = sc.add_source(clock(args.rebalance_instants()));
    let universe =
        common::build_cap_weighted_universe(&mut sc, circ_market_cap, rebalance_clock, args.index_size);

    // Hold the rebalance-day universe fixed between rebalances by re-emitting it
    // on the daily close pulse (clock = the close view, data = the universe).
    let daily_universe = sc.push(ResampleView::<f64, 1>::new(), (st.close, universe));

    // Summed circulating market cap of the current constituents.
    let index_circ_market_cap = sc.push(
        Apply::<(ViewPort<ArrayValue<f64, 1>>, ViewPort<ArrayValue<f64, 1>>), f64, 0, _>::new(
            |(u, c): (ArrayView<f64, 1>, ArrayView<f64, 1>)| {
                let (us, cs) = (u.to_contiguous(), c.to_contiguous());
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

    // Frictionless cap-weighted index NAV via the native Benchmark trader.
    let (upper, lower) = common::build_price_limits(&mut sc, st.close, 0.10);
    let index = sc.push(
        Benchmark::new(n, 1.0, true),
        (universe, st.close, st.adjusts, upper, lower),
    );
    let index_value = sc.push(
        Map::new(|a: ArrayView<f64, 1>| Array::scalar(a.to_contiguous().iter().sum::<f64>())),
        index,
    );

    let h_mc = sc.push(record(&clk), index_circ_market_cap);
    let h_nav = sc.push(record(&clk), index_value);

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
