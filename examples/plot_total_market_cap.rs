//! Cap-weighted A-shares index: total circulating market cap + NAV plot.
//!
//! Tracks a cap-weighted A-shares index: at every rebalance the top
//! `--index-size` stocks by circulating market cap form the universe (weights
//! ∝ circulating cap, renormalised). Two views are written to CSV:
//!
//! 1. the summed circulating market cap of the current constituents (daily), and
//! 2. the index total-return NAV from the frictionless native `benchmark` trader
//!    (unit cash; dividends are credited to cash and reinvested at the next
//!    rebalance), traded over a synthetic quote book built from the closes.
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

use tradingflow::data::{Array, ArrayView};
use tradingflow::graph::{Builder, Pool};
use tradingflow::operators::array::array_binary_map;
use tradingflow::operators::elem::mul;
use tradingflow::operators::series::record_all;
use tradingflow::operators::trader::fixed::benchmark;
use tradingflow::sources::sync::signal_iter;
use tradingflow::time::UnixTime;

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

    let mut sc = Builder::new(UnixTime);

    let st = common::build_stacked(&mut sc, &symbols, &args);
    let circ_market_cap = sc.segment(mul(), (st.close, st.circ_shares));

    let rebalance_signal = sc.source(signal_iter(args.rebalance_instants().into_iter()));
    // Recomputed once per rebalance pulse and retained in between, so the
    // wire itself holds the rebalance-day universe fixed until the next pulse.
    let universe = common::build_cap_weighted_universe(
        &mut sc,
        circ_market_cap,
        rebalance_signal,
        args.index_size,
    );
    let daily = st.daily;

    // Summed circulating market cap of the current constituents.
    let masked_circ_sum = |u: ArrayView<f64, 1>, c: ArrayView<f64, 1>| -> f64 {
        let (us, cs) = (u.to_contiguous(), c.to_contiguous());
        let mut s = 0.0;
        for i in 0..us.len() {
            if us[i] > 0.0 && cs[i].is_finite() {
                s += cs[i];
            }
        }
        s
    };
    let index_circ_market_cap = sc.segment(
        array_binary_map(move |u, c| Array::scalar(masked_circ_sum(u, c))),
        (universe, circ_market_cap),
    );

    // Frictionless cap-weighted index NAV via the native benchmark trader,
    // over the synthetic quote book built from the daily closes.
    let (price_signal, flags, bids, asks) = common::build_quotes(
        &mut sc,
        daily,
        st.close,
        common::PRICE_LIMIT,
        common::TICK_SIZE,
        common::DELIST_DAYS,
    );
    let (_positions, _cash, index_value) = sc.segment(
        benchmark(true, 1.0),
        (
            (price_signal, flags, bids, asks),
            (st.div_signals, st.share_divs, st.cash_divs),
            (rebalance_signal, universe),
        ),
    );

    let h_mc = sc.segment(record_all(), (daily, index_circ_market_cap));
    let h_nav = sc.segment(record_all(), (daily, index_value));

    let mut session = sc.build();
    let mut pool = Pool::new(args.threads);
    let total = session.size_hint();
    session.run(&mut pool, common::progress(total)).await;
    eprintln!();

    // Trim warmup output before `begin` so only the live index window is shown.
    let begin_ns = args.begin().as_offset().as_nanos();
    let (mc_ts, mc_v) = common::read_scalar_series(&session, h_mc);
    let (nav_ts, nav_v) = common::read_scalar_series(&session, h_nav);
    let keep = |ts: &[i64], v: &[f64]| -> (Vec<i64>, Vec<f64>) {
        ts.iter()
            .zip(v.iter())
            .filter(|(t, _)| **t >= begin_ns)
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
