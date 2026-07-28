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
//! The IC metric is a Python (`tradingflow`) operator (no cvxpy), so this needs
//! `--features python` and a venv with NumPy (a standard GIL venv is fine).
//!
//! ```text
//! cargo run --example factor_ic --features python -- --index-size 1000
//! python examples/plot_factor_ic.py target/factor_ic.csv
//! ```

#[path = "common/mod.rs"]
mod common;

use clap::Parser;

use tradingflow::data::Retention;
use tradingflow::graph::Builder;
use tradingflow::operators::rolling::series_lag;
use tradingflow::operators::series::record_all;
use tradingflow::time::UnixTime;

use common::ic::{ic_series, ic_stats};
use common::strategy::Market;
use common::universe::mask_to_universe;

/// Factor IC evaluation on the A-shares cross-sectional panel.
#[derive(Parser)]
struct Args {
    #[command(flatten)]
    common: common::CommonArgs,
}

#[tokio::main]
async fn main() {
    let Args { common: args } = Args::parse();
    let symbols = common::load_symbols(&args.data_dir);
    eprintln!(
        "loaded {} symbols; index_size={}",
        symbols.len(),
        args.index_size
    );

    let mut sc = Builder::new(UnixTime);

    let m = Market::build(&mut sc, &symbols, &args, Retention::unbounded());
    let names = m.features.names.clone();
    let ic_handles: Vec<_> = m
        .features
        .handles
        .iter()
        .map(|&feature| {
            // Lag one trading day, then NaN out the stocks outside the
            // universe so they don't dilute the correlation; the IC metric
            // reads the masked factor on the rebalance clock.
            let feature_series = sc.segment(record_all(), (m.daily, feature));
            let lagged = sc.segment(series_lag(1), feature_series);
            let masked = mask_to_universe(&mut sc, lagged, m.universe);
            ic_series(
                &mut sc,
                (m.rebalance_clock, masked),
                (m.daily, m.target),
                m.n,
            )
        })
        .collect();

    let session = common::run(sc, &args).await;

    // Per-feature IC stats + long-format CSV.
    let mut cols: Vec<(String, Vec<i64>, Vec<f64>)> = Vec::new();
    for (name, h) in names.iter().zip(ic_handles.iter()) {
        let (ts, v) = common::read_scalar_series(&session, *h);
        match ic_stats(&v) {
            None => println!("{name:>18}: no valid values"),
            Some(s) => println!(
                "{name:>18}: mean={:+.4} std={:.4} IR={:+.4} ({} periods)",
                s.mean, s.std, s.ir, s.n
            ),
        }
        cols.push((name.clone(), ts, v));
    }

    let path = "target/factor_ic.csv";
    common::write_long_csv(path, &cols);
    println!("wrote {path}\nplot with:  python examples/plot_factor_ic.py {path}");
}
