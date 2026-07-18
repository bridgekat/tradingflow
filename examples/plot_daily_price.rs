//! Daily-price plot, reading the consolidated long **parquet** panels via
//! [`ParquetPanelSource`].
//!
//! Loads `daily_prices.parquet` / `dividends.parquet`, `Select`s the target
//! stock's row out of the cross-section and `Filter`s out the "no data" ticks,
//! then computes the forward-adjusted close with a 252-day moving average and
//! Bollinger bands using only **native** Rust operators, writing the recorded
//! series to a tidy CSV for `examples/plot.py` (matplotlib).
//!
//! ```text
//! cargo run --example plot_daily_price [SYMBOL]   # default 000009.SZ
//! python examples/plot.py target/plot_daily_price.csv
//! ```

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;

#[path = "common/mod.rs"]
mod common;

use tradingflow::data::{Array, ArrayView, Instant};
use tradingflow::operators::constant::array_cell;
use tradingflow::operators::num::{add, multiply, sqrt, subtract};
use tradingflow::operators::rolling::{Window, rolling_mean, rolling_variance};
use tradingflow::operators::stocks::forward_adjust;
use tradingflow::operators::structural::{filter, record};
use tradingflow::operators::transform::select_at;
use tradingflow::sources::panel::*;
use tradingflow::clock::WallClock;
use tradingflow::graph::{Builder, Pool};

const WINDOW: usize = 252;
const MULTIPLE: f64 = 2.0;

use clap::Parser;

/// Daily-price plot: forward-adjusted close, moving average, Bollinger bands.
#[derive(Parser)]
struct Args {
    /// Stock symbol to plot, e.g. 000009.SZ.
    symbol: String,
}

#[tokio::main]
async fn main() {
    let symbol = Args::parse().symbol;
    let data_dir = "examples/data";
    let prices_pq = format!("{data_dir}/daily_prices.parquet");
    let dividends_pq = format!("{data_dir}/dividends.parquet");
    for p in [&prices_pq, &dividends_pq] {
        if !std::path::Path::new(p).exists() {
            eprintln!(
                "data not found: {p}\n(run the crawler with --export-long parquet; see examples/README.md)"
            );
            std::process::exit(1);
        }
    }

    let symbols = common::load_symbols(data_dir);
    let idx = symbols
        .iter()
        .position(|s| s == &symbol)
        .unwrap_or_else(|| panic!("{symbol} not in symbol_list.csv"));

    let mut sc = Builder::new(WallClock);

    // Panel sources: close from prices, (share, cash) from dividends.
    let price_src = parquet_panel_source(prices_pq, vec!["prices.close".into()], symbols.clone());
    let price_panel = sc.source(price_src);
    let div_src = parquet_panel_source(
        dividends_pq,
        vec!["dividends.share".into(), "dividends.cash".into()],
        symbols.clone(),
    );
    let div_panel = sc.source(div_src);

    // Select the target stock; close (scalar) and volume (scalar) from its row
    // (rank-1 `[K]` → rank-0 scalar via the squeezing `Select`).
    let prices = sc.segment(select_at(idx, 0), price_panel);
    let prices = sc.segment(
        filter(|a: ArrayView<f64, 1>| a.to_contiguous().iter().any(|x| x.is_finite())),
        prices,
    );
    let dividends = sc.segment(select_at(idx, 0), div_panel);
    let dividends = sc.segment(
        filter(|a: ArrayView<f64, 1>| a.to_contiguous().iter().any(|x| x.is_finite())),
        dividends,
    );
    let closes = sc.segment(select_at(0, 0), prices);

    // Forward-adjusted close (scalar close `0`, dividends row `1`), recorded into
    // a Series for the rolling stats.
    let adj_closes = sc.segment(forward_adjust(), (closes, dividends));
    let adj_series = sc.segment(record(), adj_closes);

    // 252-day MA + rolling std → Bollinger bands (scalar series → rank-0).
    let ma = sc.segment(rolling_mean(Window::Count(WINDOW)), adj_series);
    let var = sc.segment(rolling_variance(Window::Count(WINDOW)), adj_series);
    let std = sc.segment(sqrt(), var);
    let multiple = sc.segment(array_cell(Array::scalar(MULTIPLE)), ());
    let band = sc.segment(multiply(), (std, multiple));
    let upper = sc.segment(add(), (ma, band));
    let lower = sc.segment(subtract(), (ma, band));

    // Record the outputs.
    let h_adj = sc.segment(record(), adj_closes);
    let h_ma = sc.segment(record(), ma);
    let h_upper = sc.segment(record(), upper);
    let h_lower = sc.segment(record(), lower);

    // Run the historical replay to completion.
    let mut session = sc.build();
    let mut pool = Pool::new(0);
    let total = session.total_num_events();
    session
        .run(&mut pool, common::progress(total, Instant::MIN))
        .await;
    eprintln!();

    // Align the recorded scalar series by timestamp and write a wide CSV.
    let cols = [
        ("adj_close", session.view(h_adj)),
        ("ma", session.view(h_ma)),
        ("upper", session.view(h_upper)),
        ("lower", session.view(h_lower)),
    ];
    let mut rows: BTreeMap<i64, [f64; 5]> = BTreeMap::new();
    for (c, (_, series)) in cols.iter().enumerate() {
        for (ts, v) in series.timestamps().iter().zip(series.data().iter()) {
            rows.entry(ts.as_nanos()).or_insert([f64::NAN; 5])[c] = *v;
        }
    }

    let n = session.view(h_adj).len();
    if n == 0 {
        eprintln!("no data for {symbol}");
        std::process::exit(1);
    }

    let mut csv = String::from("timestamp_ns,adj_close,ma,upper,lower,volume\n");
    for (ts, vals) in &rows {
        write!(csv, "{ts}").unwrap();
        for v in vals {
            write!(csv, ",{v}").unwrap();
        }
        csv.push('\n');
    }
    let path = "target/plot_daily_price.csv";
    fs::write(path, csv).expect("write csv");
    println!("{symbol}: {n} trading days -> {path}\nplot with:  python examples/plot.py {path}");
}
