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

#[path = "common/mod.rs"]
mod common;

use tradingflow::clock::UnixClock;
use tradingflow::data::ArrayView;
use tradingflow::graph::{Builder, Pool};
use tradingflow::operators::array;
use tradingflow::operators::array::select_at;
use tradingflow::operators::elem::{add, mul, sqrt, sub};
use tradingflow::operators::event::filter;
use tradingflow::operators::feature::stock::forward_adjust;
use tradingflow::operators::rolling;
use tradingflow::operators::series::record_all;
use tradingflow::sources::panel::*;

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

    let mut sc = Builder::new(UnixClock);

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

    // Forward-adjusted close (scalar close plus the two scalar dividend legs
    // split out of the `[share, cash]` row), recorded into a Series for the
    // rolling stats.
    let share_divs = sc.segment(select_at(0, 0), dividends);
    let cash_divs = sc.segment(select_at(1, 0), dividends);
    let (_multipliers, adj_closes) = sc.segment(forward_adjust(), (closes, share_divs, cash_divs));
    let adj_series = sc.segment(record_all(), adj_closes);

    // 252-day MA + rolling std → Bollinger bands (scalar series → rank-0).
    let ma = sc.segment(rolling::series_mean(WINDOW, 1), adj_series);
    let var = sc.segment(rolling::series_var(WINDOW, 1), adj_series);
    let std = sc.segment(sqrt(), var);
    let multiple = sc.segment(array::scalar(MULTIPLE), ());
    let band = sc.segment(mul(), (std, multiple));
    let upper = sc.segment(add(), (ma, band));
    let lower = sc.segment(sub(), (ma, band));

    // Record the outputs.
    let h_adj = sc.segment(record_all(), adj_closes);
    let h_ma = sc.segment(record_all(), ma);
    let h_upper = sc.segment(record_all(), upper);
    let h_lower = sc.segment(record_all(), lower);

    // Run the historical replay to completion.
    let mut session = sc.build();
    let mut pool = Pool::new(0);
    let total = session.size_hint();
    session.run(&mut pool, common::progress(total)).await;
    eprintln!();

    // Read the recorded series and align them by timestamp into a wide CSV.
    let series = [
        ("adj_close", h_adj),
        ("ma", h_ma),
        ("upper", h_upper),
        ("lower", h_lower),
    ]
    .map(|(name, h)| {
        let (ts, v) = common::read_scalar_series(&session, h);
        (name.to_string(), ts, v)
    });

    let n = series[0].1.len();
    if n == 0 {
        eprintln!("no data for {symbol}");
        std::process::exit(1);
    }

    let path = "target/plot_daily_price.csv";
    common::write_wide_csv(path, &series);
    println!("{symbol}: {n} trading days -> {path}\nplot with:  python examples/plot.py {path}");
}
