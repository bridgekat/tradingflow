//! Daily-price plot, reading the consolidated long **parquet** panels via
//! [`panel::Parquet`].
//!
//! Loads `daily_prices.parquet` / `dividends.parquet`, selects the target
//! stock's element out of each cross-section and its own tick signal,
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

use tradingflow::data::utils::{Axis, Schema};
use tradingflow::graph::{Builder, Pool};
use tradingflow::operators::array::{map, select_at};
use tradingflow::operators::elem::{add, sqrt, sub};
use tradingflow::operators::feature::forward_adjust;
use tradingflow::operators::rolling;
use tradingflow::operators::series::record_all;
use tradingflow::operators::signal::as_signal_map;
use tradingflow::sources::panel;
use tradingflow::time::UnixTime;

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

    let mut sc = Builder::new(UnixTime);

    // Panel sources: close from prices, (share, cash) from dividends. Each is
    // a `([N] signal, M × [N])` stream over the whole-market symbol axis (the
    // axis schema must cover every label its table carries).
    let universe = Schema::new(symbols.clone());
    let (price_signals, prices) = sc.source(panel::parquet(
        prices_pq,
        [("symbol".into(), Axis::Labeled(universe.clone()))],
        vec!["prices.close".into()],
    ));
    let (div_signals, divs) = sc.source(panel::parquet(
        dividends_pq,
        [("symbol".into(), Axis::Labeled(universe))],
        vec!["dividends.share".into(), "dividends.cash".into()],
    ));

    // Squeeze the target stock's element out of each `[N]` field, and its own
    // tick signal out of each `[N]` signal.
    let closes = sc.segment(select_at::<f64, 1, 0>(idx, 0), prices[0]);
    let ticks = sc.segment(
        as_signal_map(select_at::<bool, 1, 0>(idx, 0)),
        price_signals,
    );
    let div_ticks = sc.segment(as_signal_map(select_at::<bool, 1, 0>(idx, 0)), div_signals);

    // Forward-adjusted close (scalar close plus the two scalar dividend legs),
    // recorded into a Series for the rolling stats.
    let share_divs = sc.segment(select_at::<f64, 1, 0>(idx, 0), divs[0]);
    let cash_divs = sc.segment(select_at::<f64, 1, 0>(idx, 0), divs[1]);
    let (_multipliers, adj_closes) = sc.segment(
        forward_adjust(),
        ((ticks, closes), (div_ticks, share_divs, cash_divs)),
    );
    let adj_series = sc.segment(record_all(), (ticks, adj_closes));

    // 252-day MA + rolling std → Bollinger bands (scalar series → rank-0).
    let ma = sc.segment(rolling::series_mean(WINDOW, 1), adj_series);
    let var = sc.segment(rolling::series_var(WINDOW, 1), adj_series);
    let std = sc.segment(sqrt(), var);
    let band = sc.segment(map(move |&x: &f64| x * MULTIPLE), std);
    let upper = sc.segment(add(), (ma, band));
    let lower = sc.segment(sub(), (ma, band));

    // Record the outputs on the stock's own tick signal.
    let h_adj = sc.segment(record_all(), (ticks, adj_closes));
    let h_ma = sc.segment(record_all(), (ticks, ma));
    let h_upper = sc.segment(record_all(), (ticks, upper));
    let h_lower = sc.segment(record_all(), (ticks, lower));

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
