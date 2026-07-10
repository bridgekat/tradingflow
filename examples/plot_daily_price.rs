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

use flowgraph::typed::{Handle, ViewPort};

use tradingflow::operators::{
    ArrayValue, Filter, ForwardAdjust, RollingMean, RollingVariance, Select, add, multiply,
    record, sqrt, subtract,
};
use tradingflow::{Scenario, ScenarioExt, WallClock};
use tradingflow::sources::ParquetPanelSource;
use tradingflow::{Array, ArrayView, Instant, Series};

const WINDOW: usize = 252;
const MULTIPLE: f64 = 2.0;

/// All symbols from `<data_dir>/symbol_list.csv` (the `symbol` column), in order.
fn load_symbols(data_dir: &str) -> Vec<String> {
    let path = format!("{data_dir}/symbol_list.csv");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut lines = text.lines();
    let header = lines.next().expect("symbol_list header");
    let col = header.split(',').position(|h| h.trim() == "symbol").expect("`symbol` column");
    lines
        .filter_map(|l| l.split(',').nth(col).map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty())
        .collect()
}

/// `Select` stock `i`'s row out of a `[N, K]` panel (→ rank-1 `[K]`) and drop the
/// all-NaN "no data" ticks.
fn pick(
    sc: &mut Scenario,
    panel: Handle<ViewPort<ArrayValue<f64, 2>>>,
    i: usize,
) -> Handle<ViewPort<ArrayValue<f64, 1>>> {
    let sel = sc.push(Select::<f64, 2, 1>::new(vec![i], 0, true), panel);
    sc.push(
        Filter::<_, 1>(|a: ArrayView<f64, 1>| a.to_contiguous().iter().any(|x| x.is_finite())),
        sel,
    )
}

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
            eprintln!("data not found: {p}\n(run the crawler with --export-long parquet; see examples/README.md)");
            std::process::exit(1);
        }
    }

    let symbols = load_symbols(data_dir);
    let idx = symbols
        .iter()
        .position(|s| s == &symbol)
        .unwrap_or_else(|| panic!("{symbol} not in symbol_list.csv"));

    let mut sc = Scenario::new(WallClock);
    let clk = sc.time();

    // Panel sources: close+volume from prices, (share, cash) from dividends.
    let price_src = ParquetPanelSource::new(
        prices_pq,
        vec!["prices.close".into(), "prices.volume".into()],
        symbols.clone(),
    );
    let price_panel = {
        let h = sc.add_source(price_src);
        sc.as_view(h)
    };
    let div_src = ParquetPanelSource::new(
        dividends_pq,
        vec!["dividends.share".into(), "dividends.cash".into()],
        symbols.clone(),
    );
    let div_panel = {
        let h = sc.add_source(div_src);
        sc.as_view(h)
    };

    // Select the target stock; close (scalar) and volume (scalar) from its row
    // (rank-1 `[K]` → rank-0 scalar via the squeezing `Select`).
    let prices = pick(&mut sc, price_panel, idx);
    let dividends = pick(&mut sc, div_panel, idx);
    let closes = sc.push(Select::<f64, 1, 0>::new(vec![0], 0, true), prices);
    let volume = sc.push(Select::<f64, 1, 0>::new(vec![1], 0, true), prices);

    // Forward-adjusted close (scalar close `0`, dividends row `1`), recorded into
    // a Series for the rolling stats.
    let adj_closes = sc.push(ForwardAdjust::<0, 1>::new(), (closes, dividends));
    let adj_series = sc.push(record(&clk), adj_closes);

    // 252-day MA + rolling std → Bollinger bands (scalar series → rank-0).
    let ma = sc.push(RollingMean::<f64, 0>::count(WINDOW), adj_series);
    let var = sc.push(RollingVariance::<f64, 0>::count(WINDOW), adj_series);
    let std = sc.push(sqrt::<f64, 0>(), var);
    let multiple_src = sc.add_const(Array::scalar(MULTIPLE));
    let multiple = sc.as_view(*multiple_src);
    let band = sc.push(multiply::<f64, 0>(), (std, multiple));
    let upper = sc.push(add::<f64, 0>(), (ma, band));
    let lower = sc.push(subtract::<f64, 0>(), (ma, band));

    // Record the outputs.
    let h_adj = sc.push(record(&clk), adj_closes);
    let h_ma = sc.push(record(&clk), ma);
    let h_upper = sc.push(record(&clk), upper);
    let h_lower = sc.push(record(&clk), lower);
    let h_vol = sc.push(record(&clk), volume);

    // Run the historical replay to completion.
    let mut session = sc.build();
    let total = session.estimated_event_count();
    session.run(common::progress(total, Instant::MIN)).await;
    eprintln!();

    // Align the recorded scalar series by timestamp and write a wide CSV.
    let cols: [(&str, &Series<f64, 0>); 5] = [
        ("adj_close", session.ref_view(h_adj)),
        ("ma", session.ref_view(h_ma)),
        ("upper", session.ref_view(h_upper)),
        ("lower", session.ref_view(h_lower)),
        ("volume", session.ref_view(h_vol)),
    ];
    let mut rows: BTreeMap<i64, [f64; 5]> = BTreeMap::new();
    for (c, (_, series)) in cols.iter().enumerate() {
        for (ts, v) in series.timestamps().iter().zip(series.values().iter()) {
            rows.entry(ts.as_nanos()).or_insert([f64::NAN; 5])[c] = *v;
        }
    }

    let n = session.ref_view(h_adj).len();
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
    println!(
        "{symbol}: {n} trading days -> {path}\nplot with:  python examples/plot.py {path}"
    );
}
