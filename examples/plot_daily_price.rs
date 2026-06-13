//! RefPort of `python/examples/plot_daily_price.py`, reading the consolidated long
//! **parquet** panels via [`ParquetPanelSource`].
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

use flowgraph::typed::{Handle, RefPort};

use tradingflow::operators::{
    Add, Filter, ForwardAdjust, Multiply, RollingMean, RollingVariance, Select, Sqrt, Subtract,
};
use tradingflow::Scenario;
use tradingflow::sources::ParquetPanelSource;
use tradingflow::{Array, Instant, Series};

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

/// `Select` stock `i`'s row out of a panel and drop the all-NaN "no data" ticks.
fn pick(sc: &mut Scenario, panel: Handle<RefPort<Array<f64>>>, i: usize) -> Handle<RefPort<Array<f64>>> {
    let sel = sc.add_operator(Select::<f64>::new(vec![i], 0, true), panel);
    sc.add_operator(Filter(|a: &Array<f64>| a.as_slice().iter().any(|x| x.is_finite())), sel)
}

#[tokio::main]
async fn main() {
    let symbol = std::env::args().nth(1).unwrap_or_else(|| "000009.SZ".to_string());
    let data_dir = "python/examples/data";
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

    let mut sc = Scenario::new();

    // Panel sources: close+volume from prices, (share, cash) from dividends.
    let price_src = ParquetPanelSource::new(
        prices_pq,
        vec!["prices.close".into(), "prices.volume".into()],
        symbols.clone(),
    );
    let price_panel = {
        let init = common::nan_array(&price_src.out_shape());
        sc.add_source(price_src, init)
    };
    let div_src = ParquetPanelSource::new(
        dividends_pq,
        vec!["dividends.share".into(), "dividends.cash".into()],
        symbols.clone(),
    );
    let div_panel = {
        let init = common::nan_array(&div_src.out_shape());
        sc.add_source(div_src, init)
    };

    // Select the target stock; close (scalar) and volume (scalar) from its row.
    let prices = pick(&mut sc, price_panel, idx);
    let dividends = pick(&mut sc, div_panel, idx);
    let closes = sc.add_operator(Select::<f64>::new(vec![0], 0, true), prices);
    let volume = sc.add_operator(Select::<f64>::new(vec![1], 0, true), prices);

    // Forward-adjusted close, recorded into a Series for the rolling stats.
    let adj_closes = sc.add_operator(ForwardAdjust::new(), (closes, dividends));
    let adj_series = sc.add_record(adj_closes);

    // 252-day MA + rolling std → Bollinger bands.
    let ma = sc.add_operator(RollingMean::<f64>::count(WINDOW), adj_series);
    let var = sc.add_operator(RollingVariance::<f64>::count(WINDOW), adj_series);
    let std = sc.add_operator(Sqrt::<f64>::new(), var);
    let multiple = sc.add_const(Array::scalar(MULTIPLE));
    let band = sc.add_operator(Multiply::<f64>::new(), (std, *multiple));
    let upper = sc.add_operator(Add::<f64>::new(), (ma, band));
    let lower = sc.add_operator(Subtract::<f64>::new(), (ma, band));

    // Record the outputs.
    let h_adj = sc.add_record(adj_closes);
    let h_ma = sc.add_record(ma);
    let h_upper = sc.add_record(upper);
    let h_lower = sc.add_record(lower);
    let h_vol = sc.add_record(volume);

    // Run the historical replay to completion.
    let mut session = sc.build();
    let total = session.estimated_event_count();
    session.run(common::progress(total, Instant::MIN)).await;
    eprintln!();

    // Align the recorded scalar series by timestamp and write a wide CSV.
    let cols: [(&str, &Series<f64>); 5] = [
        ("adj_close", session.value(h_adj)),
        ("ma", session.value(h_ma)),
        ("upper", session.value(h_upper)),
        ("lower", session.value(h_lower)),
        ("volume", session.value(h_vol)),
    ];
    let mut rows: BTreeMap<i64, [f64; 5]> = BTreeMap::new();
    for (c, (_, series)) in cols.iter().enumerate() {
        for (ts, v) in series.timestamps().iter().zip(series.values().iter()) {
            rows.entry(ts.as_nanos()).or_insert([f64::NAN; 5])[c] = *v;
        }
    }

    let n = session.value(h_adj).len();
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
