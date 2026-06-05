//! Port of `python/examples/plot_daily_price.py` to the Rust flow engine.
//!
//! Loads daily prices + dividends for an A-shares stock from CSV, computes the
//! forward-adjusted close with a 252-day moving average and Bollinger bands
//! using only **native** Rust operators (no Python), and writes the recorded
//! series to a tidy CSV for `examples/plot.py` (matplotlib).
//!
//! Runs on default features (no `pyflow` / Python needed):
//!
//! ```text
//! cargo run --example plot_daily_price          # default symbol 000009.SZ
//! python examples/plot.py target/plot_daily_price.csv
//! ```

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;

use tradingflow::data::Duration;

use tradingflow::flow::{
    Add, ForwardAdjust, Multiply, RollingMean, RollingVariance, Scenario, Select, Sqrt, Subtract,
};
use tradingflow::sources::CsvSource;
use tradingflow::{Array, Series};

const WINDOW: usize = 252;
const MULTIPLE: f64 = 2.0;

#[tokio::main]
async fn main() {
    let symbol = std::env::args().nth(1).unwrap_or_else(|| "000009.SZ".to_string());
    let dir = "python/examples/data/a_shares_history";
    let prices_csv = format!("{dir}/{symbol}.daily_prices.csv");
    let dividends_csv = format!("{dir}/{symbol}.dividends.csv");
    if !std::path::Path::new(&prices_csv).exists() {
        eprintln!("data not found: {prices_csv}");
        std::process::exit(1);
    }

    let mut sc = Scenario::new();

    // Sources: close+volume from prices, (share, cash) from dividends.
    let prices = sc.add_source(
        CsvSource::new(
            prices_csv,
            "date".into(),
            vec!["prices.close".into(), "prices.volume".into()],
            Duration::ZERO,
        ),
        Array::zeros(&[2]),
    );
    let dividends = sc.add_source(
        CsvSource::new(
            dividends_csv,
            "date".into(),
            vec!["dividends.share".into(), "dividends.cash".into()],
            Duration::ZERO,
        ),
        Array::zeros(&[2]),
    );

    // close (scalar) and volume (scalar) extracted from the price row.
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
    let band = sc.add_operator(Multiply::<f64>::new(), (std, multiple));
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
    session.run(|_, _| {}).await;

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
