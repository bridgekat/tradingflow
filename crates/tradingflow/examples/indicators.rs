//! Moving averages and Bollinger bands over real market history, read from
//! CSV panels and logged to a long-format CSV.

use chrono::{DateTime, NaiveDate};
use clap::Parser;
use indicatif::ProgressBar;
use tradingflow::{
    data::{Axis, Duration, Instant, Schema},
    graph::{Builder, Pool},
    operators::{array, feature, rolling, series, signal},
    ports::SeriesPortHandle,
    sources::panel,
    time::UnixTime,
};

/// The symbols shipped in the example data.
const SYMBOLS: [&str; 6] = [
    "000001.SZ",
    "000002.SZ",
    "000858.SZ",
    "600000.SH",
    "600519.SH",
    "601398.SH",
];

/// Moving averages and Bollinger bands over real market history.
#[derive(Parser)]
struct Args {
    /// Directory containing the data CSV files.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data"))]
    data_dir: String,
    /// Path of the indicator CSV to write.
    #[arg(long, default_value = "indicators.csv")]
    output: String,
    /// First date to compute (inclusive), e.g. 2017-01-01.
    #[arg(long, value_parser = parse_date)]
    start: Option<Instant>,
    /// Last date to compute (exclusive).
    #[arg(long, value_parser = parse_date)]
    end: Option<Instant>,
    /// Window of the fast moving average, in trading days.
    #[arg(long, default_value_t = 10)]
    ma_fast: usize,
    /// Window of the slow moving average, in trading days.
    #[arg(long, default_value_t = 60)]
    ma_slow: usize,
    /// Window of the Bollinger basis and its standard deviation. The defaults
    /// keep all three windows distinct, so no two logged curves coincide.
    #[arg(long, default_value_t = 20)]
    bb_window: usize,
    /// Half-width of the Bollinger bands, in standard deviations.
    #[arg(long, default_value_t = 2.0)]
    bb_width: f64,
}

/// Parses a `YYYY-MM-DD` date into its midnight [`Instant`].
fn parse_date(s: &str) -> Result<Instant, String> {
    let date = NaiveDate::parse_from_str(s, "%Y-%m-%d").map_err(|e| e.to_string())?;
    let ns = date
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp_nanos_opt()
        .unwrap();
    Ok(Instant::from_offset(Duration::from_nanos(ns)))
}

/// Formats an [`Instant`] back into a `YYYY-MM-DD` date.
fn format_date(t: Instant) -> String {
    let dt = DateTime::from_timestamp_nanos(t.as_offset().as_nanos());
    dt.date_naive().format("%Y-%m-%d").to_string()
}

/// Formats one cell: a non-finite value (a window that has not filled yet)
/// is written as an empty cell, which every CSV reader takes as missing.
fn cell(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.6}")
    } else {
        String::new()
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let dir = &args.data_dir;

    // Create the thread pool.
    let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());

    // Build the graph. Each table is a long-format CSV panel source over the
    // shared symbol axis: one signal array marking the rows present at each
    // date, plus one carried cross-section per value column.
    let mut b = Builder::new(UnixTime);
    let schema = Schema::new(SYMBOLS);

    let (price_signals, prices) = b.source(
        panel::csv(
            format!("{dir}/prices.csv"),
            "date",
            [("symbol".into(), Axis::Labeled(schema.clone()))],
            vec!["close".into()],
        )
        .with_time_range(args.start, args.end),
    );
    let (div_signals, divs) = b.source(
        panel::csv(
            format!("{dir}/dividends.csv"),
            "date",
            [("symbol".into(), Axis::Labeled(schema.clone()))],
            vec!["share".into(), "cash".into()],
        )
        .with_time_range(args.start, args.end),
    );
    let (close, share_divs, cash_divs) = (prices[0], divs[0], divs[1]);

    // One scalar pulse per trading day (any symbol has a row). Every rolling
    // operator below ingests exactly one sample per pulse, so the windows are
    // counted in trading days rather than calendar days.
    let daily = b.op(signal::any(), price_signals);

    // Forward-adjust closes for dividends, over the whole cross-section: the
    // close leg gates per element on the price panel's signal, the dividend
    // leg on the dividend panel's.
    let (_multipliers, adj_close) = b.op(
        feature::forward_adjust(),
        ((price_signals, close), (div_signals, share_divs, cash_divs)),
    );

    // The two moving averages. `min_count` is set to the full window, so each
    // line stays `NaN` until its window has filled rather than starting from a
    // one-sample average.
    let ma_fast = b.op(
        rolling::mean(args.ma_fast, args.ma_fast),
        (daily, adj_close),
    );
    let ma_slow = b.op(
        rolling::mean(args.ma_slow, args.ma_slow),
        (daily, adj_close),
    );

    // The Bollinger bands: a moving average basis, plus and minus a multiple
    // of the rolling sample standard deviation over the same window. The basis
    // is its own node so that `--bb-window` can differ from the moving average
    // windows; setting it equal to one of them just computes the same mean
    // twice, which is why the defaults keep all three distinct.
    let basis = b.op(
        rolling::mean(args.bb_window, args.bb_window),
        (daily, adj_close),
    );
    let sd = b.op(
        rolling::std_dev(args.bb_window, args.bb_window),
        (daily, adj_close),
    );
    let width = args.bb_width;
    let upper = b.op(
        array::binary_map(move |&m: &f64, &s: &f64| m + width * s),
        (basis, sd),
    );
    let lower = b.op(
        array::binary_map(move |&m: &f64, &s: &f64| m - width * s),
        (basis, sd),
    );

    // Record every wire we want to log. All of them record on the same daily
    // pulse, so the series share one date axis.
    let columns: [(&str, SeriesPortHandle<f64, 1>); 7] = [
        ("close", b.op(series::record_all(), (daily, close))),
        ("adj_close", b.op(series::record_all(), (daily, adj_close))),
        ("ma_fast", b.op(series::record_all(), (daily, ma_fast))),
        ("ma_slow", b.op(series::record_all(), (daily, ma_slow))),
        ("bb_basis", b.op(series::record_all(), (daily, basis))),
        ("bb_upper", b.op(series::record_all(), (daily, upper))),
        ("bb_lower", b.op(series::record_all(), (daily, lower))),
    ];

    // Run the event loop until all sources are exhausted, with a progress bar
    // over the estimated total row count.
    let mut g = b.build();
    let bar = ProgressBar::new(g.size_hint().unwrap_or(0) as u64);
    g.run(&mut pool, |g, _| bar.set_position(g.num_events() as u64))
        .await;
    bar.finish();

    // Write the long-format CSV: one row per (date, symbol), one column per
    // recorded wire. A symbol that did not trade on a given date carries its
    // previous cross-section, so its row repeats the last known values.
    let series: Vec<_> = columns.iter().map(|&(_, h)| g.view(h)).collect();
    let instants = series[0].instants();

    let mut csv = String::from("date,symbol");
    for (name, _) in &columns {
        csv += ",";
        csv += name;
    }
    csv.push('\n');

    for (i, &t) in instants.iter().enumerate() {
        let date = format_date(t);
        let sections: Vec<_> = series.iter().map(|s| s.at(i).1).collect();
        for (j, symbol) in schema.labels().iter().enumerate() {
            csv += &date;
            csv += ",";
            csv += symbol;
            for section in &sections {
                csv += ",";
                csv += &cell(section[[j]]);
            }
            csv.push('\n');
        }
    }

    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&args.output, csv).unwrap();
    println!(
        "wrote {} rows ({} dates x {} symbols) to {}",
        instants.len() * SYMBOLS.len(),
        instants.len(),
        SYMBOLS.len(),
        args.output,
    );
}
