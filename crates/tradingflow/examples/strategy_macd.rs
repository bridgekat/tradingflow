//! MACD crossover strategy over real market history, read from CSV panels.

use chrono::{DateTime, NaiveDate};
use clap::Parser;
use indicatif::ProgressBar;
use tradingflow::{
    data::{Axis, Duration, Instant, Schema},
    graph::{Builder, Pool},
    operators::{array, elem, feature, metric, rolling, series, signal, trader},
    sources::panel,
    time::UnixTime,
};

/// Trading days per year, for annualizing the daily statistics.
const PERIODS_PER_YEAR: f64 = 252.0;

/// The symbols shipped in the example data.
const SYMBOLS: [&str; 6] = [
    "000001.SZ",
    "000002.SZ",
    "000858.SZ",
    "600000.SH",
    "600519.SH",
    "601398.SH",
];

/// MACD crossover strategy over real market history, read from CSV panels.
#[derive(Parser)]
struct Args {
    /// Directory containing the data CSV files.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data"))]
    data_dir: String,
    /// Path of the NAV curve CSV to write.
    #[arg(long, default_value = "strategy_macd.csv")]
    output: String,
    /// First date to trade (inclusive), e.g. 2017-01-01.
    #[arg(long, value_parser = parse_date)]
    start: Option<Instant>,
    /// Last date to trade (exclusive).
    #[arg(long, value_parser = parse_date)]
    end: Option<Instant>,
    /// Initial cash.
    #[arg(long, default_value_t = 1_000_000.0)]
    initial_cash: f64,
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

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let dir = &args.data_dir;
    let n = SYMBOLS.len();

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
            [("symbol".into(), Axis::Labeled(schema))],
            vec!["share".into(), "cash".into()],
        )
        .with_time_range(args.start, args.end),
    );
    let (close, share_divs, cash_divs) = (prices[0], divs[0], divs[1]);

    // One scalar pulse per trading day (any symbol has a row).
    let daily = b.op(signal::any(), price_signals);

    // Forward-adjust closes for dividends, over the whole cross-section: the
    // close leg gates per element on the price panel's signal, the dividend
    // leg on the dividend panel's.
    let (_multipliers, adj_close) = b.op(
        feature::forward_adjust(),
        ((price_signals, close), (div_signals, share_divs, cash_divs)),
    );

    // The MACD indicator on adjusted closes, composed from built-in operators.
    let ema_fast = b.op(rolling::mean_exp(12, 1), (daily, adj_close)); // EMA(12)
    let ema_slow = b.op(rolling::mean_exp(26, 1), (daily, adj_close)); // EMA(26)
    let macd = b.op(elem::sub(), (ema_fast, ema_slow)); // EMA(12) - EMA(26)
    let smooth = b.op(rolling::mean_exp(9, 1), (daily, macd)); // EMA(9) of MACD
    let held = b.op(elem::gt(), (macd, smooth)); // MACD > smooth
    let weights = b.op(elem::indicator(1.0 / n as f64, 0.0), held); // 1/N position if held

    // Simulate frictionless trading using `weight`.
    // Here we assume: best bid = best ask = prices, with dividends credited.
    // Execution is delayed by one period after the rebalance signal.
    let flags = b.val(array::constant(vec![true; n]));
    let (bids, asks) = (close, close);
    let (_positions, _cash, nav) = b.op(
        trader::fixed::benchmark(true, args.initial_cash),
        (
            (daily, flags, bids, asks),
            (div_signals, share_divs, cash_divs),
            (daily, weights),
        ),
    );

    // Record the NAV curve and its daily performance statistics.
    let nav_series = b.op(series::record_all(), (daily, nav));
    let comp_return = b.op(metric::performance::comp_return(), (daily, nav));
    let sharpe = b.op(metric::performance::return_sharpe(), (daily, nav));
    let vol = b.op(metric::performance::return_vol(), (daily, nav));

    // Run the event loop until all sources are exhausted, with a progress bar
    // over the estimated total row count.
    let mut g = b.build();
    let bar = ProgressBar::new(g.size_hint().unwrap_or(0) as u64);
    g.run(&mut pool, |g, _| bar.set_position(g.num_events() as u64))
        .await;
    bar.finish();

    // Log the NAV curve.
    let series = g.view(nav_series);
    let (instants, values) = (series.instants(), series.to_contiguous());
    let mut csv = String::from("date,nav\n");
    for (&t, &nav) in instants.iter().zip(values.iter()) {
        csv += &format!("{},{nav}\n", format_date(t));
    }
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&args.output, csv).unwrap();
    println!("wrote {} NAV points to {}", instants.len(), args.output);

    // Print summary statistics (daily metrics annualized).
    let final_nav = *values.last().unwrap();
    let years = (*instants.last().unwrap() - instants[0]).as_days() as f64 / 365.25;
    let mut peak = f64::MIN;
    let mdd = values.iter().fold(0.0f64, |mdd, &nav| {
        peak = peak.max(nav);
        mdd.min(nav / peak - 1.0)
    });
    println!("final NAV:         {final_nav:.0} ({years:.1} years)");
    println!(
        "total return:      {:+.2}%",
        (final_nav / args.initial_cash - 1.0) * 100.0
    );
    println!(
        "annual return:     {:+.2}%",
        ((1.0 + *g.view(comp_return)).powf(PERIODS_PER_YEAR) - 1.0) * 100.0
    );
    println!(
        "annual volatility: {:.2}%",
        *g.view(vol) * PERIODS_PER_YEAR.sqrt() * 100.0
    );
    println!(
        "annual Sharpe:     {:.3}",
        *g.view(sharpe) * PERIODS_PER_YEAR.sqrt()
    );
    println!("max drawdown:      {:.2}%", mdd * 100.0);
}
