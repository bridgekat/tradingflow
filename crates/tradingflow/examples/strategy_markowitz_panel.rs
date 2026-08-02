//! Ridge + Markowitz strategy over real market history, read from CSV panels.

use chrono::{DateTime, NaiveDate};
use clap::Parser;
use indicatif::ProgressBar;

use tradingflow::{
    data::{Axis, Duration, Instant, Schema},
    graph::{Builder, Pool},
    operators::{
        array, elem, feature, metric, portfolio, predictor, rolling, series, signal, stats, trader,
    },
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

/// Trading days per year, for annualizing the daily statistics.
const DAYS_PER_YEAR: f64 = 252.0;

/// Ridge + Markowitz strategy over real market history, read from CSV panels.
#[derive(Parser)]
struct Args {
    /// Directory containing the merged long-format CSV tables.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data"))]
    data_dir: String,
    /// Path of the NAV curve CSV to write.
    #[arg(long, default_value = "target/strategy_markowitz_panel.csv")]
    output: String,
    /// First date to backtest (inclusive), e.g. 2018-01-01.
    #[arg(long, value_parser = parse_date)]
    start: Option<Instant>,
    /// Last date to backtest (exclusive).
    #[arg(long, value_parser = parse_date)]
    end: Option<Instant>,
    /// Initial cash.
    #[arg(long, default_value_t = 1_000_000.0)]
    initial_cash: f64,
    /// Markowitz risk-aversion coefficient.
    #[arg(long, default_value_t = 25.0)]
    risk_aversion: f64,
    /// Ridge regression L2 penalty.
    #[arg(long, default_value_t = 0.01)]
    ridge_alpha: f64,
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
            [("symbol".into(), Axis::Labeled(schema.clone()))],
            vec!["share".into(), "cash".into()],
        )
        .with_time_range(args.start, args.end),
    );
    let (_equity_signals, equity) = b.source(
        panel::csv(
            format!("{dir}/equity_structures.csv"),
            "date",
            [("symbol".into(), Axis::Labeled(schema))],
            vec!["total".into()],
        )
        .with_time_range(args.start, args.end),
    );
    let (close, share_divs, cash_divs) = (prices[0], divs[0], divs[1]);
    let total_shares = equity[0];

    // One scalar pulse per trading day (any symbol has a row).
    let daily = b.op(signal::any(), price_signals);

    // Forward-adjust closes for dividends, then take daily log returns —
    // the prediction target, and an input to the volatility feature.
    let (_multipliers, adj_close) = b.op(
        feature::forward_adjust(),
        ((price_signals, close), (div_signals, share_divs, cash_divs)),
    );
    let log_price = b.op(elem::ln(), adj_close);
    let log_return = b.op(rolling::diff(1), (daily, log_price));

    // A few cross-sectional features, standardized across the cross-section:
    // 21-day momentum, 63-day volatility, and log market cap.
    let momentum = b.op(rolling::diff(21), (daily, log_price));
    let volatility = b.op(rolling::std_dev(63, 21), (daily, log_return));
    let market_cap = b.op(elem::mul(), (close, total_shares));
    let size = b.op(elem::ln(), market_cap);
    let f1 = b.op(stats::standardize(), momentum);
    let f2 = b.op(stats::standardize(), volatility);
    let f3 = b.op(stats::standardize(), size);
    let features = b.op(array::stack(1), &[f1, f2, f3][..]); // [N, 3] panel

    // Predict next-day moments: an incremental Ridge regression of the
    // features against next-day log returns for the means, and a shrunk
    // sample covariance for the risk. Both sample daily and refit monthly.
    let universe = b.val(array::constant(vec![1.0; n]));
    let config = predictor::Config {
        target_offset: 1,      // features[t] predict target[t + 1]
        refit_every: 21,       // refit monthly, reuse the fit in between
        max_periods: None,     // use all available history for the fit
        min_periods: Some(63), // a quarter of coverage before predicting
        ..predictor::Config::default()
    };
    let (_, predicted_returns) = b.op(
        predictor::mean::ridge_incr(config, args.ridge_alpha),
        (daily, features, log_return, daily, universe),
    );
    let (_, predicted_cov) = b.op(
        predictor::variance::shrinkage(
            predictor::Config {
                max_periods: Some(252), // one-year covariance window
                ..config
            },
            predictor::variance::Target::SingleIndex,
        ),
        (daily, features, log_return, daily, universe),
    );

    // The Markowitz optimizer trades predicted return against predicted risk:
    // maximize `μᵀx - δ·xᵀΣx` subject to long-only weights summing to at most
    // one. Its default `Config` maps the predictors' log-return moments to the
    // linear ones optimization needs.
    let (rebalance, weights) = b.op(
        portfolio::mean_variance::markowitz(
            portfolio::Config::default(),
            portfolio::mean_variance::Mode::MinMeanVariance,
            args.risk_aversion,
            true,  // long-only
            false, // may hold cash
            n - 1, // covariance factor rank
        ),
        (daily, universe, predicted_returns, predicted_cov),
    );

    // Frictionless trading at the last known close (assuming best bid = best
    // ask = close, so a suspended symbol trades at its carried price), with
    // real dividend events credited to the book.
    let flags = b.val(array::constant(vec![true; n]));
    let (_positions, _cash, nav) = b.op(
        trader::fixed::benchmark(false, args.initial_cash),
        (
            (daily, flags, close, close),
            (div_signals, share_divs, cash_divs),
            (rebalance, weights),
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
        ((1.0 + *g.view(comp_return)).powf(DAYS_PER_YEAR) - 1.0) * 100.0
    );
    println!(
        "annual volatility: {:.2}%",
        *g.view(vol) * DAYS_PER_YEAR.sqrt() * 100.0
    );
    println!(
        "annual Sharpe:     {:.3}",
        *g.view(sharpe) * DAYS_PER_YEAR.sqrt()
    );
    println!("max drawdown:      {:.2}%", mdd * 100.0);
}
