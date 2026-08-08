//! A miniature factor model strategy, using Python operators.
//!
//! The strategy reads the same CSV panels as the MACD example, and does the
//! following:
//!
//! 1. Calculate a few simple features from the price and volume series.
//! 2. Feed the features and past returns into a mean predictor
//!    (the "alpha model") to obtain a cross-sectional vector of expected
//!    future returns.
//! 3. Feed the features and past returns into a covariance predictor
//!    (the "risk model") to obtain a cross-section of factor exposures,
//!    factor covariance matrix and specific variances. Together they define
//!    a full covariance matrix of future returns.
//! 4. Feed both the mean vector and covariance matrix components into a
//!    mean-variance portfolio optimizer to obtain a vector of weights.
//! 5. Simulate frictionless trading using the weights, output NAV curves
//!    and stats.
//!
//! Steps 4 and 5 repeat once per `--risk-aversion` value. The whole sweep
//! runs in a single pass over the data: the variants share the feature, alpha
//! and risk model nodes, and fork only at the optimizer. The optimizers are
//! run in parallel: they are scheduled on independent threads, and CVXPY
//! releases the Python GIL during backend solves (although it makes little
//! difference on the small example dataset).
//!
//! This example requires an embedded Python interpreter with NumPy, SciPy
//! and CVXPY; see the repository README for details. `OPENBLAS_NUM_THREADS=1`
//! is required if NumPy is linked against OpenBLAS.

use chrono::{DateTime, NaiveDate};
use clap::Parser;
use indicatif::ProgressBar;
use pyo3::prelude::*;
use std::ffi::CString;
use tradingflow::{
    data::{Axis, Duration, Instant, Retention, Schema},
    graph::{Builder, OperatorExt, Pool},
    operators::{array, elem, feature, metric, rolling, series, signal, stats, trader},
    ports::{ArrayPort, SignalPort},
    python::{py_operator_module, py_params},
    sources::{panel, sync},
    time::UnixTime,
};

/// Rolling windows, as calendar retention bounds.
const MONTH: Retention = Retention::duration(Duration::from_days(30));
const QUARTER: Retention = Retention::duration(Duration::from_days(90));
const YEAR: Retention = Retention::duration(Duration::from_days(365));

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

/// A miniature factor model strategy, using Python operators.
#[derive(Parser)]
struct Args {
    /// Directory containing the data CSV files.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data"))]
    data_dir: String,
    /// Directory containing the Python operator modules.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/python"))]
    python_ops_dir: String,
    /// Path of the NAV curve CSV to write: one column per sweep variant,
    /// plus the cap-weighted index.
    #[arg(long, default_value = "strategy_factor.csv")]
    output: String,
    /// First date to trade (inclusive), e.g. 2017-01-01. This strategy needs
    /// a warm-up period, so it reads all data even before the date.
    #[arg(long, value_parser = parse_date, default_value = "2017-01-01")]
    start: Instant,
    /// Last date to trade (exclusive).
    #[arg(long, value_parser = parse_date)]
    end: Option<Instant>,
    /// Rebalance every this many calendar days. A rebalance falling between
    /// trading days executes at the next trading day's quotes.
    #[arg(long, default_value_t = 30)]
    rebalance_every: usize,
    /// Sampling periods a stock needs before either model emits a prediction.
    #[arg(long, default_value_t = 120)]
    min_periods: usize,
    /// Ridge regression regularization term for the alpha model.
    #[arg(long, default_value_t = 0.01)]
    ridge_l2: f64,
    /// Halflife (in number of trading days) of the exponential decay for
    /// the factor covariances in the risk model.
    #[arg(long, default_value_t = 250.0)]
    covariance_halflife: f64,
    /// Halflife (in number of trading days) of the exponential decay for
    /// the specific variances in the risk model.
    #[arg(long, default_value_t = 150.0)]
    specific_halflife: f64,
    /// Measure risk relative to the cap-weighted index, rather than in
    /// absolute terms.
    #[arg(long)]
    benchmark_relative: bool,
    /// Risk aversion parameter sweep: the optimizer maximizes
    /// `expected returns - risk aversion * variance of returns`.
    #[arg(long, value_delimiter = ',', default_value = "0.5,1,2,5,10,25,50,100")]
    risk_aversion: Vec<f64>,
    /// Initial cash.
    #[arg(long, default_value_t = 1_000_000.0)]
    initial_cash: f64,
}

impl Args {
    /// The rebalance calendar: every `--rebalance-every` calendar days from
    /// `--start` up to `--end` (or today when open-ended).
    pub fn rebalance_instants(&self) -> Vec<Instant> {
        let now = Instant::from_offset(Duration::from_nanos(
            chrono::Utc::now().timestamp_nanos_opt().unwrap(),
        ));
        let end = self.end.unwrap_or(now).min(now);
        let step = Duration::from_days(self.rebalance_every as i64);
        let mut instants = Vec::new();
        let mut t = self.start;
        while t < end {
            instants.push(t);
            t = t.saturating_add(step);
        }
        instants
    }
}

/// The performance summary of one strategy variant.
struct Summary {
    final_nav: f64,
    total_return: f64,
    annual_return: f64,
    annual_vol: f64,
    annual_sharpe: f64,
    max_drawdown: f64,
}

impl Summary {
    /// Summarizes one strategy variant.
    pub fn new(nav: &[f64], initial_cash: f64, comp_return: f64, sharpe: f64, vol: f64) -> Self {
        let mut peak = f64::MIN;
        let max_drawdown = nav.iter().fold(0.0f64, |mdd, &v| {
            peak = peak.max(v);
            mdd.min(v / peak - 1.0)
        });
        let final_nav = *nav.last().unwrap();
        Self {
            final_nav,
            total_return: final_nav / initial_cash - 1.0,
            annual_return: (1.0 + comp_return).powf(PERIODS_PER_YEAR) - 1.0,
            annual_vol: vol * PERIODS_PER_YEAR.sqrt(),
            annual_sharpe: sharpe * PERIODS_PER_YEAR.sqrt(),
            max_drawdown,
        }
    }
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

    // Starts the embedded interpreter and puts `dir` on its `sys.path`, so the
    // operator modules resolve like installed packages.
    Python::initialize();
    let code = CString::new(format!(
        "import sys; sys.path.append({:?})",
        args.python_ops_dir
    ))
    .unwrap();
    Python::attach(|py| {
        py.run(&code, None, None).expect("cannot extend sys.path");
    });

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
            vec!["close".into(), "volume".into()],
        )
        .with_time_range(None, args.end),
    );
    let (div_signals, divs) = b.source(
        panel::csv(
            format!("{dir}/dividends.csv"),
            "date",
            [("symbol".into(), Axis::Labeled(schema.clone()))],
            vec!["share".into(), "cash".into()],
        )
        .with_time_range(None, args.end),
    );
    let (_equity_signals, equities) = b.source(
        panel::csv(
            format!("{dir}/equity_structures.csv"),
            "date",
            [("symbol".into(), Axis::Labeled(schema.clone()))],
            vec!["total".into(), "circulating".into()],
        )
        .with_time_range(None, args.end),
    );
    let (close, volume) = (prices[0], prices[1]);
    let (share_divs, cash_divs) = (divs[0], divs[1]);
    let (total_shares, circulating_shares) = (equities[0], equities[1]);

    // One scalar pulse per trading day (any symbol has a row).
    let daily = b.op(signal::any(), price_signals);

    // The rebalance calendar, on which the models emit and the book turns over.
    let rebalance = b.source(sync::signal_iter(args.rebalance_instants().into_iter()));

    // Forward-adjust closes for dividends.
    let (_multipliers, adj_close) = b.op(
        feature::forward_adjust(),
        ((price_signals, close), (div_signals, share_divs, cash_divs)),
    );
    let returns = b.op(rolling::pct_change(1), (daily, adj_close));
    let returns_demeaned = b.op(stats::demean(), returns);
    let log_adj_close = b.op(elem::ln(), adj_close);
    let log_returns = b.op(rolling::diff(1), (daily, log_adj_close));

    // Calculate all features.
    let market = b.val(array::constant(vec![1.0; n]));
    let mom_1m = b.op(rolling::diff(MONTH), (daily, log_adj_close));
    let mom_12m = b.op(rolling::diff(YEAR), (daily, log_adj_close));
    let mom_12m_1m = b.op(elem::sub(), (mom_12m, mom_1m));
    let volatility = b.op(rolling::std_dev(QUARTER, 20), (daily, log_returns));
    let market_cap = b.op(elem::mul(), (close, total_shares));
    let size = b.op(elem::ln(), market_cap);
    let daily_turnover = b.op(elem::div(), (volume, circulating_shares));
    let turnover = b.op(rolling::mean(MONTH, 5), (daily, daily_turnover));

    // Calculate a cap-weighted index.
    let circulating_cap = b.op(elem::mul(), (close, circulating_shares));
    let index_weights = b.op(elem::fill_nan(0.0).then(stats::scale(1.0)), circulating_cap);

    // Tracks the cap-weighted index portfolio for baseline.
    let flags = b.val(array::constant(vec![true; n]));
    let (bids, asks) = (close, close);
    let (_positions, _cash, index_nav) = b.op(
        trader::fixed::benchmark(true, args.initial_cash),
        (
            (daily, flags, bids, asks),
            (div_signals, share_divs, cash_divs),
            (rebalance, index_weights),
        ),
    );
    let index_nav_series = b.op(series::record_all(), (daily, index_nav));
    let index_comp_return = b.op(metric::performance::comp_return(), (daily, index_nav));
    let index_sharpe = b.op(metric::performance::return_sharpe(), (daily, index_nav));
    let index_vol = b.op(metric::performance::return_vol(), (daily, index_nav));

    // Construct the alpha feature matrix.
    // All features are cross-sectionally standardized.
    // If number of assets grow large, consider winsorizing the features.
    let alpha_columns = [
        b.op(stats::standardize(), mom_1m),
        b.op(stats::standardize(), mom_12m_1m),
        b.op(stats::standardize(), volatility),
        b.op(stats::standardize(), turnover),
    ];
    let alpha_features = b.op(array::stack(1), &alpha_columns);

    // Construct the risk feature matrix.
    let risk_columns = [
        market,
        b.op(stats::standardize(), size),
        b.op(stats::standardize(), mom_12m_1m),
    ];
    let risk_features = b.op(array::stack(1), &risk_columns);

    // The mean predictor (alpha model).
    //
    // Prediction is updated per rebalance signal, while observing one sample
    // of alpha feature matrix and realized return vector per trading day.
    type AlphaInputs = (
        SignalPort<0>,     // sample (trading day) signal
        ArrayPort<f64, 2>, // raw alpha features
        ArrayPort<f64, 1>, // realized returns (demeaned)
        SignalPort<0>,     // rebalance signal
    );
    type AlphaOutputs = ArrayPort<f64, 1>; // predicted returns (demeaned)
    let mu = b.op(
        py_operator_module::<AlphaInputs, AlphaOutputs>(
            "alpha_model",
            py_params(|d| {
                d.set_item("target_offset", 1)?;
                d.set_item("min_periods", args.min_periods)?;
                d.set_item("ridge_l2", args.ridge_l2)?;
                Ok(())
            }),
        ),
        (daily, alpha_features, returns_demeaned, rebalance),
    );

    // The covariance predictor (risk model).
    //
    // Emits the three objects of the covariance matrix `Σ = X F Xᵀ + D`
    // on three ports, updated per rebalance signal, while observing one sample
    // of risk feature matrix and realized return vector per trading day.
    type RiskInputs = (
        SignalPort<0>,     // sample (trading day) signal
        ArrayPort<f64, 2>, // raw risk features
        ArrayPort<f64, 1>, // realized returns
        SignalPort<0>,     // rebalance signal
    );
    type RiskOutputs = (
        ArrayPort<f64, 2>, // risk factor exposures: X
        ArrayPort<f64, 2>, // factor covariance matrix: F
        ArrayPort<f64, 1>, // specific variances: diagonal D
    );
    let (exposures, covariance, specific) = b.op(
        py_operator_module::<RiskInputs, RiskOutputs>(
            "risk_model",
            py_params(|d| {
                d.set_item("target_offset", 1)?;
                d.set_item("min_periods", args.min_periods)?;
                d.set_item("covariance_halflife", args.covariance_halflife)?;
                d.set_item("specific_halflife", args.specific_halflife)?;
                Ok(())
            }),
        ),
        (daily, risk_features, returns, rebalance),
    );

    // The risk aversion parameter sweep.
    let mut variants = Vec::new();
    for risk_aversion in args.risk_aversion.iter() {
        // The portfolio optimizer.
        type PortfolioInputs = (
            SignalPort<0>,     // rebalance signal
            ArrayPort<f64, 1>, // index weights
            ArrayPort<f64, 1>, // predicted returns (demeaned)
            ArrayPort<f64, 2>, // risk factor exposures: X
            ArrayPort<f64, 2>, // factor covariance matrix: F
            ArrayPort<f64, 1>, // specific variances: diagonal D
        );
        type PortfolioOutputs = ArrayPort<f64, 1>; // portfolio weights
        let weights = b.op(
            py_operator_module::<PortfolioInputs, PortfolioOutputs>(
                "portfolio",
                py_params(|d| {
                    d.set_item("benchmark_relative", args.benchmark_relative)?;
                    d.set_item("risk_aversion", risk_aversion)?;
                    d.set_item("long_only", true)?; // aim for long-only portfolios
                    d.set_item("full_position", true)?; // aim for fully invested portfolios
                    Ok(())
                }),
            ),
            (
                rebalance,
                index_weights,
                mu,
                exposures,
                covariance,
                specific,
            ),
        );

        // Simulate frictionless trading using `weight`.
        let (_positions, _cash, nav) = b.op(
            trader::fixed::benchmark(true, args.initial_cash),
            (
                (daily, flags, bids, asks),
                (div_signals, share_divs, cash_divs),
                (rebalance, weights),
            ),
        );
        let nav_series = b.op(series::record_all(), (daily, nav));
        let comp_return = b.op(metric::performance::comp_return(), (daily, nav));
        let sharpe = b.op(metric::performance::return_sharpe(), (daily, nav));
        let vol = b.op(metric::performance::return_vol(), (daily, nav));
        variants.push((nav_series, comp_return, sharpe, vol));
    }

    // Run the event loop until all sources are exhausted, with a progress bar
    // over the estimated total row count.
    let mut g = b.build();
    let bar = ProgressBar::new(g.size_hint().unwrap_or(0) as u64);
    g.run(&mut pool, |g, _| bar.set_position(g.num_events() as u64))
        .await;
    bar.finish();

    // Read the recorded NAV curves back out. Every curve is recorded on the
    // same daily pulse, so they all share one timestamp axis.
    let index_series = g.view(index_nav_series);
    let instants: Vec<Instant> = index_series.instants().to_vec();
    let index_values: Vec<f64> = index_series.to_contiguous().to_vec();
    let curves: Vec<Vec<f64>> = variants
        .iter()
        .map(|&(nav_series, ..)| {
            let series = g.view(nav_series);
            assert_eq!(series.instants(), &instants[..]);
            series.to_contiguous().to_vec()
        })
        .collect();

    // Log every NAV curve: one column per sweep variant, plus the benchmark.
    let mut csv = String::from("date");
    for risk_aversion in &args.risk_aversion {
        csv += &format!(",nav_ra{risk_aversion}");
    }
    csv += ",index_nav\n";
    for (i, &t) in instants.iter().enumerate() {
        csv += &format_date(t);
        for curve in &curves {
            csv += &format!(",{}", curve[i]);
        }
        csv += &format!(",{}\n", index_values[i]);
    }
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&args.output, csv).unwrap();
    println!(
        "wrote {} × {} NAV points to {}",
        instants.len(),
        curves.len() + 1,
        args.output
    );

    // Print one summary row per variant (daily metrics annualized), with the
    // cap-weighted index on the last row for reference.
    let summaries: Vec<Summary> = variants
        .iter()
        .zip(curves.iter())
        .map(|(&(_, comp_return, sharpe, vol), nav)| {
            Summary::new(
                nav,
                args.initial_cash,
                *g.view(comp_return),
                *g.view(sharpe),
                *g.view(vol),
            )
        })
        .collect();

    let index_summary = Summary::new(
        &index_values,
        args.initial_cash,
        *g.view(index_comp_return),
        *g.view(index_sharpe),
        *g.view(index_vol),
    );

    let years = (*instants.last().unwrap() - instants[0]).as_days() as f64 / 365.25;
    println!();
    println!(
        "{} variants over {years:.1} years from {:.0} initial cash:",
        summaries.len(),
        args.initial_cash
    );
    println!(
        "{:>10}  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "aversion", "final NAV", "total ret.", "ann. ret.", "ann. vol.", "Sharpe", "max DD"
    );
    let print_row = |label: String, s: &Summary| {
        println!(
            "{label:>10}  {:>12.0}  {:>10}  {:>10}  {:>10}  {:>+10.3}  {:>10}",
            s.final_nav,
            format!("{:+.2}%", s.total_return * 100.0),
            format!("{:+.2}%", s.annual_return * 100.0),
            format!("{:.2}%", s.annual_vol * 100.0),
            s.annual_sharpe,
            format!("{:.2}%", s.max_drawdown * 100.0),
        );
    };
    for (risk_aversion, summary) in args.risk_aversion.iter().zip(summaries.iter()) {
        print_row(format!("{risk_aversion}"), summary);
    }
    print_row("index".into(), &index_summary);
}
