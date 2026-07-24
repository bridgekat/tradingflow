//! Financial-data plot, reading the consolidated long **parquet** panels via
//! [`ParquetPanelSource`] / [`ParquetFinancialReportPanelSource`].
//!
//! Loads daily prices, equity-structure events, and quarterly financial reports
//! for one A-shares stock by `Select`ing that stock's row out of each
//! cross-sectional panel (and `Filter`ing out the "no data" ticks), then computes
//! market cap, annualized income/cash-flow, TTM net profit, and the E/P, B/P, ROE
//! ratios with native operators, writing the recorded series to a tidy CSV for
//! `examples/plot_financial_data.py` (matplotlib).
//!
//! The report panels are aligned on the report `date` (point-in-time), i.e.
//! `ParquetFinancialReportPanelSource` with `use_effective_date = false`. Binary ops
//! carry the last-known value of the slower input, so ratios update on every tick
//! of either input; the wide CSV has `NaN` where a column did not tick.
//!
//! ```text
//! cargo run --example plot_financial_data [SYMBOL]   # default 000001.SZ
//! python examples/plot_financial_data.py target/plot_financial_data.csv
//! ```

#[path = "common/mod.rs"]
mod common;

use tradingflow::clock::UnixClock;
use tradingflow::data::{Array, ArrayView, Duration, Instant};
use tradingflow::graph::{Builder, Pool};
use tradingflow::operators::array::{array_map, select, select_at};
use tradingflow::operators::elem::{div, mul, neg};
use tradingflow::operators::rolling;
use tradingflow::operators::series::record_all;
use tradingflow::operators::stocks::annualize;
use tradingflow::operators::structural::filter;
use tradingflow::ports::ArrayPortHandle;
use tradingflow::sources::panel::*;

const COLS: [&str; 10] = [
    "market_cap",
    "assets",
    "equity",
    "parent_equity",
    "op_income",
    "net_profit",
    "cash_flow",
    "ep_ratio",
    "bp_ratio",
    "roe",
];

use clap::Parser;

/// Financial-data plot: market cap + annualized statement metrics.
#[derive(Parser)]
struct Args {
    /// Stock symbol to plot, e.g. 000001.SZ.
    symbol: String,
}

#[tokio::main]
async fn main() {
    let symbol = Args::parse().symbol;
    let data_dir = "examples/data";
    let prices_pq = format!("{data_dir}/daily_prices.parquet");
    if !std::path::Path::new(&prices_pq).exists() {
        eprintln!(
            "data not found: {prices_pq}\n(run the crawler with --export-long parquet; see examples/README.md)"
        );
        std::process::exit(1);
    }

    let symbols = common::load_symbols(data_dir);
    let idx = symbols
        .iter()
        .position(|s| s == &symbol)
        .unwrap_or_else(|| panic!("{symbol} not in symbol_list.csv"));

    let mut sc = Builder::new(UnixClock);

    // ------------------------------------------------------------------
    // Panel sources → select the target stock.
    // ------------------------------------------------------------------
    let daily = |sc: &mut Builder<Instant, UnixClock>,
                 kind: &str,
                 cols: Vec<String>|
     -> ArrayPortHandle<f64, 1> {
        let s = parquet_panel_source(format!("{data_dir}/{kind}.parquet"), cols, symbols.clone());
        let panel = sc.source(s);
        let sel = sc.segment(select_at(idx, 0), panel);
        sc.segment(
            filter(|a: ArrayView<f64, 1>| a.to_contiguous().iter().any(|x| x.is_finite())),
            sel,
        )
    };

    let report = |sc: &mut Builder<Instant, UnixClock>,
                  kind: &str,
                  cols: Vec<String>,
                  with_report_date: bool|
     -> ArrayPortHandle<f64, 1> {
        let s = parquet_financial_report_panel_source(
            format!("{data_dir}/{kind}.parquet"),
            cols,
            symbols.clone(),
        )
        .with_report_date(with_report_date);
        let panel = sc.source(s);
        let sel = sc.segment(select_at(idx, 0), panel);
        sc.segment(
            filter(|a: ArrayView<f64, 1>| a.to_contiguous().iter().any(|x| x.is_finite())),
            sel,
        )
    };

    let prices = daily(&mut sc, "daily_prices", vec!["prices.close".into()]); // [close]
    let equity = daily(&mut sc, "equity_structures", vec!["shares.total".into()]); // [total]
    // Balance sheet (point-in-time): assets, equity, and the parent-equity parts.
    let balance = report(
        &mut sc,
        "balance_sheets",
        vec![
            "balance_sheet.assets".into(),
            "balance_sheet.equity".into(),
            "balance_sheet.equity.capital".into(),
            "balance_sheet.equity.reserves".into(),
            "balance_sheet.equity.parent_interests".into(),
        ],
        false,
    ); // [5]
    // Income / cash flow (YTD → annualize): with_report_date prepends [year, day_of_year].
    let income = report(
        &mut sc,
        "income_statements",
        vec![
            "income_statement.profit.operating.income".into(),
            "income_statement.profit".into(),
        ],
        true,
    ); // [year, day_of_year, op_income, net_profit]
    let cf = report(
        &mut sc,
        "cash_flow_statements",
        vec!["cash_flow_statement.change".into()],
        true,
    ); // [year, day_of_year, change]

    // ------------------------------------------------------------------
    // Operators (identical to the CSV version).
    // ------------------------------------------------------------------
    let close = sc.segment(select_at(0, 0), prices);
    let total_shares = sc.segment(select_at(0, 0), equity);
    let market_cap = sc.segment(mul(), (close, total_shares));

    let assets = sc.segment(select_at(0, 0), balance);
    let neg_equity = sc.segment(select_at(1, 0), balance);
    let equity_val = sc.segment(neg(), neg_equity);
    let neg_peq = sc.segment(select(vec![2, 3, 4], 0), balance);
    let parent_equity = sc.segment(
        array_map(|a: ArrayView<f64, 1>| {
            Array::<f64, 0>::scalar(-a.to_contiguous().iter().sum::<f64>())
        }),
        neg_peq,
    );

    let income_ann = sc.segment(annualize(), income); // [op_income, net_profit]
    let op_income = sc.segment(select_at(0, 0), income_ann);
    let net_profit = sc.segment(select_at(1, 0), income_ann);
    let cf_ann = sc.segment(annualize(), cf); // [change]
    let cash_flow = sc.segment(select_at(0, 0), cf_ann);

    let net_profit_series = sc.segment(record_all(), net_profit);
    let net_profit_ttm = sc.segment(
        rolling::series_mean(Duration::from_days(365), 1),
        net_profit_series,
    );

    let ep = sc.segment(div(), (net_profit_ttm, market_cap));
    let bp = sc.segment(div(), (parent_equity, market_cap));
    let roe = sc.segment(div(), (net_profit_ttm, parent_equity));

    let records = [
        sc.segment(record_all(), market_cap),
        sc.segment(record_all(), assets),
        sc.segment(record_all(), equity_val),
        sc.segment(record_all(), parent_equity),
        sc.segment(record_all(), op_income),
        sc.segment(record_all(), net_profit),
        sc.segment(record_all(), cash_flow),
        sc.segment(record_all(), ep),
        sc.segment(record_all(), bp),
        sc.segment(record_all(), roe),
    ];

    // ------------------------------------------------------------------
    // Run.
    // ------------------------------------------------------------------
    let mut session = sc.build();
    let mut pool = Pool::new(0);
    let total = session.size_hint();
    session.run(&mut pool, common::progress(total)).await;
    eprintln!();

    // Read the recorded series and align them by timestamp into a wide CSV.
    let series: Vec<_> = COLS
        .iter()
        .zip(records)
        .map(|(name, h)| {
            let (ts, v) = common::read_scalar_series(&session, h);
            (name.to_string(), ts, v)
        })
        .collect();

    let n_mc = series[0].1.len();
    if n_mc == 0 {
        eprintln!("no data for {symbol}");
        std::process::exit(1);
    }

    let path = "target/plot_financial_data.csv";
    common::write_wide_csv(path, &series);
    println!(
        "{symbol}: {n_mc} market-cap rows -> {path}\nplot with:  python examples/plot_financial_data.py {path}"
    );
}
