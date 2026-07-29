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

use tradingflow::data::{Array, ArrayView, Duration, Instant};
use tradingflow::graph::{Builder, Pool};
use tradingflow::operators::array::{array_map, select, select_at, slice_reshape};
use tradingflow::operators::elem::{as_, div, forward_fill_nan, mul, neg};
use tradingflow::operators::feature::stock::annualize;
use tradingflow::operators::rolling;
use tradingflow::operators::series::record_all;
use tradingflow::operators::signal::as_signal_map;
use tradingflow::ports::{ArrayPortHandle, SignalPortHandle};
use tradingflow::sources::panel::*;
use tradingflow::time::UnixTime;

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

    let mut sc = Builder::new(UnixTime);

    // ------------------------------------------------------------------
    // Panel sources → select the target stock's row and its own tick signal:
    // the source's `[N, 1]` row signal selects to a `[1]` signal array (this
    // stock's dates; broadcastable over the row for the annualizer) and
    // reduces to a rank-0 pulse for the records.
    // ------------------------------------------------------------------
    let daily = |sc: &mut Builder<Instant, UnixTime>,
                 kind: &str,
                 cols: Vec<String>|
     -> (
        SignalPortHandle<0>,
        SignalPortHandle<1>,
        ArrayPortHandle<f64, 1>,
    ) {
        let s = parquet_panel_source(format!("{data_dir}/{kind}.parquet"), cols, symbols.clone());
        let (rows, panel) = sc.source(s);
        let row = sc.segment(select_at(idx, 0), panel);
        let tick1 = sc.segment(as_signal_map(slice_reshape((idx, ..))), rows);
        let tick = sc.segment(as_signal_map(slice_reshape((idx, 0usize))), rows);
        (tick, tick1, row)
    };

    let report = |sc: &mut Builder<Instant, UnixTime>,
                  kind: &str,
                  cols: Vec<String>,
                  with_report_date: bool|
     -> (
        SignalPortHandle<0>,
        SignalPortHandle<1>,
        ArrayPortHandle<f64, 1>,
    ) {
        let s = parquet_financial_report_panel_source(
            format!("{data_dir}/{kind}.parquet"),
            cols,
            symbols.clone(),
        )
        .with_report_date(with_report_date);
        let (rows, panel) = sc.source(s);
        let row = sc.segment(select_at(idx, 0), panel);
        let tick1 = sc.segment(as_signal_map(slice_reshape((idx, ..))), rows);
        let tick = sc.segment(as_signal_map(slice_reshape((idx, 0usize))), rows);
        (tick, tick1, row)
    };

    let (price_ticks, _, prices) = daily(&mut sc, "daily_prices", vec!["prices.close".into()]); // [close]
    let (_, _, equity) = daily(&mut sc, "equity_structures", vec!["shares.total".into()]); // [total]
    // Balance sheet (point-in-time): assets, equity, and the parent-equity parts.
    let (balance_ticks, _, balance) = report(
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
    let (income_ticks, income_ticks1, income) = report(
        &mut sc,
        "income_statements",
        vec![
            "income_statement.profit.operating.income".into(),
            "income_statement.profit".into(),
        ],
        true,
    ); // [year, day_of_year, op_income, net_profit]
    let (cf_ticks, cf_ticks1, cf) = report(
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
    // Carry the last known share count per element (the equity row retains
    // only the last equity-panel date's batch, NaN elsewhere).
    let shares_carried = sc.segment(forward_fill_nan(), total_shares);
    let market_cap = sc.segment(mul(), (close, shares_carried));

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

    // Split the panel's leading [year, day_of_year] f64 columns off each report
    // row and cast them to the calendar inputs (broadcast `[1]` → `[K]`); the
    // annualizer reads them on the report tick signal only.
    let income_year_f = sc.segment(select(vec![0], 0), income);
    let income_year = sc.segment(as_(), income_year_f);
    let income_doy_f = sc.segment(select(vec![1], 0), income);
    let income_doy = sc.segment(as_(), income_doy_f);
    let income_ytd = sc.segment(select(vec![2, 3], 0), income);
    let income_ann = sc.segment(
        annualize(),
        (income_ticks1, income_ytd, income_year, income_doy),
    ); // [op_income, net_profit]
    let op_income = sc.segment(select_at(0, 0), income_ann);
    let net_profit = sc.segment(select_at(1, 0), income_ann);
    let cf_year_f = sc.segment(select(vec![0], 0), cf);
    let cf_year = sc.segment(as_(), cf_year_f);
    let cf_doy_f = sc.segment(select(vec![1], 0), cf);
    let cf_doy = sc.segment(as_(), cf_doy_f);
    let cf_ytd = sc.segment(select(vec![2], 0), cf);
    let cf_ann = sc.segment(annualize(), (cf_ticks1, cf_ytd, cf_year, cf_doy)); // [change]
    let cash_flow = sc.segment(select_at(0, 0), cf_ann);

    let net_profit_series = sc.segment(record_all(), (income_ticks, net_profit));
    let net_profit_ttm = sc.segment(
        rolling::series_mean(Duration::from_days(365), 1),
        net_profit_series,
    );

    // Carry the report-cadence levels per element for the daily ratios.
    let ttm_carried = sc.segment(forward_fill_nan(), net_profit_ttm);
    let peq_carried = sc.segment(forward_fill_nan(), parent_equity);
    let ep = sc.segment(div(), (ttm_carried, market_cap));
    let bp = sc.segment(div(), (peq_carried, market_cap));
    let roe = sc.segment(div(), (ttm_carried, peq_carried));

    let records = [
        sc.segment(record_all(), (price_ticks, market_cap)),
        sc.segment(record_all(), (balance_ticks, assets)),
        sc.segment(record_all(), (balance_ticks, equity_val)),
        sc.segment(record_all(), (balance_ticks, parent_equity)),
        sc.segment(record_all(), (income_ticks, op_income)),
        sc.segment(record_all(), (income_ticks, net_profit)),
        sc.segment(record_all(), (cf_ticks, cash_flow)),
        sc.segment(record_all(), (price_ticks, ep)),
        sc.segment(record_all(), (price_ticks, bp)),
        sc.segment(record_all(), (price_ticks, roe)),
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
