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

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;

#[path = "common/mod.rs"]
mod common;

use tradingflow::graph::{Handle, ViewPort};

use tradingflow::data::Duration;
use tradingflow::operators::{
    ArrayValue, Window, annualize, as_view, divide, filter, map, multiply, negate, record,
    rolling_mean, select,
};
use tradingflow::sources::{ParquetFinancialReportPanelSource, ParquetPanelSource};
use tradingflow::{Array, ArrayView, Instant, Series};
use tradingflow::{Scenario, WallClock};

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

fn load_symbols(data_dir: &str) -> Vec<String> {
    let path = format!("{data_dir}/symbol_list.csv");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut lines = text.lines();
    let header = lines.next().expect("symbol_list header");
    let col = header
        .split(',')
        .position(|h| h.trim() == "symbol")
        .expect("`symbol` column");
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
    let sel = sc.push(select(vec![i], 0, true), panel);
    sc.push(
        filter(|a: ArrayView<f64, 1>| a.to_contiguous().iter().any(|x| x.is_finite())),
        sel,
    )
}

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

    let symbols = load_symbols(data_dir);
    let idx = symbols
        .iter()
        .position(|s| s == &symbol)
        .unwrap_or_else(|| panic!("{symbol} not in symbol_list.csv"));

    let mut sc = Scenario::new(WallClock, Instant::MIN);

    // ------------------------------------------------------------------
    // Panel sources → select the target stock.
    // ------------------------------------------------------------------
    let daily = |sc: &mut Scenario,
                 kind: &str,
                 cols: Vec<String>|
     -> Handle<ViewPort<ArrayValue<f64, 1>>> {
        let s =
            ParquetPanelSource::new(format!("{data_dir}/{kind}.parquet"), cols, symbols.clone());
        let panel = sc.add_source(s);
        let panel = sc.push(as_view(), panel);
        pick(sc, panel, idx)
    };
    let report = |sc: &mut Scenario,
                  kind: &str,
                  cols: Vec<String>,
                  with_report_date: bool|
     -> Handle<ViewPort<ArrayValue<f64, 1>>> {
        let s = ParquetFinancialReportPanelSource::new(
            format!("{data_dir}/{kind}.parquet"),
            cols,
            symbols.clone(),
        )
        .with_report_date(with_report_date);
        let panel = sc.add_source(s);
        let panel = sc.push(as_view(), panel);
        pick(sc, panel, idx)
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
    let close = sc.push(select(vec![0], 0, true), prices);
    let total_shares = sc.push(select(vec![0], 0, true), equity);
    let market_cap = sc.push(multiply(), (close, total_shares));

    let assets = sc.push(select(vec![0], 0, true), balance);
    let neg_equity = sc.push(select(vec![1], 0, true), balance);
    let equity_val = sc.push(negate(), neg_equity);
    let neg_peq = sc.push(select(vec![2, 3, 4], 0, false), balance);
    let parent_equity = sc.push(
        map(|a: ArrayView<f64, 1>| Array::scalar(-a.to_contiguous().iter().sum::<f64>())),
        neg_peq,
    );

    let income_ann = sc.push(annualize(), income); // [op_income, net_profit]
    let op_income = sc.push(select(vec![0], 0, true), income_ann);
    let net_profit = sc.push(select(vec![1], 0, true), income_ann);
    let cf_ann = sc.push(annualize(), cf); // [change]
    let cash_flow = sc.push(select(vec![0], 0, true), cf_ann);

    let net_profit_series = sc.push(record(), net_profit);
    let net_profit_ttm = sc.push(
        rolling_mean(Window::TimeDelta(Duration::from_days(365))),
        net_profit_series,
    );

    let ep = sc.push(divide(), (net_profit_ttm, market_cap));
    let bp = sc.push(divide(), (parent_equity, market_cap));
    let roe = sc.push(divide(), (net_profit_ttm, parent_equity));

    let records = [
        sc.push(record(), market_cap),
        sc.push(record(), assets),
        sc.push(record(), equity_val),
        sc.push(record(), parent_equity),
        sc.push(record(), op_income),
        sc.push(record(), net_profit),
        sc.push(record(), cash_flow),
        sc.push(record(), ep),
        sc.push(record(), bp),
        sc.push(record(), roe),
    ];

    // ------------------------------------------------------------------
    // Run.
    // ------------------------------------------------------------------
    let mut session = sc.build();
    let total = session.total_num_events();
    session.run(common::progress(total, Instant::MIN)).await;
    eprintln!();

    let mut rows: BTreeMap<i64, [f64; 10]> = BTreeMap::new();
    for (c, h) in records.iter().enumerate() {
        let series: &Series<f64, 0> = session.ref_view(*h);
        for (ts, v) in series.timestamps().iter().zip(series.values().iter()) {
            rows.entry(ts.as_nanos()).or_insert([f64::NAN; 10])[c] = *v;
        }
    }

    let n_mc = session.ref_view(records[0]).len();
    if n_mc == 0 {
        eprintln!("no data for {symbol}");
        std::process::exit(1);
    }

    let mut csv = String::from("timestamp_ns");
    for c in COLS {
        write!(csv, ",{c}").unwrap();
    }
    csv.push('\n');
    for (ts, vals) in &rows {
        write!(csv, "{ts}").unwrap();
        for v in vals {
            write!(csv, ",{v}").unwrap();
        }
        csv.push('\n');
    }
    let path = "target/plot_financial_data.csv";
    fs::write(path, csv).expect("write csv");
    println!(
        "{symbol}: {} report+price rows -> {path}\nplot with:  python examples/plot_financial_data.py {path}",
        rows.len()
    );
}
