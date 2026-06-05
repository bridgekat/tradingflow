//! Port of `python/examples/plot_financial_data.py` to the Rust flow engine.
//!
//! Loads daily prices, equity-structure events, and quarterly financial
//! reports (balance sheet / income statement / cash-flow statement) for one
//! A-shares stock, computes market cap, annualized income/cash-flow, TTM net
//! profit, and the E/P, B/P, ROE valuation ratios using only **native** Rust
//! operators (no Python), and writes the recorded series to a tidy CSV for
//! `examples/plot_financial_data.py` (matplotlib).
//!
//! Mixed-cadence design: prices tick daily, financial reports tick on their
//! (point-in-time) report dates. Binary ops (`Multiply`/`Divide`) carry the
//! last-known value of the slower input, so the ratios update on every tick of
//! either input — exactly as the Python original. The wide CSV therefore has
//! `NaN` in a column wherever that series did not tick; the plot script masks
//! `NaN` per column so each (daily or quarterly) curve renders on its own
//! cadence.
//!
//! Runs on default features (no `pyflow` / Python needed):
//!
//! ```text
//! cargo run --example plot_financial_data           # default symbol 000001.SZ
//! python examples/plot_financial_data.py target/plot_financial_data.csv
//! ```

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;

use tradingflow::data::Duration;
use tradingflow::flow::{Annualize, Divide, Map, Multiply, Negate, RollingMean, Scenario, Select};
use tradingflow::sources::CsvSource;
use tradingflow::sources::stocks::FinancialReportSource;
use tradingflow::{Array, Series};

/// Columns written to the wide CSV, in order.
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

#[tokio::main]
async fn main() {
    let symbol = std::env::args().nth(1).unwrap_or_else(|| "000001.SZ".to_string());
    let dir = "python/examples/data/a_shares_history";
    let prices_csv = format!("{dir}/{symbol}.daily_prices.csv");
    let equity_csv = format!("{dir}/{symbol}.equity_structures.csv");
    let balance_csv = format!("{dir}/{symbol}.balance_sheets.csv");
    let income_csv = format!("{dir}/{symbol}.income_statements.csv");
    let cf_csv = format!("{dir}/{symbol}.cash_flow_statements.csv");
    if !std::path::Path::new(&prices_csv).exists() {
        eprintln!("data not found: {prices_csv}");
        std::process::exit(1);
    }

    let mut sc = Scenario::new();

    // ------------------------------------------------------------------
    // Sources
    // ------------------------------------------------------------------

    // Daily close price (scalar).
    let prices = sc.add_source(
        CsvSource::new(prices_csv, "date".into(), vec!["prices.close".into()], Duration::ZERO),
        Array::zeros(&[1]),
    );
    // Total shares from equity-structure events (scalar, irregular).
    let equity = sc.add_source(
        CsvSource::new(equity_csv, "date".into(), vec!["shares.total".into()], Duration::ZERO),
        Array::zeros(&[1]),
    );

    // Balance sheet (point-in-time, no annualization): assets, equity, and the
    // three parent-equity components. Equity is stored negated (assets =
    // liabilities + equity convention), so we negate it back.
    let balance = sc.add_source(
        FinancialReportSource::new(
            balance_csv,
            "date".into(),
            "notice_date".into(),
            vec![
                "balance_sheet.assets".into(),
                "balance_sheet.equity".into(),
                "balance_sheet.equity.capital".into(),
                "balance_sheet.equity.reserves".into(),
                "balance_sheet.equity.parent_interests".into(),
            ],
            false, // with_report_date: no [year, day_of_year] prefix needed
            false, // use_effective_date: align on report date (point-in-time)
            Duration::ZERO,
        ),
        Array::zeros(&[5]),
    );

    // Income statement (YTD cumulative → annualize): operating income, net
    // profit. with_report_date=true prepends [year, day_of_year] for Annualize.
    let income = sc.add_source(
        FinancialReportSource::new(
            income_csv,
            "date".into(),
            "notice_date".into(),
            vec![
                "income_statement.profit.operating.income".into(),
                "income_statement.profit".into(),
            ],
            true,
            false,
            Duration::ZERO,
        ),
        Array::zeros(&[4]), // [year, day_of_year, op_income, net_profit]
    );

    // Cash-flow statement (YTD cumulative → annualize): net change in cash.
    let cf = sc.add_source(
        FinancialReportSource::new(
            cf_csv,
            "date".into(),
            "notice_date".into(),
            vec!["cash_flow_statement.change".into()],
            true,
            false,
            Duration::ZERO,
        ),
        Array::zeros(&[3]), // [year, day_of_year, change]
    );

    // ------------------------------------------------------------------
    // Operators
    // ------------------------------------------------------------------

    let close = sc.add_operator(Select::<f64>::new(vec![0], 0, true), prices);
    let total_shares = sc.add_operator(Select::<f64>::new(vec![0], 0, true), equity);
    let market_cap = sc.add_operator(Multiply::<f64>::new(), (close, total_shares));

    let assets = sc.add_operator(Select::<f64>::new(vec![0], 0, true), balance);
    let neg_equity = sc.add_operator(Select::<f64>::new(vec![1], 0, true), balance);
    let equity_val = sc.add_operator(Negate::<f64>::new(), neg_equity);

    // parent_equity = -(capital + reserves + parent_interests).
    let neg_peq = sc.add_operator(Select::<f64>::new(vec![2, 3, 4], 0, false), balance);
    let parent_equity = sc.add_operator(
        Map::new(|a: &Array<f64>| Array::scalar(-a.as_slice().iter().sum::<f64>())),
        neg_peq,
    );

    // Annualize income & cash flow (strips the [year, day_of_year] prefix).
    let income_ann = sc.add_operator(Annualize::new(), income); // [op_income, net_profit]
    let op_income = sc.add_operator(Select::<f64>::new(vec![0], 0, true), income_ann);
    let net_profit = sc.add_operator(Select::<f64>::new(vec![1], 0, true), income_ann);
    let cf_ann = sc.add_operator(Annualize::new(), cf); // [change]
    let cash_flow = sc.add_operator(Select::<f64>::new(vec![0], 0, true), cf_ann);

    // TTM net profit: rolling mean of annualized quarterly values over 365
    // days on the report-date axis (equals the last-4-quarter mean).
    let net_profit_series = sc.add_record(net_profit);
    let net_profit_ttm =
        sc.add_operator(RollingMean::<f64>::time_delta(Duration::from_days(365)), net_profit_series);

    // Valuation ratios.
    let ep = sc.add_operator(Divide::<f64>::new(), (net_profit_ttm, market_cap));
    let bp = sc.add_operator(Divide::<f64>::new(), (parent_equity, market_cap));
    let roe = sc.add_operator(Divide::<f64>::new(), (net_profit_ttm, parent_equity));

    // Record everything for output.
    let records = [
        sc.add_record(market_cap),
        sc.add_record(assets),
        sc.add_record(equity_val),
        sc.add_record(parent_equity),
        sc.add_record(op_income),
        sc.add_record(net_profit),
        sc.add_record(cash_flow),
        sc.add_record(ep),
        sc.add_record(bp),
        sc.add_record(roe),
    ];

    // ------------------------------------------------------------------
    // Run
    // ------------------------------------------------------------------

    let mut session = sc.build();
    session.run(|_, _| {}).await;

    // Align the recorded series by timestamp into a wide CSV (NaN where a
    // column did not tick at that timestamp; the plot script masks per column).
    let mut rows: BTreeMap<i64, [f64; 10]> = BTreeMap::new();
    for (c, h) in records.iter().enumerate() {
        let series: &Series<f64> = session.value(*h);
        for (ts, v) in series.timestamps().iter().zip(series.values().iter()) {
            rows.entry(ts.as_nanos()).or_insert([f64::NAN; 10])[c] = *v;
        }
    }

    let n_mc = session.value(records[0]).len();
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
