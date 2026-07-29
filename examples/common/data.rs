//! The stacked cross-sectional market panel: parquet sources → `[num_stocks]`
//! per-field panels, wired entirely in cross-section.
//!
//! Every panel source emits a `([N, 1] row signal, [N, K] values)` event
//! stream: signal element `i` pulses on the dates stock `i` has a row, and the
//! one row signal broadcasts over every `[N, K]` value column. Since
//! [`forward_adjust`] and [`annualize`] gate per element on the signal array,
//! they run **once over the whole cross-section** — there is no per-stock
//! fan-out at all. Occurrence is the signal alone: the `NaN`s padding a sparse
//! cross-section sit under clear signal elements and are never read, so a
//! `NaN` under a *set* signal would be a data error (both operators panic on
//! one). Per-stock carry ("latest known value") is a cross-sectional
//! [`elem::forward_fill_nan`] over the panel's batch face; batch-level
//! cadences (`daily`, `reports`) are [`any`] reductions of the row signals.

use tradingflow::data::{Array, ArrayView, Duration, Instant};
use tradingflow::graph::Builder;
use tradingflow::operators::{
    array::{array_map, select, select_at},
    elem,
    feature::stock::*,
    signal::{any, as_signal_map},
};
use tradingflow::ports::{ArrayPortHandle, SignalPortHandle};
use tradingflow::sources::panel::*;
use tradingflow::time::UnixTime;

use super::args::CommonArgs;

/// Cross-sectional `(num_stocks,)` panels produced by [`build_stacked`].
///
/// Daily price fields are the **batch face** of the daily stream: on a `daily`
/// pulse an element is that day's value, `NaN` where the stock did not trade;
/// between pulses the wire retains the last trading day's cross-section.
/// The fundamental fields are point-in-time carried state (effective-date
/// aligned, forward-filled per element). Income / cash-flow flows are
/// **annualized** (YTD → [`annualize`]); a trailing-twelve-month figure is a
/// 365-day rolling mean of the annualized report stream (see
/// [`factors`](super::features)). The parquet stores assets debit-positive and
/// liabilities / expense items credit-**negative** — the factor formulas negate
/// them where a positive magnitude is wanted.
pub struct Stacked {
    /// One pulse per trading day (the daily-prices panel's dates).
    pub daily: SignalPortHandle<0>,
    /// One pulse per income-statement effective date (any stock reporting).
    pub reports: SignalPortHandle<0>,
    /// One pulse per cash-flow-statement effective date.
    pub cf_reports: SignalPortHandle<0>,
    pub close: ArrayPortHandle<f64, 1>, // unadjusted close (daily batch face)
    pub volume: ArrayPortHandle<f64, 1>, // 成交量 shares (daily batch face)
    pub open: ArrayPortHandle<f64, 1>,  // 开盘价 unadjusted (daily batch face)
    pub high: ArrayPortHandle<f64, 1>,  // 最高价 unadjusted (daily batch face)
    pub low: ArrayPortHandle<f64, 1>,   // 最低价 unadjusted (daily batch face)
    pub amount: ArrayPortHandle<f64, 1>, // 成交额 turnover value (daily batch face)
    pub adjusted_close: ArrayPortHandle<f64, 1>, // close * forward-adjust factor (daily batch face)
    pub adjusts: ArrayPortHandle<f64, 1>, // forward-adjust factor (carried)
    pub div_signals: SignalPortHandle<1>,
    pub share_divs: ArrayPortHandle<f64, 1>, // 送股比例 shares per share held
    pub cash_divs: ArrayPortHandle<f64, 1>,  // 派息 cash per share held
    pub total_shares: ArrayPortHandle<f64, 1>, // (carried)
    pub circ_shares: ArrayPortHandle<f64, 1>, // (carried)
    pub parent_equity: ArrayPortHandle<f64, 1>, // 归母净资产, positive (carried)
    pub net_profit: ArrayPortHandle<f64, 1>, // 净利润, annualized, positive (carried)
    pub operating_profit: ArrayPortHandle<f64, 1>, // 营业利润, annualized, positive (carried)
    pub revenue: ArrayPortHandle<f64, 1>,    // 营业收入, annualized, positive (carried)
    pub operating_cost: ArrayPortHandle<f64, 1>, // 营业成本, annualized, NEGATIVE (carried)
    pub total_assets: ArrayPortHandle<f64, 1>, // 总资产, positive (carried)
    pub total_liab: ArrayPortHandle<f64, 1>, // 总负债, NEGATIVE (carried)
    pub current_assets: ArrayPortHandle<f64, 1>, // 流动资产, positive (carried)
    pub current_liab: ArrayPortHandle<f64, 1>, // 流动负债, NEGATIVE (carried)
    pub cash: ArrayPortHandle<f64, 1>,       // 货币资金, positive (carried)
    pub inventories: ArrayPortHandle<f64, 1>, // 存货, positive (carried)
    pub receivables: ArrayPortHandle<f64, 1>, // 应收票据及账款, positive (carried)
    pub net_operating_cashflow: ArrayPortHandle<f64, 1>, // 经营现金流净额, annualized (carried)
    // The same five annualized flows on their report cadence (pair with
    // `reports` / `cf_reports`; an element is non-NaN exactly when that stock's
    // report became effective on the pulse) — the inputs of the TTM means.
    pub net_profit_events: ArrayPortHandle<f64, 1>,
    pub operating_profit_events: ArrayPortHandle<f64, 1>,
    pub revenue_events: ArrayPortHandle<f64, 1>,
    pub operating_cost_events: ArrayPortHandle<f64, 1>,
    pub net_operating_cashflow_events: ArrayPortHandle<f64, 1>,
}

/// Load the consolidated long-format parquet panels and wire the
/// cross-sectional panel. One [`ParquetPanelSource`] /
/// [`ParquetFinancialReportPanelSource`] per data kind (one sequential scan
/// each); the financial reports align on the look-ahead-safe effective date
/// `max(report, notice)` (`use_effective_date`, zero fallback).
pub fn build_stacked(
    sc: &mut Builder<Instant, UnixTime>,
    symbols: &[String],
    args: &CommonArgs,
) -> Stacked {
    let dir = &args.data_dir;
    let start = Some(args.data_start());
    let end = Some(args.end());
    let universe: Vec<String> = symbols.to_vec();

    // Panel sources emit `([N, 1] row signal, [N, K])` streams reflecting only
    // each date's rows (absent symbols NaN); all carry-forward is downstream.
    let daily_panel = |sc: &mut Builder<Instant, UnixTime>,
                       kind: &str,
                       cols: Vec<String>|
     -> (SignalPortHandle<2>, ArrayPortHandle<f64, 2>) {
        let s = parquet_panel_source(format!("{dir}/{kind}.parquet"), cols, universe.clone())
            .with_time_range(start, end);
        sc.source(s)
    };
    let report_panel = |sc: &mut Builder<Instant, UnixTime>,
                        kind: &str,
                        cols: Vec<String>,
                        with_report_date: bool|
     -> (SignalPortHandle<2>, ArrayPortHandle<f64, 2>) {
        let s = parquet_financial_report_panel_source(
            format!("{dir}/{kind}.parquet"),
            cols,
            universe.clone(),
        )
        .with_report_date(with_report_date)
        .use_effective_date(Duration::ZERO)
        .with_time_range(start, end);
        sc.source(s)
    };

    let (daily_rows, prices) = daily_panel(
        sc,
        "daily_prices",
        vec![
            "prices.close".into(),  // 0
            "prices.volume".into(), // 1 (shares)
            "prices.open".into(),   // 2
            "prices.high".into(),   // 3
            "prices.low".into(),    // 4
            "prices.amount".into(), // 5 (turnover value, 成交额)
        ],
    );
    let (div_rows, divs) = daily_panel(
        sc,
        "dividends",
        vec!["dividends.share".into(), "dividends.cash".into()],
    );
    let (_equity_rows, equity) = daily_panel(
        sc,
        "equity_structures",
        vec!["shares.total".into(), "shares.circulating".into()],
    );
    let (_balance_rows, balance) = report_panel(
        sc,
        "balance_sheets",
        vec![
            // [0..3]: parent-equity components (summed and negated below).
            "balance_sheet.equity.capital".into(),
            "balance_sheet.equity.reserves".into(),
            "balance_sheet.equity.parent_interests".into(),
            // [3..10]: point-in-time balance stocks for the fundamental factors
            // (assets debit-positive, liabilities credit-negative).
            "balance_sheet.assets".into(),         // 3 total assets
            "balance_sheet.liab".into(),           // 4 total liabilities (negative)
            "balance_sheet.assets.current".into(), // 5 current assets
            "balance_sheet.liab.current".into(),   // 6 current liabilities (negative)
            "balance_sheet.assets.current.cash".into(), // 7 cash & equivalents
            "balance_sheet.assets.current.inventories".into(), // 8 inventories
            "balance_sheet.assets.current.receivables.notes_and_accounts".into(), // 9 应收票据及账款
        ],
        false,
    );
    let (report_rows, income) = report_panel(
        sc,
        "income_statements",
        vec![
            "income_statement.profit".into(),           // 0 net profit
            "income_statement.profit.operating".into(), // 1 operating profit
            "income_statement.profit.operating.income.revenue".into(), // 2 revenue
            "income_statement.profit.operating.expenses.costs".into(), // 3 operating cost (negative)
        ],
        true,
    );
    let (cf_report_rows, cashflow) = report_panel(
        sc,
        "cash_flow_statements",
        vec![
            "cash_flow_statement.change.operating".into(), // 0 operating cash flow
            "cash_flow_statement.change.investing".into(), // 1 investing cash flow
            "cash_flow_statement.change.financing".into(), // 2 financing cash flow
        ],
        true,
    );

    // Batch-level cadences (any row this date) and `[N]` per-stock row signals
    // for the squeezed per-field wires.
    let daily = sc.segment(any(), daily_rows);
    let reports = sc.segment(any(), report_rows);
    let cf_reports = sc.segment(any(), cf_report_rows);
    let daily_rows1 = sc.segment(as_signal_map(select_at(0, 1)), daily_rows);
    let div_rows1 = sc.segment(as_signal_map(select_at(0, 1)), div_rows);

    // Daily price fields: squeezing column selects of the `[N, 6]` panel.
    let close = sc.segment(select_at(0, 1), prices);
    let volume = sc.segment(select_at(1, 1), prices);
    let open = sc.segment(select_at(2, 1), prices);
    let high = sc.segment(select_at(3, 1), prices);
    let low = sc.segment(select_at(4, 1), prices);
    let amount = sc.segment(select_at(5, 1), prices);

    // Forward adjustment, whole cross-section at once: the close leg on the
    // daily row signal, both dividend legs on the dividend panel's row signal;
    // a per-stock event is that stock's signal element, and a NaN under a set
    // signal is a missing field of the fired row.
    let share_divs = sc.segment(select_at(0, 1), divs);
    let cash_divs = sc.segment(select_at(1, 1), divs);
    let (adjusts, adjusted_close) = sc.segment(
        forward_adjust(),
        ((daily_rows1, close), (div_rows1, share_divs, cash_divs)),
    );

    // Point-in-time carried panels: forward-fill each element of the batch
    // face, then select columns. (The fill node's cone is its own panel's
    // dates, so it ingests one batch per report date.)
    let equity_carried = sc.segment(elem::forward_fill_nan(), equity);
    let total_shares = sc.segment(select_at(0, 1), equity_carried);
    let circ_shares = sc.segment(select_at(1, 1), equity_carried);

    let balance_carried = sc.segment(elem::forward_fill_nan(), balance);
    // parent_equity = -(capital + reserves + parent_interests) (cols 0..3).
    let parent_equity = sc.segment(
        array_map(|a: ArrayView<f64, 2>| {
            let n = a.extents()[0];
            let rows = a.to_contiguous();
            let k = a.extents()[1];
            Array::<f64, 1>::from_parts(
                [n],
                (0..n)
                    .map(|i| -rows[i * k..i * k + 3].iter().sum::<f64>())
                    .collect::<Vec<_>>()
                    .into(),
            )
        }),
        balance_carried,
    );
    let total_assets = sc.segment(select_at(3, 1), balance_carried);
    let total_liab = sc.segment(select_at(4, 1), balance_carried);
    let current_assets = sc.segment(select_at(5, 1), balance_carried);
    let current_liab = sc.segment(select_at(6, 1), balance_carried);
    let cash = sc.segment(select_at(7, 1), balance_carried);
    let inventories = sc.segment(select_at(8, 1), balance_carried);
    let receivables = sc.segment(select_at(9, 1), balance_carried);

    // Annualized income / cash flows (YTD → annualize), whole cross-section at
    // once: the `[N, 1]` report row signal and the `[year, day_of_year]`
    // calendar columns (non-squeezed `[N, 1]` selects, cast to u16) all
    // broadcast against the `[N, K]` YTD columns — one row signal gates every
    // field of the report.
    let income_year_f = sc.segment(select(vec![0], 1), income);
    let income_year = sc.segment(elem::as_(), income_year_f);
    let income_doy_f = sc.segment(select(vec![1], 1), income);
    let income_doy = sc.segment(elem::as_(), income_doy_f);
    let income_ytd = sc.segment(select(vec![2, 3, 4, 5], 1), income);
    let income_ann = sc.segment(
        annualize(),
        (report_rows, income_ytd, income_year, income_doy),
    );
    let cf_year_f = sc.segment(select(vec![0], 1), cashflow);
    let cf_year = sc.segment(elem::as_(), cf_year_f);
    let cf_doy_f = sc.segment(select(vec![1], 1), cashflow);
    let cf_doy = sc.segment(elem::as_(), cf_doy_f);
    let cf_ytd = sc.segment(select(vec![2, 3, 4], 1), cashflow);
    let cf_ann = sc.segment(annualize(), (cf_report_rows, cf_ytd, cf_year, cf_doy));

    // Report event columns (batch face on the report signals) and their
    // carried counterparts (forward-filled per element).
    let np_e = sc.segment(select_at(0, 1), income_ann);
    let op_e = sc.segment(select_at(1, 1), income_ann);
    let rev_e = sc.segment(select_at(2, 1), income_ann);
    let cost_e = sc.segment(select_at(3, 1), income_ann);
    let ocf_e = sc.segment(select_at(0, 1), cf_ann);
    let income_carried = sc.segment(elem::forward_fill_nan(), income_ann);
    let cf_carried = sc.segment(elem::forward_fill_nan(), cf_ann);

    Stacked {
        daily,
        reports,
        cf_reports,
        close,
        volume,
        open,
        high,
        low,
        amount,
        adjusted_close,
        adjusts,
        div_signals: div_rows1,
        share_divs,
        cash_divs,
        total_shares,
        circ_shares,
        parent_equity,
        net_profit: sc.segment(select_at(0, 1), income_carried),
        operating_profit: sc.segment(select_at(1, 1), income_carried),
        revenue: sc.segment(select_at(2, 1), income_carried),
        operating_cost: sc.segment(select_at(3, 1), income_carried),
        total_assets,
        total_liab,
        current_assets,
        current_liab,
        cash,
        inventories,
        receivables,
        net_operating_cashflow: sc.segment(select_at(0, 1), cf_carried),
        net_profit_events: np_e,
        operating_profit_events: op_e,
        revenue_events: rev_e,
        operating_cost_events: cost_e,
        net_operating_cashflow_events: ocf_e,
    }
}
