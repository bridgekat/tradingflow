//! The cross-sectional market panel: parquet sources → `[num_stocks]`
//! per-field panels, wired entirely in cross-section.
//!
//! Every panel source emits a `([N] signal, M × [N] value arrays)` stream over
//! the whole-market symbol axis: signal element `i` pulses on the timestamps
//! stock `i` has a row, and one signal is shared by every value field of that
//! table. Since [`forward_adjust`] and [`annualize`] gate per element on the
//! signal array, they run **once over the whole cross-section** — there is no
//! per-stock fan-out at all.
//!
//! # Carried state vs. event face
//!
//! The source *retains* each cell: a stock with no row on a timestamp keeps
//! its last emitted value, so the fundamental fields are point-in-time carried
//! state for free (no [`elem::forward_fill_nan`] needed). The daily price
//! fields instead want the **event face** — that day's value, `NaN` where the
//! stock did not trade — which [`event_face`] recovers by masking the carried
//! wire with the signal. The distinction matters downstream: a suspended
//! stock has no quote (`NaN` face) but does have a last known price (carried).
//!
//! A `NaN` under a *set* signal element is a null cell of a row that did fire
//! — missing or erroneous source data, which [`forward_adjust`] and
//! [`annualize`] reject.

use tradingflow::data::utils::{Axis, Schema};
use tradingflow::data::{ArrayView, Duration, Instant};
use tradingflow::graph::{Builder, Segment};
use tradingflow::operators::{
    elem,
    feature::{annualize, forward_adjust},
    signal::any,
};
use tradingflow::ports::{ArrayPort, ArrayPortHandle, SignalPort, SignalPortHandle};
use tradingflow::sources::panel;
use tradingflow::time::UnixTime;

use super::args::CommonArgs;

/// Operator signature for [`event_face`].
struct EventFace;

impl Segment for EventFace {
    type Inputs = (SignalPort<1>, ArrayPort<f64, 1>);
    type Outputs = ArrayPort<f64, 1>;
    type Context = Instant;
    type State = tradingflow::data::Array<f64, 1>;

    fn init(self, (_, values): (ArrayView<'_, bool, 1>, ArrayView<'_, f64, 1>)) -> Self::State {
        tradingflow::data::Array::full(values.extents(), f64::NAN)
    }

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, bool, 1>, ArrayView<'a, f64, 1>),
        state: &'b mut Self::State,
    ) -> ArrayView<'a, f64, 1> {
        // The face of the last pulse is retained between pulses, exactly as
        // the values wire it faces is.
        state.view()
    }

    fn compute<'a, 'b: 'a>(
        (signal, values): (ArrayView<'a, bool, 1>, ArrayView<'a, f64, 1>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, f64, 1> {
        let signal = signal.to_contiguous();
        let values = values.to_contiguous();
        for ((dst, &set), &value) in state.data_mut().iter_mut().zip(&*signal).zip(&*values) {
            *dst = if set { value } else { f64::NAN };
        }
        state.view()
    }
}

/// The event face of a carried panel field: this pulse's value where the
/// signal is set, `NaN` where the cell is stale. Retained between pulses.
fn event_face() -> impl Segment<
    Inputs = (SignalPort<1>, ArrayPort<f64, 1>),
    Outputs = ArrayPort<f64, 1>,
    Context = Instant,
> {
    EventFace
}

/// Cross-sectional `(num_stocks,)` panels produced by [`build_stacked`].
///
/// Daily price fields are the **event face** of the daily stream: on a `daily`
/// pulse an element is that day's value, `NaN` where the stock did not trade;
/// between pulses the wire retains the last trading day's cross-section.
/// The fundamental fields are point-in-time carried state (effective-date
/// aligned by the exporter, retained per element by the source). Income /
/// cash-flow flows are **annualized** (YTD → [`annualize`]); a
/// trailing-twelve-month figure is a 365-day rolling mean of the annualized
/// report stream (see [`factors`](super::features)). The parquet stores assets
/// debit-positive and liabilities / expense items credit-**negative** — the
/// factor formulas negate them where a positive magnitude is wanted.
pub struct Stacked {
    /// One pulse per trading day (the daily-prices panel's dates).
    pub daily: SignalPortHandle<0>,
    /// Per-stock daily row signal: element `i` pulses on stock `i`'s trading
    /// days. This — not a `NaN` test — is what states whether a stock traded.
    pub daily_signals: SignalPortHandle<1>,
    /// One pulse per income-statement effective date (any stock reporting).
    pub reports: SignalPortHandle<0>,
    /// One pulse per cash-flow-statement effective date.
    pub cf_reports: SignalPortHandle<0>,
    pub close: ArrayPortHandle<f64, 1>, // unadjusted close (daily event face)
    pub volume: ArrayPortHandle<f64, 1>, // 成交量 shares (daily event face)
    pub open: ArrayPortHandle<f64, 1>,  // 开盘价 unadjusted (daily event face)
    pub high: ArrayPortHandle<f64, 1>,  // 最高价 unadjusted (daily event face)
    pub low: ArrayPortHandle<f64, 1>,   // 最低价 unadjusted (daily event face)
    pub amount: ArrayPortHandle<f64, 1>, // 成交额 turnover value (daily event face)
    pub adjusted_close: ArrayPortHandle<f64, 1>, // close * forward-adjust factor (daily event face)
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
    // The same five annualized flows, kept as separate fields because they are
    // the TTM means' inputs (sampled on `reports` / `cf_reports`). `annualize`
    // already retains per element, so these *are* the carried wires above —
    // an element holds that stock's latest annualized flow, not a per-pulse
    // event face.
    pub net_profit_events: ArrayPortHandle<f64, 1>,
    pub operating_profit_events: ArrayPortHandle<f64, 1>,
    pub revenue_events: ArrayPortHandle<f64, 1>,
    pub operating_cost_events: ArrayPortHandle<f64, 1>,
    pub net_operating_cashflow_events: ArrayPortHandle<f64, 1>,
}

/// Load the consolidated long-format parquet panels and wire the
/// cross-sectional panel — one [`panel::Parquet`] source per data kind (one
/// sequential scan each), over the whole-market symbol axis.
///
/// The financial-report tables are already keyed by the look-ahead-safe
/// effective date `max(report, notice)`, with restatements dropped and
/// `report_year` / `report_day_of_year` materialised as value columns, by
/// `examples/export_parquet.py`.
pub fn build_stacked(
    sc: &mut Builder<Instant, UnixTime>,
    symbols: &[String],
    args: &CommonArgs,
) -> Stacked {
    let dir = &args.data_dir;
    let start = Some(args.data_start());
    // The source's window is half-open; the backtest's `end` is inclusive.
    let end = Some(args.end() + Duration::from_days(1));
    // Whole-market axis: the schema must cover every label its table carries.
    let universe = Schema::new(symbols.to_vec());

    let panel = |sc: &mut Builder<Instant, UnixTime>,
                 kind: &str,
                 columns: Vec<String>|
     -> (SignalPortHandle<1>, Box<[ArrayPortHandle<f64, 1>]>) {
        let source = panel::parquet(
            format!("{dir}/{kind}.parquet"),
            [("symbol".into(), Axis::Labeled(universe.clone()))],
            columns,
        )
        .with_time_range(start, end);
        sc.source(source)
    };

    let (daily_signals, prices) = panel(
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
    let (div_signals, divs) = panel(
        sc,
        "dividends",
        vec!["dividends.share".into(), "dividends.cash".into()],
    );
    let (_equity_signals, equity) = panel(
        sc,
        "equity_structures",
        vec!["shares.total".into(), "shares.circulating".into()],
    );
    let (_balance_signals, balance) = panel(
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
    );
    let (report_signals, income) = panel(
        sc,
        "income_statements",
        vec![
            "report_year".into(),                                      // 0
            "report_day_of_year".into(),                               // 1
            "income_statement.profit".into(),                          // 2 net profit
            "income_statement.profit.operating".into(),                // 3 operating profit
            "income_statement.profit.operating.income.revenue".into(), // 4 revenue
            "income_statement.profit.operating.expenses.costs".into(), // 5 operating cost (negative)
        ],
    );
    let (cf_report_signals, cashflow) = panel(
        sc,
        "cash_flow_statements",
        vec![
            "report_year".into(),                          // 0
            "report_day_of_year".into(),                   // 1
            "cash_flow_statement.change.operating".into(), // 2 operating cash flow
            "cash_flow_statement.change.investing".into(), // 3 investing cash flow
            "cash_flow_statement.change.financing".into(), // 4 financing cash flow
        ],
    );

    // Batch-level cadences: any stock has a row on this timestamp.
    let daily = sc.segment(any(), daily_signals);
    let reports = sc.segment(any(), report_signals);
    let cf_reports = sc.segment(any(), cf_report_signals);

    // Daily price fields, as the event face of the carried panel.
    let face = |sc: &mut Builder<Instant, UnixTime>, values| {
        sc.segment(event_face(), (daily_signals, values))
    };
    let close = face(sc, prices[0]);
    let volume = face(sc, prices[1]);
    let open = face(sc, prices[2]);
    let high = face(sc, prices[3]);
    let low = face(sc, prices[4]);
    let amount = face(sc, prices[5]);

    // Forward adjustment, whole cross-section at once: the close leg on the
    // daily signal, both dividend legs on the dividend panel's signal; a
    // per-stock event is that stock's signal element, and a NaN under a set
    // element is a missing field of the row that fired.
    let share_divs = divs[0];
    let cash_divs = divs[1];
    let (adjusts, adjusted_close) = sc.segment(
        forward_adjust(),
        ((daily_signals, close), (div_signals, share_divs, cash_divs)),
    );

    // Point-in-time carried fields come straight off the source — a cell with
    // no row on a timestamp keeps its last reported value.
    let total_shares = equity[0];
    let circ_shares = equity[1];

    // parent_equity = -(capital + reserves + parent_interests).
    let capital_reserves = sc.segment(elem::add(), (balance[0], balance[1]));
    let parent_sum = sc.segment(elem::add(), (capital_reserves, balance[2]));
    let parent_equity = sc.segment(elem::neg(), parent_sum);

    let total_assets = balance[3];
    let total_liab = balance[4];
    let current_assets = balance[5];
    let current_liab = balance[6];
    let cash = balance[7];
    let inventories = balance[8];
    let receivables = balance[9];

    // Annualized income / cash flows (YTD → annualize), once per flow: the
    // `[N]` report signal and the `[N]` calendar fields (cast to u16) gate and
    // date every element of the cross-section.
    let income_year = sc.segment(elem::as_(), income[0]);
    let income_doy = sc.segment(elem::as_(), income[1]);
    let income_ann = |sc: &mut Builder<Instant, UnixTime>, ytd| {
        sc.segment(annualize(), (report_signals, ytd, income_year, income_doy))
    };
    let net_profit_ann = income_ann(sc, income[2]);
    let operating_profit_ann = income_ann(sc, income[3]);
    let revenue_ann = income_ann(sc, income[4]);
    let operating_cost_ann = income_ann(sc, income[5]);

    let cf_year = sc.segment(elem::as_(), cashflow[0]);
    let cf_doy = sc.segment(elem::as_(), cashflow[1]);
    let net_operating_cashflow_ann = sc.segment(
        annualize(),
        (cf_report_signals, cashflow[2], cf_year, cf_doy),
    );

    Stacked {
        daily,
        daily_signals,
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
        div_signals,
        share_divs,
        cash_divs,
        total_shares,
        circ_shares,
        parent_equity,
        net_profit: net_profit_ann,
        operating_profit: operating_profit_ann,
        revenue: revenue_ann,
        operating_cost: operating_cost_ann,
        total_assets,
        total_liab,
        current_assets,
        current_liab,
        cash,
        inventories,
        receivables,
        net_operating_cashflow: net_operating_cashflow_ann,
        net_profit_events: net_profit_ann,
        operating_profit_events: operating_profit_ann,
        revenue_events: revenue_ann,
        operating_cost_events: operating_cost_ann,
        net_operating_cashflow_events: net_operating_cashflow_ann,
    }
}
