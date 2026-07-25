//! The stacked cross-sectional market panel: parquet sources → `[num_stocks]`
//! per-field panels, via one fused per-stock segment.

use tradingflow::clock::UnixClock;
use tradingflow::data::{Array, ArrayView, Duration, Instant};
use tradingflow::graph::Builder;
use tradingflow::operators::{
    array::{array_map, select, select_at, stack, unstack},
    elem, event,
    metrics::*,
    stats::*,
    stocks::*,
    traders::*,
};
use tradingflow::ports::{ArrayPort, ArrayPortHandle};
use tradingflow::sources::panel::*;

use super::args::CommonArgs;

/// Cross-sectional `(num_stocks,)` panels produced by [`build_stacked`].
///
/// The fundamental fields below are point-in-time (effective-date-aligned,
/// carried forward). Income / cash-flow flows are **annualized** (YTD →
/// `Annualize`); a trailing-twelve-month figure is a 365-day rolling mean of the
/// annualized series (see [`factors`](super::features)). The parquet stores assets
/// debit-positive and liabilities / expense items credit-**negative** — the factor
/// formulas negate them where a positive magnitude is wanted.
pub struct Stacked {
    pub close: ArrayPortHandle<f64, 1>, // unadjusted close (StackSync)
    pub volume: ArrayPortHandle<f64, 1>, // 成交量 shares (StackSync)
    pub open: ArrayPortHandle<f64, 1>,  // 开盘价 unadjusted (StackSync)
    pub high: ArrayPortHandle<f64, 1>,  // 最高价 unadjusted (StackSync)
    pub low: ArrayPortHandle<f64, 1>,   // 最低价 unadjusted (StackSync)
    pub amount: ArrayPortHandle<f64, 1>, // 成交额 turnover value (StackSync)
    pub adjusted_close: ArrayPortHandle<f64, 1>, // close * forward-adjust factor (StackSync)
    pub adjusts: ArrayPortHandle<f64, 1>, // forward-adjust factor (Stack)
    pub total_shares: ArrayPortHandle<f64, 1>, // (Stack)
    pub circ_shares: ArrayPortHandle<f64, 1>, // (Stack)
    pub parent_equity: ArrayPortHandle<f64, 1>, // 归母净资产, positive (Stack)
    pub net_profit: ArrayPortHandle<f64, 1>, // 净利润, annualized, positive
    pub operating_profit: ArrayPortHandle<f64, 1>, // 营业利润, annualized, positive
    pub revenue: ArrayPortHandle<f64, 1>, // 营业收入, annualized, positive
    pub operating_cost: ArrayPortHandle<f64, 1>, // 营业成本, annualized, NEGATIVE (a deduction)
    pub total_assets: ArrayPortHandle<f64, 1>, // 总资产, positive
    pub total_liab: ArrayPortHandle<f64, 1>, // 总负债, NEGATIVE (credit side)
    pub current_assets: ArrayPortHandle<f64, 1>, // 流动资产, positive
    pub current_liab: ArrayPortHandle<f64, 1>, // 流动负债, NEGATIVE (credit side)
    pub cash: ArrayPortHandle<f64, 1>,  // 货币资金, positive
    pub inventories: ArrayPortHandle<f64, 1>, // 存货, positive
    pub receivables: ArrayPortHandle<f64, 1>, // 应收票据及账款, positive
    pub net_operating_cashflow: ArrayPortHandle<f64, 1>, // 经营现金流净额, annualized
}

/// Predicate for the per-stock `Gate`: the row has ≥1 finite entry, i.e.
/// the stock actually has data this tick (vs. an all-NaN "no data" cross-section
/// the panel emits on a date where other stocks ticked but this one didn't).
fn any_finite(a: ArrayView<'_, f64, 1>) -> bool {
    a.to_contiguous().iter().any(|x| x.is_finite())
}

/// Load the consolidated long-format parquet panels and stack into the
/// cross-sectional panel. One [`ParquetPanelSource`] / [`ParquetFinancialReportPanelSource`] per
/// data kind (one sequential scan each) replaces the per-symbol CSV fan-in.
/// Each panel fans out through a single [`Split`] node (`1 → N` rows), every
/// stock's whole transform chain (NaN `Gate` + column `Select`s +
/// `ForwardAdjust` + `Annualize` + ...) is **fused into one segment** via
/// `tradingflow::segment!` — one scheduling unit per stock instead of ~19 nodes,
/// with identical per-operator notify/cutoff semantics (each sub-operator keeps
/// its own gate inside the fused node) — and `StackSync` (NaN-fill non-trading
/// slots) / `Stack` (carry last-known) recombine into `[N]` panels. The financial
/// reports align on the look-ahead-safe effective date `max(report, notice)`
/// (`use_effective_date`, zero fallback).
pub fn build_stacked(
    sc: &mut Builder<Instant, UnixClock>,
    symbols: &[String],
    args: &CommonArgs,
) -> Stacked {
    let dir = &args.data_dir;
    let start = Some(args.data_start());
    let end = Some(args.end());
    let n = symbols.len();
    let universe: Vec<String> = symbols.to_vec();

    // Panel sources emit pure StackSync cross-sections (only each date's rows;
    // the carry-forward / NaN-fill is the downstream `Stack` / `StackSync`'s job).
    // Reports align on the **effective date** `max(report, notice)` — the
    // look-ahead-safe point-in-time a backtest may use them (`use_effective_date`).
    // A panel source cell lends its `[N, K]` panel as an `ArrayPort<f64, 2>`
    // view edge, which feeds `Split` directly.
    let daily_panel = |sc: &mut Builder<Instant, UnixClock>,
                       kind: &str,
                       cols: Vec<String>|
     -> ArrayPortHandle<f64, 2> {
        let s = parquet_panel_source(format!("{dir}/{kind}.parquet"), cols, universe.clone())
            .with_time_range(start, end);
        sc.source(s)
    };
    let report_panel = |sc: &mut Builder<Instant, UnixClock>,
                        kind: &str,
                        cols: Vec<String>,
                        with_report_date: bool|
     -> ArrayPortHandle<f64, 2> {
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

    let prices_panel = daily_panel(
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
    let div_panel = daily_panel(
        sc,
        "dividends",
        vec!["dividends.share".into(), "dividends.cash".into()],
    );
    let equity_panel = daily_panel(
        sc,
        "equity_structures",
        vec!["shares.total".into(), "shares.circulating".into()],
    );
    let balance_panel = report_panel(
        sc,
        "balance_sheets",
        vec![
            // [0..3]: parent-equity components (summed and negated in-segment).
            "balance_sheet.equity.capital".into(),
            "balance_sheet.equity.reserves".into(),
            "balance_sheet.equity.parent_interests".into(),
            // [3..8]: point-in-time balance stocks for the fundamental factors
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
    let income_panel = report_panel(
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
    let cashflow_panel = report_panel(
        sc,
        "cash_flow_statements",
        vec![
            "cash_flow_statement.change.operating".into(), // 0 operating cash flow
            "cash_flow_statement.change.investing".into(), // 1 investing cash flow
            "cash_flow_statement.change.financing".into(), // 2 financing cash flow
        ],
        true,
    );

    // One `unstack` per panel: the `1 → N` row fan-out as a single node each. The
    // rank-2 `[N, K]` panel unstacks along axis 0 into `N` rank-1 `[K]` row views.
    let prices_rows = sc.segment(unstack(0), prices_panel);
    let div_rows = sc.segment(unstack(0), div_panel);
    let equity_rows = sc.segment(unstack(0), equity_panel);
    let balance_rows = sc.segment(unstack(0), balance_panel);
    let income_rows = sc.segment(unstack(0), income_panel);
    let cashflow_rows = sc.segment(unstack(0), cashflow_panel);

    let mut closes = Vec::with_capacity(n);
    let mut volumes = Vec::with_capacity(n);
    let mut adjusted_closes = Vec::with_capacity(n);
    let mut adjust_factors = Vec::with_capacity(n);
    let mut totals = Vec::with_capacity(n);
    let mut circs = Vec::with_capacity(n);
    let mut parent_equities = Vec::with_capacity(n);
    let mut incomes = Vec::with_capacity(n); // annualized [profit, operating, revenue, cost]
    let mut balances = Vec::with_capacity(n); // [assets, liab, current_assets, current_liab, cash]
    let mut cashflows = Vec::with_capacity(n); // annualized [operating, investing, financing]
    let mut price_extras = Vec::with_capacity(n); // daily [open, high, low, amount]

    for i in 0..n {
        // The whole per-stock transform chain, fused into ONE graph node via
        // `segment!`. Inputs are the stock's **zero-copy row views** of the six
        // panels (from `Split`); the leading `Gate(any_finite)` per panel drops
        // the all-NaN "no data" ticks, recovering that stock's real event stream.
        // `Gate` retains the last passed row in its own state and re-presents a
        // view of it whenever it gates out, so it honours the no-notify⟹unchanged
        // contract (its un-notified output is the last passed value, never the
        // gated-out row). Views materialize at the computing/selecting operators,
        // whose owned outputs retain last values for the carry-style `Stack`
        // joins; every sub-operator keeps its own notify gate inside the fused
        // node, so the cutoff and `ForwardAdjust`'s price/dividend message-passing
        // are unchanged. The segment is identical for every stock (the stock index
        // lives only in the input wiring), so it monomorphizes once.
        //
        // The many fundamental columns are emitted as GROUPED arrays
        // (`income_ann` [4], `balance_extras` [7], `cf_ann` [3]) rather than one
        // output each — both to keep the segment within the tuple-arity limit
        // and to move less. The `Stack` joins below build the `(N, K)` panels;
        // a squeezing column `Select` recovers each field as `(N,)`.
        // Per-stock scalar fields are rank-0 `[]` views; grouped fields
        // (`prices_extras` [4], `income_ann` [4], `balance_extras` [7], `cf_ann`
        // [3]) are rank-1 `[K]` views. Each `Split` row is an ordinary by-value
        // `ArrayPort<f64, 1>`, consumed directly by the `Gate`/view operators.
        let seg = tradingflow::segment!(|prices_row: ArrayPort<f64, 1>,
                                       div_row: ArrayPort<f64, 1>,
                                       equity_row: ArrayPort<f64, 1>,
                                       balance_row: ArrayPort<f64, 1>,
                                       income_row: ArrayPort<f64, 1>,
                                       cashflow_row: ArrayPort<f64, 1>|
            -> (
            ArrayPort<f64, 0>, // close
            ArrayPort<f64, 0>, // volume
            ArrayPort<f64, 0>, // adjusted_close
            ArrayPort<f64, 0>, // adjusts
            ArrayPort<f64, 0>, // total_shares
            ArrayPort<f64, 0>, // circ_shares
            ArrayPort<f64, 0>, // parent_equity
            ArrayPort<f64, 1>, // income_ann [4]
            ArrayPort<f64, 1>, // balance_extras [7]
            ArrayPort<f64, 1>, // cf_ann [3]
            ArrayPort<f64, 1>  // prices_extras [4]
        ) {
            let prices = event::filter(any_finite) @ prices_row; // [close, volume, open, high, low, amount]
            let dividends = event::filter(any_finite) @ div_row; // [share, cash]
            let equity = event::filter(any_finite) @ equity_row; // [total, circulating]
            let balance = event::filter(any_finite) @ balance_row; // [cap, res, parent, assets, liab, cur_a, cur_l, cash]
            let income = event::filter(any_finite) @ income_row; // [year, doy, profit, operating, revenue, cost]
            let cashflow = event::filter(any_finite) @ cashflow_row; // [year, doy, operating, investing, financing]
            // Terminal column picks `Select` out of the retaining `Gate`'s stable
            // storage; squeezing one index drops the
            // axis (rank-1 row → rank-0 scalar). `close` feeds `ForwardAdjust` /
            // `multiply` and materializes via the owned `Select`.
            let close = select_at(0, 0) @ prices;
            let volume = select_at(1, 0) @ prices;
            // [open, high, low, amount] as a contiguous rank-1 view of cols 2..6.
            let prices_extras = select(vec![2, 3, 4, 5], 0) @ prices;
            let adjusts =
                forward_adjust().with_output_prices(false) @ (close, dividends);
            let adjusted_close = elem::mul() @ (close, adjusts);
            let total_shares = select_at(0, 0) @ equity;
            let circ_shares = select_at(1, 0) @ equity;
            // parent_equity = -(capital + reserves + parent_interests) (cols 0..3).
            let parent_equity = array_map(|a: ArrayView<f64, 1>| {
                Array::<f64, 0>::scalar(-a.to_contiguous()[..3].iter().sum::<f64>())
            }) @ balance;
            // Annualized income / cash flows (YTD → Annualize) and the balance
            // stocks [assets, liab, current_assets, current_liab, cash, inv, rec]
            // as a contiguous rank-1 view of cols 3..10.
            let income_ann = annualize() @ income;
            let cf_ann = annualize() @ cashflow;
            let balance_extras = select(vec![3, 4, 5, 6, 7, 8, 9], 0) @ balance;
            (
                close,
                volume,
                adjusted_close,
                adjusts,
                total_shares,
                circ_shares,
                parent_equity,
                income_ann,
                balance_extras,
                cf_ann,
                prices_extras,
            )
        });
        let (
            close,
            volume,
            adjusted_close,
            adjusts,
            total_shares,
            circ_shares,
            parent_equity,
            income_ann,
            balance_extras,
            cf_ann,
            prices_extras,
        ) = sc.segment(
            seg,
            (
                prices_rows[i],
                div_rows[i],
                equity_rows[i],
                balance_rows[i],
                income_rows[i],
                cashflow_rows[i],
            ),
        );

        closes.push(close);
        volumes.push(volume);
        adjusted_closes.push(adjusted_close);
        adjust_factors.push(adjusts);
        totals.push(total_shares);
        circs.push(circ_shares);
        parent_equities.push(parent_equity);
        incomes.push(income_ann);
        balances.push(balance_extras);
        cashflows.push(cf_ann);
        price_extras.push(prices_extras);
    }

    // The per-stock segment outputs are by-value `ArrayPort` handles; the carry
    // joins (`Stack`/`StackSync`) take a slice of them directly.

    // Cross-sectional grouped panels (rank-1 `[K]` rows → rank-2 `[N, K]`); a
    // squeezing column `Select` (axis 1) recovers each field as rank-1 `[N]`.
    let income_xs = sc.segment(stack::<f64, 1, 2>(0), &incomes[..]); // (N, 4)
    let balance_xs = sc.segment(stack::<f64, 1, 2>(0), &balances[..]); // (N, 7)
    let cf_xs = sc.segment(stack::<f64, 1, 2>(0), &cashflows[..]); // (N, 3)
    let px_xs = sc.segment(event::stack_sync::<f64, 1, 2>(0), &price_extras[..]); // (N, 4) [open, high, low, amount]

    Stacked {
        // Per-stock scalars (rank-0) → rank-1 `[N]` cross-sections.
        close: sc.segment(event::stack_sync(0), &closes[..]),
        volume: sc.segment(event::stack_sync(0), &volumes[..]),
        adjusted_close: sc.segment(event::stack_sync(0), &adjusted_closes[..]),
        adjusts: sc.segment(stack(0), &adjust_factors[..]),
        total_shares: sc.segment(stack(0), &totals[..]),
        circ_shares: sc.segment(stack(0), &circs[..]),
        parent_equity: sc.segment(stack(0), &parent_equities[..]),
        net_profit: sc.segment(select_at(0, 1), income_xs),
        operating_profit: sc.segment(select_at(1, 1), income_xs),
        revenue: sc.segment(select_at(2, 1), income_xs),
        operating_cost: sc.segment(select_at(3, 1), income_xs),
        total_assets: sc.segment(select_at(0, 1), balance_xs),
        total_liab: sc.segment(select_at(1, 1), balance_xs),
        current_assets: sc.segment(select_at(2, 1), balance_xs),
        current_liab: sc.segment(select_at(3, 1), balance_xs),
        cash: sc.segment(select_at(4, 1), balance_xs),
        inventories: sc.segment(select_at(5, 1), balance_xs),
        receivables: sc.segment(select_at(6, 1), balance_xs),
        net_operating_cashflow: sc.segment(select_at(0, 1), cf_xs),
        open: sc.segment(select_at(0, 1), px_xs),
        high: sc.segment(select_at(1, 1), px_xs),
        low: sc.segment(select_at(2, 1), px_xs),
        amount: sc.segment(select_at(3, 1), px_xs),
    }
}
