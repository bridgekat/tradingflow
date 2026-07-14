//! The stacked cross-sectional market panel: parquet sources → `[num_stocks]`
//! per-field panels, via one fused per-stock segment.

use tradingflow::graph::{Handle, RefViewPort, ViewPort};

use tradingflow::data::Duration;
use tradingflow::operators::{
    ArrayValue, annualize, apply, as_view, deref_array_view, forward_adjust, gate, multiply,
    ref_array_views, select, slice_view, split, stack, stack_sync,
};
use tradingflow::sources::{ParquetFinancialReportPanelSource, ParquetPanelSource};
use tradingflow::{Array, ArrayView, Scenario};

use super::AvH;
use super::args::CommonArgs;

/// Cross-sectional `(num_stocks,)` panels produced by [`build_stacked`].
///
/// The fundamental fields below are point-in-time (effective-date-aligned,
/// carried forward). Income / cash-flow flows are **annualized** (YTD →
/// `Annualize`); a trailing-twelve-month figure is a 365-day rolling mean of the
/// annualized series (see [`factors`](super::factors)). The parquet stores assets
/// debit-positive and liabilities / expense items credit-**negative** — the factor
/// formulas negate them where a positive magnitude is wanted.
pub struct Stacked {
    pub close: AvH,                  // unadjusted close (StackSync)
    pub volume: AvH,                 // 成交量 shares (StackSync)
    pub open: AvH,                   // 开盘价 unadjusted (StackSync)
    pub high: AvH,                   // 最高价 unadjusted (StackSync)
    pub low: AvH,                    // 最低价 unadjusted (StackSync)
    pub amount: AvH,                 // 成交额 turnover value (StackSync)
    pub adjusted_close: AvH,         // close * forward-adjust factor (StackSync)
    pub adjusts: AvH,                // forward-adjust factor (Stack)
    pub total_shares: AvH,           // (Stack)
    pub circ_shares: AvH,            // (Stack)
    pub parent_equity: AvH,          // 归母净资产, positive (Stack)
    pub net_profit: AvH,             // 净利润, annualized, positive
    pub operating_profit: AvH,       // 营业利润, annualized, positive
    pub revenue: AvH,                // 营业收入, annualized, positive
    pub operating_cost: AvH,         // 营业成本, annualized, NEGATIVE (a deduction)
    pub total_assets: AvH,           // 总资产, positive
    pub total_liab: AvH,             // 总负债, NEGATIVE (credit side)
    pub current_assets: AvH,         // 流动资产, positive
    pub current_liab: AvH,           // 流动负债, NEGATIVE (credit side)
    pub cash: AvH,                   // 货币资金, positive
    pub inventories: AvH,            // 存货, positive
    pub receivables: AvH,            // 应收票据及账款, positive
    pub net_operating_cashflow: AvH, // 经营现金流净额, annualized
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
pub fn build_stacked(sc: &mut Scenario, symbols: &[String], args: &CommonArgs) -> Stacked {
    let dir = &args.data_dir;
    let start = Some(args.data_start());
    let end = Some(args.end());
    let n = symbols.len();
    let universe: Vec<String> = symbols.to_vec();

    // Panel sources emit pure StackSync cross-sections (only each date's rows;
    // the carry-forward / NaN-fill is the downstream `Stack` / `StackSync`'s job).
    // Reports align on the **effective date** `max(report, notice)` — the
    // look-ahead-safe point-in-time a backtest may use them (`use_effective_date`).
    // Panel sources emit a whole-array `RefPort<Array<f64, 2>>`; bridge to the
    // view currency with `as_view` so the rank-2 panel feeds `Split`.
    let daily_panel = |sc: &mut Scenario,
                       kind: &str,
                       cols: Vec<String>|
     -> Handle<ViewPort<ArrayValue<f64, 2>>> {
        let s = ParquetPanelSource::new(format!("{dir}/{kind}.parquet"), cols, universe.clone())
            .with_time_range(start, end);
        let h = sc.add_source(s);
        sc.push(as_view(), h)
    };
    let report_panel = |sc: &mut Scenario,
                        kind: &str,
                        cols: Vec<String>,
                        with_report_date: bool|
     -> Handle<ViewPort<ArrayValue<f64, 2>>> {
        let s = ParquetFinancialReportPanelSource::new(
            format!("{dir}/{kind}.parquet"),
            cols,
            universe.clone(),
        )
        .with_report_date(with_report_date)
        .use_effective_date(Duration::ZERO)
        .with_time_range(start, end);
        let h = sc.add_source(s);
        sc.push(as_view(), h)
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

    // One `Split` per panel: the `1 → N` row fan-out as a single node each. The
    // rank-2 `[N, K]` panel splits along axis 0 into `N` rank-1 `[K]` row views.
    let prices_rows = sc.push(split(n), prices_panel);
    let div_rows = sc.push(split(n), div_panel);
    let equity_rows = sc.push(split(n), equity_panel);
    let balance_rows = sc.push(split(n), balance_panel);
    let income_rows = sc.push(split(n), income_panel);
    let cashflow_rows = sc.push(split(n), cashflow_panel);

    let mut close_v = Vec::with_capacity(n);
    let mut volume_v = Vec::with_capacity(n);
    let mut adj_close_v = Vec::with_capacity(n);
    let mut adjusts_v = Vec::with_capacity(n);
    let mut total_v = Vec::with_capacity(n);
    let mut circ_v = Vec::with_capacity(n);
    let mut peq_v = Vec::with_capacity(n);
    let mut income_v = Vec::with_capacity(n); // annualized [profit, operating, revenue, cost]
    let mut bal_v = Vec::with_capacity(n); // [assets, liab, current_assets, current_liab, cash]
    let mut cf_v = Vec::with_capacity(n); // annualized [operating, investing, financing]
    let mut px_v = Vec::with_capacity(n); // daily [open, high, low, amount]

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
        // [3]) are rank-1 `[K]` views. Each `Split` row is a by-reference
        // `RefViewPort<ArrayValue<f64, 1>>`; `DerefArrayView` re-derives it as the
        // by-value `ViewPort` the `Gate`/view operators consume.
        let seg = tradingflow::segment!(|prices_row: RefViewPort<ArrayValue<f64, 1>>,
                                       div_row: RefViewPort<ArrayValue<f64, 1>>,
                                       equity_row: RefViewPort<ArrayValue<f64, 1>>,
                                       balance_row: RefViewPort<ArrayValue<f64, 1>>,
                                       income_row: RefViewPort<ArrayValue<f64, 1>>,
                                       cashflow_row: RefViewPort<ArrayValue<f64, 1>>|
            -> (
            ViewPort<ArrayValue<f64, 0>>, // close
            ViewPort<ArrayValue<f64, 0>>, // volume
            ViewPort<ArrayValue<f64, 0>>, // adjusted_close
            ViewPort<ArrayValue<f64, 0>>, // adjusts
            ViewPort<ArrayValue<f64, 0>>, // total_shares
            ViewPort<ArrayValue<f64, 0>>, // circ_shares
            ViewPort<ArrayValue<f64, 0>>, // parent_equity
            ViewPort<ArrayValue<f64, 1>>, // income_ann [4]
            ViewPort<ArrayValue<f64, 1>>, // balance_extras [7]
            ViewPort<ArrayValue<f64, 1>>, // cf_ann [3]
            ViewPort<ArrayValue<f64, 1>>  // prices_extras [4]
        ) {
            // Each `Split` row is a by-reference `RefViewPort`; `DerefArrayView`
            // re-derives the by-value `ViewPort` the `Gate` consumes.
            let prices = gate(any_finite) @ deref_array_view() @ prices_row; // [close, volume, open, high, low, amount]
            let dividends = gate(any_finite) @ deref_array_view() @ div_row; // [share, cash]
            let equity = gate(any_finite) @ deref_array_view() @ equity_row; // [total, circulating]
            let balance = gate(any_finite) @ deref_array_view() @ balance_row; // [cap, res, parent, assets, liab, cur_a, cur_l, cash]
            let income = gate(any_finite) @ deref_array_view() @ income_row; // [year, doy, profit, operating, revenue, cost]
            let cashflow = gate(any_finite) @ deref_array_view() @ cashflow_row; // [year, doy, operating, investing, financing]
            // Terminal column picks stay zero-copy views (`SliceView`) into the
            // retaining `Gate`'s stable storage; squeezing one index drops the
            // axis (rank-1 row → rank-0 scalar). `close` feeds `ForwardAdjust` /
            // `multiply` and materializes via the owned `Select`.
            let close = select(vec![0], 0, true) @ prices;
            let volume = slice_view(vec![1], 0, true) @ prices;
            // [open, high, low, amount] as a contiguous rank-1 view of cols 2..6.
            let prices_extras = slice_view(vec![2, 3, 4, 5], 0, false) @ prices;
            let adjusts =
                forward_adjust().with_output_prices(false) @ (close, dividends);
            let adjusted_close = multiply() @ (close, adjusts);
            let total_shares = slice_view(vec![0], 0, true) @ equity;
            let circ_shares = slice_view(vec![1], 0, true) @ equity;
            // parent_equity = -(capital + reserves + parent_interests) (cols 0..3).
            let parent_equity = apply::<ViewPort<ArrayValue<f64, 1>>, f64, 0, _>(
                |a: ArrayView<f64, 1>| Array::scalar(-a.to_contiguous()[..3].iter().sum::<f64>()),
            ) @ balance;
            // Annualized income / cash flows (YTD → Annualize) and the balance
            // stocks [assets, liab, current_assets, current_liab, cash, inv, rec]
            // as a contiguous rank-1 view of cols 3..10.
            let income_ann = annualize() @ income;
            let cf_ann = annualize() @ cashflow;
            let balance_extras = slice_view(vec![3, 4, 5, 6, 7, 8, 9], 0, false) @ balance;
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
        ) = sc.push(
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

        close_v.push(close);
        volume_v.push(volume);
        adj_close_v.push(adjusted_close);
        adjusts_v.push(adjusts);
        total_v.push(total_shares);
        circ_v.push(circ_shares);
        peq_v.push(parent_equity);
        income_v.push(income_ann);
        bal_v.push(balance_extras);
        cf_v.push(cf_ann);
        px_v.push(prices_extras);
    }

    // The per-stock segment outputs are by-value `ViewPort` handles; the carry
    // joins (`Stack`/`StackSync`) consume by-reference `RefViewPort` rows, so
    // bridge each vector through `RefArrayView` (the value→reference adapter — the
    // engine has none inside `segment!`) before stacking.
    let income_refs = ref_array_views(sc, &income_v);
    let bal_refs = ref_array_views(sc, &bal_v);
    let cf_refs = ref_array_views(sc, &cf_v);
    let px_refs = ref_array_views(sc, &px_v);
    let close_refs = ref_array_views(sc, &close_v);
    let volume_refs = ref_array_views(sc, &volume_v);
    let adj_close_refs = ref_array_views(sc, &adj_close_v);
    let adjusts_refs = ref_array_views(sc, &adjusts_v);
    let total_refs = ref_array_views(sc, &total_v);
    let circ_refs = ref_array_views(sc, &circ_v);
    let peq_refs = ref_array_views(sc, &peq_v);

    // Cross-sectional grouped panels (rank-1 `[K]` rows → rank-2 `[N, K]`); a
    // squeezing column `Select` (axis 1) recovers each field as rank-1 `[N]`.
    let income_xs = sc.push(stack::<f64, 1, 2>(0), &income_refs[..]); // (N, 4)
    let balance_xs = sc.push(stack::<f64, 1, 2>(0), &bal_refs[..]); // (N, 7)
    let cf_xs = sc.push(stack::<f64, 1, 2>(0), &cf_refs[..]); // (N, 3)
    let px_xs = sc.push(stack_sync::<f64, 1, 2>(0), &px_refs[..]); // (N, 4) [open, high, low, amount]

    Stacked {
        // Per-stock scalars (rank-0) → rank-1 `[N]` cross-sections.
        close: sc.push(stack_sync(0), &close_refs[..]),
        volume: sc.push(stack_sync(0), &volume_refs[..]),
        adjusted_close: sc.push(stack_sync(0), &adj_close_refs[..]),
        adjusts: sc.push(stack(0), &adjusts_refs[..]),
        total_shares: sc.push(stack(0), &total_refs[..]),
        circ_shares: sc.push(stack(0), &circ_refs[..]),
        parent_equity: sc.push(stack(0), &peq_refs[..]),
        net_profit: sc.push(select(vec![0], 1, true), income_xs),
        operating_profit: sc.push(select(vec![1], 1, true), income_xs),
        revenue: sc.push(select(vec![2], 1, true), income_xs),
        operating_cost: sc.push(select(vec![3], 1, true), income_xs),
        total_assets: sc.push(select(vec![0], 1, true), balance_xs),
        total_liab: sc.push(select(vec![1], 1, true), balance_xs),
        current_assets: sc.push(select(vec![2], 1, true), balance_xs),
        current_liab: sc.push(select(vec![3], 1, true), balance_xs),
        cash: sc.push(select(vec![4], 1, true), balance_xs),
        inventories: sc.push(select(vec![5], 1, true), balance_xs),
        receivables: sc.push(select(vec![6], 1, true), balance_xs),
        net_operating_cashflow: sc.push(select(vec![0], 1, true), cf_xs),
        open: sc.push(select(vec![0], 1, true), px_xs),
        high: sc.push(select(vec![1], 1, true), px_xs),
        low: sc.push(select(vec![2], 1, true), px_xs),
        amount: sc.push(select(vec![3], 1, true), px_xs),
    }
}
