//! Shared scaffolding for the A-shares cross-sectional examples.
//!
//! Pulled in by each strategy/plot example via `#[path = "common/mod.rs"] mod
//! common;`. It uses no Python operators (builds with the default feature set):
//! it builds the data pipeline (columnar parquet panel + financial-report sources, the stacked
//! cross-sectional panel, the canonical 7-factor feature set, the cap-weighted
//! universe, the log-return target, and price limits) entirely from native
//! operators. Examples add their `flowops` predictor/portfolio
//! operators (and any cvxpy ones) on top.
//!
//! Differences from the Python original, all intentional:
//!
//! * Calendar/timezone handling is the caller's job in Rust (the core has no
//!   tz database), so rebalance dates are generated here from a plain
//!   `Duration::from_days` step, and date strings are turned into `Instant`s
//!   via [`instant_from_days`] — which reproduces `CsvSource`'s default
//!   UTC-midnight parse exactly (`utc_to_tai(days * 86_400e9)`).
//! * The Python lambdas (`calculate_index_weights`, cross-sectional demean,
//!   price-limit rounding) become native Rust closures fed to `Map`/`Apply`.
//! * All symbols in `symbol_list.csv` are loaded (matching the original);
//!   selection of the top-`index_size` constituents happens in-graph.

#![allow(dead_code)] // not every example uses every helper

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;

use flowgraph::typed::{Handle, RefPort, RefViewPort, ViewPort};

use tradingflow::data::Duration;
use tradingflow::operators::{
    AnnualizeView, Apply, ArrayValue, Divide, ForwardAdjustViewDiv, Gate, Lag, Log, Map, Multiply,
    Percentile, Resample, RollingMean, RollingVariance, Select, SelectView, SliceView, Split, Sqrt,
    Stack, StackSync, StackSyncView, StackView, Subtract, Winsorize,
};
use tradingflow::{Scenario, Session};
use tradingflow::sources::{ParquetPanelSource, ReportPanelSource};
use tradingflow::{Array, ArraySlice, Instant, Retention, Series, utc_to_tai};

/// Rows kept beyond a consumer's exact count look-back, absorbing the
/// amortized-compaction slack and any off-by-one at the window boundary.
pub const RETAIN_MARGIN: usize = 8;

pub mod backtest;
pub mod factors;
pub mod pv_factors;
pub mod universe;

// ===========================================================================
// Calendar / Instant helpers
// ===========================================================================

/// Days since 1970-01-01 for a proleptic-Gregorian date (Howard Hinnant's
/// algorithm). Valid for any date; result may be negative before the epoch.
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// `Instant` at UTC midnight of the given days-since-epoch count, matching
/// `CsvSource`'s default (`is_utc = true`) date parse exactly.
pub fn instant_from_days(days: i64) -> Instant {
    Instant::from_nanos(utc_to_tai(days * 86_400 * 1_000_000_000))
}

/// Parse a `YYYY-MM-DD` string into days since 1970-01-01 (a `clap` value parser,
/// so a malformed date yields a usage error rather than a panic).
pub fn parse_date_days(s: &str) -> Result<i64, String> {
    let parts: Vec<&str> = s.split('-').collect();
    let err = || format!("invalid date `{s}` (expected YYYY-MM-DD)");
    if parts.len() != 3 {
        return Err(err());
    }
    let y: i64 = parts[0].parse().map_err(|_| err())?;
    let m: i64 = parts[1].parse().map_err(|_| err())?;
    let d: i64 = parts[2].parse().map_err(|_| err())?;
    Ok(days_from_civil(y, m, d))
}

/// Civil date `(year, month, day)` from days-since-1970 — the inverse of
/// [`days_from_civil`] (Howard Hinnant's algorithm).
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `YYYY-MM-DD` for an event [`Instant`]. The event timestamps are UTC midnight,
/// so the ~37 s TAI leap offset never crosses a day boundary — a plain
/// TAI-nanos→days floor is exact for display.
fn date_str(ts: Instant) -> String {
    let (y, m, d) = civil_from_days(ts.as_nanos().div_euclid(86_400 * 1_000_000_000));
    format!("{y:04}-{m:02}-{d:02}")
}

// ===========================================================================
// Progress bar (tqdm-like)
// ===========================================================================

/// A `tqdm`-style progress callback (backed by [`indicatif`]) for
/// [`Session::run`](tradingflow::Session::run)'s `on_flush`.
///
/// Progress is measured in **long-table rows**: the panel sources emit one event
/// per narrow row, so the engine's `on_flush` count *is* the row count (no shared
/// counter needed). `total` is
/// [`estimated_event_count`](tradingflow::Session::estimated_event_count) in
/// the same row unit; `Some(n)` → a bounded bar (percent / rate / ETA), else a
/// spinner. `{per_sec}` is therefore rows/s. `begin` sets `{prefix}` (warm-up
/// before it, running after); `{msg}` is the current event date. The bar uses a
/// terminal-width `{wide_bar}` with Unicode sub-cell fill, and finalises itself
/// when the callback drops at the end of `run`:
/// ```ignore
/// let total = session.estimated_event_count();
/// session.run(common::progress(total, args.begin())).await;
/// eprintln!(); // move past the finished bar line before printing results
/// ```
pub fn progress(total: Option<usize>, begin: Instant) -> impl FnMut(Instant, usize) {
    use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

    // Finish (leave) the bar when the callback is dropped — i.e. when `run`
    // returns — so the final state persists without the caller managing it.
    struct FinishOnDrop(ProgressBar);
    impl Drop for FinishOnDrop {
        fn drop(&mut self) {
            self.0.finish();
        }
    }

    let pb = match total {
        Some(t) if t > 0 => {
            let pb = ProgressBar::new(t as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "{prefix} |{wide_bar}| {pos}/{len} [{elapsed}<{eta}, {per_sec}]",
                )
                .unwrap()
                .with_key(
                    "per_sec",
                    |s: &indicatif::ProgressState, w: &mut dyn std::fmt::Write| {
                        write!(w, "{:.0} events/s", s.per_sec()).unwrap()
                    },
                )
                .progress_chars("█▉▊▋▌▍▎▏ "),
            );
            pb
        }
        _ => {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template("{prefix} {spinner} {pos} [{elapsed}, {per_sec}]")
                    .unwrap()
                    .with_key(
                        "per_sec",
                        |s: &indicatif::ProgressState, w: &mut dyn std::fmt::Write| {
                            write!(w, "{:.0} events/s", s.per_sec()).unwrap()
                        },
                    ),
            );
            pb
        }
    };
    // Cap redraws at ~20 fps regardless of how often the callback fires.
    pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(20));

    let _begin_ns = begin.as_nanos();
    let guard = FinishOnDrop(pb);
    move |ts: Instant, processed: usize| {
        let pb = &guard.0;
        let rows = processed as u64;
        // Grow the length if the estimate undershot (keeps the percentage sane).
        if let Some(len) = pb.length()
            && rows > len
        {
            pb.set_length(rows);
        }
        pb.set_position(rows);
        pb.set_prefix(date_str(ts));
    }
}

// ===========================================================================
// CLI args
// ===========================================================================

/// CLI args shared by every cross-sectional A-shares example. Embed it in an
/// example's own `#[derive(Parser)]` with `#[command(flatten)]`, so each example
/// declares exactly the extra args it needs (e.g. `--window` for the
/// feature-based ones, a `SYMBOL` positional for the single-stock plots) and
/// gets its own `--help`. All of these are **required** except `--sample-begin`
/// and `--threads` — there are no hidden "magic" defaults for the universe size,
/// rebalance cadence, or backtest window.
#[derive(clap::Args)]
pub struct CommonArgs {
    /// Data directory: the `examples/data` parquet tables + `symbol_list.csv`.
    #[arg(long)]
    pub data_dir: String,

    /// Backtest start date, e.g. 2022-01-01.
    #[arg(short = 'b', long = "begin", value_name = "DATE", value_parser = parse_date_days)]
    pub begin_days: i64,

    /// Backtest end date, e.g. 2024-12-31.
    #[arg(short = 'e', long = "end", value_name = "DATE", value_parser = parse_date_days)]
    pub end_days: i64,

    /// Number of stocks in the cap-weighted universe.
    #[arg(long)]
    pub index_size: usize,

    /// Rebalance every N calendar days.
    #[arg(long)]
    pub rebalance_days: i64,

    /// Warm-up sampling start; if omitted, defaults to `begin` − 400 days (enough
    /// to populate the 365-day TTM and the rolling feature windows).
    #[arg(long = "sample-begin", value_name = "DATE", value_parser = parse_date_days)]
    pub sample_begin: Option<i64>,

    /// Worker threads for the flowgraph `Pool` (0 = serial). `> 0` lets
    /// independent solve-bound operators (e.g. one cvxpy portfolio per
    /// risk-aversion) overlap via GIL release.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
}

impl CommonArgs {
    pub fn begin(&self) -> Instant {
        instant_from_days(self.begin_days)
    }
    pub fn end(&self) -> Instant {
        instant_from_days(self.end_days)
    }
    pub fn data_start(&self) -> Instant {
        let days = match self.sample_begin {
            Some(s) => s.min(self.begin_days),
            None => self.begin_days - 400,
        };
        instant_from_days(days)
    }

    /// Rebalance trigger instants: every `rebalance_days` calendar days from
    /// `begin` through `end` inclusive (mirrors the Python `np.arange`).
    pub fn rebalance_instants(&self) -> Vec<Instant> {
        let mut out = Vec::new();
        let mut d = self.begin_days;
        while d <= self.end_days {
            out.push(instant_from_days(d));
            d += self.rebalance_days;
        }
        out
    }
}

/// Load all stock symbols from `<data_dir>/symbol_list.csv` (the `symbol`
/// column), in file order.
pub fn load_symbols(data_dir: &str) -> Vec<String> {
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

// ===========================================================================
// Stacked cross-sectional panel
// ===========================================================================

/// Cross-sectional `(num_stocks,)` panels produced by [`build_stacked`].
///
/// The fundamental fields below are point-in-time (effective-date-aligned,
/// carried forward). Income / cash-flow flows are **annualized** (YTD →
/// `Annualize`); a trailing-twelve-month figure is a 365-day rolling mean of the
/// annualized series (see [`factors`]). The parquet stores assets debit-positive
/// and liabilities / expense items credit-**negative** — the factor formulas
/// negate them where a positive magnitude is wanted.
pub struct Stacked {
    pub close: Handle<RefPort<Array<f64>>>,          // unadjusted close (StackSync)
    pub volume: Handle<RefPort<Array<f64>>>,         // 成交量 shares (StackSync)
    pub open: Handle<RefPort<Array<f64>>>,           // 开盘价 unadjusted (StackSync)
    pub high: Handle<RefPort<Array<f64>>>,           // 最高价 unadjusted (StackSync)
    pub low: Handle<RefPort<Array<f64>>>,            // 最低价 unadjusted (StackSync)
    pub amount: Handle<RefPort<Array<f64>>>,         // 成交额 turnover value (StackSync)
    pub adjusted_close: Handle<RefPort<Array<f64>>>, // close * forward-adjust factor (StackSync)
    pub adjusts: Handle<RefPort<Array<f64>>>,        // forward-adjust factor (Stack)
    pub total_shares: Handle<RefPort<Array<f64>>>,   // (Stack)
    pub circ_shares: Handle<RefPort<Array<f64>>>,    // (Stack)
    pub parent_equity: Handle<RefPort<Array<f64>>>,  // 归母净资产, positive (Stack)
    pub net_profit: Handle<RefPort<Array<f64>>>,     // 净利润, annualized, positive
    pub operating_profit: Handle<RefPort<Array<f64>>>, // 营业利润, annualized, positive
    pub revenue: Handle<RefPort<Array<f64>>>,        // 营业收入, annualized, positive
    pub operating_cost: Handle<RefPort<Array<f64>>>, // 营业成本, annualized, NEGATIVE (a deduction)
    pub total_assets: Handle<RefPort<Array<f64>>>,   // 总资产, positive
    pub total_liab: Handle<RefPort<Array<f64>>>,     // 总负债, NEGATIVE (credit side)
    pub current_assets: Handle<RefPort<Array<f64>>>, // 流动资产, positive
    pub current_liab: Handle<RefPort<Array<f64>>>,   // 流动负债, NEGATIVE (credit side)
    pub cash: Handle<RefPort<Array<f64>>>,           // 货币资金, positive
    pub inventories: Handle<RefPort<Array<f64>>>,    // 存货, positive
    pub receivables: Handle<RefPort<Array<f64>>>,    // 应收票据及账款, positive
    pub net_operating_cashflow: Handle<RefPort<Array<f64>>>, // 经营现金流净额, annualized
}

/// Predicate for the per-stock `FilterView`: the row has ≥1 finite entry, i.e.
/// the stock actually has data this tick (vs. an all-NaN "no data" cross-section
/// the panel emits on a date where other stocks ticked but this one didn't).
fn any_finite(a: ArraySlice<'_, f64>) -> bool {
    a.as_slice().iter().any(|x| x.is_finite())
}

/// An all-NaN `Array<f64>` of `shape` — the correct "no data yet" initial cell
/// value for a panel source. The per-row sources only `write` rows that have an
/// event, so an unwritten row must read as NaN (not `0.0`) for the per-stock
/// [`pick`] `Filter` to drop it before the stock's first observation.
pub fn nan_array(shape: &[usize]) -> Array<f64> {
    Array::from_vec(shape, vec![f64::NAN; shape.iter().product()])
}

/// Load the consolidated long-format parquet panels and stack into the
/// cross-sectional panel. One [`ParquetPanelSource`] / [`ReportPanelSource`] per
/// data kind (one sequential scan each) replaces the per-symbol CSV fan-in.
/// Each panel fans out through a single [`Split`] node (`1 → N` rows), every
/// stock's whole transform chain (NaN `Filter` + column `Select`s +
/// `ForwardAdjust` + `Annualize` + ...) is **fused into one segment** via the
/// `flowgraph::segment!` arrow notation — one scheduling unit per stock instead
/// of ~19 nodes, with identical per-operator notify/cutoff semantics (each
/// sub-operator keeps its own gate inside the fused node) — and `StackSync`
/// (NaN-fill non-trading slots) / `Stack` (carry last-known) recombine into
/// `[N]` panels. The financial reports align on the look-ahead-safe effective
/// date `max(report, notice)` (`use_effective_date`, zero fallback).
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
    let daily_panel = |sc: &mut Scenario, kind: &str, cols: Vec<String>| -> Handle<RefPort<Array<f64>>> {
        let s = ParquetPanelSource::new(format!("{dir}/{kind}.parquet"), cols, universe.clone())
            .with_time_range(start, end);
        let init = nan_array(&s.out_shape());
        sc.add_source(s, init)
    };
    let report_panel = |sc: &mut Scenario,
                        kind: &str,
                        cols: Vec<String>,
                        with_report_date: bool|
     -> Handle<RefPort<Array<f64>>> {
        let s = ReportPanelSource::new(format!("{dir}/{kind}.parquet"), cols, universe.clone())
            .with_report_date(with_report_date)
            .use_effective_date(Duration::ZERO)
            .with_time_range(start, end);
        let init = nan_array(&s.out_shape());
        sc.add_source(s, init)
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
            "balance_sheet.assets".into(),              // 3 total assets
            "balance_sheet.liab".into(),                // 4 total liabilities (negative)
            "balance_sheet.assets.current".into(),      // 5 current assets
            "balance_sheet.liab.current".into(),        // 6 current liabilities (negative)
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
            "income_statement.profit".into(),                          // 0 net profit
            "income_statement.profit.operating".into(),                // 1 operating profit
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

    // One `Split` per panel: the `1 → N` row fan-out as a single node each.
    let prices_rows = sc.add_operator(Split::<f64>::new(n), prices_panel);
    let div_rows = sc.add_operator(Split::<f64>::new(n), div_panel);
    let equity_rows = sc.add_operator(Split::<f64>::new(n), equity_panel);
    let balance_rows = sc.add_operator(Split::<f64>::new(n), balance_panel);
    let income_rows = sc.add_operator(Split::<f64>::new(n), income_panel);
    let cashflow_rows = sc.add_operator(Split::<f64>::new(n), cashflow_panel);

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
        // the `segment!` arrow notation. Inputs are the stock's **zero-copy
        // row views** of the six panels (from `Split`); the leading
        // `Gate(any_finite)` per panel drops the all-NaN "no data" ticks,
        // recovering that stock's real event stream exactly as the old
        // per-stock `pick` chains did. `Gate` retains the last passed row in
        // its own state and re-presents a view of it whenever it gates out, so
        // it honours the no-notify⟹unchanged contract (its un-notified output
        // is the last passed value, never the gated-out row). Views materialize
        // at the computing/selecting operators, whose owned outputs retain last
        // values for the carry-style `Stack` joins; every sub-operator keeps
        // its own notify gate inside the fused node, so the cutoff and
        // `ForwardAdjust`'s price/dividend message-passing are unchanged. The
        // segment is identical for every stock (the stock index lives only in
        // the input wiring), so it monomorphizes once.
        //
        // The many fundamental columns are emitted as GROUPED arrays
        // (`income_ann` [4], `balance_extras` [5], `cf_ann` [3]) rather than one
        // output each — both to keep the segment within the tuple-arity limit
        // and to move less. The `Stack` joins below build the `(N, K)` panels;
        // a squeezing column `Select` recovers each field as `(N,)`.
        let seg = flowgraph::segment!(|prices_row: RefViewPort<ArrayValue<f64>>,
                                       div_row: RefViewPort<ArrayValue<f64>>,
                                       equity_row: RefViewPort<ArrayValue<f64>>,
                                       balance_row: RefViewPort<ArrayValue<f64>>,
                                       income_row: RefViewPort<ArrayValue<f64>>,
                                       cashflow_row: RefViewPort<ArrayValue<f64>>|
            -> (
            RefPort<Array<f64>>,
            RefViewPort<ArrayValue<f64>>,
            RefPort<Array<f64>>,
            RefPort<Array<f64>>,
            RefViewPort<ArrayValue<f64>>,
            RefViewPort<ArrayValue<f64>>,
            RefPort<Array<f64>>,
            RefPort<Array<f64>>,
            RefViewPort<ArrayValue<f64>>,
            RefPort<Array<f64>>,
            RefViewPort<ArrayValue<f64>>
        ) {
            let prices = prices_row => Gate(any_finite); // [close, volume, open, high, low, amount]
            let dividends = div_row => Gate(any_finite); // [share, cash]
            let equity = equity_row => Gate(any_finite); // [total, circulating]
            let balance = balance_row => Gate(any_finite); // [cap, res, parent, assets, liab, cur_a, cur_l, cash]
            let income = income_row => Gate(any_finite); // [year, doy, profit, operating, revenue, cost]
            let cashflow = cashflow_row => Gate(any_finite); // [year, doy, operating, investing, financing]
            // Terminal column picks (whose only consumer is their cross-section
            // join) stay zero-copy views (`SliceView`) into the retaining
            // `Gate`'s stable storage. `close` is NOT terminal — it feeds
            // `ForwardAdjust` and `Multiply` (owned inputs) — so it materializes
            // here via the owned `SelectView`.
            let close = prices => SelectView::<f64>::new(vec![0], 0, true);
            let volume = prices => SliceView::new(vec![1], 0, true);
            // [open, high, low, amount] as a contiguous view of cols 2..6.
            let prices_extras = prices => SliceView::new(vec![2, 3, 4, 5], 0, false);
            let adjusts = (close, dividends)
                => ForwardAdjustViewDiv::default().with_output_prices(false);
            let adjusted_close = (close, adjusts) => Multiply::<f64>::new();
            let total_shares = equity => SliceView::new(vec![0], 0, true);
            let circ_shares = equity => SliceView::new(vec![1], 0, true);
            // parent_equity = -(capital + reserves + parent_interests) (cols 0..3).
            let parent_equity = balance
                => Apply::<ViewPort<ArrayValue<f64>>, Array<f64>, _>::new(
                    |a: ArraySlice<f64>| Array::scalar(-a.as_slice()[..3].iter().sum::<f64>()),
                );
            // Annualized income / cash flows (YTD → Annualize) and the balance
            // stocks [assets, liab, current_assets, current_liab, cash] as a
            // contiguous view of cols 3..8.
            let income_ann = income => AnnualizeView::default();
            let cf_ann = cashflow => AnnualizeView::default();
            let balance_extras = balance => SliceView::new(vec![3, 4, 5, 6, 7, 8, 9], 0, false);
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
        ) = sc.add_operator(
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

    // Cross-sectional grouped panels; a squeezing column `Select` (axis 1)
    // recovers each field as `(N,)`.
    let income_xs = sc.add_operator(Stack::<f64>::new(0), &income_v[..]); // (N, 4)
    let balance_xs = sc.add_operator(StackView::<f64>::new(0), &bal_v[..]); // (N, 5)
    let cf_xs = sc.add_operator(Stack::<f64>::new(0), &cf_v[..]); // (N, 3)
    let px_xs = sc.add_operator(StackSyncView::<f64>::new(0), &px_v[..]); // (N, 4) [open, high, low, amount]

    Stacked {
        close: sc.add_operator(StackSync::<f64>::new(0), &close_v[..]),
        volume: sc.add_operator(StackSyncView::<f64>::new(0), &volume_v[..]),
        adjusted_close: sc.add_operator(StackSync::<f64>::new(0), &adj_close_v[..]),
        adjusts: sc.add_operator(Stack::<f64>::new(0), &adjusts_v[..]),
        total_shares: sc.add_operator(StackView::<f64>::new(0), &total_v[..]),
        circ_shares: sc.add_operator(StackView::<f64>::new(0), &circ_v[..]),
        parent_equity: sc.add_operator(Stack::<f64>::new(0), &peq_v[..]),
        net_profit: sc.add_operator(Select::<f64>::new(vec![0], 1, true), income_xs),
        operating_profit: sc.add_operator(Select::<f64>::new(vec![1], 1, true), income_xs),
        revenue: sc.add_operator(Select::<f64>::new(vec![2], 1, true), income_xs),
        operating_cost: sc.add_operator(Select::<f64>::new(vec![3], 1, true), income_xs),
        total_assets: sc.add_operator(Select::<f64>::new(vec![0], 1, true), balance_xs),
        total_liab: sc.add_operator(Select::<f64>::new(vec![1], 1, true), balance_xs),
        current_assets: sc.add_operator(Select::<f64>::new(vec![2], 1, true), balance_xs),
        current_liab: sc.add_operator(Select::<f64>::new(vec![3], 1, true), balance_xs),
        cash: sc.add_operator(Select::<f64>::new(vec![4], 1, true), balance_xs),
        inventories: sc.add_operator(Select::<f64>::new(vec![5], 1, true), balance_xs),
        receivables: sc.add_operator(Select::<f64>::new(vec![6], 1, true), balance_xs),
        net_operating_cashflow: sc.add_operator(Select::<f64>::new(vec![0], 1, true), cf_xs),
        open: sc.add_operator(Select::<f64>::new(vec![0], 1, true), px_xs),
        high: sc.add_operator(Select::<f64>::new(vec![1], 1, true), px_xs),
        low: sc.add_operator(Select::<f64>::new(vec![2], 1, true), px_xs),
        amount: sc.add_operator(Select::<f64>::new(vec![3], 1, true), px_xs),
    }
}

// ===========================================================================
// Feature set
// ===========================================================================

/// The canonical 7-factor cross-sectional feature panel.
pub struct Features {
    /// Per-feature live handles (each `(num_stocks,)`), in column order.
    pub names: Vec<String>,
    pub handles: Vec<Handle<RefPort<Array<f64>>>>,
    /// Trading-day-aligned `Record` of the `(num_stocks, n_features)` panel.
    pub series: Handle<RefPort<Series<f64>>>,
}

/// Build the canonical factor panel (3 percentile-ranked fundamentals plus
/// momentum / volatility / turnover-MA / volume-ratio), recorded on the daily
/// trading pulse. The returned panel `Series` is recorded under
/// `feature_retention` (pass [`Retention::UNBOUNDED`] when a consumer needs full
/// history); the internal rolling-input records are bounded to their own windows.
pub fn build_features(
    sc: &mut Scenario,
    st: &Stacked,
    window: usize,
    feature_retention: Retention,
) -> Features {
    let win_ret = Retention::count(window + RETAIN_MARGIN);

    // Fundamentals.
    let market_cap = sc.add_operator(Multiply::<f64>::new(), (st.close, st.total_shares));
    let bp = sc.add_operator(Divide::<f64>::new(), (st.parent_equity, market_cap));
    // TTM net profit: a 365-day rolling mean → retain a 365-day (+margin) window.
    let net_profit_series =
        sc.add_record_retained(st.net_profit, Retention::duration(Duration::from_days(380)));
    let net_profit_ttm = sc.add_operator(
        RollingMean::<f64>::time_delta(Duration::from_days(365)),
        net_profit_series,
    );
    let ttm_roe = sc.add_operator(Divide::<f64>::new(), (net_profit_ttm, st.parent_equity));

    // Momentum: window-period log return of adjusted close.
    let log_adj = sc.add_operator(Log::<f64>::new(), st.adjusted_close);
    let log_adj_series = sc.add_record_retained(log_adj, win_ret);
    let log_adj_lag = sc.add_operator(Lag::<f64>::new(window, f64::NAN), log_adj_series);
    let momentum = sc.add_operator(Subtract::<f64>::new(), (log_adj, log_adj_lag));

    // Volatility: rolling std of daily log returns.
    let log_var = sc.add_operator(RollingVariance::<f64>::count(window), log_adj_series);
    let volatility = sc.add_operator(Sqrt::<f64>::new(), log_var);

    // Turnover MA.
    let turnover = sc.add_operator(Divide::<f64>::new(), (st.volume, st.circ_shares));
    let turnover_series = sc.add_record_retained(turnover, win_ret);
    let turnover_ma = sc.add_operator(RollingMean::<f64>::count(window), turnover_series);

    // Volume ratio.
    let volume_series = sc.add_record_retained(st.volume, win_ret);
    let volume_ma = sc.add_operator(RollingMean::<f64>::count(window), volume_series);
    let volume_ratio = sc.add_operator(Divide::<f64>::new(), (st.volume, volume_ma));

    let p = 0.01;
    let names = vec![
        "rank_market_cap".to_string(),
        "rank_bp".to_string(),
        "rank_ttm_roe".to_string(),
        format!("momentum_ma_{window}"),
        format!("volatility_{window}"),
        format!("turnover_ma_{window}"),
        format!("volume_ratio_{window}"),
    ];
    let handles = vec![
        sc.add_operator(Percentile::<f64>::new(), market_cap),
        sc.add_operator(Percentile::<f64>::new(), bp),
        sc.add_operator(Percentile::<f64>::new(), ttm_roe),
        sc.add_operator(Winsorize::<f64>::new(p), momentum),
        sc.add_operator(Winsorize::<f64>::new(p), volatility),
        sc.add_operator(Winsorize::<f64>::new(p), turnover_ma),
        sc.add_operator(Winsorize::<f64>::new(p), volume_ratio),
    ];

    // Stack the 7 features into (N, 7) and re-emit on the daily close pulse.
    let stacked_features = sc.add_operator(Stack::<f64>::new(1), &handles[..]);
    let sampled = sc.add_operator(
        Resample::<Array<f64>, Array<f64>>::new(),
        (st.adjusted_close, stacked_features),
    );
    let series = sc.add_record_retained(sampled, feature_retention);

    Features {
        names,
        handles,
        series,
    }
}

// ===========================================================================
// CICC handbook factor feature panel
// ===========================================================================

/// Which factor panel a strategy regresses on.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FeatureSet {
    /// The canonical 7-factor [`build_features`] panel (back-compat default).
    Canonical,
    /// A curated, style-diverse ~24-factor subset of the CICC handbooks
    /// (value / quality / growth / size / momentum / reversal / low-vol /
    /// liquidity / illiquidity / price-volume-correlation / chip), chosen from
    /// the single-factor study. Within-style collinearity is handled by the
    /// predictor's pool-standardized Ridge (`alpha`).
    Cicc,
    /// All 145 CICC factors (fundamental + price-volume). Heavily collinear by
    /// design — leans entirely on Ridge to regularise.
    All,
}

impl FeatureSet {
    pub fn parse(s: &str) -> FeatureSet {
        match s {
            "canonical" => FeatureSet::Canonical,
            "cicc" => FeatureSet::Cicc,
            "all" => FeatureSet::All,
            other => panic!("unknown --feature-set {other:?} (expected canonical|cicc|all)"),
        }
    }
}

/// Curated style-diverse subset of CICC handbook factors used by
/// [`FeatureSet::Cicc`]. Names must match the catalog entries in
/// [`factors::build_factor_catalog`] / [`pv_factors::build_pv_catalog`].
const CICC_SUBSET: &[&str] = &[
    // Value / quality / growth / size (fundamental)
    "BP_LR", "SP_TTM", "EP_TTM", "ROE_TTM", "GPM_TTM", "CFOA", "APR_TTM", "TA_YOY", "Ln_MC",
    // Momentum / reversal (price-volume)
    "mmt_normal_M", "mmt_intraday_M", "mmt_overnight_M", "mmt_range_M", "mmt_route_M",
    // Volatility
    "vol_std_1M", "vol_up_std_1M", "vol_w_downshadow_std_1M",
    // Liquidity / illiquidity
    "liq_turn_std_1M", "liq_amihud_avg_1M", "liq_vstd_1M",
    // Price-volume correlation
    "corr_price_turn_1M", "corr_ret_turnd_1M",
    // Chip distribution
    "distribution_loss_l", "distribution_ret_avg",
];

/// Build a cross-sectionally percentile-ranked factor panel from the CICC
/// handbook catalogs, recorded on the daily close pulse — a drop-in [`Features`]
/// for the mean / mean-variance predictors. Each selected factor is ranked
/// ([`Percentile`], robust to the heavy tails / disparate scales of raw factors),
/// stacked into `(N, F)`, resampled onto the daily pulse, and recorded. `set`
/// must be [`FeatureSet::Cicc`] or [`FeatureSet::All`] (use [`build_features`]
/// for [`FeatureSet::Canonical`]).
pub fn build_factor_features(
    sc: &mut Scenario,
    st: &Stacked,
    set: FeatureSet,
    feature_retention: Retention,
) -> Features {
    // Both catalogs (fundamental + price-volume); select columns by name. Each
    // catalog entry is already the model-ready feature — cross-sectionally ranked
    // and its missing values imputed to the neutral median (so a stock with
    // partial factor coverage stays predictable rather than dropped by the
    // predictor's all-features-finite mask).
    let fund = factors::build_factor_catalog(sc, st);
    let pv = pv_factors::build_pv_catalog(sc, st);
    let mut all_names = fund.names;
    let mut all_feat = fund.feature;
    all_names.extend(pv.names);
    all_feat.extend(pv.feature);

    let selected: Vec<usize> = match set {
        FeatureSet::All => (0..all_names.len()).collect(),
        FeatureSet::Cicc => CICC_SUBSET
            .iter()
            .map(|nm| {
                all_names
                    .iter()
                    .position(|x| x == nm)
                    .unwrap_or_else(|| panic!("CICC_SUBSET names an unknown factor {nm:?}"))
            })
            .collect(),
        FeatureSet::Canonical => panic!("build_factor_features called with Canonical; use build_features"),
    };

    let mut names = Vec::with_capacity(selected.len());
    let mut handles = Vec::with_capacity(selected.len());
    for &i in &selected {
        names.push(all_names[i].clone());
        handles.push(all_feat[i]);
    }

    // Stack into (N, F), resample onto the daily close pulse, and record under
    // the caller's retention (the predictor only reads a short trailing window).
    let stacked_features = sc.add_operator(Stack::<f64>::new(1), &handles[..]);
    let sampled = sc.add_operator(
        Resample::<Array<f64>, Array<f64>>::new(),
        (st.adjusted_close, stacked_features),
    );
    let series = sc.add_record_retained(sampled, feature_retention);

    Features { names, handles, series }
}

/// Build the strategy feature panel for `set`, dispatching to [`build_features`]
/// (canonical) or [`build_factor_features`] (CICC subsets). `window` is only
/// used by the canonical set.
pub fn build_strategy_features(
    sc: &mut Scenario,
    st: &Stacked,
    window: usize,
    set: FeatureSet,
    feature_retention: Retention,
) -> Features {
    match set {
        FeatureSet::Canonical => build_features(sc, st, window, feature_retention),
        _ => build_factor_features(sc, st, set, feature_retention),
    }
}

// ===========================================================================
// Universe / target / limits
// ===========================================================================

/// Market-cap-weighted index weights for the top-`k` stocks (proportional to
/// cap, normalised to 1; others zero).
pub fn calculate_index_weights(mc: &[f64], k: usize) -> Vec<f64> {
    let n = mc.len();
    let mut w = vec![0.0f64; n];
    let mut valid: Vec<usize> = (0..n)
        .filter(|&i| mc[i].is_finite() && mc[i] > 0.0)
        .collect();
    if valid.is_empty() {
        return w;
    }
    let k = k.min(valid.len());
    valid.sort_by(|&a, &b| mc[b].partial_cmp(&mc[a]).unwrap());
    let top = &valid[..k];
    let mut s = 0.0;
    for &i in top {
        w[i] = mc[i];
        s += mc[i];
    }
    if s > 0.0 {
        for &i in top {
            w[i] /= s;
        }
    }
    w
}

/// Cap-weighted top-`index_size` universe, recomputed on each rebalance tick.
/// `market_cap` is the per-stock circulating market cap; `rebalance_clock` is
/// the `Handle<RefPort<()>>` of a [`clock`](tradingflow::sources::clock) source.
pub fn build_cap_weighted_universe(
    sc: &mut Scenario,
    market_cap: Handle<RefPort<Array<f64>>>,
    rebalance_clock: Handle<RefPort<()>>,
    index_size: usize,
) -> Handle<RefPort<Array<f64>>> {
    use tradingflow::operators::Clocked;
    let k = index_size;
    sc.add_operator(
        Clocked::new(Map::new(move |m: &Array<f64>| {
            Array::from_vec(m.shape(), calculate_index_weights(m.as_slice(), k))
        })),
        (rebalance_clock, market_cap),
    )
}

/// Winsorized daily log returns: `(target, target_series, demeaned_series)`.
/// The covariance predictor consumes `target_series` (raw); the mean predictor
/// consumes `demeaned_series` (cross-sectionally demeaned). Both series are
/// recorded under `target_retention` — size it to the deepest consumer
/// look-back (the incremental mean predictor reads a single trailing pair, the
/// shrinkage covariance reads its `max_periods` window); pass
/// [`Retention::UNBOUNDED`] when full history is needed.
pub fn build_log_return_target(
    sc: &mut Scenario,
    log_adj: Handle<RefPort<Array<f64>>>,
    target_retention: Retention,
) -> (Handle<RefPort<Array<f64>>>, Handle<RefPort<Series<f64>>>, Handle<RefPort<Series<f64>>>) {
    use tradingflow::operators::Diff;
    let log_returns = sc.add_operator(Diff::<f64>::new(), log_adj);
    let target = sc.add_operator(Winsorize::<f64>::new(0.01), log_returns);
    let target_series = sc.add_record_retained(target, target_retention);
    let demeaned = sc.add_operator(Map::new(demean), target);
    let demeaned_series = sc.add_record_retained(demeaned, target_retention);
    (target, target_series, demeaned_series)
}

/// Cross-sectional demean preserving NaN.
fn demean(r: &Array<f64>) -> Array<f64> {
    let s = r.as_slice();
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for &x in s {
        if x.is_finite() {
            sum += x;
            cnt += 1;
        }
    }
    let mean = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
    Array::from_vec(
        r.shape(),
        s.iter()
            .map(|&x| if x.is_finite() { x - mean } else { x })
            .collect(),
    )
}

/// Constant ±`limit_pct` daily price limits from the previous close, rounded to
/// 0.01 yuan. Returns `(upper, lower)`; first tick is NaN (no prior close).
pub fn build_price_limits(
    sc: &mut Scenario,
    close: Handle<RefPort<Array<f64>>>,
    limit_pct: f64,
) -> (Handle<RefPort<Array<f64>>>, Handle<RefPort<Array<f64>>>) {
    // Only a 1-step lag reads this record; keep a tiny trailing window.
    let close_series = sc.add_record_retained(close, Retention::count(1 + RETAIN_MARGIN));
    let prev_close = sc.add_operator(Lag::<f64>::new(1, f64::NAN), close_series);
    let up = limit_pct;
    let dn = limit_pct;
    let upper = sc.add_operator(
        Map::new(move |c: &Array<f64>| {
            Array::from_vec(
                c.shape(),
                c.as_slice()
                    .iter()
                    .map(|&x| ((x * (1.0 + up)) * 100.0).round() / 100.0)
                    .collect(),
            )
        }),
        prev_close,
    );
    let lower = sc.add_operator(
        Map::new(move |c: &Array<f64>| {
            Array::from_vec(
                c.shape(),
                c.as_slice()
                    .iter()
                    .map(|&x| ((x * (1.0 - dn)) * 100.0).round() / 100.0)
                    .collect(),
            )
        }),
        prev_close,
    );
    (upper, lower)
}

// ===========================================================================
// Output helpers
// ===========================================================================

/// Read a recorded **scalar** series into `(timestamps_ns, values)`.
pub fn read_scalar_series(session: &Session, h: Handle<RefPort<Series<f64>>>) -> (Vec<i64>, Vec<f64>) {
    let s: &Series<f64> = session.value(h);
    let ts = s.timestamps().iter().map(|t| t.as_nanos()).collect();
    let vals = s.values().to_vec();
    (ts, vals)
}

/// Write labelled scalar series in long format (`series,timestamp_ns,value`)
/// so the plot scripts can group by series and handle independent cadences.
pub fn write_long_csv(path: &str, series: &[(String, Vec<i64>, Vec<f64>)]) {
    let mut csv = String::from("series,timestamp_ns,value\n");
    for (label, ts, vals) in series {
        for (t, v) in ts.iter().zip(vals.iter()) {
            writeln!(csv, "{label},{t},{v}").unwrap();
        }
    }
    fs::write(path, csv).unwrap_or_else(|e| panic!("write {path}: {e}"));
}

/// Align labelled scalar series by timestamp into a wide CSV (NaN-filled).
pub fn write_wide_csv(path: &str, series: &[(String, Vec<i64>, Vec<f64>)]) {
    let ncols = series.len();
    let mut rows: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for (c, (_, ts, vals)) in series.iter().enumerate() {
        for (t, v) in ts.iter().zip(vals.iter()) {
            rows.entry(*t).or_insert_with(|| vec![f64::NAN; ncols])[c] = *v;
        }
    }
    let mut csv = String::from("timestamp_ns");
    for (label, _, _) in series {
        write!(csv, ",{label}").unwrap();
    }
    csv.push('\n');
    for (t, vals) in &rows {
        write!(csv, "{t}").unwrap();
        for v in vals {
            write!(csv, ",{v}").unwrap();
        }
        csv.push('\n');
    }
    fs::write(path, csv).unwrap_or_else(|e| panic!("write {path}: {e}"));
}
