//! Shared scaffolding for the A-shares cross-sectional examples — the Rust
//! port of `python/examples/common.py`.
//!
//! Pulled in by each strategy/plot example via `#[path = "common/mod.rs"] mod
//! common;`. It is **pure native** (no Python / `pyflow`): it builds the data
//! pipeline (per-stock CSV + financial-report sources, the stacked
//! cross-sectional panel, the canonical 7-factor feature set, the cap-weighted
//! universe, the log-return target, and price limits) entirely from native
//! flow operators. Examples add their `flowops` predictor/portfolio/trader
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
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use flowgraph::typed::Handle;

use tradingflow::data::Duration;
use tradingflow::flow::{
    Annualize, Divide, Filter, ForwardAdjust, Lag, Log, Map, Multiply, Percentile, Resample,
    RollingMean, RollingVariance, Scenario, Select, Session, Sqrt, Stack, StackSync, Subtract,
    Winsorize,
};
use tradingflow::sources::{ParquetPanelSource, ReportPanelSource};
use tradingflow::{Array, Instant, Series, utc_to_tai};

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

/// Parse a `YYYY-MM-DD` string into days since 1970-01-01.
pub fn parse_date_days(s: &str) -> i64 {
    let mut it = s.split('-');
    let y: i64 = it.next().unwrap().parse().expect("year");
    let m: i64 = it.next().unwrap().parse().expect("month");
    let d: i64 = it.next().unwrap().parse().expect("day");
    days_from_civil(y, m, d)
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
/// [`Session::run`](tradingflow::flow::Session::run)'s `on_flush`.
///
/// Progress is measured in **long-table rows scanned**: the position is read
/// from the shared [`progress_counter`](tradingflow::flow::Session::progress_counter)
/// (which the panel sources bump as they read rows), *not* from the emit count
/// passed to `on_flush` (one emit = a whole cross-section = many rows). `total`
/// is [`estimated_event_count`](tradingflow::flow::Session::estimated_event_count)
/// in the same row unit; `Some(n)` → a bounded bar (percent / rate / ETA), else a
/// spinner. `{per_sec}` is therefore rows/s. `begin` sets `{prefix}` (warm-up
/// before it, running after); `{msg}` is the current event date. The bar uses a
/// terminal-width `{wide_bar}` with Unicode sub-cell fill, and finalises itself
/// when the callback drops at the end of `run`:
/// ```ignore
/// let total = session.estimated_event_count();
/// let counter = session.progress_counter();
/// session.run(common::progress(total, args.begin(), counter)).await;
/// eprintln!(); // move past the finished bar line before printing results
/// ```
pub fn progress(total: Option<usize>, begin: Instant, counter: Arc<AtomicU64>) -> impl FnMut(Instant, usize) {
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
                    "{prefix:>7} {elapsed_precise} [{wide_bar}] {human_pos}/{human_len} rows {percent:>3}% {per_sec} eta {eta} {msg}",
                )
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏ "),
            );
            pb
        }
        _ => {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template("{prefix:>7} {elapsed_precise} {spinner} {human_pos} rows {per_sec} {msg}")
                    .unwrap(),
            );
            pb
        }
    };
    // Cap redraws at ~12 fps regardless of how often the callback fires.
    pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(12));

    let begin_ns = begin.as_nanos();
    let guard = FinishOnDrop(pb);
    move |ts: Instant, _emits: usize| {
        let pb = &guard.0;
        pb.set_prefix(if ts.as_nanos() < begin_ns { "warmup" } else { "running" });
        let rows = counter.load(Ordering::Relaxed);
        // Grow the length if the estimate undershot (keeps the percentage sane).
        if let Some(len) = pb.length() {
            if rows > len {
                pb.set_length(rows);
            }
        }
        pb.set_position(rows);
        pb.set_message(date_str(ts));
    }
}

// ===========================================================================
// CLI args
// ===========================================================================

/// Shared CLI configuration for the cross-sectional examples.
pub struct Args {
    pub data_dir: String,
    pub index_size: usize,
    pub rebalance_days: i64,
    pub window: usize,
    pub begin_days: i64,
    pub end_days: i64,
    pub data_start_days: i64,
    /// Worker threads for the flow `Pool` (`--threads`, default 0 = serial).
    /// `> 0` lets independent solve-bound operators (e.g. one cvxpy portfolio
    /// per risk-aversion) overlap via GIL release.
    pub threads: usize,
}

impl Args {
    /// Parse `--key value` flags from the process args, with example-friendly
    /// defaults (a bounded universe and short windows so a run completes in
    /// seconds on the bundled data).
    pub fn from_env() -> Self {
        let mut data_dir = "python/examples/data".to_string();
        let mut index_size = 30usize;
        let mut rebalance_days = 30i64;
        let mut window = 20usize;
        let mut begin = "2022-01-01".to_string();
        let mut end = "2024-12-31".to_string();
        let mut sample_begin: Option<String> = None;
        let mut threads = 0usize;

        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i + 1 < args.len() {
            let v = args[i + 1].clone();
            match args[i].as_str() {
                "--data-dir" => data_dir = v,
                "--index-size" => index_size = v.parse().expect("--index-size"),
                "--rebalance-days" => rebalance_days = v.parse().expect("--rebalance-days"),
                "--window" => window = v.parse().expect("--window"),
                "--begin" => begin = v,
                "--end" => end = v,
                "--sample-begin" => sample_begin = Some(v),
                "--threads" => threads = v.parse().expect("--threads"),
                _ => {}
            }
            i += 2;
        }

        let begin_days = parse_date_days(&begin);
        let end_days = parse_date_days(&end);
        // Default warmup: enough to populate the 365-day TTM and the rolling
        // feature windows before trading starts.
        let data_start_days = match sample_begin {
            Some(s) => parse_date_days(&s).min(begin_days),
            None => begin_days - 400,
        };

        Args {
            data_dir,
            index_size,
            rebalance_days,
            window,
            begin_days,
            end_days,
            data_start_days,
            threads,
        }
    }

    pub fn begin(&self) -> Instant {
        instant_from_days(self.begin_days)
    }
    pub fn end(&self) -> Instant {
        instant_from_days(self.end_days)
    }
    pub fn data_start(&self) -> Instant {
        instant_from_days(self.data_start_days)
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
pub struct Stacked {
    pub close: Handle<Array<f64>>,          // unadjusted close (StackSync)
    pub volume: Handle<Array<f64>>,         // (StackSync)
    pub adjusted_close: Handle<Array<f64>>, // close * forward-adjust factor (StackSync)
    pub adjusts: Handle<Array<f64>>,        // forward-adjust factor (Stack)
    pub total_shares: Handle<Array<f64>>,   // (Stack)
    pub circ_shares: Handle<Array<f64>>,    // (Stack)
    pub parent_equity: Handle<Array<f64>>,  // (Stack)
    pub net_profit: Handle<Array<f64>>,     // annualized net profit (Stack)
}

/// Predicate for the per-stock `Filter`: the row has ≥1 finite entry, i.e. the
/// stock actually has data this tick (vs. an all-NaN "no data" cross-section the
/// panel emits on a date where other stocks ticked but this one didn't).
fn any_finite(a: &Array<f64>) -> bool {
    a.as_slice().iter().any(|x| x.is_finite())
}

/// `Select` stock `i`'s row out of a `[N, K]` panel and drop the all-NaN
/// "no data" ticks, recovering that stock's real event stream (so the existing
/// per-stock operators, incl. message-passing `ForwardAdjust`, are unchanged).
fn pick(sc: &mut Scenario, panel: Handle<Array<f64>>, i: usize) -> Handle<Array<f64>> {
    let sel = sc.add_operator(Select::<f64>::new(vec![i], 0, true), panel);
    sc.add_operator(Filter(any_finite), sel)
}

/// Load the consolidated long-format parquet panels and stack into the
/// cross-sectional panel. One [`ParquetPanelSource`] / [`ReportPanelSource`] per
/// data kind (one sequential scan each) replaces the per-symbol CSV fan-in;
/// each stock is then recovered with [`pick`] (`Select` + NaN `Filter`) and the
/// per-stock transforms run unchanged, before `StackSync` (NaN-fill non-trading
/// slots) / `Stack` (carry last-known) recombine into `[N]` panels — the
/// `1 → N → 1` fan-out. The financial reports align on the look-ahead-safe
/// effective date `max(report, notice)` (`use_effective_date`, zero fallback).
pub fn build_stacked(sc: &mut Scenario, symbols: &[String], args: &Args) -> Stacked {
    let dir = &args.data_dir;
    let start = Some(args.data_start());
    let end = Some(args.end());
    let n = symbols.len();
    let universe: Vec<String> = symbols.to_vec();

    // Panel sources emit pure StackSync cross-sections (only each date's rows;
    // the carry-forward / NaN-fill is the downstream `Stack` / `StackSync`'s job).
    // Reports align on the **effective date** `max(report, notice)` — the
    // look-ahead-safe point-in-time a backtest may use them (`use_effective_date`).
    let daily_panel = |sc: &mut Scenario, kind: &str, cols: Vec<String>| -> Handle<Array<f64>> {
        let s = ParquetPanelSource::new(format!("{dir}/{kind}.parquet"), cols, universe.clone())
            .with_time_range(start, end);
        let init = Array::zeros(&s.out_shape());
        sc.add_source(s, init)
    };
    let report_panel =
        |sc: &mut Scenario, kind: &str, cols: Vec<String>, with_report_date: bool| -> Handle<Array<f64>> {
            let s = ReportPanelSource::new(format!("{dir}/{kind}.parquet"), cols, universe.clone())
                .with_report_date(with_report_date)
                .use_effective_date(Duration::ZERO)
                .with_time_range(start, end);
            let init = Array::zeros(&s.out_shape());
            sc.add_source(s, init)
        };

    let prices_panel = daily_panel(sc, "daily_prices", vec!["prices.close".into(), "prices.volume".into()]);
    let div_panel = daily_panel(sc, "dividends", vec!["dividends.share".into(), "dividends.cash".into()]);
    let equity_panel = daily_panel(sc, "equity_structures", vec!["shares.total".into(), "shares.circulating".into()]);
    let balance_panel = report_panel(
        sc,
        "balance_sheets",
        vec![
            "balance_sheet.equity.capital".into(),
            "balance_sheet.equity.reserves".into(),
            "balance_sheet.equity.parent_interests".into(),
        ],
        false,
    );
    let income_panel = report_panel(sc, "income_statements", vec!["income_statement.profit".into()], true);

    let mut close_v = Vec::with_capacity(n);
    let mut volume_v = Vec::with_capacity(n);
    let mut adj_close_v = Vec::with_capacity(n);
    let mut adjusts_v = Vec::with_capacity(n);
    let mut total_v = Vec::with_capacity(n);
    let mut circ_v = Vec::with_capacity(n);
    let mut peq_v = Vec::with_capacity(n);
    let mut np_v = Vec::with_capacity(n);

    for i in 0..n {
        let prices = pick(sc, prices_panel, i); // [close, volume]
        let dividends = pick(sc, div_panel, i); // [share, cash]
        let equity = pick(sc, equity_panel, i); // [total, circulating]
        let balance = pick(sc, balance_panel, i); // [capital, reserves, parent_interests]
        let income = pick(sc, income_panel, i); // [year, day_of_year, profit]

        let close = sc.add_operator(Select::<f64>::new(vec![0], 0, true), prices);
        let volume = sc.add_operator(Select::<f64>::new(vec![1], 0, true), prices);
        let adjusts = sc.add_operator(
            ForwardAdjust::new().with_output_prices(false),
            (close, dividends),
        );
        let adjusted_close = sc.add_operator(Multiply::<f64>::new(), (close, adjusts));
        let total_shares = sc.add_operator(Select::<f64>::new(vec![0], 0, true), equity);
        let circ_shares = sc.add_operator(Select::<f64>::new(vec![1], 0, true), equity);
        let income_ann = sc.add_operator(Annualize::new(), income); // [profit]
        let net_profit = sc.add_operator(Select::<f64>::new(vec![0], 0, true), income_ann);
        // parent_equity = -(capital + reserves + parent_interests).
        let parent_equity = sc.add_operator(
            Map::new(|a: &Array<f64>| Array::scalar(-a.as_slice().iter().sum::<f64>())),
            balance,
        );

        close_v.push(close);
        volume_v.push(volume);
        adj_close_v.push(adjusted_close);
        adjusts_v.push(adjusts);
        total_v.push(total_shares);
        circ_v.push(circ_shares);
        peq_v.push(parent_equity);
        np_v.push(net_profit);
    }

    Stacked {
        close: sc.add_operator(StackSync::<f64>::new(0), &close_v[..]),
        volume: sc.add_operator(StackSync::<f64>::new(0), &volume_v[..]),
        adjusted_close: sc.add_operator(StackSync::<f64>::new(0), &adj_close_v[..]),
        adjusts: sc.add_operator(Stack::<f64>::new(0), &adjusts_v[..]),
        total_shares: sc.add_operator(Stack::<f64>::new(0), &total_v[..]),
        circ_shares: sc.add_operator(Stack::<f64>::new(0), &circ_v[..]),
        parent_equity: sc.add_operator(Stack::<f64>::new(0), &peq_v[..]),
        net_profit: sc.add_operator(Stack::<f64>::new(0), &np_v[..]),
    }
}

// ===========================================================================
// Feature set
// ===========================================================================

/// The canonical 7-factor cross-sectional feature panel.
pub struct Features {
    /// Per-feature live handles (each `(num_stocks,)`), in column order.
    pub names: Vec<String>,
    pub handles: Vec<Handle<Array<f64>>>,
    /// Trading-day-aligned `Record` of the `(num_stocks, n_features)` panel.
    pub series: Handle<Series<f64>>,
}

/// Build the canonical factor panel (3 percentile-ranked fundamentals plus
/// momentum / volatility / turnover-MA / volume-ratio), recorded on the daily
/// trading pulse.
pub fn build_features(sc: &mut Scenario, st: &Stacked, args: &Args) -> Features {
    let window = args.window;

    // Fundamentals.
    let market_cap = sc.add_operator(Multiply::<f64>::new(), (st.close, st.total_shares));
    let bp = sc.add_operator(Divide::<f64>::new(), (st.parent_equity, market_cap));
    let net_profit_series = sc.add_record(st.net_profit);
    let net_profit_ttm = sc.add_operator(
        RollingMean::<f64>::time_delta(Duration::from_days(365)),
        net_profit_series,
    );
    let ttm_roe = sc.add_operator(Divide::<f64>::new(), (net_profit_ttm, st.parent_equity));

    // Momentum: window-period log return of adjusted close.
    let log_adj = sc.add_operator(Log::<f64>::new(), st.adjusted_close);
    let log_adj_series = sc.add_record(log_adj);
    let log_adj_lag = sc.add_operator(Lag::<f64>::new(window, f64::NAN), log_adj_series);
    let momentum = sc.add_operator(Subtract::<f64>::new(), (log_adj, log_adj_lag));

    // Volatility: rolling std of daily log returns.
    let log_var = sc.add_operator(RollingVariance::<f64>::count(window), log_adj_series);
    let volatility = sc.add_operator(Sqrt::<f64>::new(), log_var);

    // Turnover MA.
    let turnover = sc.add_operator(Divide::<f64>::new(), (st.volume, st.circ_shares));
    let turnover_series = sc.add_record(turnover);
    let turnover_ma = sc.add_operator(RollingMean::<f64>::count(window), turnover_series);

    // Volume ratio.
    let volume_series = sc.add_record(st.volume);
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
    let series = sc.add_record(sampled);

    Features { names, handles, series }
}

// ===========================================================================
// Universe / target / limits
// ===========================================================================

/// Market-cap-weighted index weights for the top-`k` stocks (proportional to
/// cap, normalised to 1; others zero).
pub fn calculate_index_weights(mc: &[f64], k: usize) -> Vec<f64> {
    let n = mc.len();
    let mut w = vec![0.0f64; n];
    let mut valid: Vec<usize> = (0..n).filter(|&i| mc[i].is_finite() && mc[i] > 0.0).collect();
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
/// the `Handle<()>` of a [`clock`](tradingflow::sources::clock) source.
pub fn build_cap_weighted_universe(
    sc: &mut Scenario,
    market_cap: Handle<Array<f64>>,
    rebalance_clock: Handle<()>,
    index_size: usize,
) -> Handle<Array<f64>> {
    use tradingflow::flow::Clocked;
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
/// consumes `demeaned_series` (cross-sectionally demeaned).
pub fn build_log_return_target(
    sc: &mut Scenario,
    log_adj: Handle<Array<f64>>,
) -> (Handle<Array<f64>>, Handle<Series<f64>>, Handle<Series<f64>>) {
    use tradingflow::flow::Diff;
    let log_returns = sc.add_operator(Diff::<f64>::new(), log_adj);
    let target = sc.add_operator(Winsorize::<f64>::new(0.01), log_returns);
    let target_series = sc.add_record(target);
    let demeaned = sc.add_operator(Map::new(demean), target);
    let demeaned_series = sc.add_record(demeaned);
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
        s.iter().map(|&x| if x.is_finite() { x - mean } else { x }).collect(),
    )
}

/// Constant ±`limit_pct` daily price limits from the previous close, rounded to
/// 0.01 yuan. Returns `(upper, lower)`; first tick is NaN (no prior close).
pub fn build_price_limits(
    sc: &mut Scenario,
    close: Handle<Array<f64>>,
    limit_pct: f64,
) -> (Handle<Array<f64>>, Handle<Array<f64>>) {
    let close_series = sc.add_record(close);
    let prev_close = sc.add_operator(Lag::<f64>::new(1, f64::NAN), close_series);
    let up = limit_pct;
    let dn = limit_pct;
    let upper = sc.add_operator(
        Map::new(move |c: &Array<f64>| {
            Array::from_vec(
                c.shape(),
                c.as_slice().iter().map(|&x| ((x * (1.0 + up)) * 100.0).round() / 100.0).collect(),
            )
        }),
        prev_close,
    );
    let lower = sc.add_operator(
        Map::new(move |c: &Array<f64>| {
            Array::from_vec(
                c.shape(),
                c.as_slice().iter().map(|&x| ((x * (1.0 - dn)) * 100.0).round() / 100.0).collect(),
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
pub fn read_scalar_series(session: &Session, h: Handle<Series<f64>>) -> (Vec<i64>, Vec<f64>) {
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
