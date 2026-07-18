//! The CICC factor catalogs (基本面因子手册 fundamental + 价量因子手册
//! price-volume) and the cross-sectional feature panel built over them.
//!
//! # Factor catalog
//!
//! Each factor is a small native subgraph over the [`Stacked`](super::Stacked)
//! cross-sectional panel, returning a `[num_stocks]` factor handle on the panel's
//! native cadence (daily for price-derived, report-effective-date for
//! fundamentals, carried forward by the panel's `Stack`).
//!
//! TTM (trailing-twelve-month) flows are a 365-day rolling mean of the annualized
//! quarterly flow (the self-recording [`ma_time`] constructor). Balance-sheet
//! stocks (assets, equity, liabilities, cash) are used at their latest
//! carried-forward value (no TTM). The parquet stores assets debit-positive and
//! liabilities / expense items credit-**negative**; the formulas negate them
//! where a positive magnitude is wanted (`debt`, `cur_liab`, gross profit =
//! revenue + cost).
//!
//! **Levels** are single shared sub-expressions (`np_ttm`, `gross_ttm`, `debt`…
//! computed once, reused), so each level factor is one `Divide`/`Log`. The
//! period-over-period **变动 (delta)** and **同比 (YoY growth)** factors need a
//! ~1-year lag of an irregular-cadence series: the level is resampled onto the
//! daily close pulse and run through the self-recording 244-day [`change`] /
//! [`growth`] (the 次新 idiom), fused into one node via a nested
//! [`segment!`](tradingflow::segment) — the form recommended for individual
//! factors. (`(cur − prev)/prev` is NOT rank-equivalent to `cur/prev` once a
//! year-ago base goes negative, so the subtraction is kept rather than dropped.)
//!
//! Catalog is config: adding a factor is adding an `add("NAME", handle)` line.
//! Still deferred: single-quarter YoY/QoQ (need quarterly deltas), z-scores
//! (rank-identical to their base under RankIC), TOE (应交税费 with a year-ago lag
//! and a cash-tax-paid term), dividend yield/payout (DP/DPR), and the regression
//! / SUE / composite (Profit/Growth/Safe/QQC) factors.
//!
//! # Price-volume catalog
//!
//! Windows are **count-based on the daily price pulse** (the recorded series only
//! tick when prices change, so `Lag(21)` / `RollingSum::count(21)` ≈ one trading
//! month and `252` ≈ one year — the same idiom as the 次新 filter). Close-to-close
//! returns use the forward-adjusted close; intraday (`close/open`) and overnight
//! (`open/prev close`) use the raw OHLC the panel now exposes.
//!
//! This first pass covers the **动量 & 反转** factors derivable from existing
//! operators (图表4). Still to come: `mmt_range`/`mmt_time_rank`/`mmt_highest_days`
//! (need rolling rank / argmax / quantile ops), `mmt_off_limit` (limit detection),
//! `mmt_report_*` (earnings-announcement dates), and the 波动率 / 流动性 /
//! 量价相关性 categories.
//!
//! # Feature panels
//!
//! [`Features`] is the whole catalog (fundamental ++ price-volume) stacked into
//! `(N, F)` and recorded on the daily pulse — the model-ready panel the
//! predictors regress on. See [`build_features`].

use tradingflow::clock::WallClock;
use tradingflow::data::{Array, ArrayView, Duration, Instant, Retention, Series, SeriesView};
use tradingflow::graph::{Builder, Operator};
use tradingflow::operators::{
    formula::*, metrics::*, num::*, rolling::*, stocks::*, structural::*, traders::*, transform::*,
};
use tradingflow::ports::{ArrayPort, ArrayPortHandle, SeriesPort, SeriesPortHandle, UnitPortHandle};

use super::data::Stacked;

/// Trading days a 变动/同比 level is lagged (the 次新 ~1-year idiom); the
/// self-recording [`change`] / [`growth`] retain exactly this look-back.
const LAG_YEAR: usize = 244;

/// A named set of **model-ready feature** handles, in column order — each factor
/// is already cross-sectionally ranked and its missing values imputed (see
/// [`rank_impute`]), so a consumer reads the value the model actually uses.
pub struct FactorSet {
    pub names: Vec<String>,
    pub feature: Vec<ArrayPortHandle<f64, 1>>,
}

/// Turn a raw factor into its model-ready feature: cross-sectionally percentile-
/// rank to `[0, 1]`, then impute a missing value (a `NaN` rank) with the neutral
/// median `0.5`. Imputation is part of the feature's definition — a stock with
/// partial factor coverage stays predictable instead of being dropped by the
/// predictor's all-features-finite mask (which, against the sparse early
/// fundamental data, would otherwise leave the book in cash / a flat NAV for
/// years). Every factor is rank-transformed, so `0.5` is the common neutral fill.
/// The rank → fill chain is fused into one node via `segment!`.
pub(super) fn rank_impute(
    sc: &mut Builder<Instant, WallClock>,
    h: ArrayPortHandle<f64, 1>,
) -> ArrayPortHandle<f64, 1> {
    sc.segment(
        tradingflow::segment!(|x: ArrayPort<f64, 1>| -> ArrayPort<f64, 1> {
            fillna(0.5) @ percentile() @ x
        }),
        h,
    )
}

/// Total market cap = unadjusted close × total shares.
pub fn market_cap(sc: &mut Builder<Instant, WallClock>, st: &Stacked) -> ArrayPortHandle<f64, 1> {
    sc.segment(multiply(), (st.close, st.total_shares))
}

/// Circulating market cap = unadjusted close × circulating shares.
pub fn circ_market_cap(sc: &mut Builder<Instant, WallClock>, st: &Stacked) -> ArrayPortHandle<f64, 1> {
    sc.segment(multiply(), (st.close, st.circ_shares))
}

/// Trailing-twelve-month of an annualized flow: a 365-day rolling mean of the
/// annualized (effective-date-aligned) series, via the self-recording
/// [`ma_time`] (record fused in, retention sized internally).
fn ttm(sc: &mut Builder<Instant, WallClock>, h: ArrayPortHandle<f64, 1>) -> ArrayPortHandle<f64, 1> {
    sc.segment(ma_time(Duration::from_days(365)), h)
}

/// 变动 (period-over-period delta): `level − level₋₁ᵧ`. The whole chain —
/// resample the level onto the daily close pulse, then the self-recording
/// 244-day [`change`] — is fused into ONE node via a `segment!`. The shared
/// `level` block is computed once outside, so it is not recomputed here.
/// The 次新 listing filter excludes names without a full prior year.
fn delta(
    sc: &mut Builder<Instant, WallClock>,
    st: &Stacked,
    level: ArrayPortHandle<f64, 1>,
) -> ArrayPortHandle<f64, 1> {
    sc.segment(
        tradingflow::segment!(|adj: ArrayPort<f64, 1>, lvl: ArrayPort<f64, 1>| -> ArrayPort<f64, 1> {
            change(LAG_YEAR) @ resample_view() @ (adj, lvl)
        }),
        (st.adjusted_close, level),
    )
}

/// 同比 (YoY growth): the faithful `(cur − prev) / prev`, i.e. the
/// self-recording [`growth`] over the same resampled daily pulse as
/// [`delta`].
fn yoy(
    sc: &mut Builder<Instant, WallClock>,
    st: &Stacked,
    level: ArrayPortHandle<f64, 1>,
) -> ArrayPortHandle<f64, 1> {
    sc.segment(
        tradingflow::segment!(|adj: ArrayPort<f64, 1>, lvl: ArrayPort<f64, 1>| -> ArrayPort<f64, 1> {
            growth(LAG_YEAR) @ resample_view() @ (adj, lvl)
        }),
        (st.adjusted_close, level),
    )
}

/// Build the fundamental factor catalog. Each entry is `(name, raw_handle)`,
/// added in category order.
pub fn build_factor_catalog(sc: &mut Builder<Instant, WallClock>, st: &Stacked) -> FactorSet {
    let mut names = Vec::new();
    let mut raw = Vec::new();
    let mut add = |name: &str, h: ArrayPortHandle<f64, 1>| {
        names.push(name.to_string());
        raw.push(h);
    };

    // Shared building blocks.
    let mc = market_cap(sc, st);
    let fc = circ_market_cap(sc, st);
    let np_ttm = ttm(sc, st.net_profit);
    let ocf_ttm = ttm(sc, st.net_operating_cashflow);
    let rev_ttm = ttm(sc, st.revenue);
    let op_ttm = ttm(sc, st.operating_profit);
    let cost_ttm = ttm(sc, st.operating_cost); // negative (a deduction)
    // Gross profit = revenue − COGS = revenue + cost_ttm (cost stored negative).
    // (`operators::add` is fully qualified — the local `add` binding shadows it.)
    let gross_ttm = sc.segment(tradingflow::operators::num::add(), (rev_ttm, cost_ttm));
    // Positive-magnitude liabilities (parquet stores them credit-negative).
    let debt = sc.segment(negate(), st.total_liab);
    let cur_liab = sc.segment(negate(), st.current_liab);
    let eps = sc.segment(divide(), (np_ttm, st.total_shares)); // 每股收益 TTM
    let cogs_ttm = sc.segment(negate(), cost_ttm); // 营业成本 (positive magnitude)
    // 应计利润 = 净利润 − 经营现金流 (earnings not backed by cash).
    let accruals_ttm = sc.segment(subtract(), (np_ttm, ocf_ttm));
    // 投入资本 ≈ 总资产 − 流动负债 (net of non-interest-bearing current liabilities).
    let invested_capital = sc.segment(subtract(), (st.total_assets, cur_liab));

    // ---- Profitability (盈利能力) ----
    let roe = sc.segment(divide(), (np_ttm, st.parent_equity)); // 净利润 TTM / 净资产
    let roa = sc.segment(divide(), (np_ttm, st.total_assets)); // 净利润 TTM / 总资产
    let cfoa = sc.segment(divide(), (ocf_ttm, st.total_assets)); // 经营现金流净额 TTM / 总资产
    // 资本回报率 ≈ 营业利润 TTM / 投入资本 (constant-tax proxy for 息前税后经营利润).
    let roic = sc.segment(divide(), (op_ttm, invested_capital));
    add("ROE_TTM", roe);
    add("ROA_TTM", roa);
    add("CFOA", cfoa);
    add("ROIC_TTM", roic);

    // ---- Valuation (估值) ----
    add("BP_LR", sc.segment(divide(), (st.parent_equity, mc))); // 净资产 / 总市值
    add("EP_TTM", sc.segment(divide(), (np_ttm, mc))); // 净利润 TTM / 总市值
    add("SP_TTM", sc.segment(divide(), (rev_ttm, mc))); // 营业收入 TTM / 总市值
    add("OCFP_TTM", sc.segment(divide(), (ocf_ttm, mc))); // 经营现金流 TTM / 总市值

    // ---- Size (规模) ----
    add("Ln_MC", sc.segment(log(), mc)); // 总市值对数
    add("Ln_FC", sc.segment(log(), fc)); // 流通市值对数
    add("FC_MC", sc.segment(divide(), (fc, mc))); // 流通市值 / 总市值

    // ---- Operating efficiency (营运效率) ----
    let at = sc.segment(divide(), (rev_ttm, st.total_assets)); // 营业收入 TTM / 总资产
    let opm = sc.segment(divide(), (op_ttm, rev_ttm)); // 营业利润率
    let gpm = sc.segment(divide(), (gross_ttm, rev_ttm)); // 毛利率
    let invt = sc.segment(divide(), (cogs_ttm, st.inventories)); // 存货周转率 = 营业成本 TTM / 存货
    let rat = sc.segment(divide(), (rev_ttm, st.receivables)); // 应收周转率 = 营业收入 TTM / 应收款
    add("AT", at);
    add("NPM_TTM", sc.segment(divide(), (np_ttm, rev_ttm))); // 净利率
    add("OPM_TTM", opm);
    add("GPM_TTM", gpm);
    add("OPtoGR_TTM", sc.segment(divide(), (op_ttm, gross_ttm))); // 营业利润 / 毛利润
    add("INVT", invt);
    add("RAT", rat);

    // ---- Safety (安全性) ----
    let debt_asset = sc.segment(divide(), (debt, st.total_assets)); // 资产负债比
    let cur = sc.segment(divide(), (st.current_assets, cur_liab)); // 流动比率
    let dte = sc.segment(divide(), (debt, st.parent_equity)); // 产权比率
    let ccr = sc.segment(divide(), (ocf_ttm, cur_liab)); // 现金流动负债比率
    add("Debt_Asset", debt_asset);
    add("CUR", cur);
    add("DTE", dte);
    add("CCR", ccr);

    // ---- Earnings quality (盈余质量) ----
    let csr = sc.segment(divide(), (st.cash, cur_liab)); // 现金比率
    // 应计利润占比 = 应计利润 / 总资产 (Sloan accruals; expect a NEGATIVE forward
    // IC — the accruals anomaly). Handbook may scale by operating profit instead.
    let apr = sc.segment(divide(), (accruals_ttm, st.total_assets));
    add("CSR", csr);
    add("APR_TTM", apr);

    // ---- 变动 (period-over-period deltas) ----
    add("ROED", delta(sc, st, roe));
    add("ROAD", delta(sc, st, roa));
    add("CFOAD", delta(sc, st, cfoa));
    add("ROICD", delta(sc, st, roic));
    add("ATD", delta(sc, st, at));
    add("OPMD", delta(sc, st, opm));
    add("GPMD", delta(sc, st, gpm));
    add("INVTD", delta(sc, st, invt));
    add("RATD", delta(sc, st, rat));
    add("DAD", delta(sc, st, debt_asset));
    add("CURD", delta(sc, st, cur));
    add("DTED", delta(sc, st, dte));
    add("CCRD", delta(sc, st, ccr));
    add("CSRD", delta(sc, st, csr));
    add("APRD", delta(sc, st, apr));

    // ---- 同比 (TTM year-over-year growth) ----
    add("NP_YOY", yoy(sc, st, np_ttm));
    add("OR_YOY", yoy(sc, st, rev_ttm));
    add("OCF_YOY", yoy(sc, st, ocf_ttm));
    add("ROE_YOY", yoy(sc, st, roe));
    add("TA_YOY", yoy(sc, st, st.total_assets));
    add("EPS_YOY", yoy(sc, st, eps));

    // ---- Reversal (alignment sanity probe) ----
    // Trailing ~1-month log return of adjusted close. Expect a NEGATIVE forward
    // IC (short-term reversal); a contemporaneous (non-lagged) wiring bug would
    // instead make this strongly POSITIVE (the factor overlapping the return).
    let log_adj = sc.segment(log(), st.adjusted_close);
    add("REV_1M", sc.segment(change(21), log_adj));

    // Each catalog entry is finalized into its model-ready feature: rank + impute.
    let feature = raw.into_iter().map(|h| rank_impute(sc, h)).collect();
    FactorSet { names, feature }
}

/// Forward (next-period) return on the rebalance clock: resample log adjusted
/// close onto the rebalance pulse, then first-difference across rebalance ticks.
/// At rebalance `t` this is the realized log return over `[t-1, t]`; paired with
/// the factor stored at `t-1` inside `InformationCoefficient`, it is the
/// next-period return the factor is meant to predict.
pub fn build_forward_return(
    sc: &mut Builder<Instant, WallClock>,
    log_adj: ArrayPortHandle<f64, 1>,
    rebalance_clock: UnitPortHandle,
) -> ArrayPortHandle<f64, 1> {
    let resampled = sc.segment(resample_clocked(), (rebalance_clock, log_adj));
    sc.segment(diff(), resampled)
}

/// A cross-sectional feature panel.
pub struct Features {
    /// Per-feature live view handles (each `(num_stocks,)`), in column order.
    pub names: Vec<String>,
    pub handles: Vec<ArrayPortHandle<f64, 1>>,
    /// Trading-day-aligned `Record` of the `(num_stocks, n_features)` panel.
    pub series: SeriesPortHandle<f64, 2>,
}

// ===========================================================================
// Price-volume factor catalog (CICC 《价量因子手册》).
// ===========================================================================

fn rsum(
    sc: &mut Builder<Instant, WallClock>,
    s: SeriesPortHandle<f64, 1>,
    n: usize,
) -> ArrayPortHandle<f64, 1> {
    sc.segment(rolling_sum(Window::Count(n)), s)
}
fn rmean(
    sc: &mut Builder<Instant, WallClock>,
    s: SeriesPortHandle<f64, 1>,
    n: usize,
) -> ArrayPortHandle<f64, 1> {
    sc.segment(rolling_mean(Window::Count(n)), s)
}
/// Elementwise map preserving shape and NaN.
fn emap(
    sc: &mut Builder<Instant, WallClock>,
    h: ArrayPortHandle<f64, 1>,
    f: fn(f64) -> f64,
) -> ArrayPortHandle<f64, 1> {
    sc.segment(
        map(move |a: ArrayView<f64, 1>| {
            let s = a.to_contiguous();
            Array::from_vec([s.len()], s.iter().map(|&x| f(x)).collect())
        }),
        h,
    )
}
/// Rolling std = sqrt of the count-window variance (variance → sqrt fused).
fn rstd(
    sc: &mut Builder<Instant, WallClock>,
    s: SeriesPortHandle<f64, 1>,
    n: usize,
) -> ArrayPortHandle<f64, 1> {
    sc.segment(
        tradingflow::segment!(|x: SeriesPort<f64, 1>| -> ArrayPort<f64, 1> {
            sqrt() @ rolling_variance(Window::Count(n)) @ x
        }),
        s,
    )
}
/// Per-stock rolling Pearson correlation of two daily handles over `n` ticks:
/// `cov(x,y) = mean(xy) − mean(x)·mean(y)`, normalized by `σx·σy`. Records each
/// input internally (some redundancy across factors, but keeps call sites simple).
fn rcorr(
    sc: &mut Builder<Instant, WallClock>,
    x: ArrayPortHandle<f64, 1>,
    y: ArrayPortHandle<f64, 1>,
    n: usize,
) -> ArrayPortHandle<f64, 1> {
    let xs = sc.segment(buffer(n), x);
    let ys = sc.segment(buffer(n), y);
    let xy = sc.segment(multiply(), (x, y));
    let xys = sc.segment(buffer(n), xy);
    let mx = rmean(sc, xs, n);
    let my = rmean(sc, ys, n);
    let mxy = rmean(sc, xys, n);
    let mxmy = sc.segment(multiply(), (mx, my));
    let cov = sc.segment(subtract(), (mxy, mxmy));
    let sx = rstd(sc, xs, n);
    let sy = rstd(sc, ys, n);
    let sxsy = sc.segment(multiply(), (sx, sy));
    sc.segment(divide(), (cov, sxsy))
}

const M: usize = 21; // ~1 trading month
const Y: usize = 252; // ~1 trading year

// ---------------------------------------------------------------------------
// WindowReduce: a per-element rolling-window reduction that needs the full
// window (not an incremental accumulator) — for time-series rank / argmax-age.
// Reads the last `window` cross-sections of the recorded series directly.
// ---------------------------------------------------------------------------
#[derive(Clone)]
struct WindowReduce {
    window: usize,
    f: fn(&[f64]) -> f64,
}
struct WindowReduceState {
    window: usize,
    f: fn(&[f64]) -> f64,
    out: Array<f64, 1>,
}
impl Operator for WindowReduce {
    type Inputs = SeriesPort<f64, 1>;
    type Outputs = ArrayPort<f64, 1>;
    type Context = Instant;
    type State = WindowReduceState;

    fn init(self) -> WindowReduceState {
        WindowReduceState {
            window: self.window,
            f: self.f,
            out: Array::zeros([0]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, series): (bool, SeriesView<'a, f64, 1>),
        _: &Instant,
        state: &'b mut WindowReduceState,
        init: bool,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        if init {
            let n = series.elem_len();
            state.out = Array::from_vec([n], vec![f64::NAN; n]);
            return (false, state.out.view());
        }
        let w = state.window;
        let len = series.len();
        if len < w {
            return (false, state.out.view());
        }
        let f = state.f;
        let n = series.elem_len();
        let slices: Vec<&[f64]> = (0..w).map(|k| series.elem(len - w + k).data()).collect();
        let out = state.out.data_mut();
        let mut buf = vec![0.0f64; w];
        for j in 0..n {
            for k in 0..w {
                buf[k] = slices[k][j];
            }
            out[j] = f(&buf);
        }
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, SeriesView<'a, f64, 1>),
        _: &Instant,
        state: &'b WindowReduceState,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        (false, state.out.view())
    }
}
fn window_reduce(
    sc: &mut Builder<Instant, WallClock>,
    s: SeriesPortHandle<f64, 1>,
    n: usize,
    f: fn(&[f64]) -> f64,
) -> ArrayPortHandle<f64, 1> {
    sc.segment(WindowReduce { window: n, f }, s)
}
/// Time-series percentile rank of the latest window value among finite values.
fn ts_rank(w: &[f64]) -> f64 {
    let cur = w[w.len() - 1];
    if !cur.is_finite() {
        return f64::NAN;
    }
    let (mut less, mut tot) = (0usize, 0usize);
    for &x in w {
        if x.is_finite() {
            tot += 1;
            if x < cur {
                less += 1;
            }
        }
    }
    if tot < 2 {
        f64::NAN
    } else {
        less as f64 / (tot - 1) as f64
    }
}
/// Age (ticks since) the window maximum; 0 if the latest value is the max.
fn argmax_age(w: &[f64]) -> f64 {
    let (mut best, mut bi) = (f64::NEG_INFINITY, None);
    for (i, &x) in w.iter().enumerate() {
        if x.is_finite() && x > best {
            best = x;
            bi = Some(i);
        }
    }
    match bi {
        Some(i) => (w.len() - 1 - i) as f64,
        None => f64::NAN,
    }
}

// Two-channel variant of WindowReduce over a recorded `(N, 2)` series (col 0 and
// col 1 per element), e.g. (amplitude, return) for the range-adjusted momentum.
#[derive(Clone)]
struct WindowReduce2 {
    window: usize,
    f: fn(&[f64], &[f64]) -> f64,
}
struct WindowReduce2State {
    window: usize,
    f: fn(&[f64], &[f64]) -> f64,
    out: Array<f64, 1>,
}
impl Operator for WindowReduce2 {
    type Inputs = SeriesPort<f64, 2>;
    type Outputs = ArrayPort<f64, 1>;
    type Context = Instant;
    type State = WindowReduce2State;

    fn init(self) -> WindowReduce2State {
        WindowReduce2State {
            window: self.window,
            f: self.f,
            out: Array::zeros([0]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, series): (bool, SeriesView<'a, f64, 2>),
        _: &Instant,
        state: &'b mut WindowReduce2State,
        init: bool,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        let n = series.elem_extents()[0]; // (N, 2) -> N
        if init {
            state.out = Array::from_vec([n], vec![f64::NAN; n]);
            return (false, state.out.view());
        }
        let w = state.window;
        let len = series.len();
        if len < w {
            return (false, state.out.view());
        }
        let f = state.f;
        let slices: Vec<&[f64]> = (0..w).map(|k| series.elem(len - w + k).data()).collect();
        let out = state.out.data_mut();
        let (mut c0, mut c1) = (vec![0.0f64; w], vec![0.0f64; w]);
        for j in 0..n {
            for k in 0..w {
                c0[k] = slices[k][j * 2];
                c1[k] = slices[k][j * 2 + 1];
            }
            out[j] = f(&c0, &c1);
        }
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, SeriesView<'a, f64, 2>),
        _: &Instant,
        state: &'b WindowReduce2State,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        (false, state.out.view())
    }
}
fn window_reduce2(
    sc: &mut Builder<Instant, WallClock>,
    s: SeriesPortHandle<f64, 2>,
    n: usize,
    f: fn(&[f64], &[f64]) -> f64,
) -> ArrayPortHandle<f64, 1> {
    sc.segment(WindowReduce2 { window: n, f }, s)
}
/// 振幅调整动量: Σ(returns on the top-20%-amplitude days) − Σ(bottom-20%-amplitude
/// days). `amp` = amplitude channel, `ret` = return channel over the window.
fn range_mom(amp: &[f64], ret: &[f64]) -> f64 {
    let mut pairs: Vec<(f64, f64)> = amp
        .iter()
        .zip(ret)
        .filter(|(a, r)| a.is_finite() && r.is_finite())
        .map(|(a, r)| (*a, *r))
        .collect();
    let m = pairs.len();
    if m < 5 {
        return f64::NAN;
    }
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let k = ((m as f64) * 0.2).round().max(1.0) as usize;
    let bottom: f64 = pairs[..k].iter().map(|p| p.1).sum();
    let top: f64 = pairs[m - k..].iter().map(|p| p.1).sum();
    top - bottom
}

// ---------------------------------------------------------------------------
// ChipDist (筹码分布): a recent transaction-volume cost distribution maintained
// from daily (adjusted close, turnover). Each day deposits `turnover` weight at
// that day's close; older chips decay by later days' turnover (× (1−turnover_t)).
// Holder return of a chip bought at price C, valued at the current close P, is
// P/C − 1. Emits all 10 distribution factors per stock in one (N, 10) output.
// Input series is the recorded (N, 2) = [adjusted_close, turnover].
// ---------------------------------------------------------------------------
const CHIP_COLS: usize = 10; // ret_avg, std, skew, kurt, max_prob_ret, bal, profit_l/s, loss_s/l
#[derive(Clone)]
struct ChipDist {
    window: usize,
}
struct ChipDistState {
    window: usize,
    out: Array<f64, 2>,
}
impl Operator for ChipDist {
    type Inputs = SeriesPort<f64, 2>;
    type Outputs = ArrayPort<f64, 2>;
    type Context = Instant;
    type State = ChipDistState;

    fn init(self) -> ChipDistState {
        ChipDistState {
            window: self.window,
            out: Array::zeros([0, CHIP_COLS]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, series): (bool, SeriesView<'a, f64, 2>),
        _: &Instant,
        state: &'b mut ChipDistState,
        init: bool,
    ) -> (bool, ArrayView<'a, f64, 2>) {
        let n = series.elem_extents()[0];
        if init {
            state.out = Array::from_vec([n, CHIP_COLS], vec![f64::NAN; n * CHIP_COLS]);
            return (false, state.out.view());
        }
        let w = state.window;
        let len = series.len();
        if len < w {
            return (false, state.out.view());
        }
        let slices: Vec<&[f64]> = (0..w).map(|k| series.elem(len - w + k).data()).collect();
        let out = state.out.data_mut();
        let (mut price, mut turn) = (vec![0.0f64; w], vec![0.0f64; w]);
        let (mut weight, mut ret) = (vec![0.0f64; w], vec![0.0f64; w]);
        for j in 0..n {
            for k in 0..w {
                price[k] = slices[k][j * 2];
                turn[k] = slices[k][j * 2 + 1];
            }
            // Survival-decay weights: weight[i] = turn[i] · Π_{l>i}(1 − turn[l]).
            let mut surv = 1.0f64;
            for i in (0..w).rev() {
                let t = if turn[i].is_finite() {
                    turn[i].clamp(0.0, 1.0)
                } else {
                    0.0
                };
                weight[i] = if price[i].is_finite() && price[i] > 0.0 {
                    t * surv
                } else {
                    0.0
                };
                surv *= 1.0 - t;
            }
            let p_cur = price[w - 1];
            let total: f64 = weight.iter().sum();
            let base = j * CHIP_COLS;
            if !(p_cur.is_finite() && p_cur > 0.0) || total <= 0.0 {
                for c in 0..CHIP_COLS {
                    out[base + c] = f64::NAN;
                }
                continue;
            }
            for i in 0..w {
                weight[i] /= total;
                ret[i] = if weight[i] > 0.0 {
                    p_cur / price[i] - 1.0
                } else {
                    0.0
                };
            }
            // Weighted moments of the holder-return distribution.
            let mut avg = 0.0;
            for i in 0..w {
                avg += weight[i] * ret[i];
            }
            let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
            for i in 0..w {
                let d = ret[i] - avg;
                let (wd, d2) = (weight[i], d * d);
                m2 += wd * d2;
                m3 += wd * d2 * d;
                m4 += wd * d2 * d2;
            }
            let std = m2.sqrt();
            let skew = if m2 > 1e-16 {
                m3 / (m2 * std)
            } else {
                f64::NAN
            };
            let kurt = if m2 > 1e-16 { m4 / (m2 * m2) } else { f64::NAN };
            // Profit/loss-band chip proportions + modal-bin return.
            let (mut bal, mut pl, mut ps, mut ls, mut ll) = (0.0, 0.0, 0.0, 0.0, 0.0);
            let mut bins = [0.0f64; 50]; // [-0.5, 0.5] in 0.02-wide bins
            for i in 0..w {
                let (r, wd) = (ret[i], weight[i]);
                if wd <= 0.0 {
                    continue;
                }
                if r > 0.10 {
                    pl += wd;
                } else if r > 0.02 {
                    ps += wd;
                } else if r >= -0.02 {
                    bal += wd;
                } else if r >= -0.10 {
                    ls += wd;
                } else {
                    ll += wd;
                }
                let b = (((r + 0.5) / 0.02).floor() as isize).clamp(0, 49) as usize;
                bins[b] += wd;
            }
            let (mut maxb, mut maxv) = (0usize, bins[0]);
            for (b, &v) in bins.iter().enumerate().skip(1) {
                if v > maxv {
                    maxv = v;
                    maxb = b;
                }
            }
            let max_prob_ret = -0.5 + (maxb as f64 + 0.5) * 0.02;
            out[base..base + CHIP_COLS].copy_from_slice(&[
                avg,
                std,
                skew,
                kurt,
                max_prob_ret,
                bal,
                pl,
                ps,
                ls,
                ll,
            ]);
        }
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, SeriesView<'a, f64, 2>),
        _: &Instant,
        state: &'b ChipDistState,
    ) -> (bool, ArrayView<'a, f64, 2>) {
        (false, state.out.view())
    }
}

/// Build the price-volume factor catalog (动量 & 反转 first pass).
pub fn build_pv_catalog(sc: &mut Builder<Instant, WallClock>, st: &Stacked) -> FactorSet {
    let mut names = Vec::new();
    let mut raw = Vec::new();
    let mut add = |name: &str, h: ArrayPortHandle<f64, 1>| {
        names.push(name.to_string());
        raw.push(h);
    };

    // --- daily building blocks (recorded on the price pulse) ---
    let lc = sc.segment(log(), st.adjusted_close); // adjusted log close
    let lc_s = sc.segment(buffer(Y), lc);
    let adjclose_s = sc.segment(buffer(Y), st.adjusted_close);
    let lo = sc.segment(log(), st.open); // raw log open
    let lcr = sc.segment(log(), st.close); // raw log close
    let lcr_s = sc.segment(buffer(Y), lcr);

    let daily_ret = sc.segment(diff(), lc); // adjusted close-to-close log return
    let dret_s = sc.segment(buffer(Y), daily_ret);
    let intraday = sc.segment(subtract(), (lcr, lo)); // log(close/open)
    let intra_s = sc.segment(buffer(Y), intraday);
    let prev_lcr = sc.segment(lag_series(1, f64::NAN), lcr_s);
    let overnight = sc.segment(subtract(), (lo, prev_lcr)); // log(open / prev close)
    let over_s = sc.segment(buffer(Y), overnight);

    let abs_ret = emap(sc, daily_ret, f64::abs);
    let absret_s = sc.segment(buffer(Y), abs_ret);
    let sign_ret = emap(sc, daily_ret, |x| {
        if x.is_finite() { x.signum() } else { f64::NAN }
    });
    let sign_s = sc.segment(buffer(Y), sign_ret);
    let xs_rank = sc.segment(percentile(), daily_ret); // daily cross-sectional rank
    let xsr_s = sc.segment(buffer(Y), xs_rank);

    // shared rolling sums (reused by the _M / _A pairs)
    let rs_intra_m = rsum(sc, intra_s, M);
    let rs_intra_y = rsum(sc, intra_s, Y);
    let rs_over_m = rsum(sc, over_s, M);
    let rs_over_y = rsum(sc, over_s, Y);

    // ---- 月度反转 / 年度动量 (close-return based) ----
    let lc_lag_m = sc.segment(lag_series(M, f64::NAN), lc_s);
    let lc_lag_y = sc.segment(lag_series(Y, f64::NAN), lc_s);
    let ret_1m = sc.segment(subtract(), (lc, lc_lag_m)); // past 1-month return
    let ret_1y = sc.segment(subtract(), (lc, lc_lag_y)); // past 1-year return
    add("mmt_normal_M", ret_1m); // 1个月收益率
    add("mmt_normal_A", sc.segment(subtract(), (lc_lag_m, lc_lag_y))); // 12个月收益率 − 1个月收益率
    let ma20 = rmean(sc, adjclose_s, 20);
    add("mmt_avg_M", sc.segment(divide(), (st.adjusted_close, ma20))); // 收盘价 / 20日均价
    let prev_close_m = sc.segment(lag_series(M, f64::NAN), adjclose_s);
    let ma_year = rmean(sc, adjclose_s, Y);
    add("mmt_avg_A", sc.segment(divide(), (prev_close_m, ma_year))); // 1月前收盘 / 1年均价

    // ---- 日内 / 隔夜动量 ----
    add("mmt_intraday_M", rs_intra_m); // 过去1月日内涨跌幅之和
    add(
        "mmt_intraday_A",
        sc.segment(subtract(), (rs_intra_y, rs_intra_m)),
    );
    add("mmt_overnight_M", rs_over_m); // 过去1月隔夜涨跌幅之和
    add(
        "mmt_overnight_A",
        sc.segment(subtract(), (rs_over_y, rs_over_m)),
    );

    // ---- 路径调整动量 (return / Σ|daily return|) ----
    let sum_abs_m = rsum(sc, absret_s, M);
    add("mmt_route_M", sc.segment(divide(), (ret_1m, sum_abs_m)));
    let sum_abs_y = rsum(sc, absret_s, Y);
    add("mmt_route_A", sc.segment(divide(), (ret_1y, sum_abs_y)));

    // ---- 信息离散度 (up% − down% = mean of sign) ----
    add("mmt_discrete_M", rmean(sc, sign_s, M));
    add("mmt_discrete_A", rmean(sc, sign_s, Y));

    // ---- 横截面 rank 动量 (daily cross-sectional rank, time-averaged) ----
    add("mmt_sec_rank_M", rmean(sc, xsr_s, 20));
    add("mmt_sec_rank_A", rmean(sc, xsr_s, Y));

    // 去涨跌停动量: sum of daily returns excluding limit days. Approximate a limit
    // day as |log return| >= ~9.5% (the ±10% board); the ±20% boards (创业板/科创板)
    // are not distinguished, a documented deviation.
    let masked_ret = emap(sc, daily_ret, |x| {
        if x.is_finite() {
            if x.abs() < 0.095 { x } else { 0.0 }
        } else {
            f64::NAN
        }
    });
    let masked_s = sc.segment(buffer(Y), masked_ret);
    add("mmt_off_limit_M", rsum(sc, masked_s, M));
    add("mmt_off_limit_A", rsum(sc, masked_s, Y));

    // 时序 rank 动量: daily time-series percentile of the log price within 1 year,
    // averaged over 20 days. 最高价距今天数: ticks since the 1-year high.
    let trank = window_reduce(sc, lc_s, Y, ts_rank);
    let trank_s = sc.segment(buffer(Y), trank);
    add("mmt_time_rank_M", rmean(sc, trank_s, 20));
    add(
        "mmt_highest_days_A",
        window_reduce(sc, adjclose_s, Y, argmax_age),
    );

    // ============ 流动性 (liquidity) — 图表28 ============
    let turnover = sc.segment(divide(), (st.volume, st.circ_shares)); // daily turnover 换手率
    let turnover_s = sc.segment(buffer(Y), turnover);
    let amount_s = sc.segment(buffer(Y), st.amount);
    let amihud = sc.segment(divide(), (abs_ret, st.amount)); // |日收益率| / 成交额
    let amihud_s = sc.segment(buffer(Y), amihud);
    // 日K线最短路径 = 2*(最高-最低) − |开盘−收盘|; shortcut 非流动 = 路径 / 成交额.
    let hl = sc.segment(subtract(), (st.high, st.low));
    let two_hl = emap(sc, hl, |x| 2.0 * x);
    let oc = sc.segment(subtract(), (st.open, st.close));
    let abs_oc = emap(sc, oc, f64::abs);
    let path = sc.segment(subtract(), (two_hl, abs_oc));
    let shortcut = sc.segment(divide(), (path, st.amount));
    let shortcut_s = sc.segment(buffer(Y), shortcut);

    for (tag, w) in [("1M", M), ("3M", 63usize), ("6M", 126usize)] {
        add(&format!("liq_turn_avg_{tag}"), rmean(sc, turnover_s, w)); // 换手率均值
        add(&format!("liq_turn_std_{tag}"), rstd(sc, turnover_s, w)); // 换手率标准差
        let sum_amt = rsum(sc, amount_s, w);
        let ret_std = rstd(sc, dret_s, w);
        add(
            &format!("liq_vstd_{tag}"),
            sc.segment(divide(), (sum_amt, ret_std)),
        ); // 成交波动比 = Σ成交额 / σ(ret)
        add(&format!("liq_amihud_avg_{tag}"), rmean(sc, amihud_s, w));
        add(&format!("liq_amihud_std_{tag}"), rstd(sc, amihud_s, w));
        add(&format!("liq_shortcut_avg_{tag}"), rmean(sc, shortcut_s, w));
        add(&format!("liq_shortcut_std_{tag}"), rstd(sc, shortcut_s, w));
    }

    // ============ 量价相关性 (price-volume correlation, 20-day) — 图表40 ============
    // sync = corr(turn_t, x_t); post (量领先) = corr(turn_t, x_{t+1}) = corr(lag(turn,1), x);
    // prior (价领先) = corr(turn_t, x_{t-1}) = corr(turn, lag(x,1)).
    let turnover_change = sc.segment(diff(), turnover);
    let turnchg_s = sc.segment(buffer(Y), turnover_change);
    let lag_turn1 = sc.segment(lag_series(1, f64::NAN), turnover_s);
    let lag_turnd1 = sc.segment(lag_series(1, f64::NAN), turnchg_s);
    let lag_price1 = sc.segment(lag_series(1, f64::NAN), adjclose_s);
    let lag_ret1 = sc.segment(lag_series(1, f64::NAN), dret_s);
    const WC: usize = 20;
    add(
        "corr_price_turn_1M",
        rcorr(sc, turnover, st.adjusted_close, WC),
    );
    add(
        "corr_price_turn_post_1M",
        rcorr(sc, lag_turn1, st.adjusted_close, WC),
    );
    add(
        "corr_price_turn_prior_1M",
        rcorr(sc, turnover, lag_price1, WC),
    );
    add("corr_ret_turn_1M", rcorr(sc, turnover, daily_ret, WC));
    add("corr_ret_turn_post_1M", rcorr(sc, lag_turn1, daily_ret, WC));
    add("corr_ret_turn_prior_1M", rcorr(sc, turnover, lag_ret1, WC));
    add(
        "corr_ret_turnd_1M",
        rcorr(sc, turnover_change, daily_ret, WC),
    );
    add(
        "corr_ret_turnd_post_1M",
        rcorr(sc, lag_turnd1, daily_ret, WC),
    );
    add(
        "corr_ret_turnd_prior_1M",
        rcorr(sc, turnover_change, lag_ret1, WC),
    );

    // ============ 波动率 (volatility) — 图表16 ============
    // Downside / upside daily returns (positive/negative clamped to 0).
    let downside = emap(sc, daily_ret, |x| {
        if x.is_finite() { x.min(0.0) } else { f64::NAN }
    });
    let downside_s = sc.segment(buffer(Y), downside);
    let upside = emap(sc, daily_ret, |x| {
        if x.is_finite() { x.max(0.0) } else { f64::NAN }
    });
    let upside_s = sc.segment(buffer(Y), upside);
    let highlow = sc.segment(divide(), (st.high, st.low)); // 日内振幅 = 最高/最低
    let highlow_s = sc.segment(buffer(Y), highlow);
    // Candlestick shadows, normalized by the low (cross-sectional price-level scale).
    let max_oc = sc.segment(max(), (st.open, st.close));
    let min_oc = sc.segment(min(), (st.open, st.close));
    let up_num = sc.segment(subtract(), (st.high, max_oc)); // 上影线 = 最高 − max(开,收)
    let upshadow = sc.segment(divide(), (up_num, st.low));
    let upshadow_s = sc.segment(buffer(Y), upshadow);
    let down_num = sc.segment(subtract(), (min_oc, st.low)); // 下影线 = min(开,收) − 最低
    let downshadow = sc.segment(divide(), (down_num, st.low));
    let downshadow_s = sc.segment(buffer(Y), downshadow);
    let wup_num = sc.segment(subtract(), (st.high, st.close)); // 威廉上影线 = 最高 − 收
    let w_upshadow = sc.segment(divide(), (wup_num, st.low));
    let w_upshadow_s = sc.segment(buffer(Y), w_upshadow);
    let wdown_num = sc.segment(subtract(), (st.close, st.low)); // 威廉下影线 = 收 − 最低
    let w_downshadow = sc.segment(divide(), (wdown_num, st.low));
    let w_downshadow_s = sc.segment(buffer(Y), w_downshadow);

    for (tag, w) in [("1M", M), ("3M", 63usize), ("6M", 126usize)] {
        add(&format!("vol_std_{tag}"), rstd(sc, dret_s, w)); // 波动率
        add(&format!("vol_down_std_{tag}"), rstd(sc, downside_s, w)); // 下行波动率
        add(&format!("vol_up_std_{tag}"), rstd(sc, upside_s, w)); // 上行波动率
        add(&format!("vol_highlow_avg_{tag}"), rmean(sc, highlow_s, w));
        add(&format!("vol_highlow_std_{tag}"), rstd(sc, highlow_s, w));
        add(&format!("vol_upshadow_avg_{tag}"), rmean(sc, upshadow_s, w));
        add(&format!("vol_upshadow_std_{tag}"), rstd(sc, upshadow_s, w));
        add(
            &format!("vol_downshadow_avg_{tag}"),
            rmean(sc, downshadow_s, w),
        );
        add(
            &format!("vol_downshadow_std_{tag}"),
            rstd(sc, downshadow_s, w),
        );
        add(
            &format!("vol_w_upshadow_avg_{tag}"),
            rmean(sc, w_upshadow_s, w),
        );
        add(
            &format!("vol_w_upshadow_std_{tag}"),
            rstd(sc, w_upshadow_s, w),
        );
        add(
            &format!("vol_w_downshadow_avg_{tag}"),
            rmean(sc, w_downshadow_s, w),
        );
        add(
            &format!("vol_w_downshadow_std_{tag}"),
            rstd(sc, w_downshadow_s, w),
        );
    }

    // 振幅调整动量 (mmt_range): pair amplitude (high/low) with the daily return as a
    // (N,2) series and reduce each window via `range_mom`. The native carry-join
    // `Stack` along axis 1 wires the two by-value view handles into an (N, 2)
    // cross-section, so `WindowReduce2` reads `[amp_i, ret_i]` pairs per element.
    let amp_ret = sc.segment(stack::<f64, 1, 2>(1), &[highlow, daily_ret]); // (N, 2)
    let amp_ret_s = sc.segment(buffer(Y), amp_ret);
    add("mmt_range_M", window_reduce2(sc, amp_ret_s, M, range_mom));
    add("mmt_range_A", window_reduce2(sc, amp_ret_s, Y, range_mom));

    // ============ 筹码分布 (chip distribution) — 图表52 ============
    // (adjusted close, turnover) -> ChipDist -> (N, 10); column-select each factor.
    let chip_in = sc.segment(stack::<f64, 1, 2>(1), &[st.adjusted_close, turnover]); // (N, 2)
    let chip_in_s = sc.segment(buffer(Y), chip_in);
    let chip = sc.segment(ChipDist { window: 250 }, chip_in_s); // (N, 10)
    for (c, name) in [
        "distribution_ret_avg",
        "distribution_ret_std",
        "distribution_ret_skew",
        "distribution_ret_kurt",
        "distribution_max_prob_ret",
        "distribution_bal",
        "distribution_profit_l",
        "distribution_profit_s",
        "distribution_loss_s",
        "distribution_loss_l",
    ]
    .into_iter()
    .enumerate()
    {
        add(name, sc.segment(select(vec![c], 1, true), chip));
    }

    // Finalize each entry into its model-ready feature: rank + impute.
    let feature = raw.into_iter().map(|h| rank_impute(sc, h)).collect();
    FactorSet { names, feature }
}

/// Build the cross-sectionally percentile-ranked feature panel: the whole CICC
/// handbook catalog (fundamental ++ price-volume), stacked into `(N, F)`,
/// resampled onto the daily close pulse and recorded under `feature_retention`
/// (pass [`Retention::UNBOUNDED`] when a consumer needs full history).
///
/// Each catalog entry is already the model-ready feature (cross-sectionally
/// ranked, missing values imputed to the neutral median — see [`rank_impute`] —
/// so a stock with partial factor coverage stays predictable rather than dropped
/// by the predictor's all-features-finite mask). The panel is heavily collinear
/// by design and leans on the predictors' pool-standardized Ridge (`alpha`) to
/// regularise.
pub fn build_features(sc: &mut Builder<Instant, WallClock>, st: &Stacked, feature_retention: Retention) -> Features {
    let fund = build_factor_catalog(sc, st);
    let pv = build_pv_catalog(sc, st);
    let mut names = fund.names;
    let mut handles = fund.feature;
    names.extend(pv.names);
    handles.extend(pv.feature);

    let stacked = sc.segment(stack(1), &handles[..]);
    let sampled = sc.segment(resample_view(), (st.adjusted_close, stacked));
    let series = sc.segment(record_bounded(feature_retention), sampled);
    Features {
        names,
        handles,
        series,
    }
}
