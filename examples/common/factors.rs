//! Fundamental factor catalog for the CICC 基本面因子手册 replication.
//!
//! Each factor is a small native sub-graph over the [`Stacked`](super::Stacked)
//! cross-sectional panel, returning a `[num_stocks]` factor handle on the panel's
//! native cadence (daily for price-derived, report-effective-date for
//! fundamentals, carried forward by the panel's `Stack`). The driver
//! ([`examples/factor_handbook.rs`]) resamples each onto the rebalance clock,
//! masks to the universe, cross-sectionally ranks ([`Percentile`]), and evaluates
//! RankIC against the forward return.
//!
//! TTM (trailing-twelve-month) flows are a 365-day rolling mean of the annualized
//! quarterly flow (the self-recording [`ma_time`] constructor) — the same idiom
//! as [`build_features`](super::build_features). Balance-sheet stocks
//! (assets, equity, liabilities, cash) are used at their latest carried-forward
//! value (no TTM). The parquet stores assets debit-positive and liabilities /
//! expense items credit-**negative**; the formulas negate them where a positive
//! magnitude is wanted (`debt`, `cur_liab`, gross profit = revenue + cost).
//!
//! **Levels** are single shared sub-expressions (`np_ttm`, `gross_ttm`, `debt`…
//! computed once, reused), so each level factor is one `Divide`/`Log`. The
//! period-over-period **变动 (delta)** and **同比 (YoY growth)** factors need a
//! ~1-year lag of an irregular-cadence series: the level is resampled onto the
//! daily close pulse and run through the self-recording 244-day [`change`] /
//! [`growth`] (the 次新 idiom), fused into one node via a nested
//! [`segment!`](tradingflow_graph::segment) — the form recommended for individual
//! factors. (`(cur − prev)/prev` is NOT rank-equivalent to `cur/prev` once a
//! year-ago base goes negative, so the subtraction is kept rather than dropped.)
//!
//! Catalog is config: adding a factor is adding an `add("NAME", handle)` line.
//! Still deferred: single-quarter YoY/QoQ (need quarterly deltas), z-scores
//! (rank-identical to their base under RankIC), TOE (应交税费 with a year-ago lag
//! and a cash-tax-paid term), dividend yield/payout (DP/DPR), and the regression
//! / SUE / composite (Profit/Growth/Safe/QQC) factors.

use tradingflow::Scenario;
use tradingflow::data::Duration;
use tradingflow::graph::typed::{PortHandle, RefPort, ViewPort};
use tradingflow::operators::{
    self, ArrayPort, ResampleView, change, diff, divide, fillna, growth, log, ma_time, multiply,
    negate, percentile, resample_clocked, subtract,
};

use super::{AvH, Stacked};

type H = AvH;

/// Trading days a 变动/同比 level is lagged (the 次新 ~1-year idiom); the
/// self-recording [`change`] / [`growth`] retain exactly this look-back.
const LAG_YEAR: usize = 244;

/// Resample `data` onto a daily-`Array` clock pulse. Aliased to keep the
/// `segment!` application sites short.
type ResampleDaily = ResampleView<f64, 1>;

/// A named set of **model-ready feature** handles, in column order — each factor
/// is already cross-sectionally ranked and its missing values imputed (see
/// [`rank_impute`]), so a consumer reads the value the model actually uses.
pub struct FactorSet {
    pub names: Vec<String>,
    pub feature: Vec<H>,
}

/// Turn a raw factor into its model-ready feature: cross-sectionally percentile-
/// rank to `[0, 1]`, then impute a missing value (a `NaN` rank) with the neutral
/// median `0.5`. Imputation is part of the feature's definition — a stock with
/// partial factor coverage stays predictable instead of being dropped by the
/// predictor's all-features-finite mask (which, against the sparse early
/// fundamental data, would otherwise leave the book in cash / a flat NAV for
/// years). Every factor is rank-transformed, so `0.5` is the common neutral fill.
/// The rank → fill chain is fused into one node via `segment!`.
pub(super) fn rank_impute(sc: &mut Scenario, h: H) -> H {
    sc.segment(
        tradingflow_graph::segment!(|x: ArrayPort<f64, 1>| -> ArrayPort<f64, 1> {
            fillna(0.5) @ percentile() @ x
        }),
        h,
    )
}

/// Total market cap = unadjusted close × total shares.
pub fn market_cap(sc: &mut Scenario, st: &Stacked) -> H {
    sc.segment(multiply(), (st.close, st.total_shares))
}

/// Circulating market cap = unadjusted close × circulating shares.
pub fn circ_market_cap(sc: &mut Scenario, st: &Stacked) -> H {
    sc.segment(multiply(), (st.close, st.circ_shares))
}

/// Trailing-twelve-month of an annualized flow: a 365-day rolling mean of the
/// annualized (effective-date-aligned) series, via the self-recording
/// [`ma_time`] (record fused in, retention sized internally).
fn ttm(sc: &mut Scenario, h: H) -> H {
    sc.segment(ma_time(Duration::from_days(365)), h)
}

fn div(sc: &mut Scenario, a: H, b: H) -> H {
    sc.segment(divide(), (a, b))
}

fn log_h(sc: &mut Scenario, h: H) -> H {
    sc.segment(log(), h)
}

fn neg(sc: &mut Scenario, h: H) -> H {
    sc.segment(negate(), h)
}

/// 变动 (period-over-period delta): `level − level₋₁ᵧ`. The whole chain —
/// resample the level onto the daily close pulse, then the self-recording
/// 244-day [`change`] — is fused into ONE node via a `segment!`. The shared
/// `level` block is computed once outside, so it is not recomputed here.
/// The 次新 listing filter excludes names without a full prior year.
fn delta(sc: &mut Scenario, st: &Stacked, level: H) -> H {
    sc.segment(
        tradingflow_graph::segment!(|adj: ArrayPort<f64, 1>, lvl: ArrayPort<f64, 1>| -> ArrayPort<f64, 1> {
            change(LAG_YEAR) @ ResampleDaily::new() @ (adj, lvl)
        }),
        (st.adjusted_close, level),
    )
}

/// 同比 (YoY growth): the faithful `(cur − prev) / prev`, i.e. the
/// self-recording [`growth`] over the same resampled daily pulse as
/// [`delta`].
fn yoy(sc: &mut Scenario, st: &Stacked, level: H) -> H {
    sc.segment(
        tradingflow_graph::segment!(|adj: ArrayPort<f64, 1>, lvl: ArrayPort<f64, 1>| -> ArrayPort<f64, 1> {
            growth(LAG_YEAR) @ ResampleDaily::new() @ (adj, lvl)
        }),
        (st.adjusted_close, level),
    )
}

/// Build the fundamental factor catalog. Each entry is `(name, raw_handle)`,
/// added in category order.
pub fn build_factor_catalog(sc: &mut Scenario, st: &Stacked) -> FactorSet {
    let mut names = Vec::new();
    let mut raw = Vec::new();
    let mut add = |name: &str, h: H| {
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
    let gross_ttm = sc.segment(operators::add(), (rev_ttm, cost_ttm));
    // Positive-magnitude liabilities (parquet stores them credit-negative).
    let debt = neg(sc, st.total_liab);
    let cur_liab = neg(sc, st.current_liab);
    let eps = div(sc, np_ttm, st.total_shares); // 每股收益 TTM
    let cogs_ttm = neg(sc, cost_ttm); // 营业成本 (positive magnitude)
    // 应计利润 = 净利润 − 经营现金流 (earnings not backed by cash).
    let accruals_ttm = sc.segment(subtract(), (np_ttm, ocf_ttm));
    // 投入资本 ≈ 总资产 − 流动负债 (net of non-interest-bearing current liabilities).
    let invested_capital = sc.segment(subtract(), (st.total_assets, cur_liab));

    // ---- Profitability (盈利能力) ----
    let roe = div(sc, np_ttm, st.parent_equity); // 净利润 TTM / 净资产
    let roa = div(sc, np_ttm, st.total_assets); // 净利润 TTM / 总资产
    let cfoa = div(sc, ocf_ttm, st.total_assets); // 经营现金流净额 TTM / 总资产
    // 资本回报率 ≈ 营业利润 TTM / 投入资本 (constant-tax proxy for 息前税后经营利润).
    let roic = div(sc, op_ttm, invested_capital);
    add("ROE_TTM", roe);
    add("ROA_TTM", roa);
    add("CFOA", cfoa);
    add("ROIC_TTM", roic);

    // ---- Valuation (估值) ----
    add("BP_LR", div(sc, st.parent_equity, mc)); // 净资产 / 总市值
    add("EP_TTM", div(sc, np_ttm, mc)); // 净利润 TTM / 总市值
    add("SP_TTM", div(sc, rev_ttm, mc)); // 营业收入 TTM / 总市值
    add("OCFP_TTM", div(sc, ocf_ttm, mc)); // 经营现金流 TTM / 总市值

    // ---- Size (规模) ----
    add("Ln_MC", log_h(sc, mc)); // 总市值对数
    add("Ln_FC", log_h(sc, fc)); // 流通市值对数
    add("FC_MC", div(sc, fc, mc)); // 流通市值 / 总市值

    // ---- Operating efficiency (营运效率) ----
    let at = div(sc, rev_ttm, st.total_assets); // 营业收入 TTM / 总资产
    let opm = div(sc, op_ttm, rev_ttm); // 营业利润率
    let gpm = div(sc, gross_ttm, rev_ttm); // 毛利率
    let invt = div(sc, cogs_ttm, st.inventories); // 存货周转率 = 营业成本 TTM / 存货
    let rat = div(sc, rev_ttm, st.receivables); // 应收周转率 = 营业收入 TTM / 应收款
    add("AT", at);
    add("NPM_TTM", div(sc, np_ttm, rev_ttm)); // 净利率
    add("OPM_TTM", opm);
    add("GPM_TTM", gpm);
    add("OPtoGR_TTM", div(sc, op_ttm, gross_ttm)); // 营业利润 / 毛利润
    add("INVT", invt);
    add("RAT", rat);

    // ---- Safety (安全性) ----
    let debt_asset = div(sc, debt, st.total_assets); // 资产负债比
    let cur = div(sc, st.current_assets, cur_liab); // 流动比率
    let dte = div(sc, debt, st.parent_equity); // 产权比率
    let ccr = div(sc, ocf_ttm, cur_liab); // 现金流动负债比率
    add("Debt_Asset", debt_asset);
    add("CUR", cur);
    add("DTE", dte);
    add("CCR", ccr);

    // ---- Earnings quality (盈余质量) ----
    let csr = div(sc, st.cash, cur_liab); // 现金比率
    // 应计利润占比 = 应计利润 / 总资产 (Sloan accruals; expect a NEGATIVE forward
    // IC — the accruals anomaly). Handbook may scale by operating profit instead.
    let apr = div(sc, accruals_ttm, st.total_assets);
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
    let log_adj = log_h(sc, st.adjusted_close);
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
    sc: &mut Scenario,
    log_adj: H,
    rebalance_clock: PortHandle<RefPort<()>>,
) -> H {
    let resampled = sc.segment(resample_clocked(), (rebalance_clock, log_adj));
    sc.segment(diff(), resampled)
}
