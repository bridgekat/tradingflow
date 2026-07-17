//! Price-volume factor catalog for the CICC 《价量因子手册》 replication.
//!
//! Reuses the same evaluation harness as the fundamental catalog
//! ([`super::factors`]): each factor is a `[num_stocks]` handle on the daily
//! pulse; the driver resamples onto the rebalance clock, masks to the universe,
//! cross-sectionally ranks, and evaluates RankIC vs the next-period return.
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

use tradingflow::graph::{PortHandle, Operator};

use tradingflow::operators::{
    ArrayPort, SeriesPort, Window, apply, diff, divide, lag_series, log, max, min, multiply,
    percentile, record_bounded, rolling_mean, rolling_sum, rolling_variance, select, sqrt, stack,
    subtract,
};
use tradingflow::{Array, ArrayView, Instant, Retention, Scenario, SeriesView};

use super::factors::{FactorSet, rank_impute};
use super::{AvH, RETAIN_MARGIN, Stacked};

type H = AvH;
type Ser = PortHandle<SeriesPort<f64, 1>>;
type Ser2 = PortHandle<SeriesPort<f64, 2>>;

/// Catalog records feed count-windowed reductions up to `Y` (one trading year)
/// or the 250-day chip window; retain that deepest look-back plus the margin.
fn rec(sc: &mut Scenario, h: H) -> Ser {
    sc.segment(record_bounded(Retention::count(Y + RETAIN_MARGIN)), h)
}

fn log_h(sc: &mut Scenario, h: H) -> H {
    sc.segment(log(), h)
}
fn sub(sc: &mut Scenario, a: H, b: H) -> H {
    sc.segment(subtract(), (a, b))
}
fn div(sc: &mut Scenario, a: H, b: H) -> H {
    sc.segment(divide(), (a, b))
}
fn lag(sc: &mut Scenario, s: Ser, n: usize) -> H {
    sc.segment(lag_series(n, f64::NAN), s)
}
fn rsum(sc: &mut Scenario, s: Ser, n: usize) -> H {
    sc.segment(rolling_sum(Window::Count(n)), s)
}
fn rmean(sc: &mut Scenario, s: Ser, n: usize) -> H {
    sc.segment(rolling_mean(Window::Count(n)), s)
}
/// Elementwise map preserving shape and NaN.
fn emap(sc: &mut Scenario, h: H, f: fn(f64) -> f64) -> H {
    sc.segment(
        tradingflow::operators::map(move |a: ArrayView<f64, 1>| {
            let s = a.to_contiguous();
            Array::from_vec([s.len()], s.iter().map(|&x| f(x)).collect())
        }),
        h,
    )
}
fn mul(sc: &mut Scenario, a: H, b: H) -> H {
    sc.segment(multiply(), (a, b))
}
/// Rolling std = sqrt of the count-window variance (variance → sqrt fused).
fn rstd(sc: &mut Scenario, s: Ser, n: usize) -> H {
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
fn rcorr(sc: &mut Scenario, x: H, y: H, n: usize) -> H {
    let win = Retention::count(n + RETAIN_MARGIN);
    let xs = sc.segment(record_bounded(win), x);
    let ys = sc.segment(record_bounded(win), y);
    let xy = mul(sc, x, y);
    let xys = sc.segment(record_bounded(win), xy);
    let mx = rmean(sc, xs, n);
    let my = rmean(sc, ys, n);
    let mxy = rmean(sc, xys, n);
    let mxmy = mul(sc, mx, my);
    let cov = sub(sc, mxy, mxmy);
    let sx = rstd(sc, xs, n);
    let sy = rstd(sc, ys, n);
    let sxsy = mul(sc, sx, sy);
    div(sc, cov, sxsy)
}

const M: usize = 21; // ~1 trading month
const Y: usize = 252; // ~1 trading year

/// Stack two rank-1 `[N]` views into an `(N, 2)` cross-section (col 0 from `a`,
/// col 1 from `b`) — the native carry-join `Stack` along axis 1, wired straight
/// from the two by-value view handles. The downstream `WindowReduce2`/`ChipDist`
/// read `[a_i, b_i]` pairs per element.
fn stack2(sc: &mut Scenario, a: H, b: H) -> PortHandle<ArrayPort<f64, 2>> {
    sc.segment(stack::<f64, 1, 2>(1), &[a, b])
}

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
            let n = series.stride();
            state.out = Array::from_vec([n], vec![f64::NAN; n]);
            return (false, state.out.view());
        }
        let w = state.window;
        let len = series.len();
        if len < w {
            return (false, state.out.view());
        }
        let f = state.f;
        let n = series.stride();
        let slices: Vec<&[f64]> = (0..w).map(|k| series.at(len - w + k)).collect();
        let out = state.out.as_mut_slice();
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
fn window_reduce(sc: &mut Scenario, s: Ser, n: usize, f: fn(&[f64]) -> f64) -> H {
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
        let n = series.extents()[0]; // (N, 2) -> N
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
        let slices: Vec<&[f64]> = (0..w).map(|k| series.at(len - w + k)).collect();
        let out = state.out.as_mut_slice();
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
fn window_reduce2(sc: &mut Scenario, s: Ser2, n: usize, f: fn(&[f64], &[f64]) -> f64) -> H {
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
        let n = series.extents()[0];
        if init {
            state.out = Array::from_vec([n, CHIP_COLS], vec![f64::NAN; n * CHIP_COLS]);
            return (false, state.out.view());
        }
        let w = state.window;
        let len = series.len();
        if len < w {
            return (false, state.out.view());
        }
        let slices: Vec<&[f64]> = (0..w).map(|k| series.at(len - w + k)).collect();
        let out = state.out.as_mut_slice();
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
pub fn build_pv_catalog(sc: &mut Scenario, st: &Stacked) -> FactorSet {
    let mut names = Vec::new();
    let mut raw = Vec::new();
    let mut add = |name: &str, h: H| {
        names.push(name.to_string());
        raw.push(h);
    };

    // --- daily building blocks (recorded on the price pulse) ---
    let lc = log_h(sc, st.adjusted_close); // adjusted log close
    let lc_s = rec(sc, lc);
    let adjclose_s = rec(sc, st.adjusted_close);
    let lo = log_h(sc, st.open); // raw log open
    let lcr = log_h(sc, st.close); // raw log close
    let lcr_s = rec(sc, lcr);

    let daily_ret = sc.segment(diff(), lc); // adjusted close-to-close log return
    let dret_s = rec(sc, daily_ret);
    let intraday = sub(sc, lcr, lo); // log(close/open)
    let intra_s = rec(sc, intraday);
    let prev_lcr = lag(sc, lcr_s, 1);
    let overnight = sub(sc, lo, prev_lcr); // log(open / prev close)
    let over_s = rec(sc, overnight);

    let abs_ret = emap(sc, daily_ret, f64::abs);
    let absret_s = rec(sc, abs_ret);
    let sign_ret = emap(sc, daily_ret, |x| {
        if x.is_finite() { x.signum() } else { f64::NAN }
    });
    let sign_s = rec(sc, sign_ret);
    let xs_rank = sc.segment(percentile(), daily_ret); // daily cross-sectional rank
    let xsr_s = rec(sc, xs_rank);

    // shared rolling sums (reused by the _M / _A pairs)
    let rs_intra_m = rsum(sc, intra_s, M);
    let rs_intra_y = rsum(sc, intra_s, Y);
    let rs_over_m = rsum(sc, over_s, M);
    let rs_over_y = rsum(sc, over_s, Y);

    // ---- 月度反转 / 年度动量 (close-return based) ----
    let lc_lag_m = lag(sc, lc_s, M);
    let lc_lag_y = lag(sc, lc_s, Y);
    let ret_1m = sub(sc, lc, lc_lag_m); // past 1-month return
    let ret_1y = sub(sc, lc, lc_lag_y); // past 1-year return
    add("mmt_normal_M", ret_1m); // 1个月收益率
    add("mmt_normal_A", sub(sc, lc_lag_m, lc_lag_y)); // 12个月收益率 − 1个月收益率
    let ma20 = rmean(sc, adjclose_s, 20);
    add("mmt_avg_M", div(sc, st.adjusted_close, ma20)); // 收盘价 / 20日均价
    let prev_close_m = lag(sc, adjclose_s, M);
    let ma_year = rmean(sc, adjclose_s, Y);
    add("mmt_avg_A", div(sc, prev_close_m, ma_year)); // 1月前收盘 / 1年均价

    // ---- 日内 / 隔夜动量 ----
    add("mmt_intraday_M", rs_intra_m); // 过去1月日内涨跌幅之和
    add("mmt_intraday_A", sub(sc, rs_intra_y, rs_intra_m));
    add("mmt_overnight_M", rs_over_m); // 过去1月隔夜涨跌幅之和
    add("mmt_overnight_A", sub(sc, rs_over_y, rs_over_m));

    // ---- 路径调整动量 (return / Σ|daily return|) ----
    let sum_abs_m = rsum(sc, absret_s, M);
    add("mmt_route_M", div(sc, ret_1m, sum_abs_m));
    let sum_abs_y = rsum(sc, absret_s, Y);
    add("mmt_route_A", div(sc, ret_1y, sum_abs_y));

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
    let masked_s = rec(sc, masked_ret);
    add("mmt_off_limit_M", rsum(sc, masked_s, M));
    add("mmt_off_limit_A", rsum(sc, masked_s, Y));

    // 时序 rank 动量: daily time-series percentile of the log price within 1 year,
    // averaged over 20 days. 最高价距今天数: ticks since the 1-year high.
    let trank = window_reduce(sc, lc_s, Y, ts_rank);
    let trank_s = rec(sc, trank);
    add("mmt_time_rank_M", rmean(sc, trank_s, 20));
    add(
        "mmt_highest_days_A",
        window_reduce(sc, adjclose_s, Y, argmax_age),
    );

    // ============ 流动性 (liquidity) — 图表28 ============
    let turnover = div(sc, st.volume, st.circ_shares); // daily turnover 换手率
    let turnover_s = rec(sc, turnover);
    let amount_s = rec(sc, st.amount);
    let amihud = div(sc, abs_ret, st.amount); // |日收益率| / 成交额
    let amihud_s = rec(sc, amihud);
    // 日K线最短路径 = 2*(最高-最低) − |开盘−收盘|; shortcut 非流动 = 路径 / 成交额.
    let hl = sub(sc, st.high, st.low);
    let two_hl = emap(sc, hl, |x| 2.0 * x);
    let oc = sub(sc, st.open, st.close);
    let abs_oc = emap(sc, oc, f64::abs);
    let path = sub(sc, two_hl, abs_oc);
    let shortcut = div(sc, path, st.amount);
    let shortcut_s = rec(sc, shortcut);

    for (tag, w) in [("1M", M), ("3M", 63usize), ("6M", 126usize)] {
        add(&format!("liq_turn_avg_{tag}"), rmean(sc, turnover_s, w)); // 换手率均值
        add(&format!("liq_turn_std_{tag}"), rstd(sc, turnover_s, w)); // 换手率标准差
        let sum_amt = rsum(sc, amount_s, w);
        let ret_std = rstd(sc, dret_s, w);
        add(&format!("liq_vstd_{tag}"), div(sc, sum_amt, ret_std)); // 成交波动比 = Σ成交额 / σ(ret)
        add(&format!("liq_amihud_avg_{tag}"), rmean(sc, amihud_s, w));
        add(&format!("liq_amihud_std_{tag}"), rstd(sc, amihud_s, w));
        add(&format!("liq_shortcut_avg_{tag}"), rmean(sc, shortcut_s, w));
        add(&format!("liq_shortcut_std_{tag}"), rstd(sc, shortcut_s, w));
    }

    // ============ 量价相关性 (price-volume correlation, 20-day) — 图表40 ============
    // sync = corr(turn_t, x_t); post (量领先) = corr(turn_t, x_{t+1}) = corr(lag(turn,1), x);
    // prior (价领先) = corr(turn_t, x_{t-1}) = corr(turn, lag(x,1)).
    let turnover_change = sc.segment(diff(), turnover);
    let turnchg_s = rec(sc, turnover_change);
    let lag_turn1 = lag(sc, turnover_s, 1);
    let lag_turnd1 = lag(sc, turnchg_s, 1);
    let lag_price1 = lag(sc, adjclose_s, 1);
    let lag_ret1 = lag(sc, dret_s, 1);
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
    let downside_s = rec(sc, downside);
    let upside = emap(sc, daily_ret, |x| {
        if x.is_finite() { x.max(0.0) } else { f64::NAN }
    });
    let upside_s = rec(sc, upside);
    let highlow = div(sc, st.high, st.low); // 日内振幅 = 最高/最低
    let highlow_s = rec(sc, highlow);
    // Candlestick shadows, normalized by the low (cross-sectional price-level scale).
    let max_oc = sc.segment(max(), (st.open, st.close));
    let min_oc = sc.segment(min(), (st.open, st.close));
    let up_num = sub(sc, st.high, max_oc); // 上影线 = 最高 − max(开,收)
    let upshadow = div(sc, up_num, st.low);
    let upshadow_s = rec(sc, upshadow);
    let down_num = sub(sc, min_oc, st.low); // 下影线 = min(开,收) − 最低
    let downshadow = div(sc, down_num, st.low);
    let downshadow_s = rec(sc, downshadow);
    let wup_num = sub(sc, st.high, st.close); // 威廉上影线 = 最高 − 收
    let w_upshadow = div(sc, wup_num, st.low);
    let w_upshadow_s = rec(sc, w_upshadow);
    let wdown_num = sub(sc, st.close, st.low); // 威廉下影线 = 收 − 最低
    let w_downshadow = div(sc, wdown_num, st.low);
    let w_downshadow_s = rec(sc, w_downshadow);

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
    // (N,2) series and reduce each window via `range_mom`.
    let amp_ret = stack2(sc, highlow, daily_ret); // (N, 2)
    let amp_ret_s = rec_2(sc, amp_ret);
    add("mmt_range_M", window_reduce2(sc, amp_ret_s, M, range_mom));
    add("mmt_range_A", window_reduce2(sc, amp_ret_s, Y, range_mom));

    // ============ 筹码分布 (chip distribution) — 图表52 ============
    // (adjusted close, turnover) -> ChipDist -> (N, 10); column-select each factor.
    let chip_in = stack2(sc, st.adjusted_close, turnover); // (N, 2)
    let chip_in_s = rec_2(sc, chip_in);
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

/// Record a rank-2 `(N, 2)` cross-section into a `Series` (the two-channel chip /
/// range inputs); retains the deepest count look-back the consumers read (`Y` /
/// the 250-day chip window) plus the margin.
fn rec_2(sc: &mut Scenario, h: PortHandle<ArrayPort<f64, 2>>) -> Ser2 {
    sc.segment(record_bounded(Retention::count(Y + RETAIN_MARGIN)), h)
}
