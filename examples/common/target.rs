//! Prediction targets and trading constraints derived from the market panel.

/// Daily price-limit band (±10% for most A-shares).
pub const PRICE_LIMIT: f64 = 0.10;
/// Half-spread of the synthetic quote book, in yuan (A-shares tick at 0.01).
pub const TICK_SIZE: f64 = 0.01;
/// Consecutive quoteless trading days after which a stock counts as delisted
/// (the panel carries no listing state).
pub const DELIST_DAYS: usize = 20;

use tradingflow::data::{Array, ArrayView, Instant, Retention};
use tradingflow::graph::Builder;
use tradingflow::operators::series::record_on;
use tradingflow::operators::{
    array::{array_map, binary_map, map},
    elem, rolling,
    stats::*,
};
use tradingflow::ports::{ArrayPortHandle, SeriesPortHandle, SignalPortHandle};
use tradingflow::time::UnixTime;

/// Cross-sectional demean preserving NaN.
fn demean(r: ArrayView<f64, 1>) -> Array<f64, 1> {
    let s = r.to_contiguous();
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for &x in s.iter() {
        if x.is_finite() {
            sum += x;
            cnt += 1;
        }
    }
    let mean = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
    Array::from_parts(
        [s.len()],
        s.iter()
            .map(|&x| if x.is_finite() { x - mean } else { x })
            .collect(),
    )
}

/// Winsorized daily log returns: `(target, target_series, demeaned_series)`.
/// The covariance predictor consumes `target_series` (raw); the mean predictor
/// consumes `demeaned_series` (cross-sectionally demeaned). Both series are
/// recorded under `target_retention` — size it to the deepest consumer
/// look-back (the incremental mean predictor reads a single trailing pair, the
/// shrinkage covariance reads its `max_periods` window); pass
/// [`Retention::unbounded()`] when full history is needed.
#[allow(clippy::type_complexity)]
pub fn build_log_return_target(
    sc: &mut Builder<Instant, UnixTime>,
    log_adj: ArrayPortHandle<f64, 1>,
    daily: SignalPortHandle<0>,
    target_retention: Retention,
) -> (
    ArrayPortHandle<f64, 1>,
    SeriesPortHandle<f64, 1>,
    SeriesPortHandle<f64, 1>,
) {
    let log_returns = sc.segment(rolling::diff(1), (daily, log_adj));
    let target = sc.segment(winsorize(0.01), log_returns);
    // Recorded on every daily pulse, so the leading all-NaN row (the first
    // close has no prior close to difference against) is kept and the panel
    // stays index-aligned with the feature panel recorded on the same signal.
    let target_series = sc.segment(record_on(target_retention, false), (daily, target));
    let demeaned = sc.segment(array_map(demean), target);
    let demeaned_series = sc.segment(record_on(target_retention, false), (daily, demeaned));
    (target, target_series, demeaned_series)
}

/// Constant ±`limit_pct` daily price limits from the previous close, rounded to
/// 0.01 yuan. Returns `(upper, lower)`; first tick is NaN (no prior close).
pub fn build_price_limits(
    sc: &mut Builder<Instant, UnixTime>,
    daily: SignalPortHandle<0>,
    close: ArrayPortHandle<f64, 1>,
    limit_pct: f64,
) -> (ArrayPortHandle<f64, 1>, ArrayPortHandle<f64, 1>) {
    // Self-recording 1-step lag (a tiny private trailing window).
    let prev_close = sc.segment(rolling::lag(1), (daily, close));
    let limit = move |scale: f64| map(move |&x: &f64| ((x * scale) * 100.0).round() / 100.0);
    let upper = sc.segment(limit(1.0 + limit_pct), prev_close);
    let lower = sc.segment(limit(1.0 - limit_pct), prev_close);
    (upper, lower)
}

/// A synthetic quote book over the daily close panel: the
/// `(price_signal, flags, bids, asks)` stream every
/// [`trader::fixed`](tradingflow::operators::trader::fixed) operator consumes.
///
/// The panel carries neither an order book nor listing state, so both are
/// modelled from the closes alone:
///
/// * **Quotes.** A stock that traded today quotes `close ∓ tick`, a one-tick
///   spread around the close, so a round trip pays `2 × tick` before fees.
/// * **Price limits.** A-shares trade inside a ±`limit_pct` band around the
///   previous close (here the previous *known* close, so the band survives a
///   suspension). A quote outside the band cannot execute, so it becomes the
///   infinity on its side: `ask = +inf` at limit-up — nobody will sell to you
///   — and `bid = -inf` at limit-down.
/// * **Suspension.** A stock with no close today has no quote at all:
///   `(-inf, +inf)`, which holds the trader's mark at the last real price
///   instead of writing the position down.
/// * **Delisting.** Lacking a listing-state column, a stock quoteless for
///   `delist_days` consecutive trading days is treated as delisted
///   (`flag = false`), which marks it to zero and blocks trading in it; a
///   later quote relists it, so a long suspension that eventually resumes is
///   written down and then written back up rather than stranding the shares.
///
/// `NaN` never reaches the trader: every stock is quoted on every price
/// pulse, and a stock with nothing to quote says so with infinities.
#[allow(clippy::type_complexity)]
pub fn build_quotes(
    sc: &mut Builder<Instant, UnixTime>,
    daily: SignalPortHandle<0>,
    close: ArrayPortHandle<f64, 1>,
    limit_pct: f64,
    tick: f64,
    delist_days: usize,
) -> (
    SignalPortHandle<0>,
    ArrayPortHandle<bool, 1>,
    ArrayPortHandle<f64, 1>,
    ArrayPortHandle<f64, 1>,
) {
    // The band is anchored on the last *known* close, so a stock resuming
    // after a suspension is banded against its pre-suspension price rather
    // than against the NaN it quoted yesterday.
    let carried = sc.segment(elem::forward_fill_nan(), close);
    let (upper, lower) = build_price_limits(sc, daily, carried, limit_pct);

    let bids = sc.segment(
        binary_map(move |&c: &f64, &lo: &f64| {
            let bid = c - tick;
            if c.is_nan() || bid < lo {
                f64::NEG_INFINITY
            } else {
                bid
            }
        }),
        (close, lower),
    );
    let asks = sc.segment(
        binary_map(move |&c: &f64, &up: &f64| {
            let ask = c + tick;
            if c.is_nan() || ask > up {
                f64::INFINITY
            } else {
                ask
            }
        }),
        (close, upper),
    );

    // Listing state: the length of the current quoteless run, as a rolling
    // sum of the "no close today" indicator over the last `delist_days`
    // pulses — it reaches `delist_days` exactly when every day in the window
    // was quoteless.
    let quoteless = sc.segment(map(|&c: &f64| f64::from(u8::from(c.is_nan()))), close);
    let run_len = sc.segment(
        rolling::sum(Retention::count(delist_days), 1),
        (daily, quoteless),
    );
    let flags = sc.segment(map(move |&r: &f64| r < delist_days as f64), run_len);

    (daily, flags, bids, asks)
}
