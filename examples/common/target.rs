//! Prediction targets and trading constraints derived from the market panel.

use tradingflow::clock::UnixClock;
use tradingflow::data::{Array, ArrayView, Instant, Retention, Series};
use tradingflow::graph::Builder;
use tradingflow::operators::series::record;
use tradingflow::operators::{
    array::{array_map, map},
    event, rolling,
    stats::*,
    traders::*,
};
use tradingflow::ports::{ArrayPortHandle, SeriesPortHandle};

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
    sc: &mut Builder<Instant, UnixClock>,
    log_adj: ArrayPortHandle<f64, 1>,
    target_retention: Retention,
) -> (
    ArrayPortHandle<f64, 1>,
    SeriesPortHandle<f64, 1>,
    SeriesPortHandle<f64, 1>,
) {
    let log_returns = sc.segment(rolling::diff(1), log_adj);
    let target = sc.segment(winsorize(0.01), log_returns);
    let target_series = sc.segment(record(target_retention, false), target);
    let demeaned = sc.segment(array_map(demean), target);
    let demeaned_series = sc.segment(record(target_retention, false), demeaned);
    (target, target_series, demeaned_series)
}

/// Constant ±`limit_pct` daily price limits from the previous close, rounded to
/// 0.01 yuan. Returns `(upper, lower)`; first tick is NaN (no prior close).
pub fn build_price_limits(
    sc: &mut Builder<Instant, UnixClock>,
    close: ArrayPortHandle<f64, 1>,
    limit_pct: f64,
) -> (ArrayPortHandle<f64, 1>, ArrayPortHandle<f64, 1>) {
    // Self-recording 1-step lag (a tiny private trailing window).
    let prev_close = sc.segment(rolling::lag(1), close);
    let limit = move |scale: f64| map(move |&x: &f64| ((x * scale) * 100.0).round() / 100.0);
    let upper = sc.segment(limit(1.0 + limit_pct), prev_close);
    let lower = sc.segment(limit(1.0 - limit_pct), prev_close);
    (upper, lower)
}
