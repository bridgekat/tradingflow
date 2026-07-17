//! Prediction targets and trading constraints derived from the market panel.

use tradingflow::graph::{PortHandle, RefPort};

use tradingflow::operators::{SeriesPort, diff, lag, map, record_bounded, winsorize};
use tradingflow::{Array, ArrayView, Retention, Scenario, Series};

use super::AvH;

/// A recorded rank-1 cross-sectional series (the target / demeaned target).
pub type SerH = PortHandle<SeriesPort<f64, 1>>;

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
    Array::from_vec(
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
/// [`Retention::UNBOUNDED`] when full history is needed.
pub fn build_log_return_target(
    sc: &mut Scenario,
    log_adj: AvH,
    target_retention: Retention,
) -> (AvH, SerH, SerH) {
    let log_returns = sc.segment(diff(), log_adj);
    let target = sc.segment(winsorize(0.01), log_returns);
    let target_series = sc.segment(record_bounded(target_retention), target);
    let demeaned = sc.segment(map(demean), target);
    let demeaned_series = sc.segment(record_bounded(target_retention), demeaned);
    (target, target_series, demeaned_series)
}

/// Constant ±`limit_pct` daily price limits from the previous close, rounded to
/// 0.01 yuan. Returns `(upper, lower)`; first tick is NaN (no prior close).
pub fn build_price_limits(sc: &mut Scenario, close: AvH, limit_pct: f64) -> (AvH, AvH) {
    // Self-recording 1-step lag (a tiny private trailing window).
    let prev_close = sc.segment(lag(1), close);
    let limit = move |scale: f64| {
        map(move |c: ArrayView<f64, 1>| {
            let s = c.to_contiguous();
            Array::from_vec(
                [s.len()],
                s.iter()
                    .map(|&x| ((x * scale) * 100.0).round() / 100.0)
                    .collect(),
            )
        })
    };
    let upper = sc.segment(limit(1.0 + limit_pct), prev_close);
    let lower = sc.segment(limit(1.0 - limit_pct), prev_close);
    (upper, lower)
}
