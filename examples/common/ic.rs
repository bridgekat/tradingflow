//! Information-coefficient evaluation.

use tradingflow::data::Instant;
use tradingflow::graph::Builder;
use tradingflow::operators::metric::predictor::mean::information_coefficient;
use tradingflow::operators::series::record_all;
use tradingflow::ports::{ArrayPortHandle, SeriesPortHandle, SignalPortHandle};
use tradingflow::time::UnixTime;

/// Record the per-evaluation-period IC of the `predict` stream against the
/// `target` stream — each a `(signal, values)` pair wired straight into the
/// metric, whose own output stream drives the record.
pub fn ic_series(
    sc: &mut Builder<Instant, UnixTime>,
    predict: (SignalPortHandle<0>, ArrayPortHandle<f64, 1>),
    target: (SignalPortHandle<0>, ArrayPortHandle<f64, 1>),
) -> SeriesPortHandle<f64, 0> {
    let ic = sc.segment(
        information_coefficient(),
        (predict.0, predict.1, target.0, target.1),
    );
    sc.segment(record_all(), ic)
}

/// Summary of an IC series: its mean, dispersion, information ratio
/// (`mean / std`), and the t-statistic of the mean being non-zero.
pub struct IcStats {
    pub mean: f64,
    pub std: f64,
    /// Information ratio — the Sharpe ratio of the IC series.
    pub ir: f64,
    pub t: f64,
    /// Number of finite observations.
    pub n: usize,
}

/// Summarize an IC series, ignoring non-finite periods (a rebalance where the
/// factor or the target had too little cross-sectional coverage). Returns `None`
/// when no period produced a finite IC.
pub fn ic_stats(v: &[f64]) -> Option<IcStats> {
    let finite: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let n = finite.len();
    if n == 0 {
        return None;
    }
    let mean = finite.iter().sum::<f64>() / n as f64;
    let var = finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    let (ir, t) = if std > 0.0 {
        (mean / std, mean / std * (n as f64).sqrt())
    } else {
        (f64::NAN, f64::NAN)
    };
    Some(IcStats {
        mean,
        std,
        ir,
        t,
        n,
    })
}

/// Trailing-window OLS of `y` on `x` with an intercept: `(beta, alpha)`.
///
/// The market beta and alpha of a strategy, computed after the run from two
/// recorded log-return series rather than in-graph — a single regression on
/// the final window is not worth a node. Rows where either series is
/// non-finite are dropped, and fewer than `min_periods` surviving rows give
/// `NaN`.
pub fn trailing_beta_alpha(
    y: &[f64],
    x: &[f64],
    max_periods: usize,
    min_periods: usize,
) -> (f64, f64) {
    let start = y.len().min(x.len()).saturating_sub(max_periods);
    let pairs: Vec<(f64, f64)> = y[start..]
        .iter()
        .zip(&x[start..])
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (*a, *b))
        .collect();

    let n = pairs.len() as f64;
    if pairs.len() < min_periods {
        return (f64::NAN, f64::NAN);
    }

    let mean_y = pairs.iter().map(|(a, _)| a).sum::<f64>() / n;
    let mean_x = pairs.iter().map(|(_, b)| b).sum::<f64>() / n;
    let covariance: f64 = pairs
        .iter()
        .map(|(a, b)| (a - mean_y) * (b - mean_x))
        .sum::<f64>();
    let variance: f64 = pairs.iter().map(|(_, b)| (b - mean_x).powi(2)).sum::<f64>();

    if variance <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let beta = covariance / variance;
    (beta, mean_y - beta * mean_x)
}
