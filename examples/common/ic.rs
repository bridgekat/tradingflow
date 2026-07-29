//! Information-coefficient evaluation.

use tradingflow::data::Instant;
use tradingflow::graph::Builder;
use tradingflow::operators::series::record_all;
use tradingflow::ports::{ArrayPortHandle, SeriesPortHandle, SignalPortHandle};
use tradingflow::time::UnixTime;

use super::models::information_coefficient;

/// Record the per-rebalance IC of the `factor` stream against the `target`
/// stream — each a `(signal, values)` pair wired straight into the Python
/// metric, whose own output stream drives the record.
pub fn ic_series(
    sc: &mut Builder<Instant, UnixTime>,
    factor: (SignalPortHandle<0>, ArrayPortHandle<f64, 1>),
    target: (SignalPortHandle<0>, ArrayPortHandle<f64, 1>),
    num_stocks: usize,
) -> SeriesPortHandle<f64, 0> {
    let ic = sc.segment(
        information_coefficient(num_stocks),
        (factor.0, factor.1, target.0, target.1),
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
