//! Information-coefficient evaluation.

use tradingflow::graph::typed::PortHandle;
use tradingflow::operators::structural::record;
use tradingflow::{ArrayPort, Scenario, SeriesPort};

use super::models::information_coefficient;

/// Record the per-rebalance IC of `factor` against `target` — both ordinary
/// `ArrayPort` views, wired straight into the Python metric.
pub fn ic_series(
    sc: &mut Scenario,
    factor: PortHandle<ArrayPort<f64, 1>>,
    target: PortHandle<ArrayPort<f64, 1>>,
    num_stocks: usize,
) -> PortHandle<SeriesPort<f64, 0>> {
    let ic = sc.segment(information_coefficient(num_stocks), (factor, target));
    sc.segment(record(), ic)
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
