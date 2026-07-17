//! Information-coefficient evaluation, shared by `factor_ic` and `factor_handbook`.
//!
//! Both examples score factors by the cross-sectional correlation of a factor
//! vector with a return vector, per rebalance; they differ only in what they feed
//! it. `factor_ic` correlates the *lagged raw* canonical features against the
//! realized return (Pearson IC); `factor_handbook` correlates *ranked* catalog
//! factors against the *ranked forward* return (Spearman / RankIC). The wiring
//! and the summary statistics are the same, and live here.

use tradingflow::Scenario;
use tradingflow::operators::structural::record;

use super::AvH;
use super::models::information_coefficient;
use super::strategy::NavH;

/// Record the per-rebalance IC of `factor` against `target` — both ordinary
/// `ArrayPort` views, wired straight into the Python metric.
pub fn ic_series(sc: &mut Scenario, factor: AvH, target: AvH, num_stocks: usize) -> NavH {
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
