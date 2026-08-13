use num_traits::Float;

use crate::data::{Instant, Scalar};
use crate::graph::{Operator, OperatorExt, cb::Arr};
use crate::operators::{signal, stats};
use crate::ports::{ArrayPort, SignalPort};

/// Cross-sectional Pearson information coefficient.
///
/// Inputs and outputs: see [module-level docs](super).
pub fn ic<T: Scalar + Float, const N: usize>() -> impl Operator<
    Inputs = (SignalPort<0>, ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, 0>,
    Context = Instant,
> {
    Arr::new(|(signal, features, targets), _| (signal, (features, targets)))
        .then(signal::on_signal(stats::corr_or_zero()))
}

/// Cross-sectional Spearman information coefficient.
///
/// This operator calculates correlation on percentiles, where both
/// percentile arrays are calculated independently instead of on
/// pairwise-complete elements.
///
/// Inputs and outputs: see [module-level docs](super).
pub fn rank_ic<T: Scalar + Float, const N: usize>() -> impl Operator<
    Inputs = (SignalPort<0>, ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, 0>,
    Context = Instant,
> {
    Arr::new(|(signal, features, targets), _| (signal, (features, targets))).then(
        signal::on_signal(
            stats::percentile()
                .par(stats::percentile())
                .then(stats::corr_or_zero()),
        ),
    )
}
