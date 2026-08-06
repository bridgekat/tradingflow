use num_traits::Float;

use crate::data::{Instant, Scalar};
use crate::graph::{Operator, OperatorExt, cb};
use crate::operators::{signal, stats};
use crate::ports::{ArrayPort, SignalPort};

/// Cross-sectional information coefficient. Can be either Pearson
/// (`ranked == false`) or Spearman (`ranked == true`) correlation.
pub fn information_coefficient<T: Scalar + Float, const N: usize>(
    ranked: bool,
    min_count: usize,
) -> impl Operator<
    Inputs = (SignalPort<0>, ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, 0>,
    Context = Instant,
> {
    cb::Arr::new(|(signal, features, targets), _| (signal, (features, targets)))
        .then(signal::on_signal(stats::corr(ranked, min_count)))
}
