use num_traits::Float;

use super::Windowed;
use crate::data::Scalar;
use crate::graph::cb::Comp;
use crate::operators::rolling::Ema;
use crate::operators::structural::buffer;

/// Exponential moving average with window-normalized weights (see [`Ema`]):
/// `ema(alpha, window) @ x`. Self-recording.
pub fn ema<T: Scalar + Float, const N: usize>(
    alpha: T,
    window: usize,
) -> Windowed<T, N, Ema<T, N>> {
    Comp(buffer(window), Ema::new(alpha, window))
}
