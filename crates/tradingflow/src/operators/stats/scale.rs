use num_traits::Float;
use std::cmp::Ordering;
use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::ArrayPort;

/// Operator signature for [`scale`] etc.
pub struct Scale<T: Scalar + Float, const N: usize> {
    compare: Ordering,
    target: f64,
    zero_degenerate: bool,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> Scale<T, N> {
    /// When `zero_degenerate` is set, a zero cross-section — one with no
    /// magnitude to rescale — maps its finite entries to `0` instead of
    /// `NaN` (see [`scale_or_zero`]).
    pub fn new(compare: Ordering, target: f64, zero_degenerate: bool) -> Self {
        Self {
            compare,
            target,
            zero_degenerate,
            _marker: PhantomData,
        }
    }
}

/// Runtime state for [`Scale`].
pub struct ScaleState<T: Scalar + Float, const N: usize> {
    compare: Ordering,
    target: T,
    zero_degenerate: bool,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Scale<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = ScaleState<T, N>;

    fn init(self, x: ArrayView<'_, T, N>) -> Self::State {
        let mut state = ScaleState {
            compare: self.compare,
            target: T::from(self.target).unwrap(),
            zero_degenerate: self.zero_degenerate,
            out: Array::zeros(x.extents()),
        };
        scale_into(
            &mut state.out,
            x,
            state.compare,
            state.target,
            state.zero_degenerate,
        );
        state
    }

    fn reset<'a, 'b: 'a>(
        _: ArrayView<'a, T, N>,
        state: &'b mut Self::State,
    ) -> ArrayView<'a, T, N> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        x: ArrayView<'a, T, N>,
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, T, N> {
        scale_into(
            &mut state.out,
            x,
            state.compare,
            state.target,
            state.zero_degenerate,
        );
        state.out.view()
    }
}

fn scale_into<T: Scalar + Float, const N: usize>(
    out: &mut Array<T, N>,
    xs: ArrayView<'_, T, N>,
    compare: Ordering,
    target: T,
    zero_degenerate: bool,
) {
    let mut sum_abs = T::zero();
    for &x in xs.iter() {
        if x.is_finite() {
            sum_abs = sum_abs + x.abs();
        }
    }
    if sum_abs.partial_cmp(&target) == Some(compare) {
        // The sum already meets the condition: pass the input through
        // unscaled.
        for (o, &x) in out.data_mut().iter_mut().zip(xs.iter()) {
            *o = if x.is_finite() { x } else { T::nan() };
        }
    } else if sum_abs > T::zero() {
        // Dividing before multiplying keeps every intermediate in range even
        // when `target / sum` alone would overflow (a denormal-small sum):
        // `|v| / sum <= 1`.
        for (o, &x) in out.data_mut().iter_mut().zip(xs.iter()) {
            *o = if x.is_finite() {
                x / sum_abs * target
            } else {
                T::nan()
            };
        }
    } else {
        // A zero (or empty) cross-section has no magnitude to rescale:
        // `NaN` ("cannot reach the target"), or `0` — hold nothing — under
        // `zero_degenerate`; a non-finite input slot stays `NaN` either way.
        for (o, &x) in out.data_mut().iter_mut().zip(xs.iter()) {
            *o = if x.is_finite() && zero_degenerate {
                T::zero()
            } else {
                T::nan()
            };
        }
    }
}

/// Cross-sectional rescale: `x / Σ|x| · target`, so the absolute values of
/// the output sum to `target` (Alpha101's `scale(x, a)`). Non-finite entries
/// (NaN or ±∞) are treated as missing: they map to NaN and are excluded from
/// the sum. When the absolute sum is zero the whole cross-section is NaN.
/// The division happens first, so a denormal-small sum still lands on the
/// target instead of overflowing through `target / Σ|x|` to ±∞.
pub fn scale<T: Scalar + Float, const N: usize>(
    target: f64,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Scale::new(Ordering::Equal, target, false)
}

/// Like [`scale`], but only scale when the sum is no greater than `target`.
pub fn scale_up<T: Scalar + Float, const N: usize>(
    target: f64,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Scale::new(Ordering::Greater, target, false)
}

/// Like [`scale`], but only scale when the sum is no less than `target`.
pub fn scale_down<T: Scalar + Float, const N: usize>(
    target: f64,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Scale::new(Ordering::Less, target, false)
}

/// Like [`scale`], but a zero cross-section maps its **finite** entries to
/// `0` — hold nothing — instead of `NaN`; non-finite entries still map to
/// `NaN`, so missing stays missing.
///
/// This is the form for a value pipeline (portfolio weights, model inputs)
/// where "no magnitude to rescale" should read as an empty book rather than
/// as undefined. Keep plain [`scale`] where `NaN` propagation is the signal
/// that there was nothing to scale.
pub fn scale_or_zero<T: Scalar + Float, const N: usize>(
    target: f64,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Scale::new(Ordering::Equal, target, true)
}

/// Like [`scale_up`], but with [`scale_or_zero`]'s treatment of a zero
/// cross-section.
pub fn scale_up_or_zero<T: Scalar + Float, const N: usize>(
    target: f64,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Scale::new(Ordering::Greater, target, true)
}

/// Like [`scale_down`], but with [`scale_or_zero`]'s treatment of a zero
/// cross-section. (With a positive `target` the zero cross-section already
/// skips as "no more than the target" and passes through; the variant only
/// differs at `target <= 0`.)
pub fn scale_down_or_zero<T: Scalar + Float, const N: usize>(
    target: f64,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Scale::new(Ordering::Less, target, true)
}
