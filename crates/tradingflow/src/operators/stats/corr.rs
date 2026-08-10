use num_traits::Float;
use std::marker::PhantomData;

use super::common::is_constant;
use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::ArrayPort;

/// Operator signature for [`corr`] etc.
pub struct Corr<T: Scalar + Float, const N: usize> {
    zero_degenerate: bool,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> Corr<T, N> {
    pub fn new(zero_degenerate: bool) -> Self {
        Self {
            zero_degenerate,
            _marker: PhantomData,
        }
    }
}

/// Runtime state for [`Corr`].
pub struct CorrState<T: Scalar + Float, const N: usize> {
    zero_degenerate: bool,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, const N: usize> Operator for Corr<T, N> {
    type Inputs = (ArrayPort<T, N>, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = CorrState<T, N>;

    fn init(self, (x, y): (ArrayView<'_, T, N>, ArrayView<'_, T, N>)) -> Self::State {
        assert_eq!(
            x.extents(),
            y.extents(),
            "stats::corr: input extents mismatch"
        );
        let mut state = CorrState {
            zero_degenerate: self.zero_degenerate,
            out: Array::full([], T::nan()),
        };
        corr_into(&mut state.out, x, y, state.zero_degenerate);
        state
    }

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, T, N>, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> ArrayView<'a, T, 0> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        (x, y): (ArrayView<'a, T, N>, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, T, 0> {
        corr_into(&mut state.out, x, y, state.zero_degenerate);
        state.out.view()
    }
}

/// Pearson correlation of two equal-length samples; `None` when either
/// side's variance is within floating-point error of zero ([`is_constant`],
/// which subsumes the exact zero) — a side whose deviations are cancellation
/// noise would correlate as a spurious `±1` instead of not at all.
fn corr_into<T: Scalar + Float, const N: usize>(
    out: &mut Array<T, 0>,
    xs: ArrayView<'_, T, N>,
    ys: ArrayView<'_, T, N>,
    zero_degenerate: bool,
) {
    let mut count = T::zero();
    let mut sum_x = T::zero();
    let mut sum_y = T::zero();
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        if x.is_finite() && y.is_finite() {
            count = count + T::one();
            sum_x = sum_x + x;
            sum_y = sum_y + y;
        }
    }
    let mean_x = sum_x / count;
    let mean_y = sum_y / count;
    let mut ssd_x = T::zero();
    let mut ssd_y = T::zero();
    let mut cov = T::zero();
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        if x.is_finite() && y.is_finite() {
            let dx = x - mean_x;
            let dy = y - mean_y;
            ssd_x = ssd_x + dx * dx;
            ssd_y = ssd_y + dy * dy;
            cov = cov + dx * dy;
        }
    }
    let var_x = ssd_x / count;
    let var_y = ssd_y / count;
    if count < T::from(2).unwrap()
        || is_constant(var_x, mean_x, count)
        || is_constant(var_y, mean_y, count)
    {
        **out = if zero_degenerate { T::zero() } else { T::nan() };
    } else {
        **out = cov / (ssd_x * ssd_y).sqrt();
    }
}

/// Cross-sectional correlation coefficient between two arrays of the same
/// extents, as a rank-0 scalar.
///
/// Elements are paired positionwise; pairs with a non-finite side (NaN or ±∞)
/// are skipped pairwise-complete, and the result is `NaN` with fewer than 2
/// valid pairs or a side whose variance is close to zero.
pub fn corr<T: Scalar + Float, const N: usize>() -> impl Operator<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, 0>,
    Context = Instant,
> {
    Corr::new(false)
}

/// Like [`corr`], but every degenerate case — too few valid pairs or a
/// no-spread side — yields `0`, the no-information coefficient, instead of
/// `NaN`.
///
/// This is the form for feeding the coefficient onward as a *value* (into an
/// arithmetic pipeline or a model input), where an unscored day should
/// contribute nothing rather than poison the computation. Keep plain [`corr`]
/// for metrics and diagnostics: there `NaN` means "not scored", and a
/// consumer counting observations (an IC table, a t-statistic) must be able
/// to tell that apart from "scored exactly zero".
pub fn corr_or_zero<T: Scalar + Float, const N: usize>() -> impl Operator<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, 0>,
    Context = Instant,
> {
    Corr::new(true)
}
