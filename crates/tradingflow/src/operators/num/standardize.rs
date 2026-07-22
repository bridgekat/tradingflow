//! [`Standardize`] — the cross-sectional z-score.

use std::marker::PhantomData;

use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Cross-sectional z-score (population std; NaN if < 2 finite or σ = 0).
#[derive(Clone)]
pub struct Standardize<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Standardize<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Standardize<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float, const N: usize> Operator for Standardize<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        let mut out = Array::zeros(x.extents());
        // Seed the output with the faithful transform of the build value; the
        // initial render does not notify.
        standardize_into(&mut out, x);
        out
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        out: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        standardize_into(out, x);
        (true, out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        out: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }
}

/// The per-call body of [`Standardize`]: z-score the cross-section of `x` into
/// `out` (all NaN if < 2 finite or σ = 0).
fn standardize_into<T: Scalar + Float, const N: usize>(
    out: &mut Array<T, N>,
    x: ArrayView<'_, T, N>,
) {
    let xs = x.to_contiguous();
    let src: &[T] = &xs;
    let n = src.len();
    let nan = T::nan();

    let mut n_valid = 0usize;
    let mut sum = T::zero();
    for &v in src.iter() {
        if !v.is_nan() {
            n_valid += 1;
            sum = sum + v;
        }
    }

    let dst = out.data_mut();
    if n_valid < 2 {
        for slot in dst.iter_mut() {
            *slot = nan;
        }
        return;
    }

    let mean = sum / T::from(n_valid).unwrap();

    let mut ssd = T::zero();
    for &v in src.iter() {
        if !v.is_nan() {
            let d = v - mean;
            ssd = ssd + d * d;
        }
    }
    let variance = ssd / T::from(n_valid).unwrap();
    let std = variance.sqrt();

    if std <= T::zero() {
        for slot in dst.iter_mut() {
            *slot = nan;
        }
        return;
    }

    for i in 0..n {
        let v = src[i];
        dst[i] = if v.is_nan() { nan } else { (v - mean) / std };
    }
}

/// Cross-sectional z-score: `(x − mean) / std`.
pub fn standardize<T: Scalar + Float, const N: usize>() -> Standardize<T, N> {
    Standardize::new()
}
