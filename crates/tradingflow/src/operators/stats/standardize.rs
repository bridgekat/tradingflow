use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::ArrayPort;

pub struct Standardize<T: Scalar + Float, const N: usize> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> Standardize<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
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

    fn init(self, x: ArrayView<'_, T, N>) -> Self::State {
        let mut out = Array::zeros(x.extents());
        standardize_into(&mut out, x);
        out
    }

    fn reset<'a, 'b: 'a>(_: ArrayView<'a, T, N>, out: &'b mut Self::State) -> ArrayView<'a, T, N> {
        out.view()
    }

    fn compute<'a, 'b: 'a>(
        x: ArrayView<'a, T, N>,
        out: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, T, N> {
        standardize_into(out, x);
        out.view()
    }
}

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
        if v.is_finite() {
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
        if v.is_finite() {
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
        dst[i] = if v.is_finite() { (v - mean) / std } else { nan };
    }
}

/// Cross-sectional z-score: `(x − mean) / std` (population std). Non-finite
/// entries (NaN or ±∞) are treated as missing and map to NaN; the whole
/// cross-section is NaN if fewer than two finite values remain or σ = 0.
pub fn standardize<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Standardize::new()
}
