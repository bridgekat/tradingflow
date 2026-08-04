use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::ArrayPort;

pub struct Demean<T: Scalar + Float, const N: usize> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> Demean<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Demean<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float, const N: usize> Operator for Demean<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self, x: ArrayView<'_, T, N>) -> Self::State {
        let mut out = Array::zeros(x.extents());
        demean_into(&mut out, x);
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
        demean_into(out, x);
        out.view()
    }
}

fn demean_into<T: Scalar + Float, const N: usize>(out: &mut Array<T, N>, x: ArrayView<'_, T, N>) {
    let mut count = 0usize;
    let mut sum = T::zero();
    for &v in x.iter() {
        if v.is_finite() {
            count += 1;
            sum = sum + v;
        }
    }
    let mean = sum / T::from(count).unwrap();
    let o = out.data_mut();
    for (i, &v) in x.iter().enumerate() {
        o[i] = if v.is_finite() { v - mean } else { T::nan() };
    }
}

/// Cross-sectional demean: `(x − mean)`. Non-finite entries (NaN or ±∞) are
/// treated as missing and map to NaN.
pub fn demean<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Demean::new()
}
