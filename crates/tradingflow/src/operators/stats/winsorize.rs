use num_traits::Float;
use std::cmp::Ordering;

use crate::data::{Array, ArrayView, Instant, Layout, Scalar};
use crate::graph::{Operator, Segment};
use crate::ports::ArrayPort;

pub struct Winsorize<T: Scalar + Float, const N: usize> {
    p: T,
}

impl<T: Scalar + Float, const N: usize> Winsorize<T, N> {
    pub fn new(p: T) -> Self {
        assert!(
            p >= T::zero() && p + p < T::one(),
            "winsorize: p must be in [0, 0.5)"
        );
        Self { p }
    }
}

pub struct WinsorizeState<T: Scalar + Float, const N: usize> {
    p: T,
    sort_buf: Vec<T>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Winsorize<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = WinsorizeState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        let mut state = WinsorizeState {
            p: self.p,
            sort_buf: vec![T::zero(); x.layout().len()],
            out: Array::zeros(x.extents()),
        };
        winsorize_into(&mut state, x);
        state
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        winsorize_into(state, x);
        (true, state.out.view())
    }
}

fn winsorize_into<T: Scalar + Float, const N: usize>(
    state: &mut WinsorizeState<T, N>,
    x: ArrayView<'_, T, N>,
) {
    let xs = x.to_contiguous();
    let src: &[T] = &xs;
    let n = src.len();

    // Only finite values define the clip bounds; ±∞ inputs are the extreme
    // outliers winsorization exists to tame, and are clamped to the finite
    // bounds below rather than allowed to become a bound themselves.
    let mut n_valid = 0usize;
    for &v in src.iter() {
        if v.is_finite() {
            state.sort_buf[n_valid] = v;
            n_valid += 1;
        }
    }
    state.sort_buf[..n_valid].sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let dst = state.out.data_mut();
    let nan = T::nan();

    if n_valid == 0 {
        for slot in dst.iter_mut() {
            *slot = nan;
        }
        return;
    }

    let p_f = state.p.to_f64().unwrap_or(0.0);
    let k = ((p_f * n_valid as f64).floor() as usize).min(n_valid - 1);
    let lo = state.sort_buf[k];
    let hi = state.sort_buf[n_valid - 1 - k];

    for i in 0..n {
        let v = src[i];
        if v.is_nan() {
            dst[i] = nan;
        } else if v < lo {
            dst[i] = lo;
        } else if v > hi {
            dst[i] = hi;
        } else {
            dst[i] = v;
        }
    }
}

pub fn winsorize<T: Scalar + Float, const N: usize>(
    p: T,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Winsorize::new(p)
}
