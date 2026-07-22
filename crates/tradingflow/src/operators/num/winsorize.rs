//! [`Winsorize`] — cross-sectional percentile clipping.

use std::cmp::Ordering;
use std::marker::PhantomData;

use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Layout, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Cross-sectional winsorization: clip non-NaN values to the `[p, 1-p]`
/// quantile range of the cross-section.
#[derive(Clone)]
pub struct Winsorize<T: Scalar + Float, const N: usize> {
    p: T,
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Winsorize<T, N> {
    pub fn new(p: T) -> Self {
        assert!(p >= T::zero(), "Winsorize requires p >= 0");
        assert!(p < T::from(0.5).unwrap(), "Winsorize requires p < 0.5");
        Self {
            p,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`Winsorize`]: the quantile, a scratch sort buffer and the
/// output buffer.
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
        // Seed the output with the faithful transform of the build value; the
        // initial render does not notify.
        winsorize_into(&mut state, x);
        state
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        winsorize_into(state, x);
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// The per-call body of [`Winsorize`]: clip the cross-section of `x` to its
/// `[p, 1-p]` quantile range into `state.out` (all NaN if nothing finite).
fn winsorize_into<T: Scalar + Float, const N: usize>(
    state: &mut WinsorizeState<T, N>,
    x: ArrayView<'_, T, N>,
) {
    let xs = x.to_contiguous();
    let src: &[T] = &xs;
    let n = src.len();

    let mut n_valid = 0usize;
    for &v in src.iter() {
        if !v.is_nan() {
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

/// Cross-sectionally clip the tails at the `p` / `1 − p` quantiles.
pub fn winsorize<T: Scalar + Float, const N: usize>(p: T) -> Winsorize<T, N> {
    Winsorize::new(p)
}
