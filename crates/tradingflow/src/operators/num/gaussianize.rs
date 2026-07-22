//! [`Gaussianize`] — the cross-sectional rank → Gaussian transform.

use std::marker::PhantomData;

use num_traits::Float;

use super::rank::rank_finite;
use crate::data::{Array, ArrayView, Instant, Layout, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Cross-sectional rank-to-Gaussian transform over the flat cross-section.
#[derive(Clone)]
pub struct Gaussianize<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Gaussianize<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Gaussianize<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Gaussianize`]: the index sort scratch buffer plus the
/// output buffer.
pub struct GaussianizeState<T: Scalar + Float, const N: usize> {
    scratch: Vec<usize>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Gaussianize<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = GaussianizeState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        let mut state = GaussianizeState {
            scratch: vec![0; x.layout().len()],
            out: Array::zeros(x.extents()),
        };
        // Seed the output with the faithful transform of the build value; the
        // initial render does not notify.
        gaussianize_into(&mut state, x);
        state
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        gaussianize_into(state, x);
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// The per-call body of [`Gaussianize`]: rank the finite entries of `x` into
/// `state.out` through the inverse normal CDF (NaN elsewhere).
fn gaussianize_into<T: Scalar + Float, const N: usize>(
    state: &mut GaussianizeState<T, N>,
    x: ArrayView<'_, T, N>,
) {
    let xs = x.to_contiguous();
    let src: &[T] = &xs;

    let GaussianizeState { scratch, out } = state;
    let dst = out.data_mut();
    let n_valid = rank_finite(src, scratch, dst);

    if n_valid > 0 {
        let nan = T::nan();
        let denom = n_valid as f64;
        for rank in 0..n_valid {
            let p = (rank as f64 + 0.5) / denom;
            let z = norm_inv(p);
            dst[scratch[rank]] = T::from(z).unwrap_or(nan);
        }
    }
}

/// Inverse standard-normal CDF via Acklam's rational approximation.
fn norm_inv(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const PLOW: f64 = 0.02425;
    const PHIGH: f64 = 1.0 - PLOW;

    if p < PLOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= PHIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Cross-sectional rank, mapped through the inverse normal CDF.
pub fn gaussianize<T: Scalar + Float, const N: usize>() -> Gaussianize<T, N> {
    Gaussianize::new()
}
