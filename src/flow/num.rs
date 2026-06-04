//! Element-wise / cross-tick / cross-sectional numeric operators — port of the
//! non-arithmetic members of [`crate::operators::num`]. Bodies are transcribed
//! verbatim; only `Input`→`Port` and the `produced` type differ.

use std::cmp::Ordering;
use std::marker::PhantomData;

use num_traits::Float;

use flowgraph::typed::Port;

use super::op::Operator;
use crate::{Array, Instant, Scalar};

// ---------------------------------------------------------------------------
// Clamp
// ---------------------------------------------------------------------------

/// Element-wise clamp to `[lo, hi]`.
#[derive(Clone)]
pub struct Clamp<T: Scalar> {
    lo: T,
    hi: T,
}

impl<T: Scalar + Float> Clamp<T> {
    pub fn new(lo: T, hi: T) -> Self {
        Self { lo, hi }
    }
}

impl<T: Scalar + Float> Operator for Clamp<T> {
    type State = (T, T);
    type Inputs = Port<Array<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Array<T>, _ts: Instant) -> ((T, T), Array<T>) {
        ((self.lo, self.hi), Array::zeros(inputs.shape()))
    }

    #[inline(always)]
    fn compute(
        state: &mut (T, T),
        inputs: &Array<T>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let (lo, hi) = *state;
        let a = inputs.as_slice();
        let out = output.as_mut_slice();
        for i in 0..out.len() {
            out[i] = lo.max(hi.min(a[i]));
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Fillna
// ---------------------------------------------------------------------------

/// Element-wise NaN replacement with a constant.
#[derive(Clone)]
pub struct Fillna<T: Scalar> {
    val: T,
}

impl<T: Scalar + Float> Fillna<T> {
    pub fn new(val: T) -> Self {
        Self { val }
    }
}

impl<T: Scalar + Float> Operator for Fillna<T> {
    type State = T;
    type Inputs = Port<Array<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Array<T>, _ts: Instant) -> (T, Array<T>) {
        (self.val, Array::zeros(inputs.shape()))
    }

    #[inline(always)]
    fn compute(
        state: &mut T,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let val = *state;
        let a = inputs.as_slice();
        let out = output.as_mut_slice();
        for i in 0..out.len() {
            out[i] = if a[i].is_nan() { val } else { a[i] };
        }
        true
    }
}

// ---------------------------------------------------------------------------
// ForwardFill
// ---------------------------------------------------------------------------

/// Forward-fills NaN with the last valid observation (per element position).
#[derive(Clone)]
pub struct ForwardFill<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> ForwardFill<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for ForwardFill<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float> Operator for ForwardFill<T> {
    type State = ();
    type Inputs = Port<Array<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Array<T>, _ts: Instant) -> ((), Array<T>) {
        let shape = inputs.shape();
        let stride: usize = shape.iter().product();
        ((), Array::from_vec(shape, vec![T::nan(); stride]))
    }

    fn compute(
        _state: &mut (),
        inputs: &Array<T>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let src = inputs.as_slice();
        let dst = output.as_mut_slice();
        for i in 0..dst.len() {
            if !src[i].is_nan() {
                dst[i] = src[i];
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Diff (cross-tick first difference)
// ---------------------------------------------------------------------------

/// Element-wise first difference across ticks: `input - input_prev`.
#[derive(Clone)]
pub struct Diff<T: Scalar + Float>(PhantomData<T>);

impl<T: Scalar + Float> Diff<T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Scalar + Float> Default for Diff<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// State: the previous input array (NaN-initialised).
pub struct DiffState<T: Scalar + Float> {
    prev: Vec<T>,
}

impl<T: Scalar + Float> Operator for Diff<T> {
    type State = DiffState<T>;
    type Inputs = Port<Array<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Array<T>, _ts: Instant) -> (DiffState<T>, Array<T>) {
        let shape = inputs.shape();
        let stride: usize = shape.iter().product();
        let out = Array::from_vec(shape, vec![T::nan(); stride]);
        (
            DiffState {
                prev: vec![T::nan(); stride],
            },
            out,
        )
    }

    fn compute(
        state: &mut DiffState<T>,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let src = inputs.as_slice();
        let dst = output.as_mut_slice();
        for i in 0..dst.len() {
            dst[i] = src[i] - state.prev[i];
        }
        state.prev.copy_from_slice(src);
        true
    }
}

// ---------------------------------------------------------------------------
// PctChange (cross-tick linear return)
// ---------------------------------------------------------------------------

/// Element-wise one-step linear return: `input / input_prev - 1`.
#[derive(Clone)]
pub struct PctChange<T: Scalar + Float>(PhantomData<T>);

impl<T: Scalar + Float> PctChange<T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Scalar + Float> Default for PctChange<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// State: the previous input array (NaN-initialised).
pub struct PctChangeState<T: Scalar + Float> {
    prev: Vec<T>,
}

impl<T: Scalar + Float> Operator for PctChange<T> {
    type State = PctChangeState<T>;
    type Inputs = Port<Array<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Array<T>, _ts: Instant) -> (PctChangeState<T>, Array<T>) {
        let shape = inputs.shape();
        let stride: usize = shape.iter().product();
        let out = Array::from_vec(shape, vec![T::nan(); stride]);
        (
            PctChangeState {
                prev: vec![T::nan(); stride],
            },
            out,
        )
    }

    fn compute(
        state: &mut PctChangeState<T>,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let src = inputs.as_slice();
        let dst = output.as_mut_slice();
        let one = T::one();
        for i in 0..dst.len() {
            dst[i] = src[i] / state.prev[i] - one;
        }
        state.prev.copy_from_slice(src);
        true
    }
}

// ---------------------------------------------------------------------------
// Gaussianize (cross-sectional rank → Gaussian)
// ---------------------------------------------------------------------------

/// Cross-sectional rank-to-Gaussian transform on a 1-D array.
#[derive(Clone)]
pub struct Gaussianize<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> Gaussianize<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for Gaussianize<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float> Operator for Gaussianize<T> {
    type State = Vec<usize>;
    type Inputs = Port<Array<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Array<T>, _ts: Instant) -> (Vec<usize>, Array<T>) {
        let n = inputs.as_slice().len();
        (vec![0; n], Array::zeros(inputs.shape()))
    }

    #[inline(always)]
    fn compute(
        state: &mut Vec<usize>,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let src = inputs.as_slice();
        let n = src.len();

        let mut n_valid = 0usize;
        for i in 0..n {
            if !src[i].is_nan() {
                state[n_valid] = i;
                n_valid += 1;
            }
        }
        state[..n_valid].sort_by(|&a, &b| src[a].partial_cmp(&src[b]).unwrap_or(Ordering::Equal));

        let dst = output.as_mut_slice();
        let nan = T::nan();
        for slot in dst.iter_mut() {
            *slot = nan;
        }
        if n_valid > 0 {
            let denom = n_valid as f64;
            for rank in 0..n_valid {
                let p = (rank as f64 + 0.5) / denom;
                let z = norm_inv(p);
                dst[state[rank]] = T::from(z).unwrap_or(nan);
            }
        }
        true
    }
}

/// Inverse standard-normal CDF via Acklam's rational approximation.
#[inline]
fn norm_inv(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
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

// ---------------------------------------------------------------------------
// Percentile (cross-sectional rank → percentile)
// ---------------------------------------------------------------------------

/// Cross-sectional rank-to-percentile transform on a 1-D array.
#[derive(Clone)]
pub struct Percentile<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> Percentile<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for Percentile<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float> Operator for Percentile<T> {
    type State = Vec<usize>;
    type Inputs = Port<Array<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Array<T>, _ts: Instant) -> (Vec<usize>, Array<T>) {
        let n = inputs.as_slice().len();
        (vec![0; n], Array::zeros(inputs.shape()))
    }

    #[inline(always)]
    fn compute(
        state: &mut Vec<usize>,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let src = inputs.as_slice();
        let n = src.len();

        let mut n_valid = 0usize;
        for i in 0..n {
            if !src[i].is_nan() {
                state[n_valid] = i;
                n_valid += 1;
            }
        }
        state[..n_valid].sort_by(|&a, &b| src[a].partial_cmp(&src[b]).unwrap_or(Ordering::Equal));

        let dst = output.as_mut_slice();
        let nan = T::nan();
        for slot in dst.iter_mut() {
            *slot = nan;
        }
        if n_valid > 0 {
            let denom = T::from(n_valid as f64).unwrap_or(nan);
            let half = T::from(0.5).unwrap_or(nan);
            for rank in 0..n_valid {
                let p = (T::from(rank as f64).unwrap_or(nan) + half) / denom;
                dst[state[rank]] = p;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Standardize (cross-sectional z-score)
// ---------------------------------------------------------------------------

/// Cross-sectional z-score (population std; NaN if < 2 finite or σ = 0).
#[derive(Clone)]
pub struct Standardize<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> Standardize<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for Standardize<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float> Operator for Standardize<T> {
    type State = ();
    type Inputs = Port<Array<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Array<T>, _ts: Instant) -> ((), Array<T>) {
        ((), Array::zeros(inputs.shape()))
    }

    #[inline(always)]
    fn compute(
        _state: &mut (),
        inputs: &Array<T>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let src = inputs.as_slice();
        let dst = output.as_mut_slice();
        let n = src.len();
        let nan = T::nan();

        let mut n_valid = 0usize;
        let mut sum = T::zero();
        for i in 0..n {
            let v = src[i];
            if !v.is_nan() {
                n_valid += 1;
                sum = sum + v;
            }
        }

        if n_valid < 2 {
            for slot in dst.iter_mut() {
                *slot = nan;
            }
            return true;
        }

        let mean = sum / T::from(n_valid).unwrap();

        let mut ssd = T::zero();
        for i in 0..n {
            let v = src[i];
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
            return true;
        }

        for i in 0..n {
            let v = src[i];
            dst[i] = if v.is_nan() { nan } else { (v - mean) / std };
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Winsorize (cross-sectional percentile clipping)
// ---------------------------------------------------------------------------

/// Cross-sectional winsorization: clip non-NaN values to the `[p, 1-p]`
/// quantile range of the cross-section.
#[derive(Clone)]
pub struct Winsorize<T: Scalar + Float> {
    p: T,
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> Winsorize<T> {
    pub fn new(p: T) -> Self {
        assert!(p >= T::zero(), "Winsorize requires p >= 0");
        assert!(p < T::from(0.5).unwrap(), "Winsorize requires p < 0.5");
        Self {
            p,
            _phantom: PhantomData,
        }
    }
}

/// State: the quantile and a scratch sort buffer.
pub struct WinsorizeState<T: Scalar + Float> {
    p: T,
    sort_buf: Vec<T>,
}

impl<T: Scalar + Float> Operator for Winsorize<T> {
    type State = WinsorizeState<T>;
    type Inputs = Port<Array<T>>;
    type Output = Array<T>;

    fn init(&self, inputs: &Array<T>, _ts: Instant) -> (WinsorizeState<T>, Array<T>) {
        let n = inputs.as_slice().len();
        (
            WinsorizeState {
                p: self.p,
                sort_buf: vec![T::zero(); n],
            },
            Array::zeros(inputs.shape()),
        )
    }

    #[inline(always)]
    fn compute(
        state: &mut WinsorizeState<T>,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let src = inputs.as_slice();
        let n = src.len();

        let mut n_valid = 0usize;
        for i in 0..n {
            if !src[i].is_nan() {
                state.sort_buf[n_valid] = src[i];
                n_valid += 1;
            }
        }
        state.sort_buf[..n_valid].sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let dst = output.as_mut_slice();
        let nan = T::nan();

        if n_valid == 0 {
            for slot in dst.iter_mut() {
                *slot = nan;
            }
            return true;
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
        true
    }
}
