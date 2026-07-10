//! Element-wise / cross-tick / cross-sectional numeric operators over the
//! strided [`ArrayView`] currency. Each homes an output [`Array<T, N>`] in its
//! state, reads the input through [`to_contiguous`](ArrayView::to_contiguous)
//! (zero-copy when the view is contiguous, materialized only when strided), and
//! lends a `ViewPort` view of the output.
//!
//! The build (`init`) call seeds the output with the faithful transform of the
//! build value (not zeros — a fabricated finite observation would leak through
//! carry readers) without notifying; the cross-tick operators ([`Diff`],
//! [`PctChange`], [`ForwardFill`]) instead seed NaN and run no per-tick state
//! update on the build call.

use std::cmp::Ordering;
use std::marker::PhantomData;

use num_traits::Float;

use flowgraph::typed::{Operator, ViewPort};

use crate::operators::op::ArrayValue;
use crate::{Array, ArrayView, Scalar};

// ---------------------------------------------------------------------------
// Clamp
// ---------------------------------------------------------------------------

/// Element-wise clamp to `[lo, hi]`.
#[derive(Clone)]
pub struct Clamp<T: Scalar, const N: usize> {
    lo: T,
    hi: T,
}

impl<T: Scalar + Float, const N: usize> Clamp<T, N> {
    pub fn new(lo: T, hi: T) -> Self {
        Self { lo, hi }
    }
}

/// Runtime state for [`Clamp`]: the bounds plus the output buffer.
pub struct ClampState<T: Scalar, const N: usize> {
    lo: T,
    hi: T,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Clamp<T, N> {
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = ClampState<T, N>;

    fn init(self) -> Self::State {
        ClampState {
            lo: self.lo,
            hi: self.hi,
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            state.out = Array::zeros(x.extents());
        }
        let (lo, hi) = (state.lo, state.hi);
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = state.out.as_mut_slice();
        for i in 0..dst.len() {
            dst[i] = lo.max(hi.min(src[i]));
        }
        (!init, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Fillna
// ---------------------------------------------------------------------------

/// Element-wise NaN replacement with a constant.
#[derive(Clone)]
pub struct Fillna<T: Scalar, const N: usize> {
    val: T,
}

impl<T: Scalar + Float, const N: usize> Fillna<T, N> {
    pub fn new(val: T) -> Self {
        Self { val }
    }
}

/// Runtime state for [`Fillna`]: the fill value plus the output buffer.
pub struct FillnaState<T: Scalar, const N: usize> {
    val: T,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Fillna<T, N> {
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = FillnaState<T, N>;

    fn init(self) -> Self::State {
        FillnaState {
            val: self.val,
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            state.out = Array::zeros(x.extents());
        }
        let val = state.val;
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = state.out.as_mut_slice();
        for i in 0..dst.len() {
            dst[i] = if src[i].is_nan() { val } else { src[i] };
        }
        (!init, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// ForwardFill
// ---------------------------------------------------------------------------

/// Forward-fills NaN with the last valid observation (per element position).
#[derive(Clone)]
pub struct ForwardFill<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> ForwardFill<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for ForwardFill<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float, const N: usize> Operator for ForwardFill<T, N> {
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    // The output buffer doubles as the fill memory: cells keep their last
    // non-NaN value across ticks because the state persists.
    type State = Array<T, N>;

    fn init(self) -> Self::State {
        Array::zeros([0; N])
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        out: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            *out = Array::full(x.extents(), T::nan());
            return (false, out.view());
        }
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = out.as_mut_slice();
        for i in 0..dst.len() {
            if !src[i].is_nan() {
                dst[i] = src[i];
            }
        }
        (true, out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        out: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }
}

// ---------------------------------------------------------------------------
// Diff (cross-tick first difference)
// ---------------------------------------------------------------------------

/// Element-wise first difference across ticks: `input - input_prev`.
#[derive(Clone)]
pub struct Diff<T: Scalar + Float, const N: usize>(PhantomData<T>);

impl<T: Scalar + Float, const N: usize> Diff<T, N> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Scalar + Float, const N: usize> Default for Diff<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Diff`]: the previous input (NaN-initialised) plus the
/// output buffer.
pub struct DiffState<T: Scalar + Float, const N: usize> {
    prev: Vec<T>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Diff<T, N> {
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = DiffState<T, N>;

    fn init(self) -> Self::State {
        DiffState {
            prev: Vec::new(),
            out: Array::zeros([0; N]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            state.prev = vec![T::nan(); x.len()];
            state.out = Array::full(x.extents(), T::nan());
            return (false, state.out.view());
        }
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = state.out.as_mut_slice();
        for i in 0..dst.len() {
            dst[i] = src[i] - state.prev[i];
        }
        state.prev.copy_from_slice(src);
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// PctChange (cross-tick linear return)
// ---------------------------------------------------------------------------

/// Element-wise one-step linear return: `input / input_prev - 1`.
#[derive(Clone)]
pub struct PctChange<T: Scalar + Float, const N: usize>(PhantomData<T>);

impl<T: Scalar + Float, const N: usize> PctChange<T, N> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Scalar + Float, const N: usize> Default for PctChange<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`PctChange`]: the previous input (NaN-initialised) plus
/// the output buffer.
pub struct PctChangeState<T: Scalar + Float, const N: usize> {
    prev: Vec<T>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for PctChange<T, N> {
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = PctChangeState<T, N>;

    fn init(self) -> Self::State {
        PctChangeState {
            prev: Vec::new(),
            out: Array::zeros([0; N]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            state.prev = vec![T::nan(); x.len()];
            state.out = Array::full(x.extents(), T::nan());
            return (false, state.out.view());
        }
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = state.out.as_mut_slice();
        let one = T::one();
        for i in 0..dst.len() {
            dst[i] = src[i] / state.prev[i] - one;
        }
        state.prev.copy_from_slice(src);
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Gaussianize (cross-sectional rank → Gaussian)
// ---------------------------------------------------------------------------

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
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = GaussianizeState<T, N>;

    fn init(self) -> Self::State {
        GaussianizeState {
            scratch: Vec::new(),
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            state.scratch = vec![0; x.len()];
            state.out = Array::zeros(x.extents());
        }
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let n = src.len();

        let mut n_valid = 0usize;
        for i in 0..n {
            if !src[i].is_nan() {
                state.scratch[n_valid] = i;
                n_valid += 1;
            }
        }
        state.scratch[..n_valid]
            .sort_by(|&a, &b| src[a].partial_cmp(&src[b]).unwrap_or(Ordering::Equal));

        let dst = state.out.as_mut_slice();
        let nan = T::nan();
        for slot in dst.iter_mut() {
            *slot = nan;
        }
        if n_valid > 0 {
            let denom = n_valid as f64;
            for rank in 0..n_valid {
                let p = (rank as f64 + 0.5) / denom;
                let z = norm_inv(p);
                dst[state.scratch[rank]] = T::from(z).unwrap_or(nan);
            }
        }
        (!init, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
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

/// Cross-sectional rank-to-percentile transform over the flat cross-section.
#[derive(Clone)]
pub struct Percentile<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Percentile<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Percentile<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Percentile`]: the index sort scratch buffer plus the
/// output buffer.
pub struct PercentileState<T: Scalar + Float, const N: usize> {
    scratch: Vec<usize>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Percentile<T, N> {
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = PercentileState<T, N>;

    fn init(self) -> Self::State {
        PercentileState {
            scratch: Vec::new(),
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            state.scratch = vec![0; x.len()];
            state.out = Array::zeros(x.extents());
        }
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let n = src.len();

        let mut n_valid = 0usize;
        for i in 0..n {
            if !src[i].is_nan() {
                state.scratch[n_valid] = i;
                n_valid += 1;
            }
        }
        state.scratch[..n_valid]
            .sort_by(|&a, &b| src[a].partial_cmp(&src[b]).unwrap_or(Ordering::Equal));

        let dst = state.out.as_mut_slice();
        let nan = T::nan();
        for slot in dst.iter_mut() {
            *slot = nan;
        }
        if n_valid > 0 {
            let denom = T::from(n_valid as f64).unwrap_or(nan);
            let half = T::from(0.5).unwrap_or(nan);
            for rank in 0..n_valid {
                let p = (T::from(rank as f64).unwrap_or(nan) + half) / denom;
                dst[state.scratch[rank]] = p;
            }
        }
        (!init, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Standardize (cross-sectional z-score)
// ---------------------------------------------------------------------------

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
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = Array<T, N>;

    fn init(self) -> Self::State {
        Array::zeros([0; N])
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        out: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            *out = Array::zeros(x.extents());
        }
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
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

        let dst = out.as_mut_slice();
        if n_valid < 2 {
            for slot in dst.iter_mut() {
                *slot = nan;
            }
            return (!init, out.view());
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
            return (!init, out.view());
        }

        for i in 0..n {
            let v = src[i];
            dst[i] = if v.is_nan() { nan } else { (v - mean) / std };
        }
        (!init, out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        out: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }
}

// ---------------------------------------------------------------------------
// Winsorize (cross-sectional percentile clipping)
// ---------------------------------------------------------------------------

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
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = WinsorizeState<T, N>;

    fn init(self) -> Self::State {
        WinsorizeState {
            p: self.p,
            sort_buf: Vec::new(),
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            state.sort_buf = vec![T::zero(); x.len()];
            state.out = Array::zeros(x.extents());
        }
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let n = src.len();

        let mut n_valid = 0usize;
        for i in 0..n {
            if !src[i].is_nan() {
                state.sort_buf[n_valid] = src[i];
                n_valid += 1;
            }
        }
        state.sort_buf[..n_valid].sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let dst = state.out.as_mut_slice();
        let nan = T::nan();

        if n_valid == 0 {
            for slot in dst.iter_mut() {
                *slot = nan;
            }
            return (!init, state.out.view());
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
        (!init, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ===========================================================================
// Constructors
// ===========================================================================

/// Element-wise clamp into `[lo, hi]` (`NaN` passes through).
pub fn clamp<T: Scalar + Float, const N: usize>(lo: T, hi: T) -> Clamp<T, N> {
    Clamp::new(lo, hi)
}

/// Replace every non-finite element with `val`.
pub fn fillna<T: Scalar + Float, const N: usize>(val: T) -> Fillna<T, N> {
    Fillna::new(val)
}

/// Carry the last finite value forward across ticks.
pub fn forward_fill<T: Scalar + Float, const N: usize>() -> ForwardFill<T, N> {
    ForwardFill::new()
}

/// Element-wise first difference across ticks: `x − x₋₁`. The `n`-tick
/// generalization over a live handle is [`change`](super::change).
pub fn diff<T: Scalar + Float, const N: usize>() -> Diff<T, N> {
    Diff::new()
}

/// Element-wise one-step linear return: `x / x₋₁ − 1`. The `n`-tick
/// generalization over a live handle is [`growth`](super::growth).
pub fn pct_change<T: Scalar + Float, const N: usize>() -> PctChange<T, N> {
    PctChange::new()
}

/// Cross-sectional rank, mapped through the inverse normal CDF.
pub fn gaussianize<T: Scalar + Float, const N: usize>() -> Gaussianize<T, N> {
    Gaussianize::new()
}

/// Cross-sectional percentile rank into `[0, 1]`.
pub fn percentile<T: Scalar + Float, const N: usize>() -> Percentile<T, N> {
    Percentile::new()
}

/// Cross-sectional z-score: `(x − mean) / std`.
pub fn standardize<T: Scalar + Float, const N: usize>() -> Standardize<T, N> {
    Standardize::new()
}

/// Cross-sectionally clip the tails at the `p` / `1 − p` quantiles.
pub fn winsorize<T: Scalar + Float, const N: usize>(p: T) -> Winsorize<T, N> {
    Winsorize::new(p)
}
