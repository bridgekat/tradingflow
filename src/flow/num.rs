//! Element-wise / cross-tick / cross-sectional numeric operators, implemented
//! directly on [`flowgraph::typed::Operator`]. The loop bodies are unchanged
//! from the legacy port; the output buffer lives in the operator state and is
//! sized/seeded on the `init` build call.

use std::cmp::Ordering;
use std::marker::PhantomData;

use num_traits::Float;

use flowgraph::typed::{Operator, RefPort};

use crate::{Array, Scalar};

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

/// Runtime state for [`Clamp`]: the bounds plus the output buffer.
pub struct ClampState<T: Scalar> {
    lo: T,
    hi: T,
    out: Array<T>,
}

impl<T: Scalar + Float> Operator for Clamp<T> {
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefPort<Array<T>>;
    type State = ClampState<T>;

    fn init(self) -> ClampState<T> {
        ClampState {
            lo: self.lo,
            hi: self.hi,
            out: Array::zeros(&[0]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, a): (bool, &'a Array<T>),
        state: &'b mut ClampState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            state.out = Array::zeros(a.shape());
            return (false, &state.out);
        }
        let (lo, hi) = (state.lo, state.hi);
        let src = a.as_slice();
        let dst = state.out.as_mut_slice();
        for i in 0..dst.len() {
            dst[i] = lo.max(hi.min(src[i]));
        }
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        state: &'b ClampState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
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

/// Runtime state for [`Fillna`]: the fill value plus the output buffer.
pub struct FillnaState<T: Scalar> {
    val: T,
    out: Array<T>,
}

impl<T: Scalar + Float> Operator for Fillna<T> {
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefPort<Array<T>>;
    type State = FillnaState<T>;

    fn init(self) -> FillnaState<T> {
        FillnaState {
            val: self.val,
            out: Array::zeros(&[0]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, a): (bool, &'a Array<T>),
        state: &'b mut FillnaState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            state.out = Array::zeros(a.shape());
            return (false, &state.out);
        }
        let val = state.val;
        let src = a.as_slice();
        let dst = state.out.as_mut_slice();
        for i in 0..dst.len() {
            dst[i] = if src[i].is_nan() { val } else { src[i] };
        }
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        state: &'b FillnaState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
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
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefPort<Array<T>>;
    // The output buffer doubles as the fill memory: cells keep their last
    // non-NaN value across ticks because the state persists.
    type State = Array<T>;

    fn init(self) -> Array<T> {
        Array::zeros(&[0])
    }

    fn compute<'a, 'b: 'a>(
        (_, a): (bool, &'a Array<T>),
        out: &'b mut Array<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            let shape = a.shape();
            let stride: usize = shape.iter().product();
            *out = Array::from_vec(shape, vec![T::nan(); stride]);
            return (false, &*out);
        }
        let src = a.as_slice();
        let dst = out.as_mut_slice();
        for i in 0..dst.len() {
            if !src[i].is_nan() {
                dst[i] = src[i];
            }
        }
        (true, &*out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        out: &'b Array<T>,
    ) -> (bool, &'a Array<T>) {
        (false, out)
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

/// Runtime state for [`Diff`]: the previous input array (NaN-initialised)
/// plus the output buffer.
pub struct DiffState<T: Scalar + Float> {
    prev: Vec<T>,
    out: Array<T>,
}

impl<T: Scalar + Float> Operator for Diff<T> {
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefPort<Array<T>>;
    type State = DiffState<T>;

    fn init(self) -> DiffState<T> {
        DiffState {
            prev: Vec::new(),
            out: Array::zeros(&[0]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, a): (bool, &'a Array<T>),
        state: &'b mut DiffState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            let shape = a.shape();
            let stride: usize = shape.iter().product();
            state.prev = vec![T::nan(); stride];
            state.out = Array::from_vec(shape, vec![T::nan(); stride]);
            return (false, &state.out);
        }
        let src = a.as_slice();
        let dst = state.out.as_mut_slice();
        for i in 0..dst.len() {
            dst[i] = src[i] - state.prev[i];
        }
        state.prev.copy_from_slice(src);
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        state: &'b DiffState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
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

/// Runtime state for [`PctChange`]: the previous input array (NaN-initialised)
/// plus the output buffer.
pub struct PctChangeState<T: Scalar + Float> {
    prev: Vec<T>,
    out: Array<T>,
}

impl<T: Scalar + Float> Operator for PctChange<T> {
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefPort<Array<T>>;
    type State = PctChangeState<T>;

    fn init(self) -> PctChangeState<T> {
        PctChangeState {
            prev: Vec::new(),
            out: Array::zeros(&[0]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, a): (bool, &'a Array<T>),
        state: &'b mut PctChangeState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            let shape = a.shape();
            let stride: usize = shape.iter().product();
            state.prev = vec![T::nan(); stride];
            state.out = Array::from_vec(shape, vec![T::nan(); stride]);
            return (false, &state.out);
        }
        let src = a.as_slice();
        let dst = state.out.as_mut_slice();
        let one = T::one();
        for i in 0..dst.len() {
            dst[i] = src[i] / state.prev[i] - one;
        }
        state.prev.copy_from_slice(src);
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        state: &'b PctChangeState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
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

/// Runtime state for [`Gaussianize`]: the index sort scratch buffer plus the
/// output buffer.
pub struct GaussianizeState<T: Scalar + Float> {
    scratch: Vec<usize>,
    out: Array<T>,
}

impl<T: Scalar + Float> Operator for Gaussianize<T> {
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefPort<Array<T>>;
    type State = GaussianizeState<T>;

    fn init(self) -> GaussianizeState<T> {
        GaussianizeState {
            scratch: Vec::new(),
            out: Array::zeros(&[0]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, a): (bool, &'a Array<T>),
        state: &'b mut GaussianizeState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            state.scratch = vec![0; a.as_slice().len()];
            state.out = Array::zeros(a.shape());
            return (false, &state.out);
        }
        let src = a.as_slice();
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
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        state: &'b GaussianizeState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
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

/// Runtime state for [`Percentile`]: the index sort scratch buffer plus the
/// output buffer.
pub struct PercentileState<T: Scalar + Float> {
    scratch: Vec<usize>,
    out: Array<T>,
}

impl<T: Scalar + Float> Operator for Percentile<T> {
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefPort<Array<T>>;
    type State = PercentileState<T>;

    fn init(self) -> PercentileState<T> {
        PercentileState {
            scratch: Vec::new(),
            out: Array::zeros(&[0]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, a): (bool, &'a Array<T>),
        state: &'b mut PercentileState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            state.scratch = vec![0; a.as_slice().len()];
            state.out = Array::zeros(a.shape());
            return (false, &state.out);
        }
        let src = a.as_slice();
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
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        state: &'b PercentileState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
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
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefPort<Array<T>>;
    type State = Array<T>;

    fn init(self) -> Array<T> {
        Array::zeros(&[0])
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, a): (bool, &'a Array<T>),
        out: &'b mut Array<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            *out = Array::zeros(a.shape());
            return (false, &*out);
        }
        let src = a.as_slice();
        let dst = out.as_mut_slice();
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
            return (true, &*out);
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
            return (true, &*out);
        }

        for i in 0..n {
            let v = src[i];
            dst[i] = if v.is_nan() { nan } else { (v - mean) / std };
        }
        (true, &*out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        out: &'b Array<T>,
    ) -> (bool, &'a Array<T>) {
        (false, out)
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

/// Runtime state for [`Winsorize`]: the quantile, a scratch sort buffer and
/// the output buffer.
pub struct WinsorizeState<T: Scalar + Float> {
    p: T,
    sort_buf: Vec<T>,
    out: Array<T>,
}

impl<T: Scalar + Float> Operator for Winsorize<T> {
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefPort<Array<T>>;
    type State = WinsorizeState<T>;

    fn init(self) -> WinsorizeState<T> {
        WinsorizeState {
            p: self.p,
            sort_buf: Vec::new(),
            out: Array::zeros(&[0]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, a): (bool, &'a Array<T>),
        state: &'b mut WinsorizeState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            state.sort_buf = vec![T::zero(); a.as_slice().len()];
            state.out = Array::zeros(a.shape());
            return (false, &state.out);
        }
        let src = a.as_slice();
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
            return (true, &state.out);
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
        (true, &state.out)
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        state: &'b WinsorizeState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
    }
}
