//! Shared fixtures for the operator integration tests: instant/array
//! constructors, NaN-aware assertions, deterministic data paths, and the two
//! auxiliary segments (a pokeable unit clock and a recompute counter) that the
//! event-semantics tests wire in as probes.

use tradingflow::data::{Array, ArrayView, Duration, Instant, Scalar, SeriesView};
use tradingflow::graph::{Operator, Port, Segment, Val};
use tradingflow::ports::ArrayPort;

/// Default tolerance for [`assert_close`]. Loose enough for the accumulators'
/// incremental arithmetic, tight enough that a wrong formula never passes.
pub const EPS: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Instants
// ---------------------------------------------------------------------------

/// An instant `n` nanoseconds after the epoch — the default stamp for tests
/// that only need distinct, ordered ticks.
pub fn nano(n: i64) -> Instant {
    Instant::from_offset(Duration::from_nanos(n))
}

/// An instant `n` days after the epoch, for the duration-windowed operators
/// whose retention is expressed in real time rather than tick counts.
pub fn day(n: i64) -> Instant {
    Instant::from_offset(Duration::from_days(n))
}

// ---------------------------------------------------------------------------
// Array constructors
// ---------------------------------------------------------------------------

/// A rank-0 (scalar) array.
pub fn scalar<T: Scalar>(v: T) -> Array<T, 0> {
    Array::scalar(v)
}

/// A rank-`N` array from its extents and row-major data.
pub fn arr<T: Scalar, const N: usize>(
    extents: [usize; N],
    data: impl Into<Box<[T]>>,
) -> Array<T, N> {
    Array::from_parts(extents, data.into())
}

/// A rank-1 array, the common cross-section shape.
pub fn arr1<T: Scalar>(data: impl Into<Box<[T]>>) -> Array<T, 1> {
    let data = data.into();
    Array::from_parts([data.len()], data)
}

// ---------------------------------------------------------------------------
// View readback
// ---------------------------------------------------------------------------

/// Materializes an array view's elements in row-major order.
pub fn vals<T: Scalar, const N: usize>(v: ArrayView<'_, T, N>) -> Vec<T> {
    v.to_contiguous().to_vec()
}

/// Materializes a series view's elements, oldest row first.
pub fn series_vals<T: Scalar, const N: usize>(v: SeriesView<'_, T, N>) -> Vec<T> {
    v.to_contiguous().to_vec()
}

/// The raw bit patterns of an array view — the comparison to use when two
/// wirings must agree *exactly*, including which `NaN` they produce.
pub fn bits<const N: usize>(v: ArrayView<'_, f64, N>) -> Vec<u64> {
    v.to_contiguous().iter().map(|x| x.to_bits()).collect()
}

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------

/// Asserts elementwise equality within [`TOL`].
///
/// `NaN` matches only `NaN` and each infinity matches only itself with the
/// same sign — both are ordinary results here (`NaN` is the missing-data
/// marker, and infinities fall out of `recip(0)`, `ln(0)`, division by zero
/// and the like), so they are values to assert on rather than errors. Note a
/// tolerance test cannot express them: `inf - inf` is `NaN`, which compares
/// false against any bound.
#[track_caller]
pub fn assert_close(actual: &[f64], expected: &[f64], ctx: &str) {
    assert_close_tol(actual, expected, EPS, ctx);
}

/// [`assert_close`] with an explicit tolerance.
#[track_caller]
pub fn assert_close_tol(actual: &[f64], expected: &[f64], tol: f64, ctx: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{ctx}: length {} != expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let ok = if e.is_nan() {
            a.is_nan()
        } else if e.is_infinite() {
            a == e
        } else {
            (a - e).abs() <= tol
        };
        assert!(
            ok,
            "{ctx}: element {i} is {a}, expected {e}\n  actual: {actual:?}\n  expected: {expected:?}"
        );
    }
}

/// Asserts two wirings produced bit-identical output.
#[track_caller]
pub fn assert_same_bits<const N: usize>(
    a: ArrayView<'_, f64, N>,
    b: ArrayView<'_, f64, N>,
    ctx: &str,
) {
    assert_eq!(bits(a), bits(b), "{ctx}: {:?} != {:?}", vals(a), vals(b));
}

// ---------------------------------------------------------------------------
// Deterministic data
// ---------------------------------------------------------------------------

/// A deterministic pseudo-random path of quarter-valued samples. Quarters keep
/// every running sum exactly representable in binary floating point, so an
/// incremental accumulator and a freshly-summed reference agree bit-for-bit
/// and exact equality is a sound assertion.
pub fn quarter_path(seed: u64, len: usize) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % 1000) as f64 / 4.0
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Probe segments
// ---------------------------------------------------------------------------

/// Operator signature for [`const_val`].
pub struct ConstVal<T: Copy + Send + Sync + 'static> {
    value: T,
}

impl<T: Copy + Send + Sync + 'static> Segment for ConstVal<T> {
    type Inputs = ();
    type Outputs = Port<Val<T>>;
    type Context = Instant;
    type State = T;

    fn init(self, _: ()) -> T {
        self.value
    }

    fn output<'a, 'b: 'a>(_: (), state: &'b mut T) -> (bool, T) {
        (true, *state)
    }

    fn compute<'a, 'b: 'a>(_: (), state: &'b mut T, _: &Instant) -> (bool, T) {
        (true, *state)
    }
}

/// A pokeable constant cell. Wired as a source it doubles as a manual clock:
/// touching its state via `state_mut` marks it dirty, so it notifies for
/// exactly the generations the test chooses.
pub fn const_val<T: Copy + Send + Sync + 'static>(
    value: T,
) -> impl Segment<Inputs = (), Outputs = Port<Val<T>>, Context = Instant, State = T> {
    ConstVal { value }
}

/// Operator signature for [`count`].
pub struct Count<const N: usize>;

impl<const N: usize> Operator for Count<N> {
    type Inputs = ArrayPort<f64, N>;
    type Outputs = Port<Val<usize>>;
    type Context = Instant;
    type State = usize;

    fn init(self, _: (bool, ArrayView<'_, f64, N>)) -> usize {
        0
    }

    fn compute<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, f64, N>),
        state: &'b mut usize,
        _: &Instant,
    ) -> (bool, usize) {
        *state += 1;
        (true, *state)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, f64, N>),
        state: &'b mut usize,
    ) -> (bool, usize) {
        (false, *state)
    }
}

/// Counts how many generations actually recomputed this node — the probe for
/// notification-propagation assertions.
pub fn count<const N: usize>()
-> impl Segment<Inputs = ArrayPort<f64, N>, Outputs = Port<Val<usize>>, Context = Instant> {
    Count
}
