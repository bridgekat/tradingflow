//! Shared fixtures for the operator integration tests: instant/array
//! constructors, NaN-aware assertions, deterministic data paths, and the two
//! auxiliary segments (a pokeable clock and an eventful-batch counter) that
//! the event-semantics tests wire in as probes.

use tradingflow::data::{Array, ArrayView, Duration, Instant, Scalar, SeriesView};
use tradingflow::graph::{Port, Segment, Val};
use tradingflow::ports::{ArrayPort, ClockPort, is_eventful};

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

/// Operator signature for [`count`].
pub struct Count<const N: usize>;

impl<const N: usize> Segment for Count<N> {
    type Inputs = ArrayPort<f64, N>;
    type Outputs = Port<Val<usize>>;
    type Context = Instant;
    type State = usize;

    fn init(self, _: ArrayView<'_, f64, N>) -> usize {
        0
    }

    fn compute<'a, 'b: 'a>(a: ArrayView<'a, f64, N>, state: &'b mut usize, _: &Instant) -> usize {
        if is_eventful(a) {
            *state += 1;
        }
        *state
    }

    fn reset<'a, 'b: 'a>(_: ArrayView<'a, f64, N>, state: &'b mut usize) -> usize {
        *state
    }
}

/// Counts how many *eventful* input batches this node has seen — the probe
/// for event-propagation assertions (a scheduled generation whose input is
/// quiescent does not count).
pub fn count<const N: usize>()
-> impl Segment<Inputs = ArrayPort<f64, N>, Outputs = Port<Val<usize>>, Context = Instant> {
    Count
}

/// Operator signature for [`clock`].
pub struct ManualClock;

impl Segment for ManualClock {
    type Inputs = ();
    type Outputs = ClockPort;
    type Context = Instant;
    type State = ();

    fn init(self, _: ()) {}

    fn reset<'a, 'b: 'a>(_: (), _: &'b mut ()) -> bool {
        false
    }

    fn compute<'a, 'b: 'a>(_: (), _: &'b mut (), _: &Instant) -> bool {
        true
    }
}

/// A pokeable manual clock: wired as a source, touching its state via
/// `state_mut` marks it dirty, so it pulses (`true`) for exactly the
/// generations the test chooses and resets to `false` in between.
pub fn clock() -> impl Segment<Inputs = (), Outputs = ClockPort, Context = Instant, State = ()> {
    ManualClock
}
