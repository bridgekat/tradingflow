//! Shared fixtures for the operator integration tests: instant/array
//! constructors, NaN-aware assertions, deterministic data paths, and the
//! auxiliary probe operators (a pokeable signal, a signal-pulse counter and a
//! scheduled-generation counter) that the event-semantics tests wire in.

use tradingflow::data::{Array, ArrayView, Duration, Instant, Scalar, SeriesView};
use tradingflow::graph::typed::{Builder, NodeHandle};
use tradingflow::graph::{Operator, Port, Val};
use tradingflow::ports::{ArrayPort, ArrayPortHandle, SignalPort, SignalPortHandle};

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

/// Asserts elementwise equality within [`EPS`].
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
// Probe operators
// ---------------------------------------------------------------------------

/// A pokeable array cell paired with its accompanying signal: each poke is one
/// pulse, and the values wire retains the last poked batch in between.
/// Returns the source handle (for `state_mut`) and the `(signal, values)`
/// stream handles (for wiring); state-only consumers wire just the values.
#[allow(clippy::type_complexity)]
pub fn event_src<const N: usize>(
    b: &mut Builder<Instant>,
    v: Array<f64, N>,
) -> (
    NodeHandle<impl Operator<State = Array<f64, N>> + use<N>>,
    (SignalPortHandle<0>, ArrayPortHandle<f64, N>),
) {
    b.source(cell(v))
}

/// A pokeable array cell paired with its accompanying signal: the signal
/// pulses on exactly the generations the cell is poked (a root with no inputs
/// is scheduled only when its own state is touched) and is `false` otherwise.
/// The manually-stepped counterpart of
/// [`sources::sync::array_series`](tradingflow::sources::sync::array_series).
pub struct Cell<T: Scalar, const N: usize> {
    value: Array<T, N>,
}

impl<T: Scalar, const N: usize> Operator for Cell<T, N> {
    type Inputs = ();
    type Outputs = (SignalPort<0>, ArrayPort<T, N>);
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self, _: ()) -> Self::State {
        self.value
    }

    fn reset<'a, 'b: 'a>(
        _: (),
        state: &'b mut Array<T, N>,
    ) -> (ArrayView<'a, bool, 0>, ArrayView<'a, T, N>) {
        (ArrayView::scalar(&false), state.view())
    }

    fn compute<'a, 'b: 'a>(
        _: (),
        state: &'b mut Array<T, N>,
        _: &Instant,
    ) -> (ArrayView<'a, bool, 0>, ArrayView<'a, T, N>) {
        (ArrayView::scalar(&true), state.view())
    }
}

/// See [`Cell`].
pub fn cell<T: Scalar, const N: usize>(
    value: impl Into<Array<T, N>>,
) -> impl Operator<
    Inputs = (),
    Outputs = (SignalPort<0>, ArrayPort<T, N>),
    Context = Instant,
    State = Array<T, N>,
> {
    Cell {
        value: value.into(),
    }
}

/// Operator signature for [`batch_face`].
pub struct BatchFace<const N: usize>;

impl<const N: usize> Operator for BatchFace<N> {
    type Inputs = (SignalPort<N>, ArrayPort<f64, N>);
    type Outputs = ArrayPort<f64, N>;
    type Context = Instant;
    type State = Array<f64, N>;

    fn init(self, (_, a): (ArrayView<'_, bool, N>, ArrayView<'_, f64, N>)) -> Array<f64, N> {
        Array::full(a.extents(), f64::NAN)
    }

    fn compute<'a, 'b: 'a>(
        (signals, a): (ArrayView<'a, bool, N>, ArrayView<'a, f64, N>),
        state: &'b mut Array<f64, N>,
        _: &Instant,
    ) -> ArrayView<'a, f64, N> {
        let signals = tradingflow::data::array::broadcast_to(signals, a.extents());
        for ((out, &c), &v) in state
            .data_mut()
            .iter_mut()
            .zip(signals.iter())
            .zip(a.iter())
        {
            *out = if c { v } else { f64::NAN };
        }
        state.view()
    }

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, bool, N>, ArrayView<'a, f64, N>),
        state: &'b mut Array<f64, N>,
    ) -> ArrayView<'a, f64, N> {
        state.view()
    }
}

/// Materializes an event stream's batch face of the last *scheduled*
/// generation: element `j` is the values wire where signal element `j` pulsed
/// and NaN elsewhere. Signals read post-reset (all-false) through
/// `g.view`, so assertions on "what arrived on this wire this tick" go
/// through this probe instead. Deliberately not `signal::collect`: the probe
/// is clockless and *carries* the face between scheduled generations so that
/// `g.view` can read it after the fact.
pub fn batch_face<const N: usize>() -> impl Operator<
    Inputs = (SignalPort<N>, ArrayPort<f64, N>),
    Outputs = ArrayPort<f64, N>,
    Context = Instant,
> {
    BatchFace
}

/// Operator signature for [`count`].
pub struct Count<const N: usize>;

impl<const N: usize> Operator for Count<N> {
    type Inputs = (SignalPort<0>, ArrayPort<f64, N>);
    type Outputs = Port<Val<usize>>;
    type Context = Instant;
    type State = usize;

    fn init(self, _: (ArrayView<'_, bool, 0>, ArrayView<'_, f64, N>)) -> usize {
        0
    }

    fn compute<'a, 'b: 'a>(
        (signal, _): (ArrayView<'a, bool, 0>, ArrayView<'a, f64, N>),
        state: &'b mut usize,
        _: &Instant,
    ) -> usize {
        if *signal {
            *state += 1;
        }
        *state
    }

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, bool, 0>, ArrayView<'a, f64, N>),
        state: &'b mut usize,
    ) -> usize {
        *state
    }
}

/// Counts an event stream's signals — the probe for event-propagation
/// assertions (a scheduled generation without a pulse does not count).
pub fn count<const N: usize>() -> impl Operator<
    Inputs = (SignalPort<0>, ArrayPort<f64, N>),
    Outputs = Port<Val<usize>>,
    Context = Instant,
> {
    Count
}

/// Operator signature for [`runs`].
pub struct Runs<const N: usize>;

impl<const N: usize> Operator for Runs<N> {
    type Inputs = ArrayPort<f64, N>;
    type Outputs = Port<Val<usize>>;
    type Context = Instant;
    type State = usize;

    fn init(self, _: ArrayView<'_, f64, N>) -> usize {
        0
    }

    fn compute<'a, 'b: 'a>(_: ArrayView<'a, f64, N>, state: &'b mut usize, _: &Instant) -> usize {
        *state += 1;
        *state
    }

    fn reset<'a, 'b: 'a>(_: ArrayView<'a, f64, N>, state: &'b mut usize) -> usize {
        *state
    }
}

/// Counts how many generations scheduled this node's cone — the probe for
/// dirty-cone/recompute assertions on *state* edges (every scheduled
/// generation counts, eventful or not).
pub fn runs<const N: usize>()
-> impl Operator<Inputs = ArrayPort<f64, N>, Outputs = Port<Val<usize>>, Context = Instant> {
    Runs
}

/// Operator signature for [`signal`].
pub struct ManualSignal;

impl Operator for ManualSignal {
    type Inputs = ();
    type Outputs = SignalPort<0>;
    type Context = Instant;
    type State = ();

    fn init(self, _: ()) {}

    fn reset<'a, 'b: 'a>(_: (), _: &'b mut ()) -> ArrayView<'a, bool, 0> {
        ArrayView::scalar(&false)
    }

    fn compute<'a, 'b: 'a>(_: (), _: &'b mut (), _: &Instant) -> ArrayView<'a, bool, 0> {
        ArrayView::scalar(&true)
    }
}

/// A pokeable manual signal: wired as a source, touching its state via
/// `state_mut` marks it dirty, so it pulses (`true`) for exactly the
/// generations the test chooses and resets to `false` in between.
pub fn signal() -> impl Operator<Inputs = (), Outputs = SignalPort<0>, Context = Instant, State = ()>
{
    ManualSignal
}
