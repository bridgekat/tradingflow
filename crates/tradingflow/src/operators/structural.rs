//! Structural operators — everything that routes, gates, records or reshapes
//! the stream rather than computing over its values.
//!
//! * **Passthrough / conversion**: [`Where`], [`Cast`].
//! * **Gating**: [`Filter`] (whole-array cutoff) and [`Gate`] (the carry-safe
//!   view gate), plus the clock-driven [`Clocked`] / [`ResampleView`] /
//!   [`ResampleClocked`].
//! * **Recording**: [`Record`] appends an `Array` stream into a `Series`,
//!   stamping each row with event time — the bridge from the [`ArrayPort`] to
//!   the [`SeriesPort`] currency.
//! * **Reshape / combine**: [`Stack`] / [`StackSync`] (N → 1 along a **new**
//!   axis, `OUT == IN + 1`), [`Concat`] / [`ConcatSync`] (N → 1 along an
//!   **existing** axis, rank-preserving), and [`Split`] (1 → N row fan-out,
//!   `OUT == IN - 1`).
//!
//! In the view currency every multi-input combine takes `ArrayPorts<T, IN>`
//! (a contiguous slice of by-value strided views, wired straight from a slice
//! of independent [`ArrayPort`] handles), so the old owned/view operator split
//! has collapsed into a single set of operators and no value↔reference
//! bridging exists anywhere. The combine into the output cross-section is
//! the irreducible panel→cross-section data movement (each input materialized
//! via `to_contiguous`); the per-stock selections upstream are
//! [`select`](super::transform::select)s.

use std::marker::PhantomData;

use bumpalo::Bump;
use num_traits::{AsPrimitive, Float};

use crate::data::layout::Strided;
use crate::data::{Array, ArrayView, Instant, Layout, Retention, Scalar, Series, SeriesView};
use crate::graph::typed::{Interface, Operator, Segment};
use crate::ports::{ArrayPort, ArrayPorts, SeriesPort, UnitPort};

// ===========================================================================
// Passthrough / conversion — Where, Cast.
// ===========================================================================

/// Element-wise conditional: keep the value where `condition` holds, else
/// replace with `fill`.
#[derive(Clone)]
pub struct Where<T: Scalar, F: Fn(T) -> bool + Clone, const N: usize> {
    condition: F,
    fill: T,
    _phantom: PhantomData<T>,
}

impl<T: Scalar, F: Fn(T) -> bool + Clone, const N: usize> Where<T, F, N> {
    pub fn new(condition: F, fill: T) -> Self {
        Self {
            condition,
            fill,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`Where`]: the predicate and fill plus the output buffer.
pub struct WhereState<T: Scalar, F, const N: usize> {
    condition: F,
    fill: T,
    out: Array<T, N>,
}

impl<T: Scalar, F: Fn(T) -> bool + Clone + Send + Sync + 'static, const N: usize> Operator
    for Where<T, F, N>
{
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = WhereState<T, F, N>;

    fn init(self) -> Self::State {
        WhereState {
            condition: self.condition,
            fill: self.fill,
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            state.out = Array::zeros(x.extents());
        }
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let out = state.out.data_mut();
        for i in 0..out.len() {
            out[i] = if (state.condition)(src[i].clone()) {
                src[i].clone()
            } else {
                state.fill.clone()
            };
        }
        (!init, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// Element-wise type conversion `ArrayView<S, N> → Array<T, N>` via
/// `AsPrimitive`.
#[derive(Clone)]
pub struct Cast<S: Scalar, T: Scalar, const N: usize> {
    _phantom: PhantomData<(S, T)>,
}

impl<S: Scalar, T: Scalar, const N: usize> Cast<S, T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<S: Scalar, T: Scalar, const N: usize> Default for Cast<S, T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S, T, const N: usize> Operator for Cast<S, T, N>
where
    S: Scalar + Copy + AsPrimitive<T>,
    T: Scalar + Copy + 'static,
{
    type Inputs = ArrayPort<S, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self) -> Self::State {
        Array::zeros([0; N])
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, S, N>),
        _: &Instant,
        out: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            *out = Array::zeros(x.extents());
        }
        let xs = x.to_contiguous();
        let src: &[S] = &xs;
        let dst = out.data_mut();
        for i in 0..dst.len() {
            dst[i] = src[i].as_();
        }
        (!init, out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, S, N>),
        _: &Instant,
        out: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }
}

/// State shared by the view-currency resamplers: the last data view
/// materialized into an owned buffer, so it survives between clock ticks while
/// the upstream view's storage may change.
pub struct ResampleViewState<T: Scalar, const N: usize> {
    out: Option<Array<T, N>>,
}

/// Clock-gated **view** passthrough whose clock is another data view (only the
/// clock's notify bit is consulted): re-emits the rank-`N` data view on every
/// tick of the rank-1 clock view, holding the last value in between — e.g.
/// resample a feature panel onto the daily-close pulse. Stays in the
/// [`ArrayView`] currency end-to-end.
#[derive(Clone)]
pub struct ResampleView<T: Scalar, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar, const N: usize> ResampleView<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Default for ResampleView<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

// A `Segment`, not an `Operator`: the gate ignores the data input's notify bit
// (only the clock's fires it), which the `Operator` any-notify gate cannot
// express.
impl<T: Scalar, const N: usize> Segment for ResampleView<T, N> {
    type Inputs = (ArrayPort<T, 1>, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = ResampleViewState<T, N>;

    fn init(self) -> Self::State {
        ResampleViewState { out: None }
    }

    fn compute<'a, 'b: 'a>(
        ((clock_fired, _), (_, x)): ((bool, ArrayView<'a, T, 1>), (bool, ArrayView<'a, T, N>)),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init || clock_fired {
            state.out = Some(x.to_array());
            return (!init, state.out.as_ref().unwrap().view());
        }
        (false, state.out.as_ref().unwrap().view())
    }
}

/// Clock-gated **view** passthrough whose clock is a unit (`RefPort<()>`) clock
/// source (e.g. a rebalance [`pulse`](crate::sources::basic::pulse): re-emits
/// the rank-`N` data view on every clock tick. The unit-clock counterpart of
/// [`ResampleView`].
#[derive(Clone)]
pub struct ResampleClocked<T: Scalar, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar, const N: usize> ResampleClocked<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Default for ResampleClocked<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar, const N: usize> Segment for ResampleClocked<T, N> {
    type Inputs = (UnitPort, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = ResampleViewState<T, N>;

    fn init(self) -> Self::State {
        ResampleViewState { out: None }
    }

    fn compute<'a, 'b: 'a>(
        ((clock_fired, _), (_, x)): ((bool, ()), (bool, ArrayView<'a, T, N>)),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init || clock_fired {
            state.out = Some(x.to_array());
            return (!init, state.out.as_ref().unwrap().view());
        }
        (false, state.out.as_ref().unwrap().view())
    }
}

// ===========================================================================
// Constructors
// ===========================================================================

/// Element-wise scalar cast `S -> T`.
pub fn cast<S: Scalar, T: Scalar, const N: usize>() -> Cast<S, T, N> {
    Cast::new()
}

/// Element-wise conditional: keep the value where `condition` holds, else
/// replace it with `fill`. (Named `keep_where` because `where` is a keyword.)
pub fn keep_where<T: Scalar, F: Fn(T) -> bool + Clone, const N: usize>(
    condition: F,
    fill: T,
) -> Where<T, F, N> {
    Where::new(condition, fill)
}

/// Re-emit an array view on every tick of a leading *view* pulse.
pub fn resample_view<T: Scalar, const N: usize>() -> ResampleView<T, N> {
    ResampleView::new()
}

/// Re-emit an array view on every tick of a leading *clock* pulse.
pub fn resample_clocked<T: Scalar, const N: usize>() -> ResampleClocked<T, N> {
    ResampleClocked::new()
}

// ===========================================================================
// Gating and recording — Filter, Gate, Record, Clocked, Count, Last.
// ===========================================================================

// ---------------------------------------------------------------------------
// Filter — whole-array gate by predicate (the cutoff operator).
// ---------------------------------------------------------------------------

/// Passes the input through when the predicate holds, else drops it (emits
/// `notify = false` → downstream gated off, previous value retained).
pub struct Filter<T: Scalar, F, const N: usize>(pub F, pub PhantomData<T>);

/// Runtime state for [`Filter`]: the predicate plus the retained output.
pub struct FilterState<T: Scalar, F, const N: usize> {
    predicate: F,
    out: Array<T, N>,
}

impl<T: Scalar, F, const N: usize> Operator for Filter<T, F, N>
where
    F: for<'x> Fn(ArrayView<'x, T, N>) -> bool + Send + Sync + 'static,
{
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = FilterState<T, F, N>;

    fn init(self) -> Self::State {
        FilterState {
            predicate: self.0,
            out: Array::zeros([0; N]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            state.out = x.to_array();
            return (false, state.out.view());
        }
        if (state.predicate)(x) {
            state.out.assign(x);
            (true, state.out.view())
        } else {
            (false, state.out.view())
        }
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Gate — view gate that honours the no-notify⟹unchanged contract.
// ---------------------------------------------------------------------------

/// View gate: emits the input row as a `ViewPort`, notifying iff the input
/// notified AND the predicate holds — the row-cutoff that drops the all-NaN "no
/// data" cross-sections a dense panel emits for an idle stock.
///
/// The TradingFlow contract is that **an operator that does not notify must not
/// change its output value** (so any consumer may treat a non-notifying input
/// as its last notified value — the carry that [`Stack`] relies
/// on). A naive forwarder would break it: gating out a *notified* all-NaN row
/// while forwarding that row changes the value under `notify = false`. So
/// `Gate` retains the last passed row in owned state and re-presents a view of
/// it whenever it gates out or its input is silent. The retained buffer is
/// overwritten **in place** (no realloc) only on a pass — i.e. only when `Gate`
/// notifies — so a view stored by an out-of-cone consumer always reads the
/// frozen last-passed value. This makes `Gate`'s output a stable backing for
/// downstream zero-copy view chains.
pub struct Gate<T: Scalar, F, const N: usize>(pub F, pub PhantomData<T>);

/// Runtime state for [`Gate`]: the predicate plus the retained last-passed row,
/// which the `ViewPort` output borrows.
pub struct GateState<T: Scalar, F, const N: usize> {
    predicate: F,
    out: Array<T, N>,
}

impl<T: Scalar, F, const N: usize> Operator for Gate<T, F, N>
where
    F: for<'x> Fn(ArrayView<'x, T, N>) -> bool + Send + Sync + 'static,
{
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = GateState<T, F, N>;

    fn init(self) -> Self::State {
        GateState {
            predicate: self.0,
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (notified, view): (bool, ArrayView<'a, T, N>),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            // Seed the retained buffer with the faithful build-time row (so the
            // first view matches what `Split` lends), but do not notify.
            state.out = view.to_array();
            return (false, state.out.view());
        }
        if notified && (state.predicate)(view) {
            // Pass: refresh the retained row in place (no realloc) and notify.
            state.out.assign(view);
            (true, state.out.view())
        } else {
            // Gate out (or upstream silent): re-present the unchanged retained
            // row under `notify = false` — the contract.
            (false, state.out.view())
        }
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Record — append an Array stream into a Series, stamping with event time.
// ---------------------------------------------------------------------------

/// Records an `Array<T, N>` stream into a `Series<T, N>`, stamping each row with
/// the event time — the graph context the driver sets before each `stabilize`.
/// The only native operator that reads time (the Python host is the other, behind
/// the `python` feature), and it needs nothing at construction:
/// [`record`] / [`record_bounded`] take
/// no clock.
///
/// An optional [`Retention`] bound (via [`with_retention`](Self::with_retention)
/// / [`record_bounded`]) caps the recorded history.
pub struct Record<T: Scalar, const N: usize> {
    retention: Retention,
    _p: PhantomData<T>,
}

impl<T: Scalar, const N: usize> Record<T, N> {
    /// An unbounded record (retains full history).
    pub fn new() -> Self {
        Self::with_retention(Retention::unbounded())
    }

    /// A record whose `Series` keeps only the history within `retention`.
    pub fn with_retention(retention: Retention) -> Self {
        Self {
            retention,
            _p: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Default for Record<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Record`]: the retention bound plus the recorded series.
pub struct RecordState<T: Scalar, const N: usize> {
    retention: Retention,
    out: Series<T, N>,
}

impl<T: Scalar, const N: usize> Operator for Record<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = SeriesPort<T, N>;
    type Context = Instant;
    type State = RecordState<T, N>;

    fn init(self) -> Self::State {
        RecordState {
            retention: self.retention,
            out: Series::new_unbounded([0; N]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        time: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, SeriesView<'a, T, N>) {
        if init {
            // The build call only sizes the series — no row is appended, so the
            // pre-first-batch context value is never stamped into it.
            state.out = Series::new(x.extents(), state.retention);
            return (false, state.out.view());
        }
        state.out.push(*time, x);
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, SeriesView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Last — most recent element of a Series as an Array.
// ---------------------------------------------------------------------------

/// Extracts the most recent element of a `Series<T, N>` as a rank-`N`
/// [`ArrayView`], substituting `fill` when the series is empty.
pub struct Last<T: Scalar, const N: usize> {
    fill: T,
}

impl<T: Scalar, const N: usize> Last<T, N> {
    pub fn new(fill: T) -> Self {
        Self { fill }
    }
}

/// Runtime state for [`Last`]: the fill value plus the output buffer.
pub struct LastState<T: Scalar, const N: usize> {
    fill: T,
    out: Array<T, N>,
}

impl<T: Scalar, const N: usize> Operator for Last<T, N> {
    type Inputs = SeriesPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = LastState<T, N>;

    fn init(self) -> Self::State {
        LastState {
            fill: self.fill.clone(),
            out: Array::zeros([0; N]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, series): (bool, SeriesView<'a, T, N>),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            state.out = Array::full(series.layout().extents(), state.fill.clone());
        }
        if series.is_empty() {
            for v in state.out.data_mut().iter_mut() {
                *v = state.fill.clone();
            }
        } else {
            state.out.assign(series.at(series.len() - 1).unwrap().1);
        }
        (!init, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, SeriesView<'a, T, N>),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Count — stateful per-tick counter (anti-corruption demonstrator).
// ---------------------------------------------------------------------------

/// Increments a counter every time it runs and emits the running count (a
/// rank-0 scalar). Used to prove gating advances state only when an input
/// actually notifies.
pub struct Count<const N: usize>;

/// Runtime state for [`Count`]: the counter plus the scalar output buffer.
pub struct CountState {
    count: i64,
    out: Array<f64, 0>,
}

impl<const N: usize> Operator for Count<N> {
    type Inputs = ArrayPort<f64, N>;
    type Outputs = ArrayPort<f64, 0>;
    type Context = Instant;
    type State = CountState;

    fn init(self) -> Self::State {
        CountState {
            count: 0,
            out: Array::scalar(0.0),
        }
    }

    fn compute<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, f64, N>),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, f64, 0>) {
        if init {
            return (false, state.out.view());
        }
        state.count += 1;
        state.out.data_mut()[0] = state.count as f64;
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, f64, N>),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, f64, 0>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Clocked — clock-gated wrapper.
// ---------------------------------------------------------------------------

/// Prepends a leading `UnitPort` clock input; runs the inner operator's compute
/// path only when the clock notifies, else the inner passthrough. Implements
/// [`Segment`] directly because its gate ignores the data inputs' notify bits.
#[derive(Debug, Clone)]
pub struct Clocked<O> {
    inner: O,
}

impl<O> Clocked<O> {
    pub fn new(inner: O) -> Self {
        Self { inner }
    }
}

impl<O: Operator> Segment for Clocked<O> {
    type Inputs = (UnitPort, O::Inputs);
    type Outputs = O::Outputs;
    // Forwarded, not pinned: `Clocked` is a gate, so it stays as
    // context-agnostic as whatever it wraps.
    type Context = O::Context;
    type State = O::State;

    fn init(self) -> O::State {
        O::init(self.inner)
    }

    fn compute<'a, 'b: 'a>(
        ((clock_fired, _), rest): ((bool, ()), <O::Inputs as Interface>::Values<'a>),
        context: &O::Context,
        state: &'b mut O::State,
        init: bool,
    ) -> <O::Outputs as Interface>::Values<'a> {
        if init || clock_fired {
            O::compute(rest, context, state, init)
        } else {
            O::passthrough(rest, context, &*state)
        }
    }
}

// ===========================================================================
// Constructors
// ===========================================================================

/// Pass the input through iff `predicate` holds, else drop the tick (emitting
/// `notify = false`). The cutoff operator: a dropped tick suppresses every
/// downstream side effect, including a [`Record`] append.
pub fn filter<T: Scalar, F, const N: usize>(predicate: F) -> Filter<T, F, N> {
    Filter(predicate, PhantomData)
}

/// Like [`filter`], but re-presents the last passed row as a stable
/// [`ArrayPort`] view (the carry-safe view gate).
pub fn gate<T: Scalar, F, const N: usize>(predicate: F) -> Gate<T, F, N> {
    Gate(predicate, PhantomData)
}

/// An unbounded [`Record`] of the input stream: `record() @ x` appends
/// every notified value of `x`, stamped with event time. Prefer
/// [`record_bounded`] whenever the consumers' look-back is known.
pub fn record<T: Scalar, const N: usize>() -> Record<T, N> {
    Record::new()
}

/// A [`Record`] keeping only the history within `retention` — the hoisted
/// shared-record form: record once, feed many windowed consumers. Size
/// `retention` to the deepest consumer look-back plus a compaction margin (see
/// the module docs).
pub fn record_bounded<T: Scalar, const N: usize>(retention: Retention) -> Record<T, N> {
    Record::with_retention(retention)
}

/// A private record sized for a count look-back of `n`.
pub fn buffer<T: Scalar, const N: usize>(n: usize) -> Record<T, N> {
    // Extra rows a private record retains beyond its consumer's exact count
    // look-back — absorbs the amortized-compaction slack plus the one-row
    // overshoot a sliding window reads while evicting.
    const COUNT_MARGIN: usize = 8;
    Record::with_retention(Retention::count(n + COUNT_MARGIN))
}

/// The most recent element of a [`Series`] as an array view, `fill` when empty.
pub fn last<T: Scalar, const N: usize>(fill: T) -> Last<T, N> {
    Last::new(fill)
}

/// Count the notified ticks seen so far.
pub fn count<const N: usize>() -> Count<N> {
    Count
}

/// Prepend a leading clock port to `inner`, running its compute path only when
/// the clock notifies.
pub fn clocked<O>(inner: O) -> Clocked<O> {
    Clocked::new(inner)
}

// ===========================================================================
// Reshape / combine — Stack, Concat, Split.
// ===========================================================================

/// Shared runtime state: the axis config, the outer × chunk layout (sized on the
/// build call), and the output buffer.
pub struct ReshapeState<T: Scalar, const OUT: usize> {
    axis: usize,
    outer_count: usize,
    chunk_size: usize,
    n_inputs: usize,
    out: Array<T, OUT>,
}

/// Interleave `inputs` (each materialized row-major) into `output` along the
/// combine layout.
#[inline(always)]
fn interleaved_copy_views<T: Scalar, const IN: usize>(
    output: &mut [T],
    inputs: &[ArrayView<T, IN>],
    n_inputs: usize,
    outer_count: usize,
    chunk_size: usize,
) {
    let stride = n_inputs * chunk_size;
    for (input_idx, arr) in inputs.iter().enumerate() {
        let src = arr.to_contiguous();
        for outer in 0..outer_count {
            let src_offset = outer * chunk_size;
            let dst_offset = outer * stride + input_idx * chunk_size;
            output[dst_offset..dst_offset + chunk_size]
                .clone_from_slice(&src[src_offset..src_offset + chunk_size]);
        }
    }
}

#[inline(always)]
fn interleaved_copy_views_selective<T: Scalar, const IN: usize>(
    output: &mut [T],
    inputs: &[ArrayView<T, IN>],
    positions: impl IntoIterator<Item = usize>,
    n_inputs: usize,
    outer_count: usize,
    chunk_size: usize,
) {
    let stride = n_inputs * chunk_size;
    for pos in positions {
        let src = inputs[pos].to_contiguous();
        for outer in 0..outer_count {
            let src_offset = outer * chunk_size;
            let dst_offset = outer * stride + pos * chunk_size;
            output[dst_offset..dst_offset + chunk_size]
                .clone_from_slice(&src[src_offset..src_offset + chunk_size]);
        }
    }
}

/// Output extents for a stack-along-new-axis (`OUT == IN + 1`): insert
/// `n_inputs` at `axis`.
fn stack_extents<const IN: usize, const OUT: usize>(
    input_extents: [usize; IN],
    axis: usize,
    n_inputs: usize,
) -> [usize; OUT] {
    let mut v = Vec::with_capacity(IN + 1);
    v.extend_from_slice(&input_extents[..axis]);
    v.push(n_inputs);
    v.extend_from_slice(&input_extents[axis..]);
    <[usize; OUT]>::try_from(v.as_slice())
        .unwrap_or_else(|_| panic!("Stack: OUT ({OUT}) must be IN ({IN}) + 1"))
}

// ---------------------------------------------------------------------------
// Stack / StackSync — new axis.
// ---------------------------------------------------------------------------

/// Stack `N` homogeneous rank-`IN` views along a **new** axis into the owned
/// rank-`OUT` (`= IN + 1`) cross-section. Reads **every** input each generation
/// (the carry join), relying on the no-notify⟹unchanged contract.
#[derive(Clone)]
pub struct Stack<T: Scalar, const IN: usize, const OUT: usize> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar, const IN: usize, const OUT: usize> Stack<T, IN, OUT> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar, const IN: usize, const OUT: usize> Operator for Stack<T, IN, OUT> {
    type Inputs = ArrayPorts<T, IN>;
    type Outputs = ArrayPort<T, OUT>;
    type Context = Instant;
    type State = ReshapeState<T, OUT>;

    fn init(self) -> Self::State {
        ReshapeState {
            axis: self.axis,
            outer_count: 0,
            chunk_size: 0,
            n_inputs: 0,
            out: Array::zeros([0; OUT]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, views): (&'a [bool], &'a [ArrayView<'a, T, IN>]),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        if init {
            assert!(!views.is_empty(), "Stack requires at least one input");
            let first = views[0].extents();
            assert!(self_axis_ok(state.axis, IN, true), "axis out of bounds");
            state.outer_count = first[..state.axis].iter().product();
            state.chunk_size = first[state.axis..].iter().product();
            state.n_inputs = views.len();
            state.out = Array::zeros(stack_extents::<IN, OUT>(first, state.axis, views.len()));
            return (false, state.out.view());
        }
        interleaved_copy_views(
            state.out.data_mut(),
            views,
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [ArrayView<'a, T, IN>]),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        (false, state.out.view())
    }
}

/// Stack `N` float views along a new axis, NaN-filling inputs that did not
/// notify this generation.
#[derive(Clone)]
pub struct StackSync<T: Scalar + Float, const IN: usize, const OUT: usize> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float, const IN: usize, const OUT: usize> StackSync<T, IN, OUT> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar + Float, const IN: usize, const OUT: usize> Operator for StackSync<T, IN, OUT> {
    type Inputs = ArrayPorts<T, IN>;
    type Outputs = ArrayPort<T, OUT>;
    type Context = Instant;
    type State = ReshapeState<T, OUT>;

    fn init(self) -> Self::State {
        ReshapeState {
            axis: self.axis,
            outer_count: 0,
            chunk_size: 0,
            n_inputs: 0,
            out: Array::zeros([0; OUT]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (flags, views): (&'a [bool], &'a [ArrayView<'a, T, IN>]),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        if init {
            assert!(!views.is_empty(), "StackSync requires at least one input");
            let first = views[0].extents();
            assert!(self_axis_ok(state.axis, IN, true), "axis out of bounds");
            state.outer_count = first[..state.axis].iter().product();
            state.chunk_size = first[state.axis..].iter().product();
            state.n_inputs = views.len();
            let mut out = Array::zeros(stack_extents::<IN, OUT>(first, state.axis, views.len()));
            for v in out.data_mut().iter_mut() {
                *v = T::nan();
            }
            state.out = out;
            return (false, state.out.view());
        }
        for v in state.out.data_mut().iter_mut() {
            *v = T::nan();
        }
        interleaved_copy_views_selective(
            state.out.data_mut(),
            views,
            (0..flags.len()).filter(|&i| flags[i]),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [ArrayView<'a, T, IN>]),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, OUT>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Concat / ConcatSync — existing axis (rank-preserving).
// ---------------------------------------------------------------------------

/// Concatenate `N` homogeneous rank-`N` views along an **existing** axis.
#[derive(Clone)]
pub struct Concat<T: Scalar, const N: usize> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar, const N: usize> Concat<T, N> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize> Operator for Concat<T, N> {
    type Inputs = ArrayPorts<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = ReshapeState<T, N>;

    fn init(self) -> Self::State {
        ReshapeState {
            axis: self.axis,
            outer_count: 0,
            chunk_size: 0,
            n_inputs: 0,
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (_, views): (&'a [bool], &'a [ArrayView<'a, T, N>]),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            assert!(!views.is_empty(), "Concat requires at least one input");
            let mut ext = views[0].extents();
            assert!(state.axis < N, "axis out of bounds");
            state.outer_count = ext[..state.axis].iter().product();
            state.chunk_size = ext[state.axis..].iter().product();
            state.n_inputs = views.len();
            ext[state.axis] *= views.len();
            state.out = Array::zeros(ext);
            return (false, state.out.view());
        }
        interleaved_copy_views(
            state.out.data_mut(),
            views,
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [ArrayView<'a, T, N>]),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// Concatenate `N` float views along an existing axis, NaN-filling inputs that
/// did not notify this generation.
#[derive(Clone)]
pub struct ConcatSync<T: Scalar + Float, const N: usize> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> ConcatSync<T, N> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Operator for ConcatSync<T, N> {
    type Inputs = ArrayPorts<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = ReshapeState<T, N>;

    fn init(self) -> Self::State {
        ReshapeState {
            axis: self.axis,
            outer_count: 0,
            chunk_size: 0,
            n_inputs: 0,
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (flags, views): (&'a [bool], &'a [ArrayView<'a, T, N>]),
        _: &Instant,
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            assert!(!views.is_empty(), "ConcatSync requires at least one input");
            let mut ext = views[0].extents();
            assert!(state.axis < N, "axis out of bounds");
            state.outer_count = ext[..state.axis].iter().product();
            state.chunk_size = ext[state.axis..].iter().product();
            state.n_inputs = views.len();
            ext[state.axis] *= views.len();
            let mut out = Array::zeros(ext);
            for v in out.data_mut().iter_mut() {
                *v = T::nan();
            }
            state.out = out;
            return (false, state.out.view());
        }
        for v in state.out.data_mut().iter_mut() {
            *v = T::nan();
        }
        interleaved_copy_views_selective(
            state.out.data_mut(),
            views,
            (0..flags.len()).filter(|&i| flags[i]),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        (true, state.out.view())
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [ArrayView<'a, T, N>]),
        _: &Instant,
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

#[inline(always)]
fn self_axis_ok(axis: usize, rank: usize, allow_equal: bool) -> bool {
    if allow_equal {
        axis <= rank
    } else {
        axis < rank
    }
}

// ---------------------------------------------------------------------------
// Split — zero-copy axis-0 fan-out into view ports (the inverse of `Stack`).
// ---------------------------------------------------------------------------

/// Split a rank-`IN` array along axis 0 into `N` per-row rank-`OUT` (`= IN - 1`)
/// views — the `1 → N` inverse of [`Stack`]. The port count is declared at
/// construction (`axis_size`); the build call asserts the input's axis-0 size
/// matches.
///
/// **Zero-copy**: each output is a strided [`ArrayView`] of the input's row,
/// re-derived from the fresh input every invocation, by value; only the
/// per-generation notify/view *planes* live in the [`Bump`]
/// arena — no row data is copied. All rows notify exactly when the input
/// notifies, and each row handle is an ordinary [`ArrayPort`] producer.
///
/// Implements [`Segment`] directly: views cannot be re-lent through `&State`, so
/// every invocation rebuilds the planes and expresses the gate in the flags.
pub struct Split<T: Scalar, const IN: usize, const OUT: usize> {
    axis_size: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar, const IN: usize, const OUT: usize> Split<T, IN, OUT> {
    /// `axis_size` is the declared input axis-0 size = the output port count.
    pub fn new(axis_size: usize) -> Self {
        assert!(axis_size > 0, "Split requires at least one output port");
        Self {
            axis_size,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Runtime state for [`Split`]: the declared port count and the per-generation
/// arena backing the notify/value planes.
pub struct SplitState {
    axis_size: usize,
    arena: Bump,
}

impl<T: Scalar, const IN: usize, const OUT: usize> Segment for Split<T, IN, OUT> {
    type Inputs = ArrayPort<T, IN>;
    type Outputs = ArrayPorts<T, OUT>;
    type Context = Instant;
    type State = SplitState;

    fn init(self) -> SplitState {
        SplitState {
            axis_size: self.axis_size,
            arena: Bump::new(),
        }
    }

    fn compute<'a, 'b: 'a>(
        (notified, x): (bool, ArrayView<'a, T, IN>),
        _: &Instant,
        state: &'b mut SplitState,
        init: bool,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let n = state.axis_size;
        let data = x.data();
        let layout = x.layout();
        let (ext, strd) = (layout.extents(), layout.strides());
        if init {
            assert!(IN >= 1, "Split requires IN >= 1");
            assert!(OUT == IN - 1, "Split: OUT ({OUT}) must be IN ({IN}) - 1");
            assert!(
                ext[0] == n,
                "Split: input axis-0 size {} != declared {n}",
                ext[0],
            );
        }
        // Each row drops axis 0, keeping the inner axes' extents/strides.
        let mut inner_ext = [0usize; OUT];
        let mut inner_str = [0usize; OUT];
        inner_ext.copy_from_slice(&ext[1..]);
        inner_str.copy_from_slice(&strd[1..]);
        let row_shape = Strided::new(inner_ext, inner_str);
        state.arena.reset();
        let alloc: &'a Bump = &state.arena;
        let flags = alloc.alloc_slice_fill_iter(std::iter::repeat_n(notified && !init, n));
        let views = alloc.alloc_slice_fill_iter(
            (0..n).map(|i| ArrayView::from_parts(row_shape, &data[i * strd[0]..])),
        );
        (&*flags, &*views)
    }
}

// ===========================================================================
// Constructors
// ===========================================================================

/// Stack `N` rank-`IN` inputs along a new `axis`, carrying un-notified inputs
/// forward at their last value.
pub fn stack<T: Scalar, const IN: usize, const OUT: usize>(axis: usize) -> Stack<T, IN, OUT> {
    Stack::new(axis)
}

/// Like [`stack`], but emits `NaN` for inputs that have not notified this tick.
pub fn stack_sync<T: Scalar + Float, const IN: usize, const OUT: usize>(
    axis: usize,
) -> StackSync<T, IN, OUT> {
    StackSync::new(axis)
}

/// Concatenate `N` inputs along an existing `axis` (carry semantics).
pub fn concat<T: Scalar, const N: usize>(axis: usize) -> Concat<T, N> {
    Concat::new(axis)
}

/// Like [`concat`](fn@concat), but emits `NaN` for inputs that have not notified.
pub fn concat_sync<T: Scalar + Float, const N: usize>(axis: usize) -> ConcatSync<T, N> {
    ConcatSync::new(axis)
}

/// Split a rank-`IN` array into `axis_size` rank-`OUT` by-value view rows.
pub fn split<T: Scalar, const IN: usize, const OUT: usize>(axis_size: usize) -> Split<T, IN, OUT> {
    Split::new(axis_size)
}
