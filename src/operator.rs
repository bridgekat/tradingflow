//! Operator trait — the unit of computation in a TradingFlow DAG.
//!
//! An operator reads from one or more input [`Series`] and writes into an
//! output [`Series`].  The implementing type **is** the mutable state (no
//! separate `State` associated type).
//!
//! The trait uses a GAT for `Inputs<'a>` so that concrete operators can
//! express heterogeneous input types (e.g. `&'a [&'a Series<f64>]`).

/// Core operator trait.
///
/// # Design rationale
///
/// * The implementing struct **is** the state — no separate `init_state()`.
///   Construct the struct with whatever initial state you need.
/// * `compute` returns `bool` (produced output?) rather than `Option<value>`
///   because the output is written directly into a pre-reserved slot in the
///   output [`Series`] for zero-copy operation.
/// * `Sized` bound required for type erasure in [`Scenario`].
pub trait Operator: Sized {
    /// Borrowed view of the operator's inputs.
    ///
    /// Typically `&'a [&'a Series<T>]` for homogeneous inputs.
    type Inputs<'a>
    where
        Self: 'a;

    /// Element type written into the output series.
    type Output: Copy;

    /// Compute one output element from the current state of the inputs.
    ///
    /// `out` points to a pre-reserved slot in the output series (length =
    /// series stride).  Write the result into `out` and return `true` to
    /// publish, or return `false` to suppress output for this timestamp.
    fn compute(
        &mut self,
        timestamp: i64,
        inputs: Self::Inputs<'_>,
        out: &mut [Self::Output],
    ) -> bool;
}
