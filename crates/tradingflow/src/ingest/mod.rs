//! The event-ingestion driver: an async, timestamp-ordered event loop around
//! a [`Graph`](crate::graph::Graph).
//!
//! [`Scenario`] couples a [`Builder`](crate::graph::Builder) with a [`Queue`] of
//! event feeds; [`Session`] is the built, self-driving graph. Feeds are
//! `futures::Stream`s of [`Event`]s; the queue merges every feed's events in
//! global timestamp order, writes payloads into graph source cells, coalesces
//! events at equal timestamps into one batch, and stabilizes the graph **at
//! most once per timestamp** (batch timestamps are strictly increasing).
//!
//! Timestamps are the TAI [`Instant`](crate::Instant) — the driver is not
//! generic over time. The one generic left is the [`Clock`] gating the merge,
//! defaulted to the real [`WallClock`]; tests substitute simulated clocks for
//! deterministic replays. (the engine crate itself is time-free: it supplies the
//! DAG engine and the graph-level context mechanism, and this module owns
//! everything that knows what a timestamp is.)
//!
//! # Stamps and frontiers
//!
//! Each [`Event`] carries a [`Stamp`] — an explicit timestamp, or an implicit
//! "now" (the wall clock at receipt) — and an optional payload. Stamps must be
//! non-decreasing within a feed. A payload-less event is a pure *frontier
//! advance*: a promise that no later event will be stamped below it, letting
//! an idle feed unblock the merge. A batch at timestamp `t` closes only once
//! every feed's frontier **and** the wall clock have moved strictly past `t`,
//! so an implicit "now" straggler still joins its batch. Consequently, avoid
//! explicit stamps far in the future: the merge holds them (and every feed
//! behind them) until the wall clock actually reaches the stamp.
//!
//! Under these rules the [`WallClock`] behaves uniformly for backtests and
//! live runs: historical timestamps sit below `now` and replay at full speed,
//! while future-dated events are released as their timestamps arrive.
//!
//! # Event time is the graph context
//!
//! The typed graph's context ([`Segment::Context`](crate::graph::Segment::Context)) is the
//! event time [`Instant`](crate::Instant) itself: [`Session::step`] sets it to
//! the batch timestamp after the batch's event writes and before its
//! stabilize, so an
//! operator declaring `type Context = Instant` observes the current event
//! time in its `compute` — no clock handle is threaded through construction.
//!
//! Why ambient (a context) rather than an in-band time *node*? A node poked
//! every batch would either mark every time-reader's cone dirty each
//! generation (defeating sparse stabilization), or change value without
//! notifying (violating the *no-notify ⟹ unchanged* contract). The context
//! keeps time out of the dependency graph: operators read it only when their
//! own inputs notify. Sparse *triggers* (a periodic rebalance tick) are a
//! different thing and belong in-band, as ordinary event feeds — see
//! [`sources::pulse()`](crate::sources::pulse()).
//!
//! No lock and no `Option` wrapper: context writes go through the graph's
//! `context_mut` (`&mut`, strictly between generations) and reads are the
//! shared `&Instant` during stabilize, so the borrow checker enforces the
//! phase separation. Before the first batch the context holds the floor
//! passed to [`Scenario::new`] — pass a value at or below every event the run
//! can produce (typically [`Instant::MIN`](crate::Instant::MIN)), so the
//! context is non-decreasing across the whole run. An operator that must tell
//! the build call apart has its `is_first_run` flag (`init` in the operator
//! conventions).

mod clock;
mod feed;
mod graph;
mod queue;
mod source;

pub use clock::{Clock, WallClock};
pub use feed::{Event, Feed, LazyFeed, Stamp, StreamFeed};
pub use graph::{Scenario, Session};
pub use queue::Queue;
pub use source::EventSource;
