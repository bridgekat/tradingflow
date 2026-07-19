//! Integration tests for `tradingflow::ingest` — the async timestamp-ordered
//! merge driving a typed computation graph.
//!
//! Feeds are in-memory `Stream`s of [`Event`]s; the clock is a deterministic
//! [`SimClock`] whose `wait_until` jumps simulated time forward (so a single
//! `pollster::block_on` replays a whole Builder without a real timer). One test
//! drives an async channel from a background thread to show real cross-thread
//! wakeups under a lightweight executor.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::thread;

use futures::stream::{self, Stream};
use tradingflow_graph::driver::{Builder, Clock, Event, Queue, Source, StreamFeed};
use tradingflow_graph::pool::Pool;
use tradingflow_graph::typed::{Operator, Port, Ref, Val};

fn pool() -> Pool {
    Pool::new(thread::available_parallelism().unwrap().get())
}

/// A simple timestamp type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Instant(i64);

/// Deterministic wall clock. `wait_until` jumps simulated time just past the
/// target on poll (so an as-fast-as-possible replay), and only if no feed was
/// ready first.
#[derive(Clone)]
struct SimClock(Arc<Mutex<Instant>>);
impl SimClock {
    fn at(t: Instant) -> Self {
        Self(Arc::new(Mutex::new(t)))
    }
}
impl Clock<Instant> for SimClock {
    fn now(&self) -> Instant {
        *self.0.lock().unwrap()
    }
    fn wait_until(&mut self, t: Instant) -> impl Future<Output = ()> {
        Jump(self.0.clone(), t)
    }
}
struct Jump(Arc<Mutex<Instant>>, Instant);
impl Future for Jump {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _: &mut Context) -> Poll<()> {
        let mut g = self.0.lock().unwrap();
        if *g <= self.1 {
            *g = Instant(self.1.0 + 1);
        }
        Poll::Ready(())
    }
}

/// A batch of `i64`s per generation, poked by-reference into a source.
type Batch = Vec<i64>;

/// Explicit feed: sorted `(timestamp, batch)` rows.
fn hist(rows: impl IntoIterator<Item = (i64, Batch)>) -> impl Stream<Item = Event<Batch, Instant>> {
    stream::iter(
        rows.into_iter()
            .map(|(n, payload)| Event::at(payload, Instant(n))),
    )
}

/// Implicit feed: wall-clock-stamped batches, in arrival order.
fn live(batches: impl IntoIterator<Item = Batch>) -> impl Stream<Item = Event<Batch, Instant>> {
    stream::iter(batches.into_iter().map(Event::now))
}

/// Sums each notified delta batch into a running total.
struct Sum;
impl Operator for Sum {
    type Inputs = Port<Ref<Batch>>;
    type Outputs = Port<Val<i64>>;
    type Context = Instant;
    type State = i64;
    fn init(self, _: (bool, &Batch)) -> i64 {
        0
    }
    fn passthrough<'a, 'b: 'a>(_: (bool, &'a Batch), sum: &'b mut i64) -> (bool, i64) {
        (true, *sum)
    }
    fn compute<'a, 'b: 'a>(
        (_, batch): (bool, &'a Batch),
        sum: &'b mut i64,
        _: &Instant,
    ) -> (bool, i64) {
        // The auto-gate only calls `compute` on a notified batch; unchanged
        // generations re-lend the running total through `passthrough`.
        *sum += batch.iter().sum::<i64>();
        (true, *sum)
    }
}

/// A stateless adder over two by-reference `i64` cells.
struct Add;
impl Operator for Add {
    type Inputs = (Port<Ref<i64>>, Port<Ref<i64>>);
    type Outputs = Port<Val<i64>>;
    type Context = Instant;
    type State = ();
    fn init(self, _: ((bool, &i64), (bool, &i64))) {}
    fn passthrough<'a, 'b: 'a>(
        ((na, a), (nb, b)): ((bool, &'a i64), (bool, &'a i64)),
        _: &'b mut (),
    ) -> (bool, i64) {
        (na || nb, a + b)
    }
    fn compute<'a, 'b: 'a>(
        inputs: ((bool, &'a i64), (bool, &'a i64)),
        state: &'b mut (),
        _: &Instant,
    ) -> (bool, i64) {
        Self::passthrough(inputs, state)
    }
}

/// A replay [`Source`]: `(timestamp, value)` rows into an `i64` cell.
#[derive(Clone)]
struct Replay(Vec<(Instant, i64)>);
impl Source for Replay {
    type Instant = Instant;
    type Payload = i64;
    type Outputs = Port<Ref<i64>>;
    type State = i64;

    fn size_hint(&self) -> Option<usize> {
        Some(self.0.len())
    }

    fn init(self) -> (i64, impl Stream<Item = Event<i64, Instant>> + 'static) {
        let default = 0;
        let it = self.0.into_iter().map(|(ts, v)| Event::at(v, ts));
        (default, stream::iter(it))
    }

    fn output(state: &mut i64) -> (bool, &i64) {
        (true, state)
    }

    fn write(payload: i64, _: Instant, state: &mut i64) -> usize {
        *state = payload;
        1
    }
}

/// A test [`Source`] driving a `Batch` cell from an explicit event stream —
/// the `add_source` form of the raw feeds these tests used before
/// `Builder::add_feed` was removed. Each stream item replaces the cell; when a
/// `log` is supplied, `(event time, item)` is also appended, capturing the
/// per-event write order that a downstream sink can't see (same-stamp events
/// fold into one generation, so only the last survives in the cell).
///
/// `init` moves the stream out by `take` — a test source drives a single
/// session, and this also lets it carry a move-only stream such as an mpsc
/// receiver.
struct VecFeed<S> {
    stream: Mutex<Option<S>>,
    log: Option<FeedLog>,
}

/// A shared write-side log of `(event time, item)` rows for a [`VecFeed`].
type FeedLog = Arc<Mutex<Vec<(Instant, Batch)>>>;

impl<S: Stream<Item = Event<Batch, Instant>> + Send + 'static> Source for VecFeed<S> {
    type Instant = Instant;
    type Payload = Batch;
    type Outputs = Port<Ref<Batch>>;
    type State = (Option<Arc<Mutex<Vec<(Instant, Vec<i64>)>>>>, Batch);

    fn init(
        self,
    ) -> (
        Self::State,
        impl Stream<Item = Event<Batch, Instant>> + 'static,
    ) {
        let state = (self.log, Batch::new());
        let stream = self
            .stream
            .lock()
            .unwrap()
            .take()
            .expect("VecFeed drives a single session");
        (state, stream)
    }

    fn output(state: &mut Self::State) -> (bool, &Batch) {
        (true, &state.1)
    }

    fn write(payload: Batch, ts: Instant, state: &mut Self::State) -> usize {
        let (log, state) = state;
        if let Some(log) = &log {
            log.lock().unwrap().push((ts, payload.clone()));
        }
        *state = payload;
        1
    }
}

/// A [`VecFeed`] that only writes its cell.
fn feed<S>(stream: S) -> VecFeed<S> {
    VecFeed {
        stream: Mutex::new(Some(stream)),
        log: None,
    }
}

/// A [`VecFeed`] that also records `(event time, item)` into `log`.
fn feed_logged<S>(stream: S, log: FeedLog) -> VecFeed<S> {
    VecFeed {
        stream: Mutex::new(Some(stream)),
        log: Some(log),
    }
}

/// A time-stamping sink operator: on each notified input, appends
/// `(event time, value)` to a shared log. The batch timestamp is the
/// graph-level context — no clock handle is threaded in, and no unwrapping:
/// the build call renders through `passthrough` (which never sees a time and
/// records nothing), and the auto-gate only calls `compute` on a notified
/// input, so `compute` always has a real timestamp.
struct Recorder(Arc<Mutex<Vec<(Instant, i64)>>>);
impl Operator for Recorder {
    type Inputs = Port<Ref<i64>>;
    type Outputs = ();
    type Context = Instant;
    type State = Arc<Mutex<Vec<(Instant, i64)>>>;

    fn init(self, _: (bool, &i64)) -> Self::State {
        self.0
    }

    fn passthrough<'a, 'b: 'a>(_: (bool, &'a i64), _: &'b mut Self::State) {}

    fn compute<'a, 'b: 'a>((_, x): (bool, &'a i64), state: &'b mut Self::State, time: &Instant) {
        state.lock().unwrap().push((*time, *x));
    }
}

/// One feed replays in order, one generation per timestamp; the sum accumulates.
#[test]
fn single_feed_orders_and_accumulates() {
    let mut b = Builder::new(SimClock::at(Instant(0)));
    let data = b.source(feed(hist([(1, vec![10]), (2, vec![20]), (3, vec![30])])));
    let sum = b.segment(Sum, data);
    let mut d = b.build();

    let mut log = Vec::new();
    pollster::block_on(d.run(&mut pool(), |d, _| log.push(d.view(sum))));
    assert_eq!(log, vec![10, 30, 60]);
}

/// Events from several feeds interleave in global timestamp order.
#[test]
fn merges_feeds_in_timestamp_order() {
    let mut b = Builder::new(SimClock::at(Instant(0)));
    b.source(feed(hist([(1, vec![10]), (3, vec![30])])));
    b.source(feed(hist([(2, vec![20])])));
    let mut d = b.build();

    let mut ts = Vec::new();
    pollster::block_on(d.run(&mut pool(), |_, t| ts.push(t)));
    assert_eq!(ts, vec![Instant(1), Instant(2), Instant(3)]);
}

/// Same-timestamp events across feeds collapse into one stabilize.
#[test]
fn batches_equal_timestamps_across_feeds() {
    let mut b = Builder::new(SimClock::at(Instant(0)));
    let a = b.source(feed(hist([(5, vec![1, 2])])));
    let c = b.source(feed(hist([(5, vec![4])])));
    let sa = b.segment(Sum, a);
    let sc = b.segment(Sum, c);
    let mut d = b.build();

    let mut gens = 0;
    pollster::block_on(d.run(&mut pool(), |_, _| gens += 1));
    // Both feeds' t=5 events fold into a single generation.
    assert_eq!(gens, 1);
    assert_eq!(d.view(sa) + d.view(sc), 7);
}

/// A future-dated explicit event is held until the wall clock moves strictly
/// past it; the clock advances to just past each event's timestamp.
#[test]
fn future_events_gated_by_wall_clock() {
    let clock = SimClock::at(Instant(0));
    let mut b = Builder::new(clock.clone());
    b.source(feed(hist([(5, vec![50]), (10, vec![100])])));
    let mut d = b.build();

    let mut nows = Vec::new();
    pollster::block_on(d.run(&mut pool(), |_, _| nows.push(clock.now())));
    assert_eq!(nows, vec![Instant(6), Instant(11)]);
}

/// An implicit feed's events are stamped with the wall clock and merge by that
/// stamp against an explicit feed's earlier event.
#[test]
fn implicit_events_stamped_with_wall_clock() {
    let mut b = Builder::new(SimClock::at(Instant(7)));
    b.source(feed(hist([(3, vec![30])])));
    b.source(feed(live([vec![70]])));
    let mut d = b.build();

    let mut ts = Vec::new();
    pollster::block_on(d.run(&mut pool(), |_, t| ts.push(t)));
    // Explicit t=3 first, then the implicit event stamped now=7.
    assert_eq!(ts, vec![Instant(3), Instant(7)]);
}

/// One feed may emit several events at the same stamp (only non-decreasing is
/// required); they land in one generation, applied in arrival order.
#[test]
fn same_stamp_run_within_one_feed() {
    let mut b = Builder::new(SimClock::at(Instant(0)));
    let log = Arc::new(Mutex::new(Vec::new()));
    b.source(feed_logged(
        hist([(5, vec![1]), (5, vec![2]), (6, vec![3])]),
        log.clone(),
    ));
    let mut d = b.build();

    let mut gens = 0;
    pollster::block_on(d.run(&mut pool(), |_, _| gens += 1));
    assert_eq!(gens, 2);
    assert_eq!(
        *log.lock().unwrap(),
        vec![
            (Instant(5), vec![1]),
            (Instant(5), vec![2]),
            (Instant(6), vec![3])
        ]
    );
}

/// Implicit events across feeds stamped within the same clock reading fold into
/// one generation: the batch closes only once the clock moves past the stamp,
/// so a same-stamp straggler still joins.
#[test]
fn same_stamp_implicit_events_batch() {
    let mut b = Builder::new(SimClock::at(Instant(7)));
    let a = b.source(feed(live([vec![1, 2]])));
    let c = b.source(feed(live([vec![4]])));
    let sa = b.segment(Sum, a);
    let sc = b.segment(Sum, c);
    let mut d = b.build();

    let mut gens = 0;
    pollster::block_on(d.run(&mut pool(), |_, _| gens += 1));
    assert_eq!(gens, 1);
    assert_eq!(d.view(sa) + d.view(sc), 7);
}

/// A payload-less message lets an idle feed unblock the merge without data:
/// feed A announces it has moved to 4 (bounds are inclusive, so anything
/// strictly above 2 closes B's t=2 batch) before ever producing its t=5 event.
#[test]
fn frontier_message_advances_without_data() {
    let mut b = Builder::new(SimClock::at(Instant(100)));
    b.source(feed(stream::iter([
        Event::frontier(Instant(4)),
        Event::at(vec![50], Instant(5)),
    ])));
    b.source(feed(hist([(2, vec![20])])));
    let mut d = b.build();

    let mut ts = Vec::new();
    pollster::block_on(d.run(&mut pool(), |_, t| ts.push(t)));
    assert_eq!(ts, vec![Instant(2), Instant(5)]);
}

/// The heap-based merge keeps global order across many feeds. Feed `i` emits one
/// event at timestamp `i`, added out of order, so the merge must step 1..=N.
#[test]
fn many_feeds_merge_in_order() {
    const N: i64 = 200;
    let mut b = Builder::new(SimClock::at(Instant(N + 1)));

    // Register some feeds at duplicated timestamps to exercise cross-feed
    // batching: each duplicated pair folds into that timestamp's single
    // generation.
    for &i in [7i64, 3, 199, 42, 0]
        .iter()
        .chain((0..N).collect::<Vec<_>>().iter())
    {
        b.source(feed(hist([(i + 1, vec![i])])));
    }
    let mut d = b.build();

    let mut ts = Vec::new();
    pollster::block_on(d.run(&mut pool(), |_, t| ts.push(t)));
    assert_eq!(ts, (1..=N).map(Instant).collect::<Vec<_>>());
}

/// The merge is graph-independent: a [`Queue`] writes into any sink
/// (here a plain `Vec`) and yields each completed batch's timestamp; the
/// caller owns the per-batch action.
#[test]
fn queue_merges_into_plain_sink() {
    let mut q = Queue::new(SimClock::at(Instant(100)));
    q.add_feed(StreamFeed::new(
        hist([(1, vec![10]), (3, vec![30])]),
        |x: Batch, t, sink: &mut Vec<(Instant, i64)>| sink.push((t, x[0])),
    ));
    q.add_feed(StreamFeed::new(
        hist([(2, vec![20])]),
        |x: Batch, t, sink: &mut Vec<(Instant, i64)>| sink.push((t, x[0])),
    ));

    let mut sink = Vec::new();
    let mut batches = Vec::new();
    pollster::block_on(async {
        while let Some(t) = q.step(&mut sink).await {
            batches.push(t);
        }
    });
    assert_eq!(
        sink,
        vec![(Instant(1), 10), (Instant(2), 20), (Instant(3), 30)]
    );
    assert_eq!(batches, vec![Instant(1), Instant(2), Instant(3)]);
}

/// A session with no feeds is immediately done.
#[test]
fn empty_graph_is_done() {
    let mut d = Builder::new(SimClock::at(Instant(0))).build();
    assert_eq!(pollster::block_on(d.step()), None);
}

/// Feeds backed by async channels driven from background threads merge in order
/// under a lightweight executor -- the async-producer pattern, with real
/// cross-thread wakeups. The clock never gates (now = max), so order is pure
/// event-time: the graph blocks on the slower feed's next row before advancing.
#[test]
fn async_channel_feeds_merge_in_order() {
    use futures::channel::mpsc;
    use std::time::Duration;

    let (mut tx_a, rx_a) = mpsc::unbounded();
    let (mut tx_c, rx_c) = mpsc::unbounded();
    let pa = thread::spawn(move || {
        for (n, v) in [(1i64, 10i64), (3, 30)] {
            thread::sleep(Duration::from_millis(5));
            tx_a.start_send(Event::at(vec![v], Instant(n))).unwrap();
        }
    });
    let pc = thread::spawn(move || {
        for (n, v) in [(2i64, 20i64), (4, 40)] {
            tx_c.start_send(Event::at(vec![v], Instant(n))).unwrap();
        }
    });

    let mut b = Builder::new(SimClock::at(Instant(i64::MAX)));
    b.source(feed(rx_a));
    b.source(feed(rx_c));
    let mut d = b.build();

    let mut ts = Vec::new();
    pollster::block_on(d.run(&mut pool(), |_, t| ts.push(t)));
    pa.join().unwrap();
    pc.join().unwrap();
    assert_eq!(ts, vec![Instant(1), Instant(2), Instant(3), Instant(4)]);
}

/// `Builder` wires sources and operators in one place: events merge in
/// timestamp order (an un-fired input keeps its stale value), the graph
/// counts logical events, and per-source estimates accumulate.
#[test]
fn builder_add_source_merges_and_counts() {
    let mut b = Builder::new(SimClock::at(Instant(0)));
    let x = b.source(Replay(vec![(Instant(1), 10), (Instant(3), 30)]));
    let y = b.source(Replay(vec![(Instant(2), 20), (Instant(3), 40)]));
    let sum = b.segment(Add, (x, y));
    let mut d = b.build();

    assert_eq!(d.size_hint(), Some(4));
    let mut log = Vec::new();
    pollster::block_on(d.run(&mut pool(), |d, t| {
        log.push((t, d.view(sum), d.num_events()))
    }));
    assert_eq!(
        log,
        vec![
            (Instant(1), 10, 1),
            (Instant(2), 30, 2),
            (Instant(3), 70, 4)
        ]
    );
    assert_eq!(*d.view(x), 30);
}

/// The running session advances the event-time context between a batch's
/// writes and its stabilize, so time-stamping segments observe the batch
/// timestamp — with no clock handle threaded through the builder.
#[test]
fn event_time_context_stamps_batches() {
    let mut b = Builder::new(SimClock::at(Instant(0)));
    let x = b.source(Replay(vec![(Instant(5), 50), (Instant(9), 90)]));
    let rows = Arc::new(Mutex::new(Vec::new()));
    b.segment(Recorder(rows.clone()), x);
    let mut d = b.build();

    let mut last = None;
    pollster::block_on(d.run(&mut pool(), |_, t| last = Some(t)));
    assert_eq!(
        *rows.lock().unwrap(),
        vec![(Instant(5), 50), (Instant(9), 90)]
    );
    assert_eq!(last, Some(Instant(9)));
}
