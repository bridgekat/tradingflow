use futures::stream::Stream;

use crate::data::{Array, ArrayView, Instant, Scalar, Series};
use crate::graph::{Event, Port, Ref, Source};
use crate::ports::{ArrayPort, UnitPort};

#[derive(Clone)]
pub struct ArraySource<T: Scalar, const N: usize> {
    default: Array<T, N>,
    series: Series<T, N>,
}

impl<T: Scalar, const N: usize> ArraySource<T, N> {
    /// Create from timestamp and flat value arrays.
    ///
    /// `values.len()` must equal `timestamps.len() * stride`.
    pub fn new(default: Array<T, N>, series: Series<T, N>) -> Self {
        Self { default, series }
    }
}

impl<T: Scalar, const N: usize> Source for ArraySource<T, N> {
    type Instant = Instant;
    type Payload = Array<T, N>;
    type Outputs = ArrayPort<T, N>;
    type State = Array<T, N>;

    fn size_hint(&self) -> Option<usize> {
        Some(self.series.len())
    }

    fn init(
        self,
    ) -> (
        Array<T, N>,
        impl Stream<Item = Event<Array<T, N>, Instant>> + 'static,
    ) {
        let it = self
            .series
            .into_iter()
            .map(|(ts, payload)| Event::at(payload, ts));
        (self.default, futures::stream::iter(it))
    }

    fn output(state: &mut Array<T, N>) -> (bool, ArrayView<'_, T, N>) {
        (true, state.view())
    }

    fn write(payload: Self::Payload, _: Self::Instant, state: &mut Array<T, N>) -> usize {
        state.assign(payload.view());
        1
    }
}

pub struct IterSource<I, T>
where
    T: Send + Sync + 'static,
    I: Iterator<Item = (Instant, T)> + 'static,
{
    iter: I,
    default: T,
}

impl<I, T> IterSource<I, T>
where
    T: Send + Sync + 'static,
    I: Iterator<Item = (Instant, T)> + 'static,
{
    pub fn new(iter: I, default: T) -> Self {
        Self { iter, default }
    }
}

impl<I, T> Source for IterSource<I, T>
where
    T: Send + Sync + 'static,
    I: Iterator<Item = (Instant, T)> + 'static,
{
    type Instant = Instant;
    type Payload = T;
    type Outputs = Port<Ref<T>>;
    type State = T;

    fn size_hint(&self) -> Option<usize> {
        self.iter.size_hint().1
    }

    fn init(self) -> (T, impl Stream<Item = Event<T, Instant>> + 'static) {
        let it = self.iter.map(|(ts, payload)| Event::at(payload, ts));
        (self.default, futures::stream::iter(it))
    }

    fn output(state: &mut T) -> (bool, &T) {
        (true, state)
    }

    fn write(payload: Self::Payload, _: Self::Instant, state: &mut T) -> usize {
        *state = payload;
        1
    }
}

#[derive(Clone)]
pub struct PulseSource {
    timestamps: Vec<Instant>,
}

impl Source for PulseSource {
    type Instant = Instant;
    type Payload = ();
    type Outputs = UnitPort;
    type State = ();

    fn size_hint(&self) -> Option<usize> {
        Some(self.timestamps.len())
    }

    fn init(self) -> ((), impl Stream<Item = Event<(), Instant>> + 'static) {
        let it = self.timestamps.into_iter().map(|ts| Event::at((), ts));
        ((), futures::stream::iter(it))
    }

    fn output(_: &mut Self::State) -> (bool, ()) {
        (true, ())
    }

    fn write(_: Self::Payload, _: Self::Instant, _: &mut ()) -> usize {
        1
    }
}

pub fn array_source<T: Scalar, const N: usize>(
    default: Array<T, N>,
    series: Series<T, N>,
) -> ArraySource<T, N> {
    ArraySource::new(default, series)
}

pub fn iter_source<I, T>(iter: I, default: T) -> IterSource<I, T>
where
    T: Send + Sync + 'static,
    I: Iterator<Item = (Instant, T)> + 'static,
{
    IterSource::new(iter, default)
}

pub fn vec_source<T: Clone + Default + Send + Sync + 'static>(
    events: Vec<(Instant, T)>,
) -> IterSource<impl Iterator<Item = (Instant, T)>, T> {
    IterSource::new(events.into_iter(), T::default())
}

pub fn pulse(timestamps: Vec<Instant>) -> PulseSource {
    PulseSource { timestamps }
}
