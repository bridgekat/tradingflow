use futures::stream::Stream;

use crate::data::{Array, Instant, Scalar, Series};
use crate::graph::{Event, Ref, Source, Val};
use crate::ports::ArrayPass;

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
    type Pass = ArrayPass<T, N>;

    fn size_hint(&self) -> Option<usize> {
        Some(self.series.len())
    }

    fn init(
        self,
    ) -> (
        Array<T, N>,
        impl Stream<Item = Event<Instant, Array<T, N>>> + Send + 'static,
        impl FnMut(Instant, Array<T, N>, &mut Array<T, N>) -> usize + Send + 'static,
    ) {
        let it = self
            .series
            .into_iter()
            .map(|(ts, payload)| Event::at(ts, payload));
        let writer = |_, payload: Array<T, N>, output: &mut Array<T, N>| {
            output.assign(payload.view());
            1
        };
        (self.default, futures::stream::iter(it), writer)
    }
}

pub struct IterSource<I, T>
where
    T: Send + Sync + 'static,
    I: Iterator<Item = (Instant, T)> + Send + 'static,
{
    iter: I,
    default: T,
}

impl<I, T> IterSource<I, T>
where
    T: Send + Sync + 'static,
    I: Iterator<Item = (Instant, T)> + Send + 'static,
{
    pub fn new(iter: I, default: T) -> Self {
        Self { iter, default }
    }
}

impl<I, T> Source for IterSource<I, T>
where
    T: Send + Sync + 'static,
    I: Iterator<Item = (Instant, T)> + Send + 'static,
{
    type Instant = Instant;
    type Payload = T;
    type Pass = Ref<T>;

    fn size_hint(&self) -> Option<usize> {
        self.iter.size_hint().1
    }

    fn init(
        self,
    ) -> (
        T,
        impl Stream<Item = Event<Instant, T>> + Send + 'static,
        impl FnMut(Instant, T, &mut T) -> usize + Send + 'static,
    ) {
        let it = self.iter.map(|(ts, payload)| Event::at(ts, payload));
        let writer = |_, payload, output: &mut T| {
            *output = payload;
            1
        };
        (self.default, futures::stream::iter(it), writer)
    }
}

#[derive(Clone)]
pub struct PulseSource {
    timestamps: Vec<Instant>,
}

impl Source for PulseSource {
    type Instant = Instant;
    type Payload = ();
    type Pass = Val<()>;

    fn size_hint(&self) -> Option<usize> {
        Some(self.timestamps.len())
    }

    fn init(
        self,
    ) -> (
        (),
        impl Stream<Item = Event<Instant, ()>> + Send + 'static,
        impl FnMut(Instant, (), &mut ()) -> usize + Send + 'static,
    ) {
        let it = self.timestamps.into_iter().map(|ts| Event::at(ts, ()));
        let writer = |_, _payload, _output: &mut ()| 1;
        ((), futures::stream::iter(it), writer)
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
    I: Iterator<Item = (Instant, T)> + Send + 'static,
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
