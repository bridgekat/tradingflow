use crate::data::{Array, ArrayView, Instant, Scalar, Series, SeriesView};
use crate::graph::{Port, Ref, Segment, Val};
use crate::ports::{ArrayPort, SeriesPort};

pub struct ConstVal<T: Copy + Send + Sync + 'static> {
    value: T,
}

impl<T: Copy + Send + Sync + 'static> ConstVal<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Copy + Send + Sync + 'static> Segment for ConstVal<T> {
    type Inputs = ();
    type Outputs = Port<Val<T>>;
    type Context = Instant;
    type State = T;

    fn init(self, _: ()) -> Self::State {
        self.value
    }

    fn output<'a, 'b: 'a>(_: (), state: &'b mut T) -> (bool, T) {
        (true, *state)
    }

    fn compute<'a, 'b: 'a>(_: (), state: &'b mut T, _: &Instant) -> (bool, T) {
        (true, *state)
    }
}

pub struct ConstRef<T: Send + Sync + 'static> {
    value: T,
}

impl<T: Send + Sync + 'static> ConstRef<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Send + Sync + 'static> Segment for ConstRef<T> {
    type Inputs = ();
    type Outputs = Port<Ref<T>>;
    type Context = Instant;
    type State = T;

    fn init(self, _: ()) -> Self::State {
        self.value
    }

    fn output<'a, 'b: 'a>(_: (), state: &'b mut T) -> (bool, &'a T) {
        (true, state)
    }

    fn compute<'a, 'b: 'a>(_: (), state: &'b mut T, _: &Instant) -> (bool, &'a T) {
        (true, state)
    }
}

pub struct ConstArray<T: Scalar, const N: usize> {
    value: Array<T, N>,
}

impl<T: Scalar, const N: usize> ConstArray<T, N> {
    pub fn new(value: Array<T, N>) -> Self {
        Self { value }
    }
}

impl<T: Scalar, const N: usize> Segment for ConstArray<T, N> {
    type Inputs = ();
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self, _: ()) -> Self::State {
        self.value
    }

    fn output<'a, 'b: 'a>(_: (), state: &'b mut Array<T, N>) -> (bool, ArrayView<'a, T, N>) {
        (true, state.view())
    }

    fn compute<'a, 'b: 'a>(
        _: (),
        state: &'b mut Array<T, N>,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        (true, state.view())
    }
}

pub struct ConstSeries<T: Scalar, const N: usize> {
    value: Series<T, N>,
}

impl<T: Scalar, const N: usize> ConstSeries<T, N> {
    pub fn new(value: Series<T, N>) -> Self {
        Self { value }
    }
}

impl<T: Scalar, const N: usize> Segment for ConstSeries<T, N> {
    type Inputs = ();
    type Outputs = SeriesPort<T, N>;
    type Context = Instant;
    type State = Series<T, N>;

    fn init(self, _: ()) -> Self::State {
        self.value
    }

    fn output<'a, 'b: 'a>(_: (), state: &'b mut Series<T, N>) -> (bool, SeriesView<'a, T, N>) {
        (true, state.view())
    }

    fn compute<'a, 'b: 'a>(
        _: (),
        state: &'b mut Series<T, N>,
        _: &Instant,
    ) -> (bool, SeriesView<'a, T, N>) {
        (true, state.view())
    }
}

/// A constant value cell.
pub fn const_val<T: Copy + Send + Sync + 'static>(value: T) -> ConstVal<T> {
    ConstVal::new(value)
}

/// A constant reference cell.
pub fn const_ref<T: Send + Sync + 'static>(value: T) -> ConstRef<T> {
    ConstRef::new(value)
}

/// A constant [`Array`] cell.
pub fn const_array<T: Scalar, const N: usize>(value: Array<T, N>) -> ConstArray<T, N> {
    ConstArray::new(value)
}

/// A constant [`Series`] cell.
pub fn const_series<T: Scalar, const N: usize>(value: Series<T, N>) -> ConstSeries<T, N> {
    ConstSeries::new(value)
}
