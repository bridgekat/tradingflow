//! Combinators and extension methods.

use std::marker::PhantomData;

use super::{Interface, Segment};

/// The identity operator `id`: the categorical identity.
/// Forwards inputs to outputs without modification.
pub struct Id<T, C>(pub PhantomData<(T, C)>);

impl<T, C> Segment for Id<T, C>
where
    T: Interface,
    C: Sync + 'static,
{
    type Inputs = T;
    type Outputs = T;
    type Context = C;
    type State = ();

    fn init(self, _: T::Values<'_>) -> Self::State {}

    fn output<'a, 'b: 'a>(inputs: T::Values<'a>, _: &'b mut Self::State) -> T::Values<'a> {
        inputs
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        _: &'b mut Self::State,
        _: &Self::Context,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        inputs
    }
}

impl<T, C> Default for Id<T, C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Feeds the outputs of `F` as inputs to `G` and produces the outputs of `G`.
pub struct Comp<F, G>(pub F, pub G);

impl<F, G> Segment for Comp<F, G>
where
    F: Segment,
    G: Segment<Inputs = F::Outputs, Context = F::Context>,
{
    type Inputs = F::Inputs;
    type Outputs = G::Outputs;
    type Context = F::Context;
    type State = (F::State, G::State);

    fn init(self, inputs: <F::Inputs as Interface>::Values<'_>) -> Self::State {
        let mut fs = self.0.init(inputs);
        let mid = F::output(<F::Inputs as Interface>::reborrow(inputs), &mut fs);
        let gs = self.1.init(mid);
        (fs, gs)
    }

    fn output<'a, 'b: 'a>(
        inputs: <F::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
    ) -> <G::Outputs as Interface>::Values<'a> {
        G::output(F::output(inputs, &mut state.0), &mut state.1)
    }

    fn compute<'a, 'b: 'a>(
        inputs: <F::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
        context: &Self::Context,
    ) -> <G::Outputs as Interface>::Values<'a> {
        let (fs, gs) = state;
        G::compute(F::compute(inputs, fs, context), gs, context)
    }
}

impl<F, G> Default for Comp<F, G>
where
    F: Segment + Default,
    G: Segment + Default,
{
    fn default() -> Self {
        Self(F::default(), G::default())
    }
}

/// Fan-out `F &&& G`: the canonical arrow for categorical products.
/// Feeds the same input to both branches and pair their outputs.
pub struct Fork<F, G>(pub F, pub G);

impl<F, G> Segment for Fork<F, G>
where
    F: Segment,
    G: Segment<Inputs = F::Inputs, Context = F::Context>,
{
    type Inputs = F::Inputs;
    type Outputs = (F::Outputs, G::Outputs);
    type Context = F::Context;
    type State = (F::State, G::State);

    fn init(self, inputs: <F::Inputs as Interface>::Values<'_>) -> Self::State {
        (self.0.init(inputs), self.1.init(inputs))
    }

    fn output<'a, 'b: 'a>(
        inputs: <F::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        (
            F::output(inputs, &mut state.0),
            G::output(inputs, &mut state.1),
        )
    }

    fn compute<'a, 'b: 'a>(
        inputs: <F::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
        context: &Self::Context,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let (fs, gs) = state;
        (
            F::compute(inputs, fs, context),
            G::compute(inputs, gs, context),
        )
    }
}

impl<F, G> Default for Fork<F, G>
where
    F: Segment + Default,
    G: Segment + Default,
{
    fn default() -> Self {
        Self(F::default(), G::default())
    }
}

/// Left projection `π₀`: the first projection for categorical products.
/// Forwards the first component of the output and drops the second.
pub struct Left<T, U, C>(pub PhantomData<(T, U, C)>);

impl<T, U, C> Segment for Left<T, U, C>
where
    T: Interface,
    U: Interface,
    C: Sync + 'static,
{
    type Inputs = (T, U);
    type Outputs = T;
    type Context = C;
    type State = ();

    fn init(self, _: <Self::Inputs as Interface>::Values<'_>) -> Self::State {}

    fn output<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        _: &'b mut Self::State,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        inputs.0
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        _: &'b mut Self::State,
        _: &Self::Context,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        inputs.0
    }
}

impl<T, U, C> Default for Left<T, U, C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Right projection `π₁`: the second projection for categorical products.
/// Forwards the second component of the output and drops the first.
pub struct Right<T, U, C>(pub PhantomData<(T, U, C)>);

impl<T, U, C> Segment for Right<T, U, C>
where
    T: Interface,
    U: Interface,
    C: Sync + 'static,
{
    type Inputs = (T, U);
    type Outputs = U;
    type Context = C;
    type State = ();

    fn init(self, _: <Self::Inputs as Interface>::Values<'_>) -> Self::State {}

    fn output<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        _: &'b mut Self::State,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        inputs.1
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        _: &'b mut Self::State,
        _: &Self::Context,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        inputs.1
    }
}

impl<T, U, C> Default for Right<T, U, C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Parallel composition `F *** G`: the arrow map of the product functor.
/// Runs both segments over a pair of inputs, producing a pair of outputs.
/// Equivalent to `Fork<Comp<Left, F>, Comp<Right, G>>`.
pub struct Par<F, G>(pub F, pub G);

impl<F, G> Segment for Par<F, G>
where
    F: Segment,
    G: Segment<Context = F::Context>,
{
    type Inputs = (F::Inputs, G::Inputs);
    type Outputs = (F::Outputs, G::Outputs);
    type Context = F::Context;
    type State = (F::State, G::State);

    fn init(self, inputs: <Self::Inputs as Interface>::Values<'_>) -> Self::State {
        let (a, b) = inputs;
        (self.0.init(a), self.1.init(b))
    }

    fn output<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let (a, b) = inputs;
        (F::output(a, &mut state.0), G::output(b, &mut state.1))
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
        context: &Self::Context,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let (a, b) = inputs;
        let (fs, gs) = state;
        (F::compute(a, fs, context), G::compute(b, gs, context))
    }
}

impl<F, G> Default for Par<F, G>
where
    F: Segment + Default,
    G: Segment + Default,
{
    fn default() -> Self {
        Self(F::default(), G::default())
    }
}

/// Arrow `arr`: the arrow map of the `Port` functor.
/// Applies the given closure to the input ports, producing output ports.
pub struct Arr<T, U, F, C>(F, PhantomData<(T, U, C)>);

impl<T, U, F, C> Segment for Arr<T, U, F, C>
where
    T: Interface,
    U: Interface,
    C: Sync + 'static,
    F: for<'a> FnMut(T::Values<'a>, &'a ()) -> U::Values<'a> + Send + 'static,
{
    type Inputs = T;
    type Outputs = U;
    type Context = C;
    type State = F;

    fn init(self, _: T::Values<'_>) -> Self::State {
        self.0
    }

    fn output<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        state(inputs, &())
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
        _: &Self::Context,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        state(inputs, &())
    }
}

impl<T, U, F, C> Arr<T, U, F, C>
where
    T: Interface,
    U: Interface,
    C: Sync + 'static,
    F: for<'a> Fn(T::Values<'a>, &'a ()) -> U::Values<'a> + Send + 'static,
{
    pub fn new(f: F) -> Self {
        Self(f, PhantomData)
    }
}

/// Equivalent to `Comp<F, Fork<Id, Comp<Arr<H>, G>>>`; used by macros to
/// reduce type complexity.
pub struct Bind<F, G, H>(pub F, pub G, pub H);

impl<F, G, H> Segment for Bind<F, G, H>
where
    F: Segment,
    G: Segment<Context = F::Context>,
    H: for<'a> Fn(
            <F::Outputs as Interface>::Values<'a>,
            &'a (),
        ) -> <G::Inputs as Interface>::Values<'a>
        + Send
        + 'static,
{
    type Inputs = F::Inputs;
    type Outputs = (F::Outputs, G::Outputs);
    type Context = F::Context;
    type State = (F::State, G::State, H);

    fn init(self, inputs: <Self::Inputs as Interface>::Values<'_>) -> Self::State {
        let mut fs = self.0.init(inputs);
        let env = F::output(<F::Inputs as Interface>::reborrow(inputs), &mut fs);
        let gs = self.1.init((self.2)(env, &()));
        (fs, gs, self.2)
    }

    fn output<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let (fs, gs, f) = state;
        let env = F::output(inputs, fs);
        (env, G::output(f(env, &()), gs))
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
        context: &Self::Context,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let (ps, ss, f) = state;
        let env = F::compute(inputs, ps, context);
        (env, G::compute(f(env, &()), ss, context))
    }
}

/// Equivalent to `Comp<F, Arr<H>>`; used by macros to reduce type complexity.
pub struct Route<F, U, T>(pub F, pub T, pub PhantomData<U>);

impl<F, T, H> Segment for Route<F, T, H>
where
    F: Segment,
    T: Interface,
    H: for<'a> Fn(<F::Outputs as Interface>::Values<'a>, &'a ()) -> T::Values<'a> + Send + 'static,
{
    type Inputs = F::Inputs;
    type Outputs = T;
    type Context = F::Context;
    type State = (F::State, H);

    fn init(self, inputs: <Self::Inputs as Interface>::Values<'_>) -> Self::State {
        (self.0.init(inputs), self.1)
    }

    fn output<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let (fs, f) = state;
        f(F::output(inputs, fs), &())
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        state: &'b mut Self::State,
        context: &Self::Context,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let (ps, f) = state;
        f(F::compute(inputs, ps, context), &())
    }
}

/// Convenient combinator methods on any [`Segment`].
pub trait SegmentExt: Segment + Sized {
    fn then<G>(self, g: G) -> Comp<Self, G>
    where
        G: Segment<Inputs = Self::Outputs, Context = Self::Context>,
    {
        Comp(self, g)
    }

    fn fork<G>(self, g: G) -> Fork<Self, G>
    where
        G: Segment<Inputs = Self::Inputs, Context = Self::Context>,
    {
        Fork(self, g)
    }

    fn par<G>(self, g: G) -> Par<Self, G>
    where
        G: Segment<Context = Self::Context>,
    {
        Par(self, g)
    }

    fn bind<S, F>(self, seg: S, route: F) -> Bind<Self, S, F>
    where
        S: Segment<Context = Self::Context>,
        F: for<'a> Fn(
                <Self::Outputs as Interface>::Values<'a>,
                &'a (),
            ) -> <S::Inputs as Interface>::Values<'a>
            + Send
            + 'static,
    {
        Bind(self, seg, route)
    }

    fn route<U, F>(self, route: F) -> Route<Self, U, F>
    where
        U: Interface,
        F: for<'a> Fn(<Self::Outputs as Interface>::Values<'a>, &'a ()) -> U::Values<'a>
            + Send
            + 'static,
    {
        Route(self, route, PhantomData)
    }
}

impl<T: Segment + Sized> SegmentExt for T {}
