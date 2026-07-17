use std::marker::PhantomData;

use super::Interface;

/// A composable morphism `Inputs -> Outputs`.
pub trait Segment {
    /// Input tree (e.g. `(RefPort<f64>, RefPorts<f64>)`).
    type Inputs: Interface;
    /// Output tree (e.g. `(RefPort<f64>, RefPorts<f64>)`).
    type Outputs: Interface;
    /// Expected graph context.
    type Context: Send + Sync + 'static;
    /// Mutable node state, must be completely owned.
    type State: Send + 'static;

    /// Typed state initialization function.
    fn init(self) -> Self::State;

    /// Typed compute function.
    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        context: &Self::Context,
        state: &'b mut Self::State,
        is_first_run: bool,
    ) -> <Self::Outputs as Interface>::Values<'a>;
}

// -- Combinators: the cartesian category structure ----------------------------

/// The identity operator `id`: the categorical identity.
/// Forwards inputs to outputs without modification.
pub struct Id<T, C>(pub PhantomData<(T, C)>);

impl<T, C> Segment for Id<T, C>
where
    T: Interface,
    C: Send + Sync + 'static,
{
    type Inputs = T;
    type Outputs = T;
    type Context = C;
    type State = ();

    fn init(self) -> Self::State {}

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        _: &Self::Context,
        _: &'b mut Self::State,
        _: bool,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        inputs
    }
}

impl<T, C> Default for Id<T, C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Sequential composition `F >>> G`: the categorical composition.
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

    fn init(self) -> Self::State {
        (self.0.init(), self.1.init())
    }

    fn compute<'a, 'b: 'a>(
        inputs: <F::Inputs as Interface>::Values<'a>,
        context: &Self::Context,
        state: &'b mut Self::State,
        is_first_run: bool,
    ) -> <G::Outputs as Interface>::Values<'a> {
        let (fs, gs) = state;
        let mid = F::compute(inputs, context, fs, is_first_run);
        G::compute(mid, context, gs, is_first_run)
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

    fn init(self) -> Self::State {
        (self.0.init(), self.1.init())
    }

    fn compute<'a, 'b: 'a>(
        inputs: <F::Inputs as Interface>::Values<'a>,
        context: &Self::Context,
        state: &'b mut Self::State,
        is_first_run: bool,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let (fs, gs) = state;
        (
            F::compute(inputs, context, fs, is_first_run),
            G::compute(inputs, context, gs, is_first_run),
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
    C: Send + Sync + 'static,
{
    type Inputs = (T, U);
    type Outputs = T;
    type Context = C;
    type State = ();

    fn init(self) -> Self::State {}

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        _: &Self::Context,
        _: &'b mut Self::State,
        _: bool,
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
    C: Send + Sync + 'static,
{
    type Inputs = (T, U);
    type Outputs = U;
    type Context = C;
    type State = ();

    fn init(self) -> Self::State {}

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        _: &Self::Context,
        _: &'b mut Self::State,
        _: bool,
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

    fn init(self) -> Self::State {
        (self.0.init(), self.1.init())
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        context: &Self::Context,
        state: &'b mut Self::State,
        is_first_run: bool,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let (a, b) = inputs;
        let (fs, gs) = state;
        (
            F::compute(a, context, fs, is_first_run),
            G::compute(b, context, gs, is_first_run),
        )
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

/// Arrow `arr`: the arrow map of the `RefPort` functor.
/// Applies the given closure to the input ports, producing output ports.
pub struct Arr<T, U, C, F>(F, PhantomData<(T, U, C)>);

impl<T, U, C, F> Segment for Arr<T, U, C, F>
where
    T: Interface,
    U: Interface,
    C: Send + Sync + 'static,
    F: for<'a> FnMut(T::Values<'a>, &'a ()) -> U::Values<'a> + Send + 'static,
{
    type Inputs = T;
    type Outputs = U;
    type Context = C;
    type State = F;

    fn init(self) -> Self::State {
        self.0
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        _: &Self::Context,
        state: &'b mut Self::State,
        _: bool,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        state(inputs, &())
    }
}

// The `new` impl repeats the `Segment` impl's bounds on purpose: the closure
// bound at the construction site is what drives closure-signature inference
// (a bound-free `new` leaves the higher-ranked signature ambiguous).
impl<T, U, C, F> Arr<T, U, C, F>
where
    T: Interface,
    U: Interface,
    C: Send + Sync + 'static,
    F: for<'a> FnMut(T::Values<'a>, &'a ()) -> U::Values<'a> + Send + 'static,
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

    fn init(self) -> Self::State {
        (self.0.init(), self.1.init(), self.2)
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        context: &Self::Context,
        state: &'b mut Self::State,
        is_first_run: bool,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let (ps, ss, f) = state;
        let env = F::compute(inputs, context, ps, is_first_run);
        (env, G::compute(f(env, &()), context, ss, is_first_run))
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

    fn init(self) -> Self::State {
        (self.0.init(), self.1)
    }

    fn compute<'a, 'b: 'a>(
        inputs: <Self::Inputs as Interface>::Values<'a>,
        context: &Self::Context,
        state: &'b mut Self::State,
        is_first_run: bool,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let (ps, f) = state;
        f(F::compute(inputs, context, ps, is_first_run), &())
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

    /// One `segment!` statement `let out = wires => seg`: route picks from the
    /// environment (`self`'s outputs) into `seg`'s inputs, and extend the
    /// environment with `seg`'s outputs, producing `(env, out)`.
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

    /// Final re-route of the environment into the result tree `U` (the `segment!`
    /// epilogue). `U` cannot be inferred from the closure (`Values` is a
    /// non-injective projection), so call as `route::<OutInterface, _>(..)`. (The
    /// annotation-free tail form needs no method of its own:
    /// `bind(seg, f).then(Right::default())` -- `Right` has no closure and is
    /// fully inferred top-down.)
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
