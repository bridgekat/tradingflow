use std::any::TypeId;
use std::marker::PhantomData;

use crate::core::ErasedCell;
use crate::typed::{NodeHandle, Pass};

use super::{
    FlatRead, FlatWrite, HandlesInterface, Interface, InterfaceHandles, PortHandle, Segment,
};

/// The erased per-node state: the (immutable) input shape, the input/output
/// scratch buffers, and the user state.
///
/// The scratch buffers are what make by-value (fat) leaves work across node
/// boundaries: deserialization gathers scattered wire views into the input
/// scratch and lends the payload from there, and serialization homes output
/// views in the output scratch and points wire slots at its fields, which
/// keep stable addresses (both are overwritten in place each run, inside the
/// heap-allocated cell). Scratch is typed at `'static` purely as storage --
/// layout is lifetime-invariant, every access re-types it at the calling
/// generation's lifetime, and cross-generation validity rests on the engine's
/// per-generation contract plus the leaves' covariance.
type NodeState<T> = (
    Box<[usize]>,
    <<T as Segment>::Inputs as Interface>::InScratch,
    <<T as Segment>::Outputs as Interface>::OutScratch,
    <T as Segment>::State,
);

pub struct Builder<C: Send + Sync + 'static> {
    inner: crate::core::Builder,
    _context: PhantomData<C>,
}

impl<C: Send + Sync + 'static> Builder<C> {
    pub fn new(context: C) -> Self {
        Self {
            inner: crate::core::Builder::new(ErasedCell::new(context)),
            _context: PhantomData,
        }
    }

    pub fn inner(&self) -> &crate::core::Builder {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut crate::core::Builder {
        &mut self.inner
    }

    pub fn view<V: Pass>(&self, handle: PortHandle<V>) -> V::View<'_>
    where
        for<'a> V::View<'a>: Copy,
    {
        let index = handle.index();
        let type_id = self.inner.slot_type_id(index);
        assert!(type_id == TypeId::of::<V>(), "slot type mismatch");
        unsafe { self.inner.slot_ptr(index).cast::<V::View<'_>>().read() }
    }

    pub fn context(&self) -> &C {
        unsafe { &*self.inner.context().get().cast::<C>() }
    }

    /// Adds a source node to the graph.
    pub fn source<S>(
        &mut self,
        source: S,
    ) -> (
        NodeHandle<S>,
        <S::Outputs as InterfaceHandles>::HandlesOwned,
    )
    where
        S: Segment<Inputs = (), Context = C>,
        S::Outputs: InterfaceHandles,
    {
        let handles = self.segment(source, ());
        (NodeHandle::new(self.inner.num_nodes() - 1), handles)
    }

    /// Adds a segment node to the graph.
    pub fn segment<T, H>(
        &mut self,
        segment: T,
        input_handles: H,
    ) -> <T::Outputs as InterfaceHandles>::HandlesOwned
    where
        H: HandlesInterface,
        T: Segment<Inputs = H::Interface, Context = C>,
        T::Outputs: InterfaceHandles,
    {
        // Get the input cell indices + the input shape (per-`ViewPorts` element
        // counts, tree-order) from the input handles.
        let mut input_indices = Vec::new();
        let mut input_shape = Vec::new();
        input_handles.to_vec(&mut input_shape, &mut input_indices);

        // Get the expected input types of the segment, sized to the actual leaf
        // count and filled in tree-order driven by `input_shape`.
        let mut input_type_ids = Vec::new();
        <T::Inputs as Interface>::type_ids_to_vec(
            &mut FlatRead::new(&input_shape),
            &mut input_type_ids,
        );

        // Validate input cell types before init.
        assert!(
            input_indices.len() == input_type_ids.len(),
            "input arity mismatch"
        );
        for (&slot, &ty) in input_indices.iter().zip(&input_type_ids) {
            assert!(self.inner.slot_type_id(slot) == ty, "input type mismatch");
        }

        // Bundle the (immutable) shape and the scratch buffers with the user
        // state so `compute` can reconstruct the nested payload trees from
        // the flat cell slices and home its result (see `NodeState`).
        let in_scratch = <T::Inputs as Interface>::new_in_scratch(&mut FlatRead::new(&input_shape));
        let state: NodeState<T> = (
            input_shape.into_boxed_slice(),
            in_scratch,
            <T::Outputs as Interface>::new_out_scratch(),
            segment.init(),
        );
        let state_cell = ErasedCell::new(state);

        // Run the compute function once to get the output shape and types.
        let (input_shape, in_scratch, out_scratch, state_mut) =
            unsafe { state_cell.get().cast::<NodeState<T>>().as_mut_unchecked() };

        let input_flags = input_indices
            .iter()
            .map(|&i| self.inner.slot_flag(i))
            .collect::<Box<_>>();
        let input_ptrs = input_indices
            .iter()
            .map(|&i| self.inner.slot_ptr(i))
            .collect::<Box<_>>();

        let inputs = unsafe {
            <T::Inputs as Interface>::values_from_flat(
                &mut FlatRead::new(input_shape),
                &mut FlatRead::new(&input_flags),
                &mut FlatRead::new(&input_ptrs),
                in_scratch,
            )
        };
        let outputs = T::compute(inputs, self.context(), state_mut, true);

        // Serialize the outputs, homing by-value views into the output
        // scratch (this build call also sizes variadic leaves' scratch).
        let mut output_shape = Vec::new();
        let mut output_flags = Vec::new();
        let mut output_ptrs = Vec::new();
        <T::Outputs as Interface>::values_to_vecs(
            outputs,
            out_scratch,
            &mut output_shape,
            &mut output_flags,
            &mut output_ptrs,
        );

        let mut output_type_ids = Vec::new();
        <T::Outputs as Interface>::type_ids_to_vec(
            &mut FlatRead::new(&output_shape),
            &mut output_type_ids,
        );

        // Construct the node and push it to the graph.
        let segment = unsafe {
            crate::core::Segment::new(
                input_type_ids.into_boxed_slice(),
                output_type_ids.into_boxed_slice(),
                compute_fn_for::<T>,
                state_cell,
                output_flags.into_boxed_slice(),
                output_ptrs.into_boxed_slice(),
            )
        };
        let output_range = self.inner.push(segment, &input_indices).unwrap();
        let output_indices = output_range.collect::<Box<[_]>>();

        // Get the output handles from the output cell indices + output shape.
        <T::Outputs as InterfaceHandles>::handles_from_flat(
            &mut FlatRead::new(&output_shape),
            &mut FlatRead::new(&output_indices),
        )
    }

    /// Finalize into a runnable [`Graph`].
    pub fn build(self) -> Graph<C> {
        Graph {
            inner: self.inner.build(),
            _context: PhantomData,
        }
    }
}

unsafe fn compute_fn_for<T: Segment>(
    in_flags: *const [bool],
    in_ptrs: *const [*const ()],
    out_flags: *mut [bool],
    out_ptrs: *mut [*const ()],
    context: *const (),
    state: *mut (),
) {
    let context = unsafe { &*context.cast::<T::Context>() };
    let (input_shape, in_scratch, out_scratch, state_mut) =
        unsafe { state.cast::<NodeState<T>>().as_mut_unchecked() };

    let inputs = unsafe {
        <T::Inputs as Interface>::values_from_flat(
            &mut FlatRead::new(input_shape.as_ref()),
            &mut FlatRead::new(in_flags.as_ref_unchecked()),
            &mut FlatRead::new(in_ptrs.as_ref_unchecked()),
            in_scratch,
        )
    };
    let outputs = T::compute(inputs, context, state_mut, false);

    // Serialize the outputs, homing by-value views into the output scratch,
    // overwriting last generation's in place (see `NodeState`): every output
    // slot is rewritten before any consumer runs, and a node that does not
    // run keeps its scratch untouched, so out-of-cone consumers' pointers
    // stay live.
    let mut flags_writer = FlatWrite::new(unsafe { out_flags.as_mut_unchecked() });
    let mut ptrs_writer = FlatWrite::new(unsafe { out_ptrs.as_mut_unchecked() });
    <T::Outputs as Interface>::values_to_flat(
        outputs,
        out_scratch,
        &mut flags_writer,
        &mut ptrs_writer,
    );

    // A changed output shape should panic, instead of leaving old dangling
    // pointers in the output cells.
    assert!(
        flags_writer.remaining() == 0 && ptrs_writer.remaining() == 0,
        "output shape changed since build (operator wrote fewer leaves than allocated)"
    );
}

pub struct Graph<C: Send + Sync + 'static> {
    inner: crate::core::Graph,
    _context: PhantomData<C>,
}

impl<C: Send + Sync + 'static> Graph<C> {
    pub fn inner(&self) -> &crate::core::Graph {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut crate::core::Graph {
        &mut self.inner
    }

    pub fn view<V: Pass>(&self, handle: PortHandle<V>) -> V::View<'_>
    where
        for<'a> V::View<'a>: Copy,
    {
        let index = handle.index();
        let type_id = self.inner.slot_type_id(index);
        assert!(type_id == TypeId::of::<V>(), "slot type mismatch");
        unsafe { self.inner.slot_ptr(index).cast::<V::View<'_>>().read() }
    }

    pub fn context(&self) -> &C {
        unsafe { &*self.inner.context().get().cast::<C>() }
    }

    pub fn context_mut(&mut self) -> &mut C {
        unsafe { &mut *self.inner.context_mut().get().cast::<C>() }
    }

    pub fn state_mut<S>(&mut self, handle: NodeHandle<S>) -> &mut S::State
    where
        S: Segment,
    {
        let index = handle.index();
        let type_id = self.inner.state_mut(index).type_id();
        assert!(
            type_id == TypeId::of::<NodeState<S>>(),
            "cell type mismatch"
        );
        let state = self.inner.state_mut(index).get();
        let (_, _, _, state_mut) = unsafe { state.cast::<NodeState<S>>().as_mut_unchecked() };
        state_mut
    }

    pub fn stabilize(&mut self, pool: &mut crate::pool::Pool) {
        self.inner.stabilize(pool);
    }
}
