use std::any::TypeId;
use std::marker::PhantomData;

use crate::core::ErasedCell;
use crate::typed::{NodeHandle, Pass};

use super::{
    FlatRead, FlatWrite, HandlesInterface, Interface, InterfaceHandles, PortHandle, Segment,
};

/// The erased per-node state: the (immutable) input shape, the input/output
/// scratch buffers, and the user state.
type NodeState<T> = (
    Box<[usize]>,
    <<T as Segment>::Inputs as Interface>::InScratch,
    <<T as Segment>::Outputs as Interface>::OutScratch,
    Option<<T as Segment>::State>,
);

/// Builder for the typed layer [`Graph`].
pub struct Builder<C: Sync> {
    inner: crate::core::Builder,
    _context: PhantomData<C>,
}

impl<C: Sync> Builder<C> {
    pub fn new() -> Self {
        Self {
            inner: crate::core::Builder::new(),
            _context: PhantomData,
        }
    }

    pub fn inner(&self) -> &crate::core::Builder {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut crate::core::Builder {
        &mut self.inner
    }

    pub fn view<V: Pass>(&self, handle: PortHandle<V>) -> V::View<'_> {
        let index = handle.index();
        let type_id = self.inner.slot_type_id(index);
        assert!(type_id == TypeId::of::<V>(), "slot type mismatch");
        unsafe { self.inner.slot_ptr(index).cast::<V::View<'_>>().read() }
    }

    pub fn push<T, H>(
        &mut self,
        segment: T,
        input_handles: H,
    ) -> (
        NodeHandle<T>,
        <T::Outputs as InterfaceHandles>::HandlesOwned,
    )
    where
        H: HandlesInterface,
        T: Segment<Inputs = H::Interface, Context = C>,
        T::Outputs: InterfaceHandles,
    {
        // Get the input cell indices + the input shape (per-`ViewPorts` element
        // counts, tree-order) from the input handles.
        let mut input_indices = Vec::new();
        let mut input_shape = Vec::new();
        input_handles.indices_to_vec(&mut input_shape, &mut input_indices);

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
        let in_scratch = <T::Inputs as Interface>::in_scratch(&mut FlatRead::new(&input_shape));
        let out_scratch = <T::Outputs as Interface>::out_scratch();
        let state: NodeState<T> = (
            input_shape.into_boxed_slice(),
            in_scratch,
            out_scratch,
            None,
        );
        let state_cell = ErasedCell::new(state);

        // Run the init functions once to get the output shape and types.
        let (input_shape, in_scratch, out_scratch, state_mut) =
            unsafe { state_cell.get().cast::<NodeState<T>>().as_mut_unchecked() };

        let input_ptrs = input_indices
            .iter()
            .map(|&i| self.inner.slot_ptr(i))
            .collect::<Box<_>>();

        let inputs = unsafe {
            <T::Inputs as Interface>::values_from_flat(
                &mut FlatRead::new(input_shape),
                &mut FlatRead::new(&input_ptrs),
                in_scratch,
            )
        };
        let outputs = T::reset(inputs, state_mut.insert(T::init(segment, inputs)));

        // Serialize the outputs, homing by-value views into the output
        // scratch (this build call also sizes variadic leaves' scratch).
        let mut output_shape = Vec::new();
        let mut output_ptrs = Vec::new();
        unsafe {
            <T::Outputs as Interface>::values_to_vecs(
                outputs,
                &mut output_shape,
                &mut output_ptrs,
                out_scratch,
            )
        };

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
                reset_fn_for::<T>,
                state_cell,
                output_ptrs.into_boxed_slice(),
            )
        };
        let (node_index, output_range) = self.inner.push(segment, &input_indices).unwrap();
        let output_indices = output_range.collect::<Box<[_]>>();

        // Get the output handles from the output cell indices + output shape.
        let node_handle = NodeHandle::new(node_index);
        let output_handles = <T::Outputs as InterfaceHandles>::handles_from_flat(
            &mut FlatRead::new(&output_shape),
            &mut FlatRead::new(&output_indices),
        );
        (node_handle, output_handles)
    }

    /// Adds a source node to the graph.
    pub fn source<T>(
        &mut self,
        source: T,
    ) -> (
        NodeHandle<T>,
        <T::Outputs as InterfaceHandles>::HandlesOwned,
    )
    where
        T: Segment<Inputs = (), Context = C>,
        T::Outputs: InterfaceHandles,
    {
        self.push(source, ())
    }

    /// Adds a constant node to the graph.
    pub fn value<T>(&mut self, value: T) -> <T::Outputs as InterfaceHandles>::HandlesOwned
    where
        T: Segment<Inputs = (), Context = C>,
        T::Outputs: InterfaceHandles,
    {
        self.push(value, ()).1
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
        self.push(segment, input_handles).1
    }

    /// Finalize into a runnable [`Graph`].
    pub fn build(self) -> Graph<C> {
        Graph {
            inner: self.inner.build(),
            _context: PhantomData,
        }
    }
}

impl<C: Sync> Default for Builder<C> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe fn compute_fn_for<T: Segment>(
    in_ptrs: *const [*const ()],
    out_ptrs: *mut [*const ()],
    state: *mut (),
    context: *const (),
) {
    let (input_shape, in_scratch, out_scratch, state_mut) =
        unsafe { state.cast::<NodeState<T>>().as_mut_unchecked() };

    let inputs = unsafe {
        <T::Inputs as Interface>::values_from_flat(
            &mut FlatRead::new(input_shape.as_ref()),
            &mut FlatRead::new(in_ptrs.as_ref_unchecked()),
            in_scratch,
        )
    };
    let state_mut = unsafe { state_mut.as_mut().unwrap_unchecked() };
    let context = unsafe { &*context.cast::<T::Context>() };
    let outputs = T::compute(inputs, state_mut, context);

    let mut ptrs_writer = FlatWrite::new(unsafe { out_ptrs.as_mut_unchecked() });
    unsafe { <T::Outputs as Interface>::values_to_flat(outputs, &mut ptrs_writer, out_scratch) };

    assert!(
        ptrs_writer.remaining() == 0,
        "output shape changed since build (operator wrote fewer leaves than allocated)"
    );
}

unsafe fn reset_fn_for<T: Segment>(
    in_ptrs: *const [*const ()],
    out_ptrs: *mut [*const ()],
    state: *mut (),
    _: *const (),
) {
    let (input_shape, in_scratch, out_scratch, state_mut) =
        unsafe { state.cast::<NodeState<T>>().as_mut_unchecked() };

    let inputs = unsafe {
        <T::Inputs as Interface>::values_from_flat(
            &mut FlatRead::new(input_shape.as_ref()),
            &mut FlatRead::new(in_ptrs.as_ref_unchecked()),
            in_scratch,
        )
    };
    let state_mut = unsafe { state_mut.as_mut().unwrap_unchecked() };
    let outputs = T::reset(inputs, state_mut);

    let mut ptrs_writer = FlatWrite::new(unsafe { out_ptrs.as_mut_unchecked() });
    unsafe { <T::Outputs as Interface>::values_to_flat(outputs, &mut ptrs_writer, out_scratch) };

    assert!(
        ptrs_writer.remaining() == 0,
        "output shape changed since build (operator wrote fewer leaves than allocated)"
    );
}

/// The typed wrapper over the core layer [`Graph`](crate::core::Graph).
pub struct Graph<C: Sync> {
    inner: crate::core::Graph,
    _context: PhantomData<C>,
}

impl<C: Sync> Graph<C> {
    pub fn inner(&self) -> &crate::core::Graph {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut crate::core::Graph {
        &mut self.inner
    }

    pub fn view<V: Pass>(&self, handle: PortHandle<V>) -> V::View<'_> {
        let index = handle.index();
        let type_id = self.inner.slot_type_id(index);
        assert!(type_id == TypeId::of::<V>(), "slot type mismatch");
        unsafe { self.inner.slot_ptr(index).cast::<V::View<'_>>().read() }
    }

    pub fn state_mut<T: Segment>(&mut self, handle: NodeHandle<T>) -> &mut T::State {
        let index = handle.index();
        let type_id = self.inner.state_mut(index).type_id();
        assert!(
            type_id == TypeId::of::<NodeState<T>>(),
            "cell type mismatch"
        );
        let state = self.inner.state_mut(index).get();
        let (_, _, _, state_mut) = unsafe { state.cast::<NodeState<T>>().as_mut_unchecked() };
        state_mut.as_mut().expect("state emplaced at build")
    }

    pub fn stabilize(&mut self, pool: &mut crate::pool::Pool, context: &C) {
        self.inner.stabilize(pool, context);
    }
}
