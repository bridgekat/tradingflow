use std::{any::TypeId, marker::PhantomData, mem::MaybeUninit};

use super::FlatRead;

/// Marker type defining the protocol for passing a type across interfaces.
///
/// # Safety
///
/// - [`Pass::View<'a>`] must be covariant in `'a`.
pub unsafe trait Pass: 'static {
    /// The view: what travels through ports.
    type View<'a>: Copy + Send + Sync + 'a;

    /// Reborrows a view to a shorter lifetime.
    fn reborrow<'a, 'b: 'a>(view: Self::View<'b>) -> Self::View<'a> {
        // SAFETY: by covariance in the contract.
        unsafe { std::mem::transmute::<Self::View<'b>, Self::View<'a>>(view) }
    }
}

/// The protocol passing `T` by `T` (pass-by-value).
pub struct Val<T: Copy + Send + Sync + 'static>(PhantomData<T>);

// # Safety
//
// `T: 'static` so it is trivially covariant in `'a`.
unsafe impl<T: Copy + Send + Sync + 'static> Pass for Val<T> {
    type View<'a> = T;
}

/// The protocol passing `T` by `&T` (pass-by-reference).
pub struct Ref<T: Sync + 'static>(PhantomData<T>);

// # Safety
//
// `&'a T` is covariant in `'a` (`T: 'static`).
unsafe impl<T: Sync + 'static> Pass for Ref<T> {
    type View<'a> = &'a T;
}

/// The protocol passing `Box<[T]>` by `&[T]`.
pub struct Slice<T: Sync + 'static>(PhantomData<T>);

// # Safety
//
// `T: 'static` so `&'a [T]` is covariant in `'a`.
unsafe impl<T: Sync + 'static> Pass for Slice<T> {
    type View<'a> = &'a [T];
}

/// Marks a single leaf in an [`Interface`] tree.
pub struct Port<V>(PhantomData<fn() -> V>);

/// Marks a variadic leaf in an [`Interface`] tree.
pub struct Ports<V>(PhantomData<fn() -> V>);

/// A type-level specification of an interface, constructed from [`Port`] and
/// [`Ports`] leaves and tuple branches.
///
/// # Safety
///
/// Across node boundaries, interface payloads needs to be de/serialized
/// from/into output slots pointed to by `*const ()` pointers. The two-way
/// conversions implemented by this trait must be mutually consistent.
/// In particular:
///
/// * [`Self::Values<'a>`] must be covariant in `'a`. This is because payloads
///   and scratch buffer contents may be transmuted into, and be used at, a
///   shorter lifetime.
/// * For any given `shape`, [`flat_len`](Self::flat_len),
///   [`type_ids_to_vec`](Self::type_ids_to_vec),
///   [`in_scratch`](Self::in_scratch),
///   [`values_to_slots`](Self::values_to_slots),
///   [`values_to_vecs`](Self::values_to_vecs) and
///   [`values_from_flat`](Self::values_from_flat) agree on the number, order,
///   and leaf types of the flat slots, so every cursor stays aligned and each
///   pointer is only ever re-typed as the `V::View` actually stored there.
/// * Output slot addresses are *stable*: [`values_to_vecs`](Self::values_to_vecs)
///   (run once at build) sizes and initializes the output slots,
///   [`slot_ptrs`](Self::slot_ptrs) (run once at build) records the
///   address of every slot, and [`values_to_slots`](Self::values_to_slots)
///   (run every generation) must rewrite exactly those locations without
///   moving or resizing the storage, so pointers derived at build stay valid
///   for the node's lifetime.
/// * Output slot pointers must carry durable *provenance*:
///   [`slot_ptrs`](Self::slot_ptrs) — and the pointer the caller passes to
///   it — must derive from the state allocation's root pointer through raw
///   projections only. A pointer recorded through a transient reference
///   would be a child of that reference's tag, and the first later write to
///   the slots would invalidate it.
/// * Each [`TypeId`] [`type_ids_to_vec`](Self::type_ids_to_vec) emits uniquely
///   identifies the wire type the matching [`values_from_flat`](Self::values_from_flat)
///   reads (this is what the wiring check is matched against).
pub unsafe trait Interface {
    /// The payload type across this interface.
    type Values<'a>: Copy + Send + 'a;

    /// The deserialization scratch buffer type (optional).
    type InScratch: Send + 'static;

    /// The serialization slot buffer type.
    type OutSlots: Send + 'static;

    /// Returns the number of pointer slots this node spans.
    /// The `shape` reader stores number of elements for variadic ports,
    /// and should be advanced each time a variadic leaf is encountered.
    fn flat_len(shape: &mut FlatRead<usize>) -> usize;

    /// Writes the [`TypeId`] of each pointer slot into `writer` in tree-order,
    /// consuming `shape` at each variadic port.
    fn type_ids_to_vec(shape: &mut FlatRead<usize>, writer: &mut Vec<TypeId>);

    /// Creates an input scratch buffer (uninitialized storage), consuming
    /// `shape` exactly as [`flat_len`](Self::flat_len) does.
    /// Called once per node at build.
    fn in_scratch(shape: &mut FlatRead<usize>) -> Self::InScratch;

    /// Creates an output slot buffer (uninitialized storage).
    /// Called once per node at build.
    fn out_slots() -> Self::OutSlots;

    /// Constructs the payload by consuming pointer slots (`ptrs`).
    /// Consumes `shape` exactly as [`flat_len`](Self::flat_len) does.
    /// The scratch buffer is assumed uninitialized and can be overwritten.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that each pointer slot originates with a
    /// [`TypeId`] matching the output generated by
    /// [`type_ids_to_vec`](Self::type_ids_to_vec) for the same `shape`.
    /// The scratch buffer must be created by
    /// [`in_scratch`](Self::in_scratch) for the same `shape`.
    unsafe fn values_from_flat<'a>(
        shape: &mut FlatRead<'a, usize>,
        ptrs: &mut FlatRead<'a, *const ()>,
        scratch: &'a mut Self::InScratch,
    ) -> Self::Values<'a>;

    /// Serializes the payload into the output slots, in place. References
    /// over slot storage are fine here: `slots` is freshly reborrowed from
    /// the state's root pointer each generation, so it is a descendant of the
    /// tag the recorded slot pointers carry, and its writes never invalidate
    /// them (see the trait-level contract on provenance).
    ///
    /// # Safety
    ///
    /// `slots` must be the output slot buffer sized and initialized by a
    /// prior [`values_to_vecs`](Self::values_to_vecs) call; this call must
    /// rewrite exactly the slots whose addresses
    /// [`slot_ptrs`](Self::slot_ptrs) recorded, without moving or
    /// resizing the storage. Panics if a variadic leaf's element count
    /// differs from the recorded one (the output shape is fixed at build).
    unsafe fn values_to_slots<'a>(values: Self::Values<'a>, slots: &'a mut Self::OutSlots);

    /// Serializes the payload into the output slots, sizing variadic
    /// storage. Writes to `shape` in the exact same format as how
    /// [`flat_len`](Self::flat_len) reads. Called once per node at build,
    /// before [`slot_ptrs`](Self::slot_ptrs).
    ///
    /// # Safety
    ///
    /// The slot buffer must be created by [`out_slots`](Self::out_slots).
    /// The callee guarantees that the resulting slots are compatible with
    /// [`TypeId`]s generated by [`type_ids_to_vec`](Self::type_ids_to_vec)
    /// called on the resulting `shape`.
    unsafe fn values_to_vecs<'a>(
        values: Self::Values<'a>,
        shape: &mut Vec<usize>,
        slots: &'a mut Self::OutSlots,
    );

    /// Records the stable address of every output slot into `ptrs`, in tree
    /// order, through raw pointer operations only. Called once per node at
    /// build, after [`values_to_vecs`](Self::values_to_vecs); the recorded
    /// pointers carry the provenance of `slots` (for inline slots) or of
    /// the variadic storage it owns, and stay valid for as long as the
    /// slots are neither moved, resized, nor dropped — which
    /// [`values_to_slots`](Self::values_to_slots) guarantees.
    ///
    /// # Safety
    ///
    /// `slots` must point to a slots buffer sized and initialized by a
    /// prior [`values_to_vecs`](Self::values_to_vecs) call.
    unsafe fn slot_ptrs(slots: *const Self::OutSlots, ptrs: &mut Vec<*const ()>);

    /// Reborrows an interface to a shorter lifetime.
    fn reborrow<'a, 'b: 'a>(values: Self::Values<'b>) -> Self::Values<'a> {
        // SAFETY: by covariance in the contract.
        unsafe { std::mem::transmute::<Self::Values<'b>, Self::Values<'a>>(values) }
    }
}

unsafe impl Interface for () {
    type Values<'a> = ();
    type InScratch = ();
    type OutSlots = ();

    fn flat_len(_shape: &mut FlatRead<usize>) -> usize {
        0
    }

    fn type_ids_to_vec(_shape: &mut FlatRead<usize>, _writer: &mut Vec<TypeId>) {}

    fn in_scratch(_shape: &mut FlatRead<usize>) {}

    fn out_slots() {}

    unsafe fn values_from_flat<'a>(
        _shape: &mut FlatRead<'a, usize>,
        _ptrs: &mut FlatRead<'a, *const ()>,
        _scratch: &'a mut Self::InScratch,
    ) {
    }

    unsafe fn values_to_slots<'a>(_values: Self::Values<'a>, _slots: &'a mut Self::OutSlots) {}

    unsafe fn values_to_vecs<'a>(
        _values: Self::Values<'a>,
        _shape: &mut Vec<usize>,
        _slots: &'a mut Self::OutSlots,
    ) {
    }

    unsafe fn slot_ptrs(_slots: *const Self::OutSlots, _ptrs: &mut Vec<*const ()>) {}
}

// -- Compound: tuple branches (arities 1-12) --------------------------------

macro_rules! impl_interface_for_tuple {
    ($($idx:tt: $T:ident),+) => {
        // SAFETY: a branch concatenates its children's flat slots in field
        // order; each method delegates to every child in that same order
        // (splitting the scratch tuples field-wise), so consistency follows
        // from the children's.
        unsafe impl<$($T: Interface,)+> Interface for ($($T,)+) {
            type Values<'a> = ($($T::Values<'a>,)+);
            type InScratch = ($($T::InScratch,)+);
            type OutSlots = ($($T::OutSlots,)+);

            fn flat_len(shape: &mut FlatRead<usize>) -> usize {
                0 $(+ $T::flat_len(shape))+
            }

            fn type_ids_to_vec(shape: &mut FlatRead<usize>, writer: &mut Vec<TypeId>) {
                $( $T::type_ids_to_vec(shape, writer); )+
            }

            fn in_scratch(shape: &mut FlatRead<usize>) -> Self::InScratch {
                ( $( $T::in_scratch(shape), )+ )
            }

            fn out_slots() -> Self::OutSlots {
                ( $( $T::out_slots(), )+ )
            }

            unsafe fn values_from_flat<'a>(
                shape: &mut FlatRead<'a, usize>,
                ptrs: &mut FlatRead<'a, *const ()>,
                scratch: &'a mut Self::InScratch,
            ) -> Self::Values<'a> {
                ( $( unsafe { $T::values_from_flat(shape, ptrs, &mut scratch.$idx) }, )+ )
            }

            unsafe fn values_to_slots<'a>(
                values: Self::Values<'a>,
                slots: &'a mut Self::OutSlots,
            ) {
                $( unsafe { $T::values_to_slots(values.$idx, &mut slots.$idx); } )+
            }

            unsafe fn values_to_vecs<'a>(
                values: Self::Values<'a>,
                shape: &mut Vec<usize>,
                slots: &'a mut Self::OutSlots,
            ) {
                $( unsafe { $T::values_to_vecs(values.$idx, shape, &mut slots.$idx); } )+
            }

            unsafe fn slot_ptrs(slots: *const Self::OutSlots, ptrs: &mut Vec<*const ()>) {
                $( unsafe { $T::slot_ptrs(&raw const(*slots).$idx, ptrs); } )+
            }
        }
    };
}

impl_interface_for_tuple!(0: A);
impl_interface_for_tuple!(0: A, 1: B);
impl_interface_for_tuple!(0: A, 1: B, 2: C);
impl_interface_for_tuple!(0: A, 1: B, 2: C, 3: D);
impl_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);

unsafe impl<V: Pass> Interface for Port<V> {
    type Values<'a> = V::View<'a>;
    type InScratch = ();
    type OutSlots = MaybeUninit<V::View<'static>>;

    fn flat_len(_shape: &mut FlatRead<usize>) -> usize {
        1
    }

    fn type_ids_to_vec(_shape: &mut FlatRead<usize>, writer: &mut Vec<TypeId>) {
        writer.push(TypeId::of::<V>());
    }

    fn in_scratch(_shape: &mut FlatRead<usize>) -> Self::InScratch {}

    fn out_slots() -> Self::OutSlots {
        MaybeUninit::uninit()
    }

    unsafe fn values_from_flat<'a>(
        _shape: &mut FlatRead<'a, usize>,
        ptrs: &mut FlatRead<'a, *const ()>,
        _scratch: &'a mut Self::InScratch,
    ) -> Self::Values<'a> {
        // Copy the view out of the producer's scratch: the payload owns its
        // copy, so a forwarding consumer re-homes it on its own output side.
        unsafe { ptrs.pop().cast::<V::View<'a>>().read() }
    }

    unsafe fn values_to_slots<'a>(values: Self::Values<'a>, slots: &'a mut Self::OutSlots) {
        // SAFETY: storage-only lifetime erasure -- layout is lifetime-invariant
        // by the [`Value`] covariance contract; consumers re-type the slot
        // back at their (shorter) generation lifetime.
        unsafe { slots.as_mut_ptr().cast::<V::View<'a>>().write(values) };
    }

    unsafe fn values_to_vecs<'a>(
        values: Self::Values<'a>,
        _shape: &mut Vec<usize>,
        slots: &'a mut Self::OutSlots,
    ) {
        // SAFETY: as in `values_to_slots` (build-time initialization).
        unsafe { slots.as_mut_ptr().cast::<V::View<'a>>().write(values) };
    }

    unsafe fn slot_ptrs(slots: *const Self::OutSlots, ptrs: &mut Vec<*const ()>) {
        ptrs.push(slots.cast());
    }
}

unsafe impl<V: Pass> Interface for Ports<V> {
    type Values<'a> = &'a [V::View<'a>];
    type InScratch = Box<[MaybeUninit<V::View<'static>>]>;
    type OutSlots = Box<[MaybeUninit<V::View<'static>>]>;

    fn flat_len(shape: &mut FlatRead<usize>) -> usize {
        *shape.pop()
    }

    fn type_ids_to_vec(shape: &mut FlatRead<usize>, writer: &mut Vec<TypeId>) {
        let n = *shape.pop();
        writer.extend(std::iter::repeat_n(TypeId::of::<V>(), n));
    }

    fn in_scratch(shape: &mut FlatRead<usize>) -> Self::InScratch {
        Box::new_uninit_slice(*shape.pop())
    }

    fn out_slots() -> Self::OutSlots {
        // Placeholder: `values_to_vecs` sizes the real storage at build, when
        // the first serialization reveals the element count.
        Box::new_uninit_slice(0)
    }

    unsafe fn values_from_flat<'a>(
        shape: &mut FlatRead<'a, usize>,
        ptrs: &mut FlatRead<'a, *const ()>,
        scratch: &'a mut Self::InScratch,
    ) -> Self::Values<'a> {
        let n = *shape.pop();
        let p = ptrs.take(n);
        debug_assert!(scratch.len() == n, "ports scratch disagrees with shape");
        for (slot, &ptr) in scratch.iter_mut().zip(p) {
            // SAFETY: `ptr` targets a valid `V::View` for `'a` by the caller's
            // contract; storing it `'static`-erased is storage-only (layout is
            // lifetime-invariant per the [`Value`] covariance contract).
            unsafe {
                slot.as_mut_ptr()
                    .cast::<V::View<'a>>()
                    .write(ptr.cast::<V::View<'a>>().read());
            }
        }
        // SAFETY: all `n` slots were initialized above; `[MaybeUninit<T>]` is
        // layout-identical to `[T]`, re-typed back at the generation lifetime.
        unsafe { &*(std::ptr::from_ref::<[_]>(scratch) as *const [V::View<'a>]) }
    }

    unsafe fn values_to_slots<'a>(values: Self::Values<'a>, slots: &'a mut Self::OutSlots) {
        // The element count is part of the output shape, which is fixed at
        // build; writing through a wrong count would target stale slots, so
        // check it explicitly.
        assert!(
            values.len() == slots.len(),
            "output shape changed since build (variadic leaf count)"
        );
        for (i, value) in values.iter().enumerate() {
            // SAFETY: storage-only lifetime erasure, as in `Port`; in-bounds
            // by the assert above. Raw writes keep the recorded pointers
            // valid.
            unsafe { slots[i].as_mut_ptr().cast::<V::View<'a>>().write(*value) };
        }
    }

    unsafe fn values_to_vecs<'a>(
        values: Self::Values<'a>,
        shape: &mut Vec<usize>,
        slots: &'a mut Self::OutSlots,
    ) {
        shape.push(values.len());
        // Size the stable storage to the now-known element count and home
        // every view into it. The box is never reallocated afterwards
        // (`values_to_slots` writes in place), so the addresses
        // `slot_ptrs` records stay valid for the node's lifetime.
        *slots = Box::new_uninit_slice(values.len());
        for (i, value) in values.iter().enumerate() {
            // SAFETY: storage-only lifetime erasure, as in `Port`.
            unsafe { slots[i].as_mut_ptr().cast::<V::View<'a>>().write(*value) };
        }
    }

    unsafe fn slot_ptrs(slots: *const Self::OutSlots, ptrs: &mut Vec<*const ()>) {
        // Raw projection through the box, so the recorded pointers carry the
        // box allocation's provenance and are not children of any (transient)
        // reference.
        let slice = unsafe { &raw const **slots };
        let base = slice.cast::<MaybeUninit<V::View<'static>>>();
        for i in 0..slice.len() {
            ptrs.push(unsafe { base.add(i).cast() });
        }
    }
}
