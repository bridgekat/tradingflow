use std::{any::TypeId, marker::PhantomData, mem::MaybeUninit};

use super::{FlatRead, FlatWrite};

/// Marks a *way to pass a value through a port*: a `'static` name `V` for a
/// payload `V::View<'a>`. `Scalar<T>` passes `T`; a user `Array<T>` might pass
/// `ArrayView<'a, T>`. `View` carries no bounds here (only `'a`) -- the leaves
/// that use it add their own (e.g. [`ViewPort`] needs `View: Copy + Send +
/// Sync`).
///
/// # Safety
///
/// * `View<'a>` must be covariant in `'a`.
pub unsafe trait ValueView: 'static {
    type View<'a>: 'a;
}

/// `T` by value: the wire carries the value itself (homed in the producer's
/// output scratch), not a reference into producer state -- so small-scalar
/// producers can be stateless.
pub struct Scalar<T>(PhantomData<T>);

// SAFETY: `T: 'static` so it is trivially covariant in `'a`.
unsafe impl<T: 'static> ValueView for Scalar<T> {
    type View<'a> = T;
}

/// `&[T]` as a view: the contiguous counterpart of [`RefViewPorts`] (one fat wire
/// slot; position *and length* free to vary per generation).
pub struct Slice<T>(PhantomData<T>);

// SAFETY: `T: 'static` so `&'a [T]` is covariant in `'a`.
unsafe impl<T: 'static> ValueView for Slice<T> {
    type View<'a> = &'a [T];
}

/// A single *view* leaf in an [`Interface`] tree: carries a borrowed fat
/// reference `V::View<'a>` -- a sub-slice, a strided view struct -- BY VALUE
/// between fused sub-segments, through one wire slot between nodes. The wire
/// pointer targets the view's home in the producer's output scratch;
/// deserialization copies it back out (hence `View: Copy`, `Send + Sync` for
/// the cross-worker scratch storage -- the bounds [`ValueView`] omits).
/// Payload type: `(bool, V::View<'_>)`.
pub struct ViewPort<V>(PhantomData<V>);

/// A runtime-length group of [`ViewPort`]-leaves over `V`: a fan-in of N
/// by-value producers, payload `(&[bool], &[V::View<'a>])`.
///
/// Wire-compatible with `ViewPort<V>` slot by slot: each element wires against
/// a plain `ViewPort<V>` producer, and each element of a `ViewPorts` *output*
/// can feed a single-`ViewPort` consumer. On input, the wire planes hold
/// pointers to N views scattered across the producers' storage, so
/// deserialization gathers them (a per-element `V::View` copy -- a small
/// `Copy` struct, never the data it views) into the consuming node's input
/// scratch and lends the payload from there as one contiguous slice. On
/// output, the payload slice is already the views' contiguous home, so the
/// wire simply carries each element's address (no scratch).
pub struct ViewPorts<V>(PhantomData<V>);

/// A single by-reference leaf over a [`ValueView`] `V`: passes `&'a V::View<'a>`
/// (the value lives in the producer's state or arena). `RefViewPort<Scalar<T>>` is the
/// plain `&T` port (value `T` in state); [`ViewPort`] is the by-value sibling.
pub struct RefViewPort<V>(PhantomData<V>);

/// A runtime-length group of [`RefViewPort`]-leaves over `V`: a fan-in of N producers,
/// payload `(&[bool], &[&V::View<'a>])`.
pub struct RefViewPorts<V>(PhantomData<V>);

/// Recursive description of a segment's inputs or outputs: a tree of
/// [`ViewPort`], [`ViewPorts`], [`RefViewPort`] and [`RefViewPorts`] leaves
/// over [`ValueView`]s.
///
/// Each leaf's payload is `(notify, value)` -- `(bool, V::View<'a>)` (by
/// value) for a [`ViewPort`], `(bool, &V::View<'a>)` for a [`RefViewPort`],
/// and parallel slices `(&[bool], &[V::View<'a>])` / `(&[bool],
/// &[&V::View<'a>])` for the variadic [`ViewPorts`] / [`RefViewPorts`]. The
/// flat wire form the core moves between nodes is two index-aligned planes: a
/// `bool` notify plane and a `*const ()` value-pointer plane.
///
/// Payloads are the intra-node currency -- segment fusion passes them
/// directly. Crossing a node boundary requires (de)serialization to the wire
/// planes, and leaves whose payload cannot borrow the planes themselves need
/// backing storage for it: the **scratch buffers** [`InScratch`](Self::InScratch)
/// and [`OutScratch`](Self::OutScratch), living in the node's state cell.
/// Pass-by-reference leaves declare `()` and pay nothing.
///
/// When the tree contains a variadic leaf, the flat layout is ambiguous on its
/// own, so the per-group element counts are carried out-of-band as a
/// **shape**: a flat `[usize]` that is the pre-order serialization of the
/// count tree (each [`ViewPorts`] / [`RefViewPorts`] contributes its count).
/// The shape is built from the input handles / `init` values and replayed via
/// a [`FlatRead<usize>`] cursor threaded alongside the plane cursors.
///
/// # Safety
///
/// The engine reconstructs payloads by re-typing `*const ()` wire pointers,
/// trusting this impl's methods to be mutually consistent. The cross-node
/// [`TypeId`] check validates agreement *between* nodes, not the internal
/// coherence of one impl, so an implementor must guarantee:
///
/// * `Values<'a>` is covariant in `'a` (payloads and scratch contents are
///   stored `'static`-erased and read back at a shorter lifetime).
/// * For any given `shape`, [`flat_len`](Self::flat_len),
///   [`type_ids_to_vec`](Self::type_ids_to_vec),
///   [`new_in_scratch`](Self::new_in_scratch),
///   [`values_to_flat`](Self::values_to_flat),
///   [`values_to_vecs`](Self::values_to_vecs) and
///   [`values_from_flat`](Self::values_from_flat) agree on the number, order,
///   and leaf types of the flat slots, so every cursor stays aligned and each
///   pointer is only ever re-typed as the `V::View` actually stored there.
/// * Each [`TypeId`] [`type_ids_to_vec`](Self::type_ids_to_vec) emits uniquely
///   identifies the wire type the matching [`values_from_flat`](Self::values_from_flat)
///   reads (this is what the wiring check is matched against).
/// * Every pointer written by [`values_to_flat`](Self::values_to_flat) /
///   [`values_to_vecs`](Self::values_to_vecs) targets a value that stays valid,
///   at a stable address, for as long as a consumer may dereference it -- in
///   particular, a pointer into `OutScratch` must stay valid until the node's
///   next run, so scratch storage may only be (re)allocated by
///   [`values_to_vecs`](Self::values_to_vecs) (the once-per-node build call);
///   [`values_to_flat`](Self::values_to_flat) and
///   [`values_from_flat`](Self::values_from_flat) must overwrite it in place.
pub unsafe trait Interface {
    /// Nested `(notify, value)` payload tree. `Copy + Send`, so it threads
    /// freely through fused bodies and across node-boundary serialization.
    type Values<'a>: Copy + Send + 'a;

    /// Node-boundary *deserialization* buffer, living in the node's state
    /// cell: a leaf that must gather scattered wire data into contiguous
    /// storage ([`ViewPorts`]) declares it here and lends its payload from it;
    /// zero-copy leaves declare `()`. Typed at `'static` purely as storage --
    /// layout is lifetime-invariant and every access re-types it at the
    /// calling generation's lifetime.
    type InScratch: Send + 'static;

    /// Node-boundary *serialization* buffer, living in the node's state cell:
    /// by-value leaves ([`ViewPort`], [`ViewPorts`]) home their views here and
    /// point wire slots at its fields; pass-by-reference leaves declare `()`
    /// and write the carried reference directly. Same `'static`-as-storage
    /// convention as [`InScratch`](Self::InScratch).
    type OutScratch: Send + 'static;

    /// Number of flat leaf slots this node spans, advancing `shape` past it.
    ///
    /// Dynamic generalization of a static arity: fixed-shape nodes ignore
    /// `shape`; a variadic leaf pops and returns its count.
    fn flat_len(shape: &mut FlatRead<usize>) -> usize;

    /// Write the [`TypeId`] of each flat leaf into `writer` in tree-order,
    /// consuming `shape` at each variadic leaf.
    fn type_ids_to_vec(shape: &mut FlatRead<usize>, writer: &mut Vec<TypeId>);

    /// Construct the input scratch, consuming `shape` exactly as
    /// [`flat_len`](Self::flat_len) does and sizing the storage consistently
    /// with what [`values_from_flat`](Self::values_from_flat) expects for that
    /// shape. Called once per node at build.
    fn new_in_scratch(shape: &mut FlatRead<usize>) -> Self::InScratch;

    /// Construct the output scratch (uninitialized storage: it is written
    /// before every read, by [`values_to_vecs`](Self::values_to_vecs) /
    /// [`values_to_flat`](Self::values_to_flat)). Called once per node at
    /// build.
    fn new_out_scratch() -> Self::OutScratch;

    /// Construct the nested payload tree by consuming the two parallel wire
    /// planes (`flags` + value pointers `ptrs`), using `shape` to size each
    /// variadic leaf. Zero-copy leaves return the consumed sub-slices (the
    /// pointer plane re-typed in place); gathering leaves copy the pointed-to
    /// views into `scratch` (overwriting last generation's in place) and lend
    /// the payload from there.
    ///
    /// # Safety
    ///
    /// Each consumed pointer must point to a valid value of the matching type,
    /// and `shape` must be the shape that produced this layout (and sized
    /// `scratch` via [`new_in_scratch`](Self::new_in_scratch)).
    unsafe fn values_from_flat<'a>(
        shape: &mut FlatRead<'a, usize>,
        flags: &mut FlatRead<'a, bool>,
        ptrs: &mut FlatRead<'a, *const ()>,
        scratch: &'a mut Self::InScratch,
    ) -> Self::Values<'a>;

    /// Serialize the payload tree onto the wire planes. Thin (by-reference)
    /// leaves write the reference they carry; a fat (by-value) leaf homes its
    /// view into `scratch` -- overwriting last generation's in place -- and
    /// writes a thin pointer to that home, which keeps a stable address until
    /// the node's next run.
    fn values_to_flat<'a>(
        values: Self::Values<'a>,
        scratch: &mut Self::OutScratch,
        flags: &mut FlatWrite<bool>,
        ptrs: &mut FlatWrite<*const ()>,
    );

    /// [`values_to_flat`](Self::values_to_flat) into growable vecs, recording
    /// the shape (the once-per-node build call). This is the one place allowed
    /// to (re)allocate `scratch` -- variadic output leaves size theirs here.
    fn values_to_vecs<'a>(
        values: Self::Values<'a>,
        scratch: &mut Self::OutScratch,
        shape: &mut Vec<usize>,
        flags: &mut Vec<bool>,
        ptrs: &mut Vec<*const ()>,
    );

    /// Whether any leaf's notify flag is set (the gate for a fused operator).
    fn any_notify(values: &Self::Values<'_>) -> bool;
}

// -- Compound: empty tuple (arity 0) ----------------------------------------

// SAFETY: the empty tree spans zero leaves; every method is a no-op, so the
// consistency obligations hold vacuously.
unsafe impl Interface for () {
    type Values<'a> = ();
    type InScratch = ();
    type OutScratch = ();

    #[inline]
    fn flat_len(_shape: &mut FlatRead<usize>) -> usize {
        0
    }

    #[inline]
    fn type_ids_to_vec(_shape: &mut FlatRead<usize>, _writer: &mut Vec<TypeId>) {}

    #[inline]
    fn new_in_scratch(_shape: &mut FlatRead<usize>) -> Self::InScratch {}

    #[inline]
    fn new_out_scratch() -> Self::OutScratch {}

    #[inline]
    unsafe fn values_from_flat<'a>(
        _shape: &mut FlatRead<'a, usize>,
        _flags: &mut FlatRead<'a, bool>,
        _ptrs: &mut FlatRead<'a, *const ()>,
        _scratch: &'a mut Self::InScratch,
    ) {
    }

    #[inline]
    fn values_to_flat<'a>(
        _values: Self::Values<'a>,
        _scratch: &mut Self::OutScratch,
        _flags: &mut FlatWrite<bool>,
        _ptrs: &mut FlatWrite<*const ()>,
    ) {
    }

    #[inline]
    fn values_to_vecs<'a>(
        _values: Self::Values<'a>,
        _scratch: &mut Self::OutScratch,
        _shape: &mut Vec<usize>,
        _flags: &mut Vec<bool>,
        _ptrs: &mut Vec<*const ()>,
    ) {
    }

    #[inline]
    fn any_notify(_: &()) -> bool {
        false
    }
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
            type OutScratch = ($($T::OutScratch,)+);

            #[inline]
            fn flat_len(shape: &mut FlatRead<usize>) -> usize {
                0 $(+ $T::flat_len(shape))+
            }

            #[inline]
            fn type_ids_to_vec(shape: &mut FlatRead<usize>, writer: &mut Vec<TypeId>) {
                $( $T::type_ids_to_vec(shape, writer); )+
            }

            #[inline]
            fn new_in_scratch(shape: &mut FlatRead<usize>) -> Self::InScratch {
                ( $( $T::new_in_scratch(shape), )+ )
            }

            #[inline]
            fn new_out_scratch() -> Self::OutScratch {
                ( $( $T::new_out_scratch(), )+ )
            }

            #[inline]
            unsafe fn values_from_flat<'a>(
                shape: &mut FlatRead<'a, usize>,
                flags: &mut FlatRead<'a, bool>,
                ptrs: &mut FlatRead<'a, *const ()>,
                scratch: &'a mut Self::InScratch,
            ) -> Self::Values<'a> {
                ( $( unsafe { $T::values_from_flat(shape, flags, ptrs, &mut scratch.$idx) }, )+ )
            }

            #[inline]
            fn values_to_flat<'a>(
                values: Self::Values<'a>,
                scratch: &mut Self::OutScratch,
                flags: &mut FlatWrite<bool>,
                ptrs: &mut FlatWrite<*const ()>,
            ) {
                $( $T::values_to_flat(values.$idx, &mut scratch.$idx, flags, ptrs); )+
            }

            #[inline]
            fn values_to_vecs<'a>(
                values: Self::Values<'a>,
                scratch: &mut Self::OutScratch,
                shape: &mut Vec<usize>,
                flags: &mut Vec<bool>,
                ptrs: &mut Vec<*const ()>,
            ) {
                $( $T::values_to_vecs(values.$idx, &mut scratch.$idx, shape, flags, ptrs); )+
            }

            #[inline]
            fn any_notify(values: &Self::Values<'_>) -> bool {
                false $(|| $T::any_notify(&values.$idx))+
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

// -- Value leaf: ViewPort<V> ------------------------------------------------

// SAFETY: one flat slot tagged `TypeId::of::<ViewPort<V>>()`; the value is
// carried by value, so serialization homes it in the one-view `OutScratch`
// (overwritten in place each run; the `'static` typing is storage-only, per
// the [`ValueView`] covariance contract, and views are `Copy` so the
// overwrite drops nothing) and points the wire slot there; deserialization
// reads it back at the same `V::View` type.
unsafe impl<V: ValueView> Interface for ViewPort<V>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Values<'a> = (bool, V::View<'a>);
    type InScratch = ();
    type OutScratch = MaybeUninit<V::View<'static>>;

    #[inline]
    fn flat_len(_shape: &mut FlatRead<usize>) -> usize {
        1
    }

    #[inline]
    fn type_ids_to_vec(_shape: &mut FlatRead<usize>, writer: &mut Vec<TypeId>) {
        writer.push(TypeId::of::<ViewPort<V>>());
    }

    #[inline]
    fn new_in_scratch(_shape: &mut FlatRead<usize>) -> Self::InScratch {}

    #[inline]
    fn new_out_scratch() -> Self::OutScratch {
        MaybeUninit::uninit()
    }

    #[inline]
    unsafe fn values_from_flat<'a>(
        _shape: &mut FlatRead<'a, usize>,
        flags: &mut FlatRead<'a, bool>,
        ptrs: &mut FlatRead<'a, *const ()>,
        _scratch: &'a mut Self::InScratch,
    ) -> Self::Values<'a> {
        // Copy the view out of the producer's scratch: the payload owns its
        // copy, so a forwarding consumer re-homes it on its own output side.
        (*flags.pop(), unsafe {
            ptrs.pop().cast::<V::View<'a>>().read()
        })
    }

    #[inline]
    fn values_to_flat<'a>(
        values: Self::Values<'a>,
        scratch: &mut Self::OutScratch,
        flags: &mut FlatWrite<bool>,
        ptrs: &mut FlatWrite<*const ()>,
    ) {
        flags.push(values.0);
        // SAFETY: storage-only lifetime erasure -- layout is lifetime-invariant
        // by the [`ValueView`] covariance contract; consumers re-type the slot
        // back at their (shorter) generation lifetime.
        unsafe { scratch.as_mut_ptr().cast::<V::View<'a>>().write(values.1) };
        ptrs.push(scratch.as_ptr().cast());
    }

    #[inline]
    fn values_to_vecs<'a>(
        values: Self::Values<'a>,
        scratch: &mut Self::OutScratch,
        _shape: &mut Vec<usize>,
        flags: &mut Vec<bool>,
        ptrs: &mut Vec<*const ()>,
    ) {
        flags.push(values.0);
        // SAFETY: as in `values_to_flat`.
        unsafe { scratch.as_mut_ptr().cast::<V::View<'a>>().write(values.1) };
        ptrs.push(scratch.as_ptr().cast());
    }

    #[inline]
    fn any_notify(values: &Self::Values<'_>) -> bool {
        values.0
    }
}

pub type Port<T> = ViewPort<Scalar<T>>;

// -- Value leaves: ViewPorts<V> ----------------------------------------------

// SAFETY: `*shape.pop()` flat slots, each tagged `TypeId::of::<ViewPort<V>>()`
// (matching what a single `ViewPort<V>` consumer expects and what a single
// `ViewPort<V>` producer emits), so a group wires against by-value producers.
// Deserialization gathers the N pointed-to views into the `InScratch` buffer
// (sized once from the shape at build, overwritten in place each run; its
// `'static` typing is storage-only, per the [`ValueView`] covariance contract,
// and views are `Copy`, so in-place overwrites drop nothing) and lends the
// payload slice from there. Serialization needs no scratch: the payload slice
// itself is the contiguous home of the N views — element `i`'s thin pointer
// is just `&values.1[i]` — and the producer keeps that storage stable for the
// generation (state / arena / its own `InScratch`), exactly the discipline
// [`RefViewPorts`] outputs already require.
unsafe impl<V: ValueView> Interface for ViewPorts<V>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Values<'a> = (&'a [bool], &'a [V::View<'a>]);
    type InScratch = Box<[MaybeUninit<V::View<'static>>]>;
    type OutScratch = ();

    #[inline]
    fn flat_len(shape: &mut FlatRead<usize>) -> usize {
        *shape.pop()
    }

    #[inline]
    fn type_ids_to_vec(shape: &mut FlatRead<usize>, writer: &mut Vec<TypeId>) {
        let n = *shape.pop();
        writer.extend(std::iter::repeat_n(TypeId::of::<ViewPort<V>>(), n));
    }

    #[inline]
    fn new_in_scratch(shape: &mut FlatRead<usize>) -> Self::InScratch {
        Box::new_uninit_slice(*shape.pop())
    }

    #[inline]
    fn new_out_scratch() -> Self::OutScratch {}

    #[inline]
    unsafe fn values_from_flat<'a>(
        shape: &mut FlatRead<'a, usize>,
        flags: &mut FlatRead<'a, bool>,
        ptrs: &mut FlatRead<'a, *const ()>,
        scratch: &'a mut Self::InScratch,
    ) -> Self::Values<'a> {
        let n = *shape.pop();
        let f = flags.take(n);
        let p = ptrs.take(n);
        let dst: &'a mut [MaybeUninit<V::View<'static>>] = scratch;
        debug_assert!(dst.len() == n, "ViewPorts scratch disagrees with shape");
        for (slot, &ptr) in dst.iter_mut().zip(p) {
            // SAFETY: `ptr` targets a valid `V::View` for `'a` by the caller's
            // contract; storing it `'static`-erased is storage-only (layout is
            // lifetime-invariant per the [`ValueView`] covariance contract).
            unsafe {
                slot.as_mut_ptr()
                    .cast::<V::View<'a>>()
                    .write(ptr.cast::<V::View<'a>>().read());
            }
        }
        // SAFETY: all `n` slots were initialized above; `[MaybeUninit<T>]` is
        // layout-identical to `[T]`, re-typed back at the generation lifetime.
        let v = unsafe {
            &*(std::ptr::from_ref::<[MaybeUninit<V::View<'static>>]>(dst) as *const [V::View<'a>])
        };
        (f, v)
    }

    #[inline]
    fn values_to_flat<'a>(
        values: Self::Values<'a>,
        _scratch: &mut Self::OutScratch,
        flags: &mut FlatWrite<bool>,
        ptrs: &mut FlatWrite<*const ()>,
    ) {
        let (f, v) = values;
        debug_assert!(f.len() == v.len(), "ViewPorts planes disagree on length");
        flags.extend(f);
        for view in v {
            ptrs.push(std::ptr::from_ref(view).cast());
        }
    }

    #[inline]
    fn values_to_vecs<'a>(
        values: Self::Values<'a>,
        _scratch: &mut Self::OutScratch,
        shape: &mut Vec<usize>,
        flags: &mut Vec<bool>,
        ptrs: &mut Vec<*const ()>,
    ) {
        let (f, v) = values;
        debug_assert!(f.len() == v.len(), "ViewPorts planes disagree on length");
        shape.push(v.len());
        flags.extend_from_slice(f);
        for view in v {
            ptrs.push(std::ptr::from_ref(view).cast());
        }
    }

    #[inline]
    fn any_notify(values: &Self::Values<'_>) -> bool {
        values.0.iter().any(|&n| n)
    }
}

pub type Ports<T> = ViewPorts<Scalar<T>>;

// -- Reference leaf: RefViewPort<V> -----------------------------------------

// `V::View: Sync` is the engine's one sharing requirement: a producer's two
// consumers may run concurrently on different workers, both dereferencing the
// `&V::View` into the producer's storage.
//
// SAFETY: one flat slot tagged `TypeId::of::<RefViewPort<V>>()`; the carried
// `&V::View` IS a thin pointer, so serialization stores it directly (zero
// scratch) and deserialization re-borrows it at the same type. Covariant via
// the [`ValueView`] contract.
unsafe impl<V: ValueView> Interface for RefViewPort<V>
where
    for<'a> V::View<'a>: Sync,
{
    type Values<'a> = (bool, &'a V::View<'a>);
    type InScratch = ();
    type OutScratch = ();

    #[inline]
    fn flat_len(_shape: &mut FlatRead<usize>) -> usize {
        1
    }

    #[inline]
    fn type_ids_to_vec(_shape: &mut FlatRead<usize>, writer: &mut Vec<TypeId>) {
        writer.push(TypeId::of::<RefViewPort<V>>());
    }

    #[inline]
    fn new_in_scratch(_shape: &mut FlatRead<usize>) -> Self::InScratch {}

    #[inline]
    fn new_out_scratch() -> Self::OutScratch {}

    #[inline]
    unsafe fn values_from_flat<'a>(
        _shape: &mut FlatRead<'a, usize>,
        flags: &mut FlatRead<'a, bool>,
        ptrs: &mut FlatRead<'a, *const ()>,
        _scratch: &'a mut Self::InScratch,
    ) -> (bool, &'a V::View<'a>) {
        (*flags.pop(), unsafe {
            ptrs.pop().cast::<V::View<'a>>().as_ref_unchecked()
        })
    }

    #[inline]
    fn values_to_flat<'a>(
        values: Self::Values<'a>,
        _scratch: &mut Self::OutScratch,
        flags: &mut FlatWrite<bool>,
        ptrs: &mut FlatWrite<*const ()>,
    ) {
        let (n, v) = values;
        flags.push(n);
        ptrs.push((v as *const V::View<'a>).cast());
    }

    #[inline]
    fn values_to_vecs<'a>(
        values: Self::Values<'a>,
        _scratch: &mut Self::OutScratch,
        _shape: &mut Vec<usize>,
        flags: &mut Vec<bool>,
        ptrs: &mut Vec<*const ()>,
    ) {
        let (n, v) = values;
        flags.push(n);
        ptrs.push((v as *const V::View<'a>).cast());
    }

    #[inline]
    fn any_notify(values: &Self::Values<'_>) -> bool {
        values.0
    }
}

pub type RefPort<T> = RefViewPort<Scalar<T>>;

// -- Reference leaves: RefViewPorts<V> --------------------------------------

// SAFETY: `*shape.pop()` flat slots, each tagged `TypeId::of::<RefViewPort<V>>()`
// (matching what a single `RefViewPort<V>` consumer expects), so a group wires
// against scalar producers. Its payload IS the two wire planes re-typed in
// place -- the layout-identity of `&V::View` and a thin pointer makes the cast
// in `values_from_flat` well-defined; covariant via the [`ValueView`] contract.
unsafe impl<V: ValueView> Interface for RefViewPorts<V>
where
    for<'a> V::View<'a>: Sync,
{
    type Values<'a> = (&'a [bool], &'a [&'a V::View<'a>]);
    type InScratch = ();
    type OutScratch = ();

    #[inline]
    fn flat_len(shape: &mut FlatRead<usize>) -> usize {
        *shape.pop()
    }

    #[inline]
    fn type_ids_to_vec(shape: &mut FlatRead<usize>, writer: &mut Vec<TypeId>) {
        let n = *shape.pop();
        writer.extend(std::iter::repeat_n(TypeId::of::<RefViewPort<V>>(), n));
    }

    #[inline]
    fn new_in_scratch(shape: &mut FlatRead<usize>) -> Self::InScratch {
        let _ = shape.pop();
    }

    #[inline]
    fn new_out_scratch() -> Self::OutScratch {}

    #[inline]
    unsafe fn values_from_flat<'a>(
        shape: &mut FlatRead<'a, usize>,
        flags: &mut FlatRead<'a, bool>,
        ptrs: &mut FlatRead<'a, *const ()>,
        _scratch: &'a mut Self::InScratch,
    ) -> Self::Values<'a> {
        let n = *shape.pop();
        let f = flags.take(n);
        let p = ptrs.take(n);
        // SAFETY: well-defined cast -- `&'a V::View<'a>` is guaranteed the same
        // layout as a (thin) raw pointer, so `[*const ()]` and `[&'a V::View]`
        // are layout-identical -- and every pointer targets a valid `V::View`
        // for `'a` by the caller's contract. This is why `RefViewPorts` is a *leaf*:
        // its payload IS the wire planes, re-typed.
        (f, unsafe {
            &*(std::ptr::from_ref(p) as *const [&'a V::View<'a>])
        })
    }

    #[inline]
    fn values_to_flat<'a>(
        values: Self::Values<'a>,
        _scratch: &mut Self::OutScratch,
        flags: &mut FlatWrite<bool>,
        ptrs: &mut FlatWrite<*const ()>,
    ) {
        let (f, v) = values;
        debug_assert!(f.len() == v.len(), "RefViewPorts planes disagree on length");
        flags.extend(f);
        // SAFETY: the reverse of the deserialization cast (ref to pointer).
        ptrs.extend(unsafe { &*(std::ptr::from_ref(v) as *const [*const ()]) });
    }

    #[inline]
    fn values_to_vecs<'a>(
        values: Self::Values<'a>,
        _scratch: &mut Self::OutScratch,
        shape: &mut Vec<usize>,
        flags: &mut Vec<bool>,
        ptrs: &mut Vec<*const ()>,
    ) {
        let (f, v) = values;
        debug_assert!(f.len() == v.len(), "RefViewPorts planes disagree on length");
        shape.push(v.len());
        flags.extend_from_slice(f);
        // SAFETY: the reverse of the deserialization cast (ref to pointer).
        ptrs.extend_from_slice(unsafe { &*(std::ptr::from_ref(v) as *const [*const ()]) });
    }

    #[inline]
    fn any_notify(values: &Self::Values<'_>) -> bool {
        values.0.iter().any(|&n| n)
    }
}

pub type RefPorts<T> = RefViewPorts<Scalar<T>>;
