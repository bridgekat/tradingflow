use std::{any::TypeId, marker::PhantomData};

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

/// `T` by value: the wire carries the value itself (homed in the stored
/// output tree), not a reference into producer state -- so small-scalar
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
/// pointer targets the view's home in the engine-stored output tree;
/// deserialization copies it back out (hence `View: Copy`, `Send + Sync` for
/// the cross-worker tree storage -- the bounds [`ValueView`] omits).
/// Payload type: `(bool, V::View<'_>)`.
pub struct ViewPort<V>(PhantomData<V>);

/// A single by-reference leaf over a [`ValueView`] `V`: passes `&'a V::View<'a>`
/// (the value lives in the producer's state or arena). `RefViewPort<Scalar<T>>` is the
/// plain `&T` port (value `T` in state); [`ViewPort`] is the by-value sibling.
pub struct RefViewPort<V>(PhantomData<V>);

/// A runtime-length group of [`RefViewPort`]-leaves over `V`: a fan-in of N producers,
/// payload `(&[bool], &[&V::View<'a>])`.
pub struct RefViewPorts<V>(PhantomData<V>);

/// Recursive description of a segment's inputs or outputs: a tree of [`RefViewPort`],
/// [`RefViewPorts`] and [`ViewPort`] leaves over [`ValueView`]s.
///
/// Each leaf's payload is `(notify, value)` -- `(bool, &V::View<'a>)` for a
/// [`RefViewPort`], parallel slices `(&[bool], &[&V::View<'a>])` for [`RefViewPorts`], and
/// `(bool, V::View<'a>)` (by value) for [`ViewPort`]. The flat wire form the
/// core moves between nodes is two index-aligned planes: a `bool` notify plane
/// and a `*const ()` value-pointer plane.
///
/// When the tree contains a [`RefViewPorts`], the flat layout is ambiguous on its own,
/// so the per-`RefViewPorts` element counts are carried out-of-band as a **shape**: a
/// flat `[usize]` that is the pre-order serialization of the count tree (each
/// `RefViewPorts` contributes its count). The shape is built from the input handles /
/// `init` values and replayed via a [`FlatRead<usize>`] cursor threaded
/// alongside the plane cursors.
///
/// # Safety
///
/// The engine erases the payload tree to `'static` storage and reconstructs it
/// by re-typing `*const ()` wire pointers, trusting this impl's methods to be
/// mutually consistent. The cross-node [`TypeId`] check validates agreement
/// *between* nodes, not the internal coherence of one impl, so an implementor
/// must guarantee:
///
/// * `Values<'a>` is covariant in `'a` (it is read back at a shorter lifetime
///   than the `'static` it is stored at).
/// * For any given `shape`, [`flat_len`](Self::flat_len),
///   [`type_ids_to_vec`](Self::type_ids_to_vec),
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
///   at a stable address, for as long as a consumer may dereference it.
pub unsafe trait Interface {
    /// Nested `(notify, value)` payload tree. `Copy + Send`, so it threads
    /// freely through fused bodies and the engine can store each node's
    /// output tree by value (see [`values_to_flat`](Self::values_to_flat)).
    type Values<'a>: Copy + Send + 'a;

    /// Number of flat leaf slots this node spans, advancing `shape` past it.
    ///
    /// Dynamic generalization of a static arity: fixed-shape nodes ignore
    /// `shape`; a [`RefViewPorts`] pops and returns its count.
    fn flat_len(shape: &mut FlatRead<usize>) -> usize;

    /// Write the [`TypeId`] of each flat leaf into `writer` in tree-order,
    /// consuming `shape` at each [`RefViewPorts`].
    fn type_ids_to_vec(shape: &mut FlatRead<usize>, writer: &mut Vec<TypeId>);

    /// Construct the nested payload tree by consuming the two parallel wire
    /// planes (`flags` + value pointers `ptrs`), using `shape` to size each
    /// [`RefViewPorts`] (whose payload is the consumed sub-slices, the pointer plane
    /// re-typed in place).
    ///
    /// # Safety
    ///
    /// Each consumed pointer must point to a valid value of the matching type,
    /// and `shape` must be the shape that produced this layout.
    unsafe fn values_from_flat<'a>(
        shape: &mut FlatRead<'a, usize>,
        flags: &mut FlatRead<'a, bool>,
        ptrs: &mut FlatRead<'a, *const ()>,
    ) -> Self::Values<'a>;

    /// Serialize the output tree onto the wire planes. `values` points at the
    /// engine-stored copy of the tree, which outlives the call: thin leaves
    /// write the reference they carry, while a fat leaf ([`ViewPort`]) writes a
    /// thin pointer INTO the stored tree itself -- the field keeps a stable
    /// address until the node's next run overwrites the tree in place.
    fn values_to_flat<'a>(
        values: &'a Self::Values<'a>,
        flags: &mut FlatWrite<bool>,
        ptrs: &mut FlatWrite<*const ()>,
    );

    /// [`values_to_flat`](Self::values_to_flat) into growable vecs, recording
    /// the shape (the build-call path).
    fn values_to_vecs<'a>(
        values: &'a Self::Values<'a>,
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

    #[inline]
    fn flat_len(_shape: &mut FlatRead<usize>) -> usize {
        0
    }

    #[inline]
    fn type_ids_to_vec(_shape: &mut FlatRead<usize>, _writer: &mut Vec<TypeId>) {}

    #[inline]
    unsafe fn values_from_flat<'a>(
        _shape: &mut FlatRead<'a, usize>,
        _flags: &mut FlatRead<'a, bool>,
        _ptrs: &mut FlatRead<'a, *const ()>,
    ) {
    }

    #[inline]
    fn values_to_flat<'a>(
        _values: &'a Self::Values<'a>,
        _flags: &mut FlatWrite<bool>,
        _ptrs: &mut FlatWrite<*const ()>,
    ) {
    }

    #[inline]
    fn values_to_vecs<'a>(
        _values: &'a Self::Values<'a>,
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
        // order; each method delegates to every child in that same order, so
        // consistency follows from the children's.
        unsafe impl<$($T: Interface,)+> Interface for ($($T,)+) {
            type Values<'a> = ($($T::Values<'a>,)+);

            #[inline]
            fn flat_len(shape: &mut FlatRead<usize>) -> usize {
                0 $(+ $T::flat_len(shape))+
            }

            #[inline]
            fn type_ids_to_vec(shape: &mut FlatRead<usize>, writer: &mut Vec<TypeId>) {
                $( $T::type_ids_to_vec(shape, writer); )+
            }

            #[inline]
            unsafe fn values_from_flat<'a>(
                shape: &mut FlatRead<'a, usize>,
                flags: &mut FlatRead<'a, bool>,
                ptrs: &mut FlatRead<'a, *const ()>,
            ) -> Self::Values<'a> {
                ( $( unsafe { $T::values_from_flat(shape, flags, ptrs) }, )+ )
            }

            #[inline]
            fn values_to_flat<'a>(
                values: &'a Self::Values<'a>,
                flags: &mut FlatWrite<bool>,
                ptrs: &mut FlatWrite<*const ()>,
            ) {
                $( $T::values_to_flat(&values.$idx, flags, ptrs); )+
            }

            #[inline]
            fn values_to_vecs<'a>(
                values: &'a Self::Values<'a>,
                shape: &mut Vec<usize>,
                flags: &mut Vec<bool>,
                ptrs: &mut Vec<*const ()>,
            ) {
                $( $T::values_to_vecs(&values.$idx, shape, flags, ptrs); )+
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
// carried by value, so serialization writes a pointer into the stored tree and
// deserialization reads it back at the same `V::View` type. `Values<'a>` is
// covariant via the [`ValueView`] covariance contract.
unsafe impl<V: ValueView> Interface for ViewPort<V>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Values<'a> = (bool, V::View<'a>);

    #[inline]
    fn flat_len(_shape: &mut FlatRead<usize>) -> usize {
        1
    }

    #[inline]
    fn type_ids_to_vec(_shape: &mut FlatRead<usize>, writer: &mut Vec<TypeId>) {
        writer.push(TypeId::of::<ViewPort<V>>());
    }

    #[inline]
    unsafe fn values_from_flat<'a>(
        _shape: &mut FlatRead<'a, usize>,
        flags: &mut FlatRead<'a, bool>,
        ptrs: &mut FlatRead<'a, *const ()>,
    ) -> Self::Values<'a> {
        // Copy the view out of the producer's stored tree: a forwarding
        // consumer re-homes it in its OWN tree, so no slot ever points into
        // another node transitively.
        (*flags.pop(), unsafe {
            ptrs.pop().cast::<V::View<'a>>().read()
        })
    }

    #[inline]
    fn values_to_flat<'a>(
        values: &'a Self::Values<'a>,
        flags: &mut FlatWrite<bool>,
        ptrs: &mut FlatWrite<*const ()>,
    ) {
        flags.push(values.0);
        ptrs.push(std::ptr::from_ref(&values.1).cast());
    }

    #[inline]
    fn values_to_vecs<'a>(
        values: &'a Self::Values<'a>,
        _shape: &mut Vec<usize>,
        flags: &mut Vec<bool>,
        ptrs: &mut Vec<*const ()>,
    ) {
        flags.push(values.0);
        ptrs.push(std::ptr::from_ref(&values.1).cast());
    }

    #[inline]
    fn any_notify(values: &Self::Values<'_>) -> bool {
        values.0
    }
}

pub type Port<T> = ViewPort<Scalar<T>>;

// -- Reference leaf: RefViewPort<V> -----------------------------------------

// `V::View: Sync` is the engine's one sharing requirement: a producer's two
// consumers may run concurrently on different workers, both dereferencing the
// `&V::View` into the producer's storage.
//
// SAFETY: one flat slot tagged `TypeId::of::<RefViewPort<V>>()`; the carried
// `&V::View` IS a thin pointer, so serialization stores it and deserialization
// re-borrows it at the same type. Covariant via the [`ValueView`] contract.
unsafe impl<V: ValueView> Interface for RefViewPort<V>
where
    for<'a> V::View<'a>: Sync,
{
    type Values<'a> = (bool, &'a V::View<'a>);

    #[inline]
    fn flat_len(_shape: &mut FlatRead<usize>) -> usize {
        1
    }

    #[inline]
    fn type_ids_to_vec(_shape: &mut FlatRead<usize>, writer: &mut Vec<TypeId>) {
        writer.push(TypeId::of::<RefViewPort<V>>());
    }

    #[inline]
    unsafe fn values_from_flat<'a>(
        _shape: &mut FlatRead<'a, usize>,
        flags: &mut FlatRead<'a, bool>,
        ptrs: &mut FlatRead<'a, *const ()>,
    ) -> (bool, &'a V::View<'a>) {
        (*flags.pop(), unsafe {
            ptrs.pop().cast::<V::View<'a>>().as_ref_unchecked()
        })
    }

    #[inline]
    fn values_to_flat<'a>(
        values: &'a Self::Values<'a>,
        flags: &mut FlatWrite<bool>,
        ptrs: &mut FlatWrite<*const ()>,
    ) {
        let (n, v) = *values;
        flags.push(n);
        ptrs.push((v as *const V::View<'a>).cast());
    }

    #[inline]
    fn values_to_vecs<'a>(
        values: &'a Self::Values<'a>,
        _shape: &mut Vec<usize>,
        flags: &mut Vec<bool>,
        ptrs: &mut Vec<*const ()>,
    ) {
        let (n, v) = *values;
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
    unsafe fn values_from_flat<'a>(
        shape: &mut FlatRead<'a, usize>,
        flags: &mut FlatRead<'a, bool>,
        ptrs: &mut FlatRead<'a, *const ()>,
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
        values: &'a Self::Values<'a>,
        flags: &mut FlatWrite<bool>,
        ptrs: &mut FlatWrite<*const ()>,
    ) {
        let (f, v) = *values;
        debug_assert!(f.len() == v.len(), "RefViewPorts planes disagree on length");
        flags.extend(f);
        // SAFETY: the reverse of the deserialization cast (ref to pointer).
        ptrs.extend(unsafe { &*(std::ptr::from_ref(v) as *const [*const ()]) });
    }

    #[inline]
    fn values_to_vecs<'a>(
        values: &'a Self::Values<'a>,
        shape: &mut Vec<usize>,
        flags: &mut Vec<bool>,
        ptrs: &mut Vec<*const ()>,
    ) {
        let (f, v) = *values;
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
