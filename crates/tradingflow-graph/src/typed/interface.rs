use std::{any::TypeId, marker::PhantomData, mem::MaybeUninit};

use super::{FlatRead, FlatWrite};

/// Marks a *way to pass a value through a port*: a `'static` name `V` for a
/// payload view `V::View<'a>`, together with the view's owned form
/// [`Owned`](Self::Owned) and how a view is [borrowed](Self::borrow) from it.
/// `Scalar<T>` passes `T` (a non-borrowing view is its own owned form, and its
/// "borrow" is a clone); a user `Array<T>` marker might pass a genuinely
/// borrowing `ArrayView<'a, T>` owned by an `Array<T>`. `View` carries no
/// bounds here (only `'a`) -- the leaves that use it add their own (e.g.
/// [`Port`] needs `View: Copy + Send + Sync`).
///
/// # Safety
///
/// * `View<'a>` must be covariant in `'a`.
pub unsafe trait Pass: 'static {
    /// The owned form: what a source node stores in state.
    type Owned: Send + Sync + 'static;

    /// The view: what travels through ports.
    type View<'a>: 'a;

    /// Borrow a view from the owned form -- an actual borrow for a borrowing
    /// policy, a clone for a non-borrowing one.
    fn borrow(owned: &Self::Owned) -> Self::View<'_>;
}

/// Pass `T` by value.
pub struct Val<T>(PhantomData<T>);

// SAFETY: `T: 'static` so it is trivially covariant in `'a`.
unsafe impl<T: Clone + Send + Sync + 'static> Pass for Val<T> {
    type View<'a> = T;
    type Owned = T;

    #[inline(always)]
    fn borrow(owned: &T) -> T {
        owned.clone()
    }
}

/// Pass `T` by reference.
pub struct Ref<T>(PhantomData<T>);

// SAFETY: `&'a T` is covariant in `'a` (`T: 'static`).
unsafe impl<T: Send + Sync + 'static> Pass for Ref<T> {
    type View<'a> = &'a T;
    type Owned = T;

    #[inline(always)]
    fn borrow(owned: &T) -> &T {
        owned
    }
}

/// Pass `Vec<T>` by `&[T]`.
pub struct Slice<T>(PhantomData<T>);

// SAFETY: `T: 'static` so `&'a [T]` is covariant in `'a`.
unsafe impl<T: Send + Sync + 'static> Pass for Slice<T> {
    type View<'a> = &'a [T];
    type Owned = Vec<T>;

    #[inline(always)]
    fn borrow(owned: &Vec<T>) -> &[T] {
        owned
    }
}

/// Marks a single leaf in an [`Interface`] tree.
pub struct Port<V>(PhantomData<fn() -> V>);

/// Marks a variadic leaf in an [`Interface`] tree.
pub struct Ports<V>(PhantomData<fn() -> V>);

/// Recursive description of a segment's inputs or outputs: a tree of
/// [`Port`] and [`Ports`] leaves.
///
/// # Safety
///
/// The engine reconstructs payloads by re-typing `*const ()` wire pointers,
/// trusting this impl's methods to be mutually consistent. The cross-node
/// [`TypeId`] check validates agreement *between* nodes, not the internal
/// coherence of one impl, so an implementor must guarantee:
///
/// * [`Self::Values<'a>`] is covariant in `'a` (payloads and scratch contents
///   are stored `'static`-erased and read back at a shorter lifetime).
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
    /// storage ([`Ports`]) declares it here and lends its payload from it;
    /// zero-copy leaves declare `()`. Typed at `'static` purely as storage --
    /// layout is lifetime-invariant and every access re-types it at the
    /// calling generation's lifetime.
    type InScratch: Send + 'static;

    /// Node-boundary *serialization* buffer, living in the node's state cell:
    /// a [`Port`] homes its view here and points the wire slot at it; a
    /// [`Ports`] output declares `()` (its payload slice is already the
    /// views' contiguous home). Same `'static`-as-storage convention as
    /// [`InScratch`](Self::InScratch).
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
    /// variadic leaf. A [`Port`] copies its view back out of the pointed-to
    /// home; the gathering [`Ports`] copies the pointed-to views into
    /// `scratch` (overwriting last generation's in place) and lends the
    /// payload from there.
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

    /// Serialize the payload tree onto the wire planes. A [`Port`] leaf
    /// homes its view into `scratch` -- overwriting last generation's in place
    /// -- and writes a thin pointer to that home, which keeps a stable address
    /// until the node's next run; a [`Ports`] output writes each element's
    /// address directly (the payload slice is the home).
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

// -- Value leaf: Port<V> ------------------------------------------------

// SAFETY: one flat slot tagged `TypeId::of::<V>()`; the value is
// carried by value, so serialization homes it in the one-view `OutScratch`
// (overwritten in place each run; the `'static` typing is storage-only, per
// the [`Value`] covariance contract, and views are `Copy` so the
// overwrite drops nothing) and points the wire slot there; deserialization
// reads it back at the same `V::View` type.
unsafe impl<V: Pass> Interface for Port<V>
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
        writer.push(TypeId::of::<V>());
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
        // by the [`Value`] covariance contract; consumers re-type the slot
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

// -- Value leaves: Ports<V> ----------------------------------------------

// SAFETY: `*shape.pop()` flat slots, each tagged `TypeId::of::<V>()`
// (matching what a single `Port<V>` consumer expects and what a single
// `Port<V>` producer emits), so a group wires against by-value producers.
// Deserialization gathers the N pointed-to views into the `InScratch` buffer
// (sized once from the shape at build, overwritten in place each run; its
// `'static` typing is storage-only, per the [`Value`] covariance contract,
// and views are `Copy`, so in-place overwrites drop nothing) and lends the
// payload slice from there. Serialization needs no scratch: the payload slice
// itself is the contiguous home of the N views — element `i`'s thin pointer
// is just `&values.1[i]` — and the producer keeps that storage stable for the
// generation (state / arena / its own `InScratch`).
unsafe impl<V: Pass> Interface for Ports<V>
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
        writer.extend(std::iter::repeat_n(TypeId::of::<V>(), n));
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
        debug_assert!(dst.len() == n, "Ports scratch disagrees with shape");
        for (slot, &ptr) in dst.iter_mut().zip(p) {
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
        debug_assert!(f.len() == v.len(), "Ports planes disagree on length");
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
        debug_assert!(f.len() == v.len(), "Ports planes disagree on length");
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
