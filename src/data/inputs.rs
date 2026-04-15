//! Recursive description of operator inputs.
//!
//! The core trait [`InputTypes`] describes an operator's `Inputs` as a tree
//! whose leaves are [`Input<T>`] for some value type `T: Send + 'static`, and
//! whose branches are tuples (heterogeneous, arities 1–12) or [`Slice<T>`]
//! (homogeneous, dynamic length).  The trait provides the bidirectional
//! mapping between the flat-per-position representation used by the graph
//! runtime and the nested structural view used by operator authors.
//!
//! # Key types
//!
//! * [`Input<T>`] — leaf marker wrapping any `T: Send + 'static`.
//! * [`Slice<T>`] — sized marker for a homogeneous dynamic-length branch.
//! * [`SliceShape<T>`] — runtime shape of a slice branch.
//! * [`SliceRefs<'a, T>`] / [`SliceProduced<'a, T>`] — zero-allocation
//!   views produced by [`InputTypes::refs_from_flat`] and
//!   [`produced_from_flat`](InputTypes::produced_from_flat).
//! * [`FlatRead`] / [`FlatWrite`] — single-pass cursors over flat buffers.
//!   Threaded through the recursive traversals so every leaf is visited
//!   exactly once (O(total_leaves) regardless of nesting depth).
//! * [`FlatShapeFromArity`] — helper for constructing flat shapes from just
//!   the arity, used by type-erased registration paths (the Python bridge).

use std::any::TypeId;
use std::marker::PhantomData;

// ===========================================================================
// Cursor-style flat readers and writers
// ===========================================================================

/// Single-pass cursor over a flat slice `&'a [T]`.
///
/// Used by [`InputTypes::refs_from_flat`] and
/// [`InputTypes::produced_from_flat`] to thread position state through a
/// recursive tree traversal without recomputing sub-tree arities at each
/// level — every leaf consumes exactly one slot, every slice branch takes
/// a precomputed count of slots.  Total traversal cost is O(total_leaves).
pub struct FlatRead<'a, T> {
    buf: &'a [T],
    idx: usize,
}

impl<'a, T> FlatRead<'a, T> {
    /// Wrap a slice; cursor starts at position 0.
    #[inline]
    pub fn new(buf: &'a [T]) -> Self {
        Self { buf, idx: 0 }
    }

    /// Number of elements remaining past the cursor.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.idx
    }

    /// Consume `n` elements from the cursor, returning a sub-slice with the
    /// outer `'a` lifetime.
    ///
    /// # Panics
    ///
    /// Panics if `n > self.remaining()`.
    #[inline]
    pub fn take(&mut self, n: usize) -> &'a [T] {
        let slice = &self.buf[self.idx..self.idx + n];
        self.idx += n;
        slice
    }
}

impl<'a, T: Copy> FlatRead<'a, T> {
    /// Consume one element, returning a copy.
    #[inline]
    pub fn next(&mut self) -> T {
        let v = self.buf[self.idx];
        self.idx += 1;
        v
    }
}

/// Single-pass cursor over a flat mutable slice `&'a mut [T]`.
///
/// Used by [`InputTypes::write_type_ids`] and
/// [`InputTypesHandles::write_node_indices`](crate::scenario::InputTypesHandles::write_node_indices)
/// to fill a caller-allocated buffer in tree-order without recomputing
/// sub-tree arities.  Total write cost is O(total_leaves).
pub struct FlatWrite<'a, T> {
    buf: &'a mut [T],
    idx: usize,
}

impl<'a, T> FlatWrite<'a, T> {
    /// Wrap a mutable slice; cursor starts at position 0.
    #[inline]
    pub fn new(buf: &'a mut [T]) -> Self {
        Self { buf, idx: 0 }
    }

    /// Number of elements remaining past the cursor.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.idx
    }

    /// Write `v` at the cursor, then advance by 1.
    ///
    /// # Panics
    ///
    /// Panics if the cursor is at end.
    #[inline]
    pub fn push(&mut self, v: T) {
        self.buf[self.idx] = v;
        self.idx += 1;
    }
}

// ===========================================================================
// InputTypes
// ===========================================================================

/// Zero-sized marker wrapping a value type in an [`InputTypes`] tree.
///
/// Every leaf of an operator's `Inputs` is `Input<T>` for some value type
/// `T: Send + 'static`.  The wrapper exists only at the type level:
/// `Input<T>::Refs<'a> = &'a T`, so compute bodies see `&T` directly with
/// no indirection.
///
/// The wrapper is what separates tree branches (tuples, slices) from tree
/// leaves (any `T`) without trait coherence conflicts: `Input<_>`, `(_, _)`,
/// and [`Slice<_>`] are distinct type constructors, so their `InputTypes`
/// impls do not overlap.
pub struct Input<T: Send + 'static>(PhantomData<fn() -> T>);

/// Recursive description of an operator's inputs.
///
/// An operator's `Inputs` is a tree whose leaves are [`Input<T>`] for some
/// value type `T`, and whose branches are tuples (heterogeneous, arities
/// 1–12) or [`Slice<T>`] (homogeneous, dynamic length).  This trait
/// provides the bidirectional mapping between the flat-per-position
/// representation used by the graph runtime and the nested structural view
/// used by operator authors.
///
/// # Traversal complexity
///
/// All cursor-style methods ([`write_type_ids`](Self::write_type_ids),
/// [`refs_from_flat`](Self::refs_from_flat),
/// [`produced_from_flat`](Self::produced_from_flat)) visit each leaf
/// exactly once and never recompute sub-tree arities during traversal —
/// total cost is **O(total_leaves)**, linear regardless of nesting depth.
///
/// [`arity`](Self::arity) is kept because callers need to pre-size the
/// flat buffer (a one-shot traversal at operator registration, not a
/// hot-path operation).
///
/// # Associated types
///
/// * [`Shape`](InputTypes::Shape) — runtime description of the tree
///   structure (e.g. slice lengths).  Required to be `Send + 'static` so
///   it can be stored inside an
///   [`ErasedOperator`](crate::operator::ErasedOperator) as
///   `Box<dyn Any + Send>` and downcast in the monomorphized compute path.
/// * [`Refs<'a>`](InputTypes::Refs) — nested immutable references
///   mirroring the tree.  Leaves expose `&'a T`; tuples expose tuples of
///   child refs; [`Slice<T>`] branches expose a lazy [`SliceRefs`] view.
///   Zero-allocation.
/// * [`Produced<'a>`](InputTypes::Produced) — nested boolean view
///   indicating which leaves produced new output in the current flush
///   cycle.
///
/// # Implementations
///
/// * [`Input<T>`] — every leaf of type `T: Send + 'static`.
/// * `()` — empty input tree (arity 0).
/// * Tuples of arity 1–12 — heterogeneous branches.
/// * [`Slice<T>`] — homogeneous dynamic-length branch.
pub trait InputTypes {
    /// Runtime shape description (e.g. slice lengths and nested shapes).
    type Shape: Send + 'static;

    /// Nested immutable references into the flat pointer buffer.
    type Refs<'a>;

    /// Nested boolean view mirroring [`Refs`](Self::Refs).
    type Produced<'a>;

    /// Total number of flat leaf positions described by `shape`.
    ///
    /// Used once at registration time to pre-size the flat buffer
    /// consumed by [`write_type_ids`](Self::write_type_ids).
    fn arity(shape: &Self::Shape) -> usize;

    /// Write the [`TypeId`] of each flat leaf into `writer` in tree-order.
    ///
    /// Advances the cursor by exactly [`arity(shape)`](Self::arity) slots.
    fn write_type_ids(shape: &Self::Shape, writer: &mut FlatWrite<TypeId>);

    /// Construct the nested reference view by consuming the next
    /// [`arity(shape)`](Self::arity) slots from `reader`.
    ///
    /// # Safety
    ///
    /// The next [`arity(shape)`](Self::arity) pointers in `reader` must
    /// each point to a valid value whose type matches the corresponding
    /// [`TypeId`] written by [`write_type_ids`](Self::write_type_ids).
    unsafe fn refs_from_flat<'a>(
        reader: &mut FlatRead<'a, *const u8>,
        shape: &'a Self::Shape,
    ) -> Self::Refs<'a>;

    /// Construct the nested boolean view by consuming the next
    /// [`arity(shape)`](Self::arity) slots from `reader`.
    fn produced_from_flat<'a>(
        reader: &mut FlatRead<'a, bool>,
        shape: &'a Self::Shape,
    ) -> Self::Produced<'a>;
}

// -- Leaf: Input<T> ----------------------------------------------------------

impl<T: Send + 'static> InputTypes for Input<T> {
    type Shape = ();
    type Refs<'a> = &'a T;
    type Produced<'a> = bool;

    #[inline(always)]
    fn arity(_: &()) -> usize {
        1
    }

    #[inline(always)]
    fn write_type_ids(_: &(), writer: &mut FlatWrite<TypeId>) {
        writer.push(TypeId::of::<T>());
    }

    #[inline(always)]
    unsafe fn refs_from_flat<'a>(reader: &mut FlatRead<'a, *const u8>, _: &()) -> &'a T {
        unsafe { &*(reader.next() as *const T) }
    }

    #[inline(always)]
    fn produced_from_flat<'a>(reader: &mut FlatRead<'a, bool>, _: &()) -> bool {
        reader.next()
    }
}

// -- Compound: empty tuple (arity 0) ----------------------------------------

impl InputTypes for () {
    type Shape = ();
    type Refs<'a> = ();
    type Produced<'a> = ();

    #[inline(always)]
    fn arity(_: &()) -> usize {
        0
    }

    #[inline(always)]
    fn write_type_ids(_: &(), _writer: &mut FlatWrite<TypeId>) {}

    #[inline(always)]
    unsafe fn refs_from_flat<'a>(_reader: &mut FlatRead<'a, *const u8>, _: &()) -> () {}

    #[inline(always)]
    fn produced_from_flat<'a>(_reader: &mut FlatRead<'a, bool>, _: &()) -> () {}
}

// -- Compound: tuple branches (arities 1-12) ---------------------------------

macro_rules! impl_input_types_for_tuple {
    ($($idx:tt: $T:ident),+ $(,)?) => {
        impl<$($T: InputTypes),+> InputTypes for ($($T,)+) {
            type Shape = ($($T::Shape,)+);
            type Refs<'a> = ($($T::Refs<'a>,)+);
            type Produced<'a> = ($($T::Produced<'a>,)+);

            #[inline]
            fn arity(shape: &Self::Shape) -> usize {
                0 $(+ <$T as InputTypes>::arity(&shape.$idx))+
            }

            #[inline]
            fn write_type_ids(shape: &Self::Shape, writer: &mut FlatWrite<TypeId>) {
                $( <$T as InputTypes>::write_type_ids(&shape.$idx, writer); )+
            }

            #[inline]
            unsafe fn refs_from_flat<'a>(
                reader: &mut FlatRead<'a, *const u8>,
                shape: &'a Self::Shape,
            ) -> Self::Refs<'a> {
                ($(
                    unsafe { <$T as InputTypes>::refs_from_flat(reader, &shape.$idx) },
                )+)
            }

            #[inline]
            fn produced_from_flat<'a>(
                reader: &mut FlatRead<'a, bool>,
                shape: &'a Self::Shape,
            ) -> Self::Produced<'a> {
                ($(
                    <$T as InputTypes>::produced_from_flat(reader, &shape.$idx),
                )+)
            }
        }
    };
}

impl_input_types_for_tuple!(0: A);
impl_input_types_for_tuple!(0: A, 1: B);
impl_input_types_for_tuple!(0: A, 1: B, 2: C);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_input_types_for_tuple!(
    0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L
);

// -- Compound: slice branch (Slice<T>) --------------------------------------

/// Runtime shape of a slice branch [`Slice<T>`] in an [`InputTypes`] tree.
///
/// Stores one `T::Shape` per slice element and a prefix-sum table of child
/// arities (length `elems.len() + 1`, last entry is
/// [`total_arity`](Self::total_arity)).  The offsets table lets
/// [`SliceRefs::get`] slice the flat pointer buffer in O(1) per element.
pub struct SliceShape<T: InputTypes> {
    elems: Box<[T::Shape]>,
    offsets: Box<[usize]>,
}

impl<T: InputTypes> SliceShape<T> {
    /// Construct a slice shape from per-element child shapes.
    ///
    /// Prefix sums are computed once here and reused for the lifetime of
    /// the operator.
    pub fn new(elems: Box<[T::Shape]>) -> Self {
        let mut offsets = Vec::with_capacity(elems.len() + 1);
        offsets.push(0);
        let mut cum = 0;
        for shape in elems.iter() {
            cum += T::arity(shape);
            offsets.push(cum);
        }
        Self {
            elems,
            offsets: offsets.into_boxed_slice(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.elems.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elems.is_empty()
    }

    /// Total flat arity of the slice — the sum of child arities (O(1),
    /// read from the cached prefix-sum table).
    #[inline]
    pub fn total_arity(&self) -> usize {
        *self.offsets.last().unwrap_or(&0)
    }
}

/// Zero-allocation nested reference view over a slice branch.
///
/// Produced by [`InputTypes::refs_from_flat`] on [`Slice<T>`] and accessed
/// via [`get`](Self::get) (O(1) per element) or [`iter`](Self::iter).
pub struct SliceRefs<'a, T: InputTypes> {
    ptrs: &'a [*const u8],
    shape: &'a SliceShape<T>,
}

impl<'a, T: InputTypes> SliceRefs<'a, T> {
    /// Construct a view from a flat pointer buffer and shape.
    ///
    /// # Safety
    ///
    /// `ptrs.len()` must equal `shape.total_arity()`, and each pointer
    /// must point to a valid value whose type matches the corresponding
    /// leaf in `shape`.
    #[inline]
    pub unsafe fn new(ptrs: &'a [*const u8], shape: &'a SliceShape<T>) -> Self {
        Self { ptrs, shape }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }

    /// Return the nested `Refs` for slice element `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn get(&self, i: usize) -> T::Refs<'a> {
        let lo = self.shape.offsets[i];
        let hi = self.shape.offsets[i + 1];
        let mut reader = FlatRead::new(&self.ptrs[lo..hi]);
        unsafe { T::refs_from_flat(&mut reader, &self.shape.elems[i]) }
    }

    /// Iterator over every element's nested `Refs`.
    pub fn iter(&self) -> impl Iterator<Item = T::Refs<'a>> + '_ {
        let ptrs = self.ptrs;
        let shape = self.shape;
        (0..shape.len()).map(move |i| {
            let lo = shape.offsets[i];
            let hi = shape.offsets[i + 1];
            let mut reader = FlatRead::new(&ptrs[lo..hi]);
            unsafe { T::refs_from_flat(&mut reader, &shape.elems[i]) }
        })
    }
}

/// Zero-allocation nested boolean view over a slice branch.
pub struct SliceProduced<'a, T: InputTypes> {
    flat: &'a [bool],
    shape: &'a SliceShape<T>,
}

impl<'a, T: InputTypes> SliceProduced<'a, T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }

    /// Return the nested `Produced` for slice element `i`.
    #[inline]
    pub fn get(&self, i: usize) -> T::Produced<'a> {
        let lo = self.shape.offsets[i];
        let hi = self.shape.offsets[i + 1];
        let mut reader = FlatRead::new(&self.flat[lo..hi]);
        T::produced_from_flat(&mut reader, &self.shape.elems[i])
    }

    pub fn iter(&self) -> impl Iterator<Item = T::Produced<'a>> + '_ {
        let flat = self.flat;
        let shape = self.shape;
        (0..shape.len()).map(move |i| {
            let lo = shape.offsets[i];
            let hi = shape.offsets[i + 1];
            let mut reader = FlatRead::new(&flat[lo..hi]);
            T::produced_from_flat(&mut reader, &shape.elems[i])
        })
    }

    /// `true` iff any leaf under this slice produced.
    #[inline]
    pub fn any(&self) -> bool {
        self.flat.iter().any(|&b| b)
    }
}

/// Zero-sized marker wrapping a homogeneous dynamic-length branch in an
/// [`InputTypes`] tree.
///
/// `Slice<T>` is the sized counterpart of the bare `[T]` type: it carries
/// the same semantics — a dynamic-length branch of `T: InputTypes`
/// elements — but is always [`Sized`], which lets it appear in any tuple
/// position.  Rust's tuple rules forbid unsized (`[T]`) fields except in
/// the last position; `Slice<T>` has no such restriction.
///
/// The wrapper exists only at the type level.  `Slice<T>::Refs<'a> =
/// SliceRefs<'a, T>`.
pub struct Slice<T: InputTypes + 'static>(PhantomData<fn() -> T>);

/// Build a runtime shape from a flat leaf count for `InputTypes` trees
/// that have no internal structure beyond a single level of tuple or slice
/// branching over [`Input<T>`] leaves.
///
/// Implemented for:
///
/// * `()` — empty shape (arity must be 0).
/// * Tuples of unit shapes (`()`, `((),)`, `((), ())`, …) — arity must
///   equal the tuple length.
/// * [`SliceShape<Input<T>>`] — arity determines the number of slice
///   elements, all of whose child shapes are unit.
///
/// Used by the Python bridge to build shapes without compile-time
/// knowledge of the operator's `Inputs` structure.
pub trait FlatShapeFromArity: Sized {
    /// Construct `Self` for the given flat leaf arity.  Panics if arity
    /// is incompatible with `Self` (e.g. 3 for a 2-tuple shape).
    fn flat_shape_from_arity(arity: usize) -> Self;
}

impl FlatShapeFromArity for () {
    fn flat_shape_from_arity(arity: usize) -> Self {
        assert_eq!(arity, 0, "empty InputTypes shape requires arity 0");
    }
}

macro_rules! impl_flat_shape_from_arity_tuple {
    ($n:expr, $($idx:tt),+) => {
        impl FlatShapeFromArity for ( $( impl_flat_shape_from_arity_tuple!(@unit $idx), )+ ) {
            fn flat_shape_from_arity(arity: usize) -> Self {
                assert_eq!(arity, $n, "flat tuple shape requires arity {}", $n);
                ( $( impl_flat_shape_from_arity_tuple!(@unit_val $idx), )+ )
            }
        }
    };
    (@unit $idx:tt) => { () };
    (@unit_val $idx:tt) => { () };
}

impl_flat_shape_from_arity_tuple!(1, 0);
impl_flat_shape_from_arity_tuple!(2, 0, 1);
impl_flat_shape_from_arity_tuple!(3, 0, 1, 2);
impl_flat_shape_from_arity_tuple!(4, 0, 1, 2, 3);
impl_flat_shape_from_arity_tuple!(5, 0, 1, 2, 3, 4);
impl_flat_shape_from_arity_tuple!(6, 0, 1, 2, 3, 4, 5);
impl_flat_shape_from_arity_tuple!(7, 0, 1, 2, 3, 4, 5, 6);
impl_flat_shape_from_arity_tuple!(8, 0, 1, 2, 3, 4, 5, 6, 7);
impl_flat_shape_from_arity_tuple!(9, 0, 1, 2, 3, 4, 5, 6, 7, 8);
impl_flat_shape_from_arity_tuple!(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl_flat_shape_from_arity_tuple!(11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
impl_flat_shape_from_arity_tuple!(12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

impl<T: InputTypes<Shape = ()> + 'static> FlatShapeFromArity for SliceShape<T> {
    fn flat_shape_from_arity(arity: usize) -> Self {
        SliceShape::new(vec![(); arity].into_boxed_slice())
    }
}

impl<T: InputTypes + 'static> InputTypes for Slice<T> {
    type Shape = SliceShape<T>;
    type Refs<'a> = SliceRefs<'a, T>;
    type Produced<'a> = SliceProduced<'a, T>;

    #[inline]
    fn arity(shape: &Self::Shape) -> usize {
        shape.total_arity()
    }

    #[inline]
    fn write_type_ids(shape: &Self::Shape, writer: &mut FlatWrite<TypeId>) {
        for i in 0..shape.len() {
            T::write_type_ids(&shape.elems[i], writer);
        }
    }

    #[inline]
    unsafe fn refs_from_flat<'a>(
        reader: &mut FlatRead<'a, *const u8>,
        shape: &'a Self::Shape,
    ) -> Self::Refs<'a> {
        let n = shape.total_arity();
        let ptrs = reader.take(n);
        SliceRefs { ptrs, shape }
    }

    #[inline]
    fn produced_from_flat<'a>(
        reader: &mut FlatRead<'a, bool>,
        shape: &'a Self::Shape,
    ) -> Self::Produced<'a> {
        let n = shape.total_arity();
        let flat = reader.take(n);
        SliceProduced { flat, shape }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ptr_of<T>(t: &T) -> *const u8 {
        t as *const T as *const u8
    }

    /// Run `refs_from_flat` for `I` over `ptrs` with `shape`.  Asserts the
    /// cursor reaches the end, catching impl bugs that consume too few or
    /// too many slots.
    unsafe fn read_refs<'a, I: InputTypes>(
        ptrs: &'a [*const u8],
        shape: &'a I::Shape,
    ) -> I::Refs<'a> {
        let mut reader = FlatRead::new(ptrs);
        let refs = unsafe { I::refs_from_flat(&mut reader, shape) };
        assert_eq!(reader.remaining(), 0, "cursor did not reach end");
        refs
    }

    fn read_produced<'a, I: InputTypes>(bits: &'a [bool], shape: &'a I::Shape) -> I::Produced<'a> {
        let mut reader = FlatRead::new(bits);
        let p = I::produced_from_flat(&mut reader, shape);
        assert_eq!(reader.remaining(), 0, "cursor did not reach end");
        p
    }

    fn write_tids<I: InputTypes>(shape: &I::Shape) -> Vec<TypeId> {
        let arity = I::arity(shape);
        let mut buf = vec![TypeId::of::<()>(); arity];
        let mut writer = FlatWrite::new(&mut buf);
        I::write_type_ids(shape, &mut writer);
        assert_eq!(writer.remaining(), 0, "writer did not reach end");
        buf
    }

    #[test]
    fn leaf_input_roundtrip() {
        let v: u32 = 42;
        let ptrs = [ptr_of(&v)];

        type L = Input<u32>;
        let shape = ();
        assert_eq!(<L as InputTypes>::arity(&shape), 1);

        let tids = write_tids::<L>(&shape);
        assert_eq!(tids[0], TypeId::of::<u32>());

        let refs = unsafe { read_refs::<L>(&ptrs, &shape) };
        assert_eq!(*refs, 42);

        let bits = [true];
        let p = read_produced::<L>(&bits, &shape);
        assert!(p);
    }

    #[test]
    fn leaf_with_tuple_value_type() {
        // A leaf whose value type happens to be a tuple — verifies the
        // wrapper semantics: Input<(A, B)> is ONE leaf carrying a compound
        // value, NOT a two-input tuple.
        let v: (u32, u64) = (7, 13);
        let ptrs = [ptr_of(&v)];

        type L = Input<(u32, u64)>;
        let shape = ();
        assert_eq!(<L as InputTypes>::arity(&shape), 1);

        let refs = unsafe { read_refs::<L>(&ptrs, &shape) };
        assert_eq!(refs.0, 7);
        assert_eq!(refs.1, 13);
    }

    #[test]
    fn pair_of_leaves_roundtrip() {
        let a: u32 = 1;
        let b: u64 = 2;
        let ptrs = [ptr_of(&a), ptr_of(&b)];

        type T = (Input<u32>, Input<u64>);
        let shape = ((), ());

        assert_eq!(<T as InputTypes>::arity(&shape), 2);

        let tids = write_tids::<T>(&shape);
        assert_eq!(tids[0], TypeId::of::<u32>());
        assert_eq!(tids[1], TypeId::of::<u64>());

        let (ra, rb) = unsafe { read_refs::<T>(&ptrs, &shape) };
        assert_eq!(*ra, 1);
        assert_eq!(*rb, 2);

        let bits = [true, false];
        let (pa, pb) = read_produced::<T>(&bits, &shape);
        assert!(pa);
        assert!(!pb);
    }

    #[test]
    fn slice_of_leaves_roundtrip() {
        let v = [10u32, 20, 30];
        let ptrs: Vec<*const u8> = v.iter().map(|x| ptr_of(x)).collect();

        type S = Slice<Input<u32>>;
        let shape: SliceShape<Input<u32>> = SliceShape::new(vec![(), (), ()].into_boxed_slice());

        assert_eq!(<S as InputTypes>::arity(&shape), 3);

        let tids = write_tids::<S>(&shape);
        assert!(tids.iter().all(|t| *t == TypeId::of::<u32>()));

        let refs = unsafe { read_refs::<S>(&ptrs, &shape) };
        assert_eq!(refs.len(), 3);
        assert_eq!(*refs.get(0), 10);
        assert_eq!(*refs.get(1), 20);
        assert_eq!(*refs.get(2), 30);
        let collected: Vec<u32> = refs.iter().map(|r| *r).collect();
        assert_eq!(collected, vec![10, 20, 30]);

        let bits = [true, false, true];
        let p = read_produced::<S>(&bits, &shape);
        assert_eq!(p.len(), 3);
        assert!(p.get(0));
        assert!(!p.get(1));
        assert!(p.get(2));
        assert!(p.any());
    }

    #[test]
    fn nested_tuple_of_slice_and_leaf_roundtrip() {
        // Shape: (Slice<Input<u32>>, Input<u64>)  —  Stack + clock-like.
        let slice_vals = [1u32, 2, 3];
        let clock: u64 = 99;
        let mut ptrs: Vec<*const u8> = slice_vals.iter().map(|x| ptr_of(x)).collect();
        ptrs.push(ptr_of(&clock));

        type InTy = (Slice<Input<u32>>, Input<u64>);
        let shape = (
            SliceShape::<Input<u32>>::new(vec![(), (), ()].into_boxed_slice()),
            (),
        );
        assert_eq!(<InTy as InputTypes>::arity(&shape), 4);

        let tids = write_tids::<InTy>(&shape);
        assert_eq!(tids[0], TypeId::of::<u32>());
        assert_eq!(tids[1], TypeId::of::<u32>());
        assert_eq!(tids[2], TypeId::of::<u32>());
        assert_eq!(tids[3], TypeId::of::<u64>());

        let (slice_refs, clock_ref) = unsafe { read_refs::<InTy>(&ptrs, &shape) };
        assert_eq!(slice_refs.len(), 3);
        assert_eq!(*slice_refs.get(0), 1);
        assert_eq!(*slice_refs.get(1), 2);
        assert_eq!(*slice_refs.get(2), 3);
        assert_eq!(*clock_ref, 99);

        let bits = [true, false, true, true];
        let (slice_p, clock_p) = read_produced::<InTy>(&bits, &shape);
        assert_eq!(slice_p.len(), 3);
        assert!(slice_p.get(0));
        assert!(!slice_p.get(1));
        assert!(slice_p.get(2));
        assert!(clock_p);
    }

    #[test]
    fn slice_of_pairs_roundtrip() {
        // Shape: Slice<(Input<u32>, Input<u64>)>  —  slice of pairs.
        let a0: u32 = 1;
        let b0: u64 = 10;
        let a1: u32 = 2;
        let b1: u64 = 20;
        let ptrs = [ptr_of(&a0), ptr_of(&b0), ptr_of(&a1), ptr_of(&b1)];

        type Pair = (Input<u32>, Input<u64>);
        type S = Slice<Pair>;

        let pair_shape: <Pair as InputTypes>::Shape = ((), ());
        let shape: SliceShape<Pair> = SliceShape::new(vec![pair_shape, ((), ())].into_boxed_slice());

        assert_eq!(<S as InputTypes>::arity(&shape), 4);

        let tids = write_tids::<S>(&shape);
        assert_eq!(tids[0], TypeId::of::<u32>());
        assert_eq!(tids[1], TypeId::of::<u64>());
        assert_eq!(tids[2], TypeId::of::<u32>());
        assert_eq!(tids[3], TypeId::of::<u64>());

        let refs = unsafe { read_refs::<S>(&ptrs, &shape) };
        assert_eq!(refs.len(), 2);
        let (pa0, pb0) = refs.get(0);
        assert_eq!(*pa0, 1);
        assert_eq!(*pb0, 10);
        let (pa1, pb1) = refs.get(1);
        assert_eq!(*pa1, 2);
        assert_eq!(*pb1, 20);
    }

    #[test]
    fn deeply_nested_roundtrip() {
        // Shape: ((Input<u8>, Input<u16>), Slice<Input<u32>>)
        let x: u8 = 1;
        let y: u16 = 2;
        let z0: u32 = 100;
        let z1: u32 = 200;
        let ptrs = [ptr_of(&x), ptr_of(&y), ptr_of(&z0), ptr_of(&z1)];

        type Inner = (Input<u8>, Input<u16>);
        type SlcT = Slice<Input<u32>>;
        type T = (Inner, SlcT);

        let inner_shape: <Inner as InputTypes>::Shape = ((), ());
        let slice_shape: SliceShape<Input<u32>> = SliceShape::new(vec![(), ()].into_boxed_slice());
        let shape = (inner_shape, slice_shape);

        assert_eq!(<T as InputTypes>::arity(&shape), 4);

        let ((rx, ry), rz) = unsafe { read_refs::<T>(&ptrs, &shape) };
        assert_eq!(*rx, 1);
        assert_eq!(*ry, 2);
        assert_eq!(rz.len(), 2);
        assert_eq!(*rz.get(0), 100);
        assert_eq!(*rz.get(1), 200);
    }

    #[test]
    fn empty_slice_arity_zero() {
        type S = Slice<Input<u32>>;
        let shape: SliceShape<Input<u32>> = SliceShape::new(Vec::new().into_boxed_slice());
        assert_eq!(<S as InputTypes>::arity(&shape), 0);

        let ptrs: &[*const u8] = &[];
        let refs = unsafe { read_refs::<S>(ptrs, &shape) };
        assert_eq!(refs.len(), 0);
        assert!(refs.is_empty());

        let bits: &[bool] = &[];
        let p = read_produced::<S>(bits, &shape);
        assert_eq!(p.len(), 0);
        assert!(!p.any());
    }
}
