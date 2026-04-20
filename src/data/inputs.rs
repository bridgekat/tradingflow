//! Recursive description of operator inputs.
//!
//! The core trait [`InputTypes`] describes an operator's `Inputs` as a tree
//! whose leaves are [`Input<T>`] for some value type `T: Send + 'static`, and
//! whose branches are tuples (heterogeneous, arities 1–12) or `[T]` slices
//! (homogeneous, `T: InputTypes`, always in the trailing position).
//!
//! # The Sized / arity correspondence
//!
//! `Sized` and "fixed compile-time arity" coincide exactly for `InputTypes`
//! implementations:
//!
//! * Every `Sized` `InputTypes` (`Input<T>`, `()`, tuples of `Sized` children)
//!   has a compile-time-constant arity.
//! * The only `!Sized` `InputTypes` is `[T]`, whose element count is a
//!   runtime value.
//!
//! This means **no `Shape` type is needed**:
//!
//! * Sized types consume a fixed, type-determined number of slots from the
//!   cursor — no runtime descriptor required.
//! * `[T]` always occupies the **tail** of the flat buffer; `refs_from_flat`
//!   simply reads all remaining slots.
//!
//! # Cursor traversal
//!
//! [`FlatRead`] / [`FlatWrite`] thread position state through the recursive
//! tree so each leaf is visited exactly once — O(total_leaves).
//!
//! # Public types
//!
//! * [`Input<T>`] — leaf marker wrapping any `T: Send + 'static`.
//! * [`SliceRefs<'a, T>`] / [`SliceProduced<'a, T>`] — zero-allocation views
//!   over a trailing slice branch.
//! * [`FlatRead`] / [`FlatWrite`] — single-pass cursors over `&[T]`, used on
//!   the [`Refs<'a>`](InputTypes::Refs) path.
//! * [`BitRead`] — single-pass cursor over a packed bit buffer `&[u64]`,
//!   used on the [`Produced<'a>`](InputTypes::Produced) path.  Symmetric to
//!   [`FlatRead<bool>`]: same `new` / `remaining` / `take` / `pop` surface,
//!   but storage is 1 bit per slot instead of 1 byte.

use std::any::TypeId;
use std::marker::PhantomData;

// ===========================================================================
// Cursor-style flat readers and writers
// ===========================================================================

/// Single-pass cursor over a flat slice `&'a [T]`.
pub struct FlatRead<'a, T> {
    buf: &'a [T],
    idx: usize,
}

impl<'a, T> FlatRead<'a, T> {
    /// Wrap a slice; cursor starts at position 0.
    #[inline(always)]
    pub fn new(buf: &'a [T]) -> Self {
        Self { buf, idx: 0 }
    }

    /// Number of elements remaining past the cursor.
    #[inline(always)]
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.idx
    }

    /// Consume `n` elements, returning a sub-slice with the outer `'a` lifetime.
    #[inline(always)]
    pub fn take(&mut self, n: usize) -> &'a [T] {
        let slice = &self.buf[self.idx..self.idx + n];
        self.idx += n;
        slice
    }

    /// Consume one element, returning a reference.
    #[inline(always)]
    pub fn pop(&mut self) -> &'a T {
        let v = &self.buf[self.idx];
        self.idx += 1;
        v
    }
}

// ===========================================================================
// Bit-packed cursor
// ===========================================================================

/// Single-pass cursor over a packed bit buffer `&'a [u64]`.
///
/// Symmetric to [`FlatRead<T>`]: same `new` / `remaining` / `take` / `pop`
/// surface but storage is 1 bit per slot.  The absolute bit offset is
/// *unbounded* (not re-normalized into `0..64`).  Bits beyond the backing
/// buffer read as `false`, which makes "all zeros" views trivial to build
/// with an empty `&[u64]`.
#[derive(Copy, Clone, Debug)]
pub struct BitRead<'a> {
    words: &'a [u64],
    /// Absolute bit position of the next bit to read.
    bit_idx: usize,
    /// Absolute bit position one past the last readable bit.
    bit_end: usize,
}

impl<'a> BitRead<'a> {
    /// Wrap `words`; cursor starts at bit 0 and covers `num_bits` bits.
    #[inline(always)]
    pub fn new(words: &'a [u64], num_bits: usize) -> Self {
        Self {
            words,
            bit_idx: 0,
            bit_end: num_bits,
        }
    }

    /// Wrap `words`; cursor covers bits `[bit_off, bit_off + num_bits)`.
    #[inline(always)]
    pub fn from_parts(words: &'a [u64], bit_off: usize, num_bits: usize) -> Self {
        Self {
            words,
            bit_idx: bit_off,
            bit_end: bit_off + num_bits,
        }
    }

    /// Number of bits remaining past the cursor.
    #[inline(always)]
    pub fn remaining(&self) -> usize {
        self.bit_end - self.bit_idx
    }

    /// Consume `n` bits, returning a sub-cursor with lifetime `'a`.
    #[inline(always)]
    pub fn take(&mut self, n: usize) -> BitRead<'a> {
        let sub = BitRead {
            words: self.words,
            bit_idx: self.bit_idx,
            bit_end: self.bit_idx + n,
        };
        self.bit_idx += n;
        sub
    }

    /// Consume one bit.  Bits beyond the backing buffer read as `false`.
    #[inline(always)]
    pub fn pop(&mut self) -> bool {
        let g = self.bit_idx;
        self.bit_idx += 1;
        let wi = g / 64;
        if wi >= self.words.len() {
            false
        } else {
            (self.words[wi] >> (g % 64)) & 1 != 0
        }
    }

    /// Iterate over local positions (0..[`remaining`](Self::remaining)) whose
    /// bit is set.  Uses `u64::trailing_zeros` for per-set-bit-constant work.
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        let words = self.words;
        let start = self.bit_idx;
        let end = self.bit_end;
        let start_word = start / 64;
        let end_word = end.div_ceil(64).min(words.len());
        let first_word = start_word.min(end_word);
        (first_word..end_word).flat_map(move |wi| {
            let word_base = wi * 64;
            let mut word = words[wi];
            if word_base < start {
                word &= u64::MAX << (start - word_base);
            }
            if word_base + 64 > end {
                let bits = end - word_base;
                if bits < 64 {
                    word &= (1u64 << bits) - 1;
                }
            }
            std::iter::from_fn(move || {
                if word == 0 {
                    None
                } else {
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    Some(word_base + bit - start)
                }
            })
        })
    }
}

// ===========================================================================

/// Build a `Produced<'static>` tree with every bit `false`.
///
/// Convenience for tests and ad-hoc operator invocations: the returned
/// tree reports "no input produced" for every leaf.  `num_bits` is only
/// needed for slice-tailed inputs (to give the `SliceProduced` its length);
/// for fully-`Sized` inputs pass 0 or the arity.
#[inline]
pub fn empty_produced<I: InputTypes + ?Sized>(num_bits: usize) -> I::Produced<'static> {
    let mut reader = BitRead::new(&[], num_bits);
    I::produced_from_flat(&mut reader)
}

/// Build a `Produced<'static>` tree from `positions` expressed as local
/// input indices.  Convenience for tests.
///
/// Packs the positions into a leaked bit buffer and traverses.  Do not use
/// on the hot path.
#[cfg(test)]
pub fn produced_from_positions<I: InputTypes + ?Sized>(
    positions: &[usize],
    num_bits: usize,
) -> I::Produced<'static> {
    let n_words = num_bits.div_ceil(64).max(1);
    let mut buf = vec![0u64; n_words];
    for &p in positions {
        buf[p / 64] |= 1u64 << (p % 64);
    }
    let words: &'static [u64] = Box::leak(buf.into_boxed_slice());
    let mut reader = BitRead::new(words, num_bits);
    I::produced_from_flat(&mut reader)
}

/// Single-pass cursor over a flat mutable slice `&'a mut [T]`.
pub struct FlatWrite<'a, T> {
    buf: &'a mut [T],
    idx: usize,
}

impl<'a, T> FlatWrite<'a, T> {
    /// Wrap a mutable slice; cursor starts at position 0.
    #[inline(always)]
    pub fn new(buf: &'a mut [T]) -> Self {
        Self { buf, idx: 0 }
    }

    /// Number of elements remaining past the cursor.
    #[inline(always)]
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.idx
    }

    /// Write `v` at the cursor position, then advance by 1.
    #[inline(always)]
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
/// `T: Send + 'static`.  `Input<T>::Refs<'a> = &'a T`, so compute bodies
/// see `&T` directly — the wrapper is invisible past the trait system.
pub struct Input<T: Send + 'static>(PhantomData<fn() -> T>);

/// Recursive description of an operator's inputs.
///
/// An operator's `Inputs` is a tree whose leaves are [`Input<T>`] and whose
/// branches are fixed-arity tuples or trailing `[T]` slices.  No `Shape`
/// type: `Sized ↔ fixed arity` is exact, so the cursor approach
/// (`refs_from_flat` drains the slice tail; everything else consumes a
/// compile-time-constant count) needs no runtime descriptor.
///
/// Methods with `where Self: Sized` are called only at operator registration,
/// never on the compute hot path.
pub trait InputTypes {
    /// Nested immutable references.
    type Refs<'a>;

    /// Nested boolean view (which leaves produced this cycle).
    type Produced<'a>;

    /// Total flat leaf count.
    ///
    /// For all `Sized` implementations this is a compile-time-constant
    /// determined by the type alone.  For `!Sized` implementations (`[T]`
    /// and tuples with a trailing `[T]`) this method is dead — the `[T]`
    /// impl panics with `unreachable!()`, and the only legitimate runtime
    /// path to registering a `!Sized` `Inputs` operator goes through
    /// `ErasedOperator::from_operator_with_type_ids`, which never calls
    /// `arity()`.
    fn arity() -> usize;

    /// Write the [`TypeId`] of each flat leaf into `writer` in tree-order.
    fn type_ids_to_flat(writer: &mut FlatWrite<TypeId>);

    /// Construct nested references by consuming slots from `reader`.
    ///
    /// For `Sized` types consumes exactly [`arity()`](Self::arity) slots.
    /// For `[T]` consumes all remaining slots.
    ///
    /// # Safety
    ///
    /// Each consumed pointer must point to a valid value of the matching type.
    unsafe fn refs_from_flat<'a>(reader: &mut FlatRead<'a, *const u8>) -> Self::Refs<'a>;

    /// Construct nested booleans by consuming bits from `reader`.
    ///
    /// Same slot-count semantics as [`refs_from_flat`](Self::refs_from_flat).
    fn produced_from_flat<'a>(reader: &mut BitRead<'a>) -> Self::Produced<'a>;
}

// -- Leaf: Input<T> ----------------------------------------------------------

impl<T: Send + 'static> InputTypes for Input<T> {
    type Refs<'a> = &'a T;
    type Produced<'a> = bool;

    #[inline(always)]
    fn arity() -> usize {
        1
    }

    #[inline(always)]
    fn type_ids_to_flat(writer: &mut FlatWrite<TypeId>) {
        writer.push(TypeId::of::<T>());
    }

    #[inline(always)]
    unsafe fn refs_from_flat<'a>(reader: &mut FlatRead<'a, *const u8>) -> &'a T {
        unsafe { &*(*reader.pop() as *const T) }
    }

    #[inline(always)]
    fn produced_from_flat<'a>(reader: &mut BitRead<'a>) -> bool {
        reader.pop()
    }
}

// -- Compound: empty tuple (arity 0) ----------------------------------------

impl InputTypes for () {
    type Refs<'a> = ();
    type Produced<'a> = ();

    #[inline(always)]
    fn arity() -> usize {
        0
    }

    #[inline(always)]
    fn type_ids_to_flat(_writer: &mut FlatWrite<TypeId>) {}

    #[inline(always)]
    unsafe fn refs_from_flat<'a>(_reader: &mut FlatRead<'a, *const u8>) {}

    #[inline(always)]
    fn produced_from_flat<'a>(_reader: &mut BitRead<'a>) {}
}

// -- Compound: tuple branches (arities 1-12) ---------------------------------

// 1-tuple: ?Sized last element only.
impl<S: InputTypes + ?Sized> InputTypes for (S,) {
    type Refs<'a> = (S::Refs<'a>,);
    type Produced<'a> = (S::Produced<'a>,);

    #[inline(always)]
    fn arity() -> usize {
        S::arity()
    }

    #[inline(always)]
    fn type_ids_to_flat(writer: &mut FlatWrite<TypeId>) {
        S::type_ids_to_flat(writer);
    }

    #[inline(always)]
    unsafe fn refs_from_flat<'a>(reader: &mut FlatRead<'a, *const u8>) -> Self::Refs<'a> {
        (unsafe { S::refs_from_flat(reader) },)
    }

    #[inline(always)]
    fn produced_from_flat<'a>(reader: &mut BitRead<'a>) -> Self::Produced<'a> {
        (S::produced_from_flat(reader),)
    }
}

// N-tuple (N >= 2): Sized prefix + ?Sized last.
macro_rules! impl_input_types_for_tuple {
    ($($idx:tt: $T:ident),+; $last_idx:tt: $S:ident) => {
        impl<$($T: InputTypes,)+ $S: InputTypes + ?Sized> InputTypes for ($($T,)+ $S,) {
            type Refs<'a> = ($($T::Refs<'a>,)+ $S::Refs<'a>,);
            type Produced<'a> = ($($T::Produced<'a>,)+ $S::Produced<'a>,);

            #[inline(always)]
            fn arity() -> usize {
                0 $(+ $T::arity())+ + $S::arity()
            }

            #[inline(always)]
            fn type_ids_to_flat(writer: &mut FlatWrite<TypeId>) {
                $( $T::type_ids_to_flat(writer); )+
                $S::type_ids_to_flat(writer);
            }

            #[inline(always)]
            unsafe fn refs_from_flat<'a>(reader: &mut FlatRead<'a, *const u8>) -> Self::Refs<'a> {
                (
                    $( unsafe { $T::refs_from_flat(reader) }, )+
                    unsafe { $S::refs_from_flat(reader) },
                )
            }

            #[inline(always)]
            fn produced_from_flat<'a>(reader: &mut BitRead<'a>) -> Self::Produced<'a> {
                (
                    $( $T::produced_from_flat(reader), )+
                    $S::produced_from_flat(reader),
                )
            }
        }
    };
}

impl_input_types_for_tuple!(0: A; 1: B);
impl_input_types_for_tuple!(0: A, 1: B; 2: C);
impl_input_types_for_tuple!(0: A, 1: B, 2: C; 3: D);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D; 4: E);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E; 5: F);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F; 6: G);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G; 7: H);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H; 8: I);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I; 9: J);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J; 10: K);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K; 11: L);

// -- Compound: trailing slice branch ([T]) -----------------------------------
//
// `[T]` is `!Sized` — it is valid only as the trailing type in an operator's
// `Inputs`.  `refs_from_flat` and `produced_from_flat` drain all remaining
// cursor slots.
//
// `T: InputTypes` (sized) guarantees every element has a compile-time-constant
// arity `T::arity()`, which `SliceRefs::get(i)` uses for O(1) access via
// simple stride arithmetic.

impl<T: InputTypes + 'static> InputTypes for [T] {
    type Refs<'a> = SliceRefs<'a, T>;
    type Produced<'a> = SliceProduced<'a, T>;

    #[inline(always)]
    fn arity() -> usize {
        unreachable!("element count is only known at runtime for [T]")
    }

    #[inline(always)]
    fn type_ids_to_flat(writer: &mut FlatWrite<TypeId>) {
        let s = T::arity();
        if s != 0 {
            while writer.remaining() > 0 {
                T::type_ids_to_flat(writer);
            }
        }
    }

    #[inline(always)]
    unsafe fn refs_from_flat<'a>(reader: &mut FlatRead<'a, *const u8>) -> SliceRefs<'a, T> {
        unsafe { SliceRefs::new(reader.take(reader.remaining())) }
    }

    #[inline(always)]
    fn produced_from_flat<'a>(reader: &mut BitRead<'a>) -> SliceProduced<'a, T> {
        let rem = reader.remaining();
        SliceProduced::from_cursor(reader.take(rem))
    }
}

/// Zero-allocation nested reference view over a trailing `[T]` branch.
pub struct SliceRefs<'a, T: InputTypes> {
    flat: &'a [*const u8],
    _marker: PhantomData<fn() -> T>,
}

impl<'a, T: InputTypes> SliceRefs<'a, T> {
    /// Construct a view from a flat pointer sub-slice.
    ///
    /// # Safety
    ///
    /// `ptrs.len()` must be a multiple of `T::arity()`, and every stride-sized
    /// sub-slice must contain valid pointers matching the leaves of `T`.
    #[inline(always)]
    pub unsafe fn new(ptrs: &'a [*const u8]) -> Self {
        Self {
            flat: ptrs,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        let s = T::arity();
        if s != 0 { self.flat.len() / s } else { 0 }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> T::Refs<'a> {
        let s = T::arity();
        let mut reader = FlatRead::new(&self.flat[i * s..]);
        unsafe { T::refs_from_flat(&mut reader) }
    }

    pub fn iter(&self) -> impl Iterator<Item = T::Refs<'a>> + '_ {
        let mut reader = FlatRead::new(self.flat);
        (0..self.len()).map(move |_| unsafe { T::refs_from_flat(&mut reader) })
    }
}

/// Zero-allocation nested boolean view over a trailing `[T]` branch.
///
/// Symmetric to [`SliceRefs`]: construction is a pure struct copy, and
/// sub-element `Produced` values are built *lazily* on demand via
/// [`get`](Self::get) / [`iter`](Self::iter) / [`any`](Self::any).
pub struct SliceProduced<'a, T: InputTypes> {
    /// Backing bit buffer.  Bits outside the logical range read as `false`.
    words: &'a [u64],
    /// Absolute starting bit offset of this slice within `words`.
    bit_off: usize,
    /// Total bit count covered by this slice (element count × `T::arity()`).
    bit_len: usize,
    _marker: PhantomData<fn() -> T>,
}

impl<'a, T: InputTypes> SliceProduced<'a, T> {
    /// Internal: take over a cursor's entire remaining range.
    #[inline(always)]
    fn from_cursor(cursor: BitRead<'a>) -> Self {
        Self {
            words: cursor.words,
            bit_off: cursor.bit_idx,
            bit_len: cursor.bit_end - cursor.bit_idx,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        let s = T::arity();
        if s != 0 { self.bit_len / s } else { 0 }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Build element `i`'s nested `Produced` on demand.  No bit is read
    /// until the caller descends into leaves.
    #[inline(always)]
    pub fn get(&self, i: usize) -> T::Produced<'a> {
        let s = T::arity();
        let mut reader = BitRead::from_parts(self.words, self.bit_off + i * s, s);
        T::produced_from_flat(&mut reader)
    }

    pub fn iter(&self) -> impl Iterator<Item = T::Produced<'a>> + '_ {
        let mut reader =
            BitRead::from_parts(self.words, self.bit_off, self.bit_len);
        (0..self.len()).map(move |_| T::produced_from_flat(&mut reader))
    }

    /// Whether any bit within this slice's range is set.  Scans O(len/64)
    /// words via `trailing_zeros`.
    pub fn any(&self) -> bool {
        let reader = BitRead::from_parts(self.words, self.bit_off, self.bit_len);
        reader.iter_set().next().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ptr_of<T>(t: &T) -> *const u8 {
        t as *const T as *const u8
    }

    unsafe fn read_refs<'a, I: InputTypes + ?Sized>(ptrs: &'a [*const u8]) -> I::Refs<'a> {
        let mut reader = FlatRead::new(ptrs);
        let refs = unsafe { I::refs_from_flat(&mut reader) };
        assert_eq!(reader.remaining(), 0, "cursor did not reach end");
        refs
    }

    /// Pack a `&[bool]` into a minimally-sized `Vec<u64>` bitset.
    fn pack_bits(bits: &[bool]) -> Vec<u64> {
        let mut out = vec![0u64; bits.len().div_ceil(64).max(1)];
        for (i, &b) in bits.iter().enumerate() {
            if b {
                out[i / 64] |= 1u64 << (i % 64);
            }
        }
        out
    }

    fn read_produced<I: InputTypes + ?Sized>(bits: &[bool]) -> I::Produced<'static> {
        // Leak the packed storage — test-only, short-lived, keeps the
        // returned `Produced<'a>` usable without lifetime gymnastics.
        let packed: &'static [u64] = Box::leak(pack_bits(bits).into_boxed_slice());
        let mut reader = BitRead::new(packed, bits.len());
        let p = I::produced_from_flat(&mut reader);
        assert_eq!(reader.remaining(), 0, "cursor did not reach end");
        p
    }

    fn write_tids<I: InputTypes>() -> Vec<TypeId> {
        let mut buf = vec![TypeId::of::<()>(); I::arity()];
        let mut writer = FlatWrite::new(&mut buf);
        I::type_ids_to_flat(&mut writer);
        assert_eq!(writer.remaining(), 0);
        buf
    }

    #[test]
    fn leaf_roundtrip() {
        let v: u32 = 42;
        let ptrs = [ptr_of(&v)];

        type L = Input<u32>;
        assert_eq!(L::arity(), 1);
        assert_eq!(write_tids::<L>(), [TypeId::of::<u32>()]);

        let refs = unsafe { read_refs::<L>(&ptrs) };
        assert_eq!(*refs, 42);
        assert!(read_produced::<L>(&[true]));
        assert!(!read_produced::<L>(&[false]));
    }

    #[test]
    fn leaf_tuple_value_type() {
        // Input<(A, B)> is ONE leaf, not a two-leaf tuple.
        let v: (u32, u64) = (7, 13);
        let ptrs = [ptr_of(&v)];

        type L = Input<(u32, u64)>;
        assert_eq!(L::arity(), 1);
        let refs = unsafe { read_refs::<L>(&ptrs) };
        assert_eq!(*refs, (7, 13));
    }

    #[test]
    fn pair_of_leaves() {
        let a: u32 = 1;
        let b: u64 = 2;
        let ptrs = [ptr_of(&a), ptr_of(&b)];

        type T = (Input<u32>, Input<u64>);
        assert_eq!(T::arity(), 2);
        assert_eq!(
            write_tids::<T>(),
            [TypeId::of::<u32>(), TypeId::of::<u64>()]
        );

        let (ra, rb) = unsafe { read_refs::<T>(&ptrs) };
        assert_eq!(*ra, 1);
        assert_eq!(*rb, 2);

        let (pa, pb) = read_produced::<T>(&[true, false]);
        assert!(pa);
        assert!(!pb);
    }

    #[test]
    fn slice_of_leaves() {
        let v = [10u32, 20, 30];
        let ptrs: Vec<*const u8> = v.iter().map(ptr_of).collect();

        let refs = unsafe { read_refs::<[Input<u32>]>(&ptrs) };
        assert_eq!(refs.len(), 3);
        assert_eq!(*refs.get(0), 10);
        assert_eq!(*refs.get(1), 20);
        assert_eq!(*refs.get(2), 30);
        let collected: Vec<u32> = refs.iter().copied().collect();
        assert_eq!(collected, vec![10, 20, 30]);

        let p = read_produced::<[Input<u32>]>(&[true, false, true]);
        assert_eq!(p.len(), 3);
        assert!(p.get(0));
        assert!(!p.get(1));
        assert!(p.iter().any(|b| b));
    }

    #[test]
    fn slice_of_pairs() {
        // [(Input<u32>, Input<u64>)] — slice of 2-leaf pairs.
        let a0: u32 = 1;
        let b0: u64 = 10;
        let a1: u32 = 2;
        let b1: u64 = 20;
        let ptrs = [ptr_of(&a0), ptr_of(&b0), ptr_of(&a1), ptr_of(&b1)];

        type Pair = (Input<u32>, Input<u64>);
        assert_eq!(Pair::arity(), 2);

        let refs = unsafe { read_refs::<[Pair]>(&ptrs) };
        assert_eq!(refs.len(), 2);
        let (pa0, pb0) = refs.get(0);
        let (pa1, pb1) = refs.get(1);
        assert_eq!((*pa0, *pb0), (1, 10));
        assert_eq!((*pa1, *pb1), (2, 20));
    }

    #[test]
    fn empty_slice() {
        let refs = unsafe { read_refs::<[Input<u32>]>(&[]) };
        assert!(refs.is_empty());
        let p = read_produced::<[Input<u32>]>(&[]);
        assert!(!p.iter().any(|b| b));
    }

    // -----------------------------------------------------------------------
    // Trailing-slice tuple tests
    //
    // These verify the (prefix..., [S]) impls: Sized branches consume fixed
    // counts, the trailing slice drains the remainder.  The notation
    // mirrors `Input<(bool, u32, [f64])>` — a tree with two scalar leaves
    // and a variable-length float tail.
    // -----------------------------------------------------------------------

    #[test]
    fn two_scalars_then_float_slice() {
        // Tree: (Input<bool>, Input<u32>, [Input<f64>])
        // Flat layout: [bool_ptr, u32_ptr, f64_0_ptr, f64_1_ptr, f64_2_ptr]
        let b: bool = true;
        let u: u32 = 42;
        let f0: f64 = 1.0;
        let f1: f64 = 2.0;
        let f2: f64 = 3.0;
        let ptrs = [
            ptr_of(&b),
            ptr_of(&u),
            ptr_of(&f0),
            ptr_of(&f1),
            ptr_of(&f2),
        ];

        type T = (Input<bool>, Input<u32>, [Input<f64>]);

        let (rb, ru, rf) = unsafe { read_refs::<T>(&ptrs) };
        assert!(*rb);
        assert_eq!(*ru, 42);
        assert_eq!(rf.len(), 3);
        assert_eq!(*rf.get(0), 1.0);
        assert_eq!(*rf.get(1), 2.0);
        assert_eq!(*rf.get(2), 3.0);
        let floats: Vec<f64> = rf.iter().copied().collect();
        assert_eq!(floats, vec![1.0, 2.0, 3.0]);

        let (pb, pu, pf) = read_produced::<T>(&[false, true, true, false, true]);
        assert!(!pb);
        assert!(pu);
        assert_eq!(pf.len(), 3);
        assert!(pf.get(0));
        assert!(!pf.get(1));
        assert!(pf.get(2));
        assert!(pf.iter().any(|b| b));
    }

    #[test]
    fn one_scalar_then_slice() {
        // Tree: (Input<u8>, [Input<u16>])
        let x: u8 = 7;
        let a: u16 = 100;
        let b: u16 = 200;
        let ptrs = [ptr_of(&x), ptr_of(&a), ptr_of(&b)];

        type T = (Input<u8>, [Input<u16>]);
        let (rx, rs) = unsafe { read_refs::<T>(&ptrs) };
        assert_eq!(*rx, 7);
        assert_eq!(rs.len(), 2);
        assert_eq!(*rs.get(0), 100);
        assert_eq!(*rs.get(1), 200);
    }

    #[test]
    fn trailing_slice_empty() {
        // Tree: (Input<bool>, [Input<f64>]) with zero floats.
        let b: bool = false;
        let ptrs = [ptr_of(&b)];

        type T = (Input<bool>, [Input<f64>]);
        let (rb, rf) = unsafe { read_refs::<T>(&ptrs) };
        assert!(!rb);
        assert_eq!(rf.len(), 0);
        assert!(rf.is_empty());
    }

    #[test]
    fn trailing_slice_of_pairs() {
        // Tree: (Input<u8>, [(Input<u32>, Input<u64>)])
        // Flat: [u8_ptr, u32_0, u64_0, u32_1, u64_1]
        let tag: u8 = 9;
        let a0: u32 = 10;
        let b0: u64 = 20;
        let a1: u32 = 30;
        let b1: u64 = 40;
        let ptrs = [
            ptr_of(&tag),
            ptr_of(&a0),
            ptr_of(&b0),
            ptr_of(&a1),
            ptr_of(&b1),
        ];

        type Pair = (Input<u32>, Input<u64>);
        type T = (Input<u8>, [Pair]);
        let (rtag, rslice) = unsafe { read_refs::<T>(&ptrs) };
        assert_eq!(*rtag, 9);
        assert_eq!(rslice.len(), 2);
        let (p0a, p0b) = rslice.get(0);
        let (p1a, p1b) = rslice.get(1);
        assert_eq!((*p0a, *p0b), (10, 20));
        assert_eq!((*p1a, *p1b), (30, 40));
    }
}
