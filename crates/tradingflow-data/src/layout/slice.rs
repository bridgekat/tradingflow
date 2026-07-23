use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

/// Slicing specifier on a single axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice {
    start: usize,
    count: Option<usize>,
    step: usize,
}

impl Slice {
    pub fn new(start: usize, count: Option<usize>, step: usize) -> Self {
        Slice { start, count, step }
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn count(&self) -> Option<usize> {
        self.count
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn len(&self, end: usize) -> usize {
        match self.count {
            Some(count) => count,
            None => end.saturating_sub(self.start).div_ceil(self.step),
        }
    }

    pub fn is_empty(&self, end: usize) -> bool {
        self.len(end) == 0
    }
}

impl From<RangeFull> for Slice {
    fn from(_: RangeFull) -> Self {
        Self::new(0, None, 1)
    }
}

impl From<(RangeFull, usize)> for Slice {
    fn from((_, step): (RangeFull, usize)) -> Self {
        Self::new(0, None, step)
    }
}

impl From<RangeFrom<usize>> for Slice {
    fn from(range: RangeFrom<usize>) -> Self {
        Self::new(range.start, None, 1)
    }
}

impl From<(RangeFrom<usize>, usize)> for Slice {
    fn from((range, step): (RangeFrom<usize>, usize)) -> Self {
        Self::new(range.start, None, step)
    }
}

impl From<RangeTo<usize>> for Slice {
    fn from(range: RangeTo<usize>) -> Self {
        Self::new(0, Some(range.end), 1)
    }
}

impl From<(RangeTo<usize>, usize)> for Slice {
    fn from((range, step): (RangeTo<usize>, usize)) -> Self {
        Self::new(0, Some(range.end.div_ceil(step)), step)
    }
}

impl From<Range<usize>> for Slice {
    fn from(range: Range<usize>) -> Self {
        let count = range.end.saturating_sub(range.start);
        Self::new(range.start, Some(count), 1)
    }
}

impl From<(Range<usize>, usize)> for Slice {
    fn from((range, step): (Range<usize>, usize)) -> Self {
        let count = range.end.saturating_sub(range.start).div_ceil(step);
        Self::new(range.start, Some(count), step)
    }
}

/// Slicing, projection or new-axis specifier on a single axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceReshape {
    Slice(Slice),
    Index(usize),
    NewAxis,
}

impl From<Slice> for SliceReshape {
    fn from(slice: Slice) -> Self {
        SliceReshape::Slice(slice)
    }
}

impl From<RangeFull> for SliceReshape {
    fn from(_: RangeFull) -> Self {
        Self::Slice(Slice::new(0, None, 1))
    }
}

impl From<(RangeFull, usize)> for SliceReshape {
    fn from((_, step): (RangeFull, usize)) -> Self {
        Self::Slice(Slice::new(0, None, step))
    }
}

impl From<RangeFrom<usize>> for SliceReshape {
    fn from(range: RangeFrom<usize>) -> Self {
        Self::Slice(Slice::new(range.start, None, 1))
    }
}

impl From<(RangeFrom<usize>, usize)> for SliceReshape {
    fn from((range, step): (RangeFrom<usize>, usize)) -> Self {
        Self::Slice(Slice::new(range.start, None, step))
    }
}

impl From<RangeTo<usize>> for SliceReshape {
    fn from(range: RangeTo<usize>) -> Self {
        Self::Slice(Slice::new(0, Some(range.end), 1))
    }
}

impl From<(RangeTo<usize>, usize)> for SliceReshape {
    fn from((range, step): (RangeTo<usize>, usize)) -> Self {
        Self::Slice(Slice::new(0, Some(range.end.div_ceil(step)), step))
    }
}

impl From<Range<usize>> for SliceReshape {
    fn from(range: Range<usize>) -> Self {
        let count = range.end.saturating_sub(range.start);
        Self::Slice(Slice::new(range.start, Some(count), 1))
    }
}

impl From<(Range<usize>, usize)> for SliceReshape {
    fn from((range, step): (Range<usize>, usize)) -> Self {
        let count = range.end.saturating_sub(range.start).div_ceil(step);
        Self::Slice(Slice::new(range.start, Some(count), step))
    }
}

impl From<usize> for SliceReshape {
    fn from(index: usize) -> Self {
        SliceReshape::Index(index)
    }
}

impl From<()> for SliceReshape {
    fn from(_: ()) -> Self {
        SliceReshape::NewAxis
    }
}

/// Converting into slicing specifiers on `N` axes.
pub trait IntoSlices<const N: usize> {
    fn into_slices(self) -> [Slice; N];
}

impl<T: Into<Slice>, const N: usize> IntoSlices<N> for [T; N] {
    fn into_slices(self) -> [Slice; N] {
        self.map(Into::into)
    }
}

/// Converting into slicing, projection or new-axis specifiers on `N` axes.
pub trait IntoSliceReshapes<const N: usize> {
    fn into_slice_reshapes(self) -> [SliceReshape; N];
}

impl<T: Into<SliceReshape>, const N: usize> IntoSliceReshapes<N> for [T; N] {
    fn into_slice_reshapes(self) -> [SliceReshape; N] {
        self.map(Into::into)
    }
}

macro_rules! impl_into_specifiers_for_tuple {
    ($N:literal; $($idx:tt: $T:ident),*) => {
        impl<$($T: Into<Slice>,)*> IntoSlices<{ $N }> for ($($T,)*) {
            fn into_slices(self) -> [Slice; $N] {
                [$(self.$idx.into(),)*]
            }
        }

        impl<$($T: Into<SliceReshape>,)*> IntoSliceReshapes<{ $N }> for ($($T,)*) {
            fn into_slice_reshapes(self) -> [SliceReshape; $N] {
                [$(self.$idx.into(),)*]
            }
        }
    };
}

impl_into_specifiers_for_tuple!(0; );
impl_into_specifiers_for_tuple!(1; 0: A);
impl_into_specifiers_for_tuple!(2; 0: A, 1: B);
impl_into_specifiers_for_tuple!(3; 0: A, 1: B, 2: C);
impl_into_specifiers_for_tuple!(4; 0: A, 1: B, 2: C, 3: D);
impl_into_specifiers_for_tuple!(5; 0: A, 1: B, 2: C, 3: D, 4: E);
impl_into_specifiers_for_tuple!(6; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_into_specifiers_for_tuple!(7; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_into_specifiers_for_tuple!(8; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_into_specifiers_for_tuple!(9; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_into_specifiers_for_tuple!(10; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_into_specifiers_for_tuple!(11; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_into_specifiers_for_tuple!(12; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
