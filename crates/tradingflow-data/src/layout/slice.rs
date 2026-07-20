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
