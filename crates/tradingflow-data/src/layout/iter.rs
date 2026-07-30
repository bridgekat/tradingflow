/// Iterator over physical offsets.
#[derive(Debug, Clone)]
pub struct Offsets<const N: usize> {
    pub(super) extents: [usize; N],
    pub(super) strides: [usize; N],
    pub(super) indices: [usize; N],
    pub(super) offset: usize,
    pub(super) remaining: usize,
}

impl<const N: usize> Iterator for Offsets<N> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        self.remaining = self.remaining.checked_sub(1)?;
        let offset = self.offset;
        let mut d = N;
        while d > 0 {
            d -= 1;
            self.indices[d] += 1;
            self.offset += self.strides[d];
            if self.indices[d] < self.extents[d] {
                break;
            }
            self.indices[d] = 0;
            self.offset -= self.extents[d] * self.strides[d];
        }
        Some(offset)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<const N: usize> ExactSizeIterator for Offsets<N> {}
impl<const N: usize> std::iter::FusedIterator for Offsets<N> {}

/// Iterator over indices and physical offsets.
#[derive(Debug, Clone)]
pub struct IndicesOffsets<const N: usize> {
    inner: Offsets<N>,
}

impl<const N: usize> Iterator for IndicesOffsets<N> {
    type Item = ([usize; N], usize);

    fn next(&mut self) -> Option<([usize; N], usize)> {
        let indices = self.inner.indices;
        let offset = self.inner.next()?;
        Some((indices, offset))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<const N: usize> ExactSizeIterator for IndicesOffsets<N> {}
impl<const N: usize> std::iter::FusedIterator for IndicesOffsets<N> {}

impl<const N: usize> Offsets<N> {
    /// Returns an iterator over indices and physical offsets.
    pub fn with_indices(self) -> IndicesOffsets<N> {
        IndicesOffsets { inner: self }
    }
}
