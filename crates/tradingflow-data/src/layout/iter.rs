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
