//! The [`Layout`] trait and concrete layout policies.

/// A generic layout policy for a rank-`N` array.
pub trait Layout<const N: usize> {
    /// Per-axis extents.
    fn extents(&self) -> [usize; N];

    /// Per-axis strides.
    fn strides(&self) -> [usize; N];

    /// Whether the layout is *row-major contiguous*.
    fn is_contiguous(&self) -> bool;

    /// The rank `N` of the layout.
    fn ndim(&self) -> usize {
        N
    }

    /// Number of elements in the `N`-dimensional index space.
    fn len(&self) -> usize {
        self.extents().iter().product()
    }

    /// Whether the layout holds no elements.
    fn is_empty(&self) -> bool {
        self.extents().contains(&0)
    }

    /// Minimal physical length that contains every addressable element —
    /// `1 + Σ (extents[d] - 1) · strides[d]`, or `0` if the shape is empty.
    /// Equals [`len`](Self::len) for a contiguous layout.
    fn span(&self) -> usize {
        if self.is_empty() {
            return 0;
        }
        let steps: usize = self
            .extents()
            .iter()
            .zip(&self.strides())
            .map(|(e, s)| (e - 1) * s)
            .sum();
        1 + steps
    }

    /// Whether `index` is in bounds on every axis (`index[d] < extents[d]`).
    fn contains(&self, index: [usize; N]) -> bool {
        index.iter().zip(&self.extents()).all(|(i, e)| i < e)
    }

    /// Physical offset of `index`, or `None` if it is out of bounds.
    fn offset_checked(&self, index: [usize; N]) -> Option<usize> {
        self.contains(index)
            .then(|| index.iter().zip(self.strides()).map(|(i, s)| i * s).sum())
    }

    /// Physical offset of `index` — `Σ index[d] · strides[d]`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds on any axis.
    fn offset(&self, index: [usize; N]) -> usize {
        self.offset_checked(index).unwrap_or_else(|| {
            panic!(
                "index {:?} out of bounds for extents {:?}",
                index,
                self.extents()
            )
        })
    }

    /// Iterates over physical offsets.
    fn offsets(&self) -> Offsets<N> {
        Offsets {
            extents: self.extents(),
            strides: self.strides(),
            indices: [0; N],
            offset: 0,
            remaining: self.len(),
        }
    }
}

/// Strided layout policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strided<const N: usize> {
    extents: [usize; N],
    strides: [usize; N],
}

impl<const N: usize> Strided<N> {
    pub fn new(extents: [usize; N], strides: [usize; N]) -> Self {
        Strided { extents, strides }
    }
}

impl<const N: usize> Layout<N> for Strided<N> {
    fn extents(&self) -> [usize; N] {
        self.extents
    }

    fn strides(&self) -> [usize; N] {
        self.strides
    }

    fn is_contiguous(&self) -> bool {
        RowMajor::new(self.extents).strides() == self.strides
    }
}

/// Row-major layout policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowMajor<const N: usize> {
    extents: [usize; N],
}

impl<const N: usize> RowMajor<N> {
    pub fn new(extents: [usize; N]) -> Self {
        RowMajor { extents }
    }
}

impl<const N: usize> Layout<N> for RowMajor<N> {
    fn extents(&self) -> [usize; N] {
        self.extents
    }

    fn strides(&self) -> [usize; N] {
        let mut strides = [0; N];
        let mut acc = 1;
        let mut d = N;
        while d > 0 {
            d -= 1;
            strides[d] = acc;
            acc *= self.extents[d];
        }
        strides
    }

    fn is_contiguous(&self) -> bool {
        true
    }
}

impl<const N: usize> From<RowMajor<N>> for Strided<N> {
    fn from(layout: RowMajor<N>) -> Self {
        Strided::new(layout.extents(), layout.strides())
    }
}

/// Column-major layout policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColMajor<const N: usize> {
    extents: [usize; N],
}

impl<const N: usize> ColMajor<N> {
    pub fn new(extents: [usize; N]) -> Self {
        ColMajor { extents }
    }
}

impl<const N: usize> Layout<N> for ColMajor<N> {
    fn extents(&self) -> [usize; N] {
        self.extents
    }

    fn strides(&self) -> [usize; N] {
        let mut strides = [0; N];
        let mut acc = 1;
        let mut d = 0;
        while d < N {
            strides[d] = acc;
            acc *= self.extents[d];
            d += 1;
        }
        strides
    }

    fn is_contiguous(&self) -> bool {
        // With at most one axis, column-major and row-major coincide.
        N <= 1
    }
}

impl<const N: usize> From<ColMajor<N>> for Strided<N> {
    fn from(layout: ColMajor<N>) -> Self {
        Strided::new(layout.extents(), layout.strides())
    }
}

/// Iterator over physical offsets.
#[derive(Debug, Clone)]
pub struct Offsets<const N: usize> {
    extents: [usize; N],
    strides: [usize; N],
    indices: [usize; N],
    offset: usize,
    remaining: usize,
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
