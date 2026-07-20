use super::Layout;

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
        N <= 1
    }
}

impl<const N: usize> From<ColMajor<N>> for Strided<N> {
    fn from(layout: ColMajor<N>) -> Self {
        Strided::new(layout.extents(), layout.strides())
    }
}
