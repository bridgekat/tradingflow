use super::{IntoSliceReshapes, IntoSlices, Offsets, Slice, SliceReshape, Strided};

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

    /// Same as [`span`](Self::span), but with an additional axis.
    /// Used internally by series.
    fn span_ext(&self, extent: usize, stride: usize) -> usize {
        match extent {
            0 => 0,
            n => (n - 1) * stride + self.span(),
        }
    }

    /// Whether `index` is in bounds on every axis (`index[d] < extents[d]`).
    fn offset_contains(&self, index: [usize; N]) -> bool {
        index.iter().zip(&self.extents()).all(|(i, e)| i < e)
    }

    /// Physical offset of `index`, or `None` if it is out of bounds.
    fn offset_checked(&self, index: [usize; N]) -> Option<usize> {
        if self.offset_contains(index) {
            Some(index.iter().zip(self.strides()).map(|(i, s)| i * s).sum())
        } else {
            None
        }
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

    /// Whether `slices` selects only in-bounds indices on every axis: a
    /// bounded, non-empty slice's last index must lie below the extent, and
    /// any other slice must start within the axis.
    fn slice_contains(&self, slices: impl IntoSlices<N>) -> bool {
        let slices = slices.into_slices();
        slices
            .iter()
            .zip(&self.extents())
            .all(|(s, e)| match s.count() {
                Some(c) if c > 0 => s.start() + (c - 1) * s.step() < *e,
                _ => s.start() <= *e,
            })
    }

    /// Base offset and layout of the sub-region selected by `slices`, or
    /// `None` if `slices` is out of bounds on any axis.
    fn slice_checked(&self, slices: impl IntoSlices<N>) -> Option<(usize, Strided<N>)> {
        let slices = slices.into_slices();
        if self.slice_contains(slices) {
            let extents = self.extents();
            let strides = self.strides();
            let offset = slices
                .iter()
                .zip(&strides)
                .map(|(sl, st)| sl.start() * st)
                .sum();
            let extents = std::array::from_fn(|d| slices[d].len(extents[d]));
            let strides = std::array::from_fn(|d| strides[d] * slices[d].step());
            Some((offset, Strided::new(extents, strides)))
        } else {
            None
        }
    }

    /// Base offset and layout of the sub-region selected by `slices`.
    ///
    /// # Panics
    ///
    /// Panics if `slices` is out of bounds on any axis.
    fn slice(&self, slices: impl IntoSlices<N>) -> (usize, Strided<N>) {
        let slices = slices.into_slices();
        self.slice_checked(slices).unwrap_or_else(|| {
            panic!(
                "slices {:?} out of bounds for extents {:?}",
                slices,
                self.extents()
            )
        })
    }

    /// Base offset and layout of the sub-region selected by `slices`, or
    /// `None` if `slices` is out of bounds on any axis.
    ///
    /// # Panics
    ///
    /// Panics if `slices` does not consume exactly `N` axes or produce
    /// exactly `M` axes.
    fn slice_reshape_checked<const M: usize, const K: usize>(
        &self,
        slices: impl IntoSliceReshapes<K>,
    ) -> Option<(usize, Strided<M>)> {
        let slices = slices.into_slice_reshapes();
        let consumed = slices
            .iter()
            .filter(|s| !matches!(s, SliceReshape::NewAxis))
            .count();
        let produced = slices
            .iter()
            .filter(|s| !matches!(s, SliceReshape::Index(_)))
            .count();
        assert_eq!(
            consumed, N,
            "slice_reshape: consumed axes ({consumed}) must match rank ({N})"
        );
        assert_eq!(
            produced, M,
            "slice_reshape: M ({M}) must match produced axes ({produced})"
        );
        let extents = self.extents();
        let strides = self.strides();
        let mut offset = 0;
        let mut new_extents = [0; M];
        let mut new_strides = [0; M];
        let (mut src, mut dst) = (0, 0);
        for spec in slices {
            match spec {
                SliceReshape::Slice(s) => {
                    let in_bounds = match s.count() {
                        Some(c) if c > 0 => s.start() + (c - 1) * s.step() < extents[src],
                        _ => s.start() <= extents[src],
                    };
                    if !in_bounds {
                        return None;
                    }
                    offset += s.start() * strides[src];
                    new_extents[dst] = s.len(extents[src]);
                    new_strides[dst] = strides[src] * s.step();
                    src += 1;
                    dst += 1;
                }
                SliceReshape::Index(i) => {
                    if i >= extents[src] {
                        return None;
                    }
                    offset += i * strides[src];
                    src += 1;
                }
                SliceReshape::NewAxis => {
                    new_extents[dst] = 1;
                    new_strides[dst] = 1;
                    dst += 1;
                }
            }
        }
        Some((offset, Strided::new(new_extents, new_strides)))
    }

    /// Base offset and layout of the sub-region selected by `slices`.
    ///
    /// # Panics
    ///
    /// Panics if `slices` does not consume exactly `N` axes or produce
    /// exactly `M` axes, or is out of bounds on any axis.
    fn slice_reshape<const M: usize, const K: usize>(
        &self,
        slices: impl IntoSliceReshapes<K>,
    ) -> (usize, Strided<M>) {
        let slices = slices.into_slice_reshapes();
        self.slice_reshape_checked(slices).unwrap_or_else(|| {
            panic!(
                "slices {:?} out of bounds for extents {:?}",
                slices,
                self.extents()
            )
        })
    }

    /// Base offset and layout with `axis` restricted to `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= N` or `slice` is out of bounds.
    fn slice_along_axis(&self, axis: usize, slice: impl Into<Slice>) -> (usize, Strided<N>) {
        assert!(axis < N, "axis {axis} out of bounds for rank {N}");
        let mut slices = [Slice::from(..); N];
        slices[axis] = slice.into();
        self.slice(slices)
    }

    /// Iterates over physical offsets.
    fn iter(&self) -> Offsets<N> {
        Offsets {
            extents: self.extents(),
            strides: self.strides(),
            indices: [0; N],
            offset: 0,
            remaining: self.len(),
        }
    }
}
