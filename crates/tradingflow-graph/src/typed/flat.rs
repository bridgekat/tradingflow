/// Single-pass cursor reader over a flat slice `&'a [T]`.
pub struct FlatRead<'a, T> {
    data: &'a [T],
    index: usize,
}

impl<'a, T> FlatRead<'a, T> {
    /// Wrap a slice; cursor starts at position 0.
    pub fn new(data: &'a [T]) -> Self {
        Self { data, index: 0 }
    }

    /// Number of elements remaining past the cursor.
    pub fn remaining(&self) -> usize {
        self.data.len() - self.index
    }

    /// Consume one element, returning a reference.
    pub fn pop(&mut self) -> &'a T {
        let v = &self.data[self.index];
        self.index += 1;
        v
    }

    /// Consume `n` elements, returning a sub-slice.
    pub fn take(&mut self, n: usize) -> &'a [T] {
        let slice = &self.data[self.index..self.index + n];
        self.index += n;
        slice
    }
}

/// Single-pass cursor writer over a flat mutable slice `&'a mut [T]`.
pub struct FlatWrite<'a, T> {
    data: &'a mut [T],
    index: usize,
}

impl<'a, T> FlatWrite<'a, T> {
    /// Wrap a mutable slice; cursor starts at position 0.
    pub fn new(data: &'a mut [T]) -> Self {
        Self { data, index: 0 }
    }

    /// Number of elements remaining past the cursor.
    pub fn remaining(&self) -> usize {
        self.data.len() - self.index
    }

    /// Write `v` at the cursor position, then advance cursor by 1.
    pub fn push(&mut self, v: T) {
        self.data[self.index] = v;
        self.index += 1;
    }

    /// Write all of `vs` at the cursor, advancing cursor past them.
    pub fn extend(&mut self, vs: &[T])
    where
        T: Copy,
    {
        self.data[self.index..self.index + vs.len()].copy_from_slice(vs);
        self.index += vs.len();
    }
}
