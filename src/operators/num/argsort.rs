//! ArgSort operator — returns indices that would sort the array.

use std::marker::PhantomData;

use num_traits::Float;

use crate::time::Instant;
use crate::{Array, Notify, Operator, Scalar};

/// Produces the indices that would sort a 1-D array from smallest to
/// largest.  Output is `Array<u64>` of the same length as the input.
///
/// NaN values are sorted to the end (assigned the highest indices).
pub struct ArgSort<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> ArgSort<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for ArgSort<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float> Operator for ArgSort<T> {
    type State = Vec<usize>;
    type Inputs = (Array<T>,);
    type Output = Array<u64>;

    fn init(self, inputs: (&Array<T>,), _timestamp: Instant) -> (Vec<usize>, Array<u64>) {
        let n = inputs.0.as_slice().len();
        let indices: Vec<usize> = (0..n).collect();
        let output = Array::from_vec(&[n], vec![0u64; n]);
        (indices, output)
    }

    #[inline(always)]
    fn compute(
        state: &mut Vec<usize>,
        inputs: (&Array<T>,),
        output: &mut Array<u64>,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        let src = inputs.0.as_slice();
        let n = src.len();

        // Reset index buffer.
        for i in 0..n {
            state[i] = i;
        }

        // Sort indices by value, NaN to the end.
        state.sort_by(|&a, &b| {
            let va = src[a];
            let vb = src[b];
            va.partial_cmp(&vb).unwrap_or_else(|| {
                // NaN handling: NaN sorts after everything.
                if va.is_nan() && vb.is_nan() {
                    std::cmp::Ordering::Equal
                } else if va.is_nan() {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            })
        });

        let dst = output.as_mut_slice();
        for i in 0..n {
            dst[i] = state[i] as u64;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let a = Array::from_vec(&[5], vec![30.0, 10.0, 50.0, 20.0, 40.0_f64]);
        let (mut s, mut o) = ArgSort::<f64>::new().init((&a,), Instant::from_nanos(0));
        ArgSort::compute(&mut s, (&a,), &mut o, Instant::from_nanos(1), &Notify::new(&[], 0));
        // sorted: 10(1), 20(3), 30(0), 40(4), 50(2)
        assert_eq!(o.as_slice(), &[1, 3, 0, 4, 2]);
    }

    #[test]
    fn with_nan() {
        let a = Array::from_vec(&[4], vec![f64::NAN, 20.0, 10.0, f64::NAN]);
        let (mut s, mut o) = ArgSort::<f64>::new().init((&a,), Instant::from_nanos(0));
        ArgSort::compute(&mut s, (&a,), &mut o, Instant::from_nanos(1), &Notify::new(&[], 0));
        // sorted: 10(2), 20(1), NaN(0), NaN(3)
        assert_eq!(o.as_slice()[0], 2);
        assert_eq!(o.as_slice()[1], 1);
        // NaN indices at end (order between NaNs is stable but unspecified)
    }

    #[test]
    fn single_element() {
        let a = Array::from_vec(&[1], vec![42.0_f64]);
        let (mut s, mut o) = ArgSort::<f64>::new().init((&a,), Instant::from_nanos(0));
        ArgSort::compute(&mut s, (&a,), &mut o, Instant::from_nanos(1), &Notify::new(&[], 0));
        assert_eq!(o.as_slice(), &[0]);
    }
}
