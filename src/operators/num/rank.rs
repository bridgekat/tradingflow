//! Rank and ArgSort operators over 1-D float arrays.
//!
//! Both share the same ordering: ascending (smallest first), with NaN
//! values pushed to the end.

use std::cmp::Ordering;
use std::marker::PhantomData;

use num_traits::Float;

use crate::Instant;
use crate::{Array, Input, InputTypes, Operator, Scalar};

/// Compare two floats so NaN sorts after every real value.  Used by
/// every operator in this module to guarantee a total order.
#[inline(always)]
fn cmp_nan_last<T: Float>(a: T, b: T) -> Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| {
        if a.is_nan() && b.is_nan() {
            Ordering::Equal
        } else if a.is_nan() {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    })
}

/// Produces the 0-based rank of each element in a 1-D array.
///
/// The smallest element is assigned rank 0, the next smallest rank 1,
/// and so on; NaN values receive the highest ranks.  Output is
/// `Array<u64>` of the same length as the input.
///
/// Conceptually, this is the inverse permutation of [`ArgSort`]:
/// `rank[argsort[i]] == i`.
pub struct Rank<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> Rank<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for Rank<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float> Operator for Rank<T> {
    type State = Vec<usize>;
    type Inputs = Input<Array<T>>;
    type Output = Array<u64>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> (Vec<usize>, Array<u64>) {
        let n = inputs.as_slice().len();
        let indices: Vec<usize> = (0..n).collect();
        let output = Array::from_vec(&[n], vec![0u64; n]);
        (indices, output)
    }

    #[inline(always)]
    fn compute(
        state: &mut Vec<usize>,
        inputs: &Array<T>,
        output: &mut Array<u64>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        let src = inputs.as_slice();
        let n = src.len();

        for i in 0..n {
            state[i] = i;
        }
        state.sort_by(|&a, &b| cmp_nan_last(src[a], src[b]));

        let dst = output.as_mut_slice();
        for rank in 0..n {
            dst[state[rank]] = rank as u64;
        }
        true
    }
}

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
    type Inputs = Input<Array<T>>;
    type Output = Array<u64>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> (Vec<usize>, Array<u64>) {
        let n = inputs.as_slice().len();
        let indices: Vec<usize> = (0..n).collect();
        let output = Array::from_vec(&[n], vec![0u64; n]);
        (indices, output)
    }

    #[inline(always)]
    fn compute(
        state: &mut Vec<usize>,
        inputs: &Array<T>,
        output: &mut Array<u64>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        let src = inputs.as_slice();
        let n = src.len();

        for i in 0..n {
            state[i] = i;
        }
        state.sort_by(|&a, &b| cmp_nan_last(src[a], src[b]));

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
    fn rank_basic() {
        let a = Array::from_vec(&[5], vec![30.0, 10.0, 50.0, 20.0, 40.0_f64]);
        let (mut s, mut o) = Rank::<f64>::new().init(&a, Instant::from_nanos(0));
        Rank::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        // 30 is 3rd smallest → rank 2, 10 → 0, 50 → 4, 20 → 1, 40 → 3.
        assert_eq!(o.as_slice(), &[2, 0, 4, 1, 3]);
    }

    #[test]
    fn rank_with_nan() {
        let a = Array::from_vec(&[4], vec![f64::NAN, 20.0, 10.0, f64::NAN]);
        let (mut s, mut o) = Rank::<f64>::new().init(&a, Instant::from_nanos(0));
        Rank::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        // 10 → rank 0, 20 → rank 1, NaNs → ranks 2 and 3 (order unspecified).
        assert_eq!(o.as_slice()[1], 1);
        assert_eq!(o.as_slice()[2], 0);
        assert!(o.as_slice()[0] >= 2 && o.as_slice()[3] >= 2);
    }

    #[test]
    fn argsort_basic() {
        let a = Array::from_vec(&[5], vec![30.0, 10.0, 50.0, 20.0, 40.0_f64]);
        let (mut s, mut o) = ArgSort::<f64>::new().init(&a, Instant::from_nanos(0));
        ArgSort::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        // sorted: 10(1), 20(3), 30(0), 40(4), 50(2)
        assert_eq!(o.as_slice(), &[1, 3, 0, 4, 2]);
    }

    #[test]
    fn argsort_with_nan() {
        let a = Array::from_vec(&[4], vec![f64::NAN, 20.0, 10.0, f64::NAN]);
        let (mut s, mut o) = ArgSort::<f64>::new().init(&a, Instant::from_nanos(0));
        ArgSort::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        // sorted: 10(2), 20(1), NaN(0), NaN(3)
        assert_eq!(o.as_slice()[0], 2);
        assert_eq!(o.as_slice()[1], 1);
        // NaN indices at end (order between NaNs is stable but unspecified)
    }

    #[test]
    fn argsort_single_element() {
        let a = Array::from_vec(&[1], vec![42.0_f64]);
        let (mut s, mut o) = ArgSort::<f64>::new().init(&a, Instant::from_nanos(0));
        ArgSort::compute(&mut s, &a, &mut o, Instant::from_nanos(1), false);
        assert_eq!(o.as_slice(), &[0]);
    }
}
