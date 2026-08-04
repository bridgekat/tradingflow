use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::ArrayPort;

pub struct GroupDemean<T: Scalar + Float, const N: usize> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> GroupDemean<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for GroupDemean<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GroupDemeanState<T: Scalar + Float, const N: usize> {
    /// Per-group `(sum, count)` scratch, indexed by group id and grown to the
    /// largest id seen.
    sums: Vec<T>,
    counts: Vec<usize>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for GroupDemean<T, N> {
    type Inputs = (ArrayPort<T, N>, ArrayPort<u32, N>);
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = GroupDemeanState<T, N>;

    fn init(self, (x, tags): (ArrayView<'_, T, N>, ArrayView<'_, u32, N>)) -> Self::State {
        assert_eq!(
            x.extents(),
            tags.extents(),
            "group_demean: one group id per element required"
        );
        let mut state = GroupDemeanState {
            sums: Vec::new(),
            counts: Vec::new(),
            out: Array::zeros(x.extents()),
        };
        group_demean_into(&mut state, x, tags);
        state
    }

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, T, N>, ArrayView<'a, u32, N>),
        state: &'b mut Self::State,
    ) -> ArrayView<'a, T, N> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        (x, tags): (ArrayView<'a, T, N>, ArrayView<'a, u32, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, T, N> {
        group_demean_into(state, x, tags);
        state.out.view()
    }
}

fn group_demean_into<T: Scalar + Float, const N: usize>(
    state: &mut GroupDemeanState<T, N>,
    x: ArrayView<'_, T, N>,
    tags: ArrayView<'_, u32, N>,
) {
    let num_groups = tags.iter().max().map_or(0, |&g| g as usize + 1);
    state.sums.clear();
    state.sums.resize(num_groups, T::zero());
    state.counts.clear();
    state.counts.resize(num_groups, 0);
    for (&v, &g) in x.iter().zip(tags.iter()) {
        if v.is_finite() {
            state.sums[g as usize] = state.sums[g as usize] + v;
            state.counts[g as usize] += 1;
        }
    }
    let o = state.out.data_mut();
    for (i, (&v, &g)) in x.iter().zip(tags.iter()).enumerate() {
        o[i] = if v.is_finite() {
            v - state.sums[g as usize] / T::from(state.counts[g as usize]).unwrap()
        } else {
            T::nan()
        };
    }
}

/// Cross-sectional demean within groups: `x − mean(x over the element's
/// group)`, the group-neutralization primitive (Alpha101's `IndNeutralize`).
/// The second input tags each element with a group id and may change over
/// time — every tick regroups against the current tags (ids should be small
/// dense integers; the scratch is sized by the largest). Non-finite entries
/// of `x` (NaN or ±∞) are treated as missing: they map to NaN and do not
/// contribute to their group's mean.
pub fn group_demean<T: Scalar + Float, const N: usize>() -> impl Operator<
    Inputs = (ArrayPort<T, N>, ArrayPort<u32, N>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    GroupDemean::new()
}
