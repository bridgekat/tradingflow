use num_traits::Float;
use std::marker::PhantomData;

use super::rank::rank_positions;
use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::ArrayPort;

/// Operator signature for [`corr`] etc.
pub struct Corr<T: Scalar + Float, const N: usize> {
    ranked: bool,
    min_count: usize,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> Corr<T, N> {
    /// When `ranked` is set, both sides are replaced by their average-tie
    /// ranks over the pairwise-valid subset before correlating, giving the
    /// Spearman form.
    pub fn new(ranked: bool, min_count: usize) -> Self {
        Self {
            ranked,
            min_count,
            _marker: PhantomData,
        }
    }
}

/// Runtime state for [`Corr`].
pub struct CorrState<T: Scalar + Float> {
    ranked: bool,
    min_count: usize,
    xs: Vec<f64>,
    ys: Vec<f64>,
    idx: Vec<usize>,
    rx: Vec<f64>,
    ry: Vec<f64>,
    out: Array<T, 0>,
}

impl<T: Scalar + Float, const N: usize> Operator for Corr<T, N> {
    type Inputs = (ArrayPort<T, N>, ArrayPort<T, N>);
    type Outputs = ArrayPort<T, 0>;
    type Context = Instant;
    type State = CorrState<T>;

    fn init(self, (x, y): (ArrayView<'_, T, N>, ArrayView<'_, T, N>)) -> Self::State {
        assert_eq!(
            x.extents(),
            y.extents(),
            "stats::corr: input extents mismatch"
        );
        let mut state = CorrState {
            ranked: self.ranked,
            min_count: self.min_count,
            xs: Vec::new(),
            ys: Vec::new(),
            idx: Vec::new(),
            rx: Vec::new(),
            ry: Vec::new(),
            out: Array::full([], T::nan()),
        };
        corr_into(&mut state, x, y);
        state
    }

    fn reset<'a, 'b: 'a>(
        _: (ArrayView<'a, T, N>, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> ArrayView<'a, T, 0> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        (x, y): (ArrayView<'a, T, N>, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, T, 0> {
        corr_into(state, x, y);
        state.out.view()
    }
}

fn corr_into<T: Scalar + Float, const N: usize>(
    state: &mut CorrState<T>,
    x: ArrayView<'_, T, N>,
    y: ArrayView<'_, T, N>,
) {
    let CorrState {
        ranked,
        min_count,
        xs,
        ys,
        idx,
        rx,
        ry,
        out,
    } = state;

    // Pairwise-complete filter: keep positions where both sides are finite.
    xs.clear();
    ys.clear();
    for (&a, &b) in x.iter().zip(y.iter()) {
        if a.is_finite() && b.is_finite() {
            xs.push(a.to_f64().unwrap());
            ys.push(b.to_f64().unwrap());
        }
    }

    let r = if xs.len() >= (*min_count).max(2) {
        if *ranked {
            idx.resize(xs.len(), 0);
            rx.resize(xs.len(), 0.0);
            ry.resize(xs.len(), 0.0);
            rank_positions(xs, idx, rx);
            rank_positions(ys, idx, ry);
            pearson(rx, ry)
        } else {
            pearson(xs, ys)
        }
    } else {
        f64::NAN
    };
    out.data_mut()[0] = T::from(r).unwrap_or(T::nan());
}

/// Pearson correlation of two equal-length samples; `NaN` when either side
/// has zero variance.
fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let (mut var_x, mut var_y, mut cov) = (0.0, 0.0, 0.0);
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let (dx, dy) = (x - mean_x, y - mean_y);
        var_x += dx * dx;
        var_y += dy * dy;
        cov += dx * dy;
    }
    if var_x <= 0.0 || var_y <= 0.0 {
        f64::NAN
    } else {
        cov / (var_x * var_y).sqrt()
    }
}

/// Cross-sectional correlation coefficient between two arrays of the same
/// extents, as a rank-0 scalar. Can be either Pearson (`ranked == false`) or
/// Spearman (`ranked == true`) correlation. Elements are paired positionwise;
/// pairs with a non-finite side (NaN or ±∞) are skipped pairwise-complete,
/// and the result is `NaN` with fewer than `max(min_count, 2)` valid pairs
/// or a zero-variance side.
pub fn corr<T: Scalar + Float, const N: usize>(
    ranked: bool,
    min_count: usize,
) -> impl Operator<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, 0>,
    Context = Instant,
> {
    Corr::new(ranked, min_count)
}
