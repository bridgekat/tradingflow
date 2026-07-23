use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Layout, Scalar, SeriesView};
use crate::graph::typed::Operator;
use crate::ports::{ArrayPort, SeriesPort};

/// Exponential moving average with window-normalized weights (output rank `NO`
/// = input element rank).
#[derive(Clone)]
pub struct Ema<T: Scalar + Float, const NO: usize> {
    alpha: T,
    window: usize,
}

impl<T: Scalar + Float, const NO: usize> Ema<T, NO> {
    pub fn new(alpha: T, window: usize) -> Self {
        assert!(
            alpha > T::zero() && alpha <= T::one(),
            "alpha must be in (0, 1]"
        );
        assert!(window >= 1, "window must be >= 1");
        Self { alpha, window }
    }

    pub fn with_span(span: usize, window: usize) -> Self {
        assert!(span >= 1, "span must be >= 1");
        let alpha = T::from(2.0).unwrap() / T::from(span + 1).unwrap();
        Self::new(alpha, window)
    }

    pub fn with_half_life(half_life: T, window: usize) -> Self {
        assert!(half_life > T::zero(), "half_life must be > 0");
        let alpha = T::one() - (-T::from(2.0).unwrap().ln() / half_life).exp();
        Self::new(alpha, window)
    }
}

/// Runtime state for [`Ema`]: the decay config, weighted-sum bookkeeping,
/// plus the output buffer.
pub struct EmaState<T: Scalar + Float, const NO: usize> {
    alpha: T,
    one_minus_alpha: T,
    decay_factor: T,
    window: usize,
    weighted_sum: Vec<T>,
    nonfinite_count: Vec<u32>,
    fill_decay: T,
    out: Array<T, NO>,
}

impl<T: Scalar + Float, const NO: usize> Operator for Ema<T, NO> {
    type Inputs = SeriesPort<T, NO>;
    type Outputs = ArrayPort<T, NO>;
    type Context = Instant;
    type State = EmaState<T, NO>;

    fn init(self, (_, series): (bool, SeriesView<'_, T, NO>)) -> Self::State {
        let one_minus_alpha = T::one() - self.alpha;
        let mut decay_factor = T::one();
        for _ in 0..self.window {
            decay_factor = decay_factor * one_minus_alpha;
        }
        let stride = series.layout().len();
        EmaState {
            alpha: self.alpha,
            one_minus_alpha,
            decay_factor,
            window: self.window,
            weighted_sum: vec![T::zero(); stride],
            nonfinite_count: vec![0; stride],
            fill_decay: T::one(),
            out: Array::from_parts(series.extents(), vec![T::nan(); stride].into()),
        }
    }

    #[expect(
        clippy::needless_range_loop,
        reason = "i walks several parallel per-column arrays plus `series.elem(..)[[i]]`"
    )]
    fn compute<'a, 'b: 'a>(
        (_, series): (bool, SeriesView<'a, T, NO>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, NO>) {
        // `end` is the total rows ever recorded (logical indices count trimmed
        // rows), which is what the warm-up logic wants — not the retained len.
        let end = series.range().end;
        let row = series.at(end - 1).1.to_contiguous();
        let stride = row.len();
        // Materialized once: the evicted row is read on every column below.
        let evicted =
            (end > state.window).then(|| series.at(end - 1 - state.window).1.to_contiguous());
        let alpha = state.alpha;
        let one_minus_alpha = state.one_minus_alpha;

        state.fill_decay = state.fill_decay * one_minus_alpha;
        let weight_sum = T::one()
            - if end >= state.window {
                state.decay_factor
            } else {
                state.fill_decay
            };

        for i in 0..stride {
            let x = row[i];
            state.weighted_sum[i] = state.weighted_sum[i] * one_minus_alpha;
            if !x.is_finite() {
                state.nonfinite_count[i] += 1;
            } else {
                state.weighted_sum[i] = state.weighted_sum[i] + alpha * x;
            }
            if let Some(evicted) = &evicted {
                let x_old = evicted[i];
                if !x_old.is_finite() {
                    state.nonfinite_count[i] -= 1;
                } else {
                    let evict_weight = alpha * state.decay_factor;
                    state.weighted_sum[i] = state.weighted_sum[i] - evict_weight * x_old;
                }
            }
        }

        if end < state.window {
            (false, state.out.view())
        } else {
            let out = state.out.data_mut();
            for i in 0..stride {
                out[i] = if state.nonfinite_count[i] == 0 && weight_sum > T::zero() {
                    state.weighted_sum[i] / weight_sum
                } else {
                    T::nan()
                };
            }
            (true, state.out.view())
        }
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, SeriesView<'a, T, NO>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, NO>) {
        (false, state.out.view())
    }
}

/// [`Ema`] over a recorded [`Series`](tradingflow_data::Series) — the primitive behind the
/// self-recording [`ema`](crate::operators::formula::ema). (Named `_series` because `ema` is taken
/// by its live-array counterpart.)
pub fn ema_series<T: Scalar + Float, const NO: usize>(alpha: T, window: usize) -> Ema<T, NO> {
    Ema::new(alpha, window)
}
