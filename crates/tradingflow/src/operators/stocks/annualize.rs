use crate::data::{Array, ArrayView, Instant};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Convert a YTD cumulative vector `[year, day_of_year, ytd_1..ytd_N]` into
/// annualized `[N]` values via days-based scaling (rank-1 in / rank-1 out).
#[derive(Clone, Default)]
pub struct Annualize;

impl Annualize {
    pub fn new() -> Self {
        Self
    }
}

/// Runtime state for [`Annualize`]: the previous-tick YTD snapshot plus the
/// output buffer.
pub struct AnnualizeState {
    prev_ytd: Vec<f64>,
    prev_year: i64,
    prev_day: f64,
    initialized: bool,
    out: Array<f64, 1>,
}

impl Operator for Annualize {
    type Inputs = ArrayPort<f64, 1>;
    type Outputs = ArrayPort<f64, 1>;
    type Context = Instant;
    type State = AnnualizeState;

    fn init(self, (_, view): (bool, ArrayView<'_, f64, 1>)) -> AnnualizeState {
        // Only the build-time input's shape is read here, to size the buffers.
        let input_len = view.to_contiguous().len();
        assert!(
            input_len >= 3,
            "Annualize: input must have shape [2 + N] with N >= 1, got length {input_len}"
        );
        let n = input_len - 2;
        AnnualizeState {
            prev_ytd: vec![0.0; n],
            prev_year: 0,
            prev_day: 0.0,
            initialized: false,
            out: Array::zeros([n]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, view): (bool, ArrayView<'a, f64, 1>),
        state: &'b mut AnnualizeState,
        _: &Instant,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        let input = view.to_contiguous();
        let year = input[0].floor() as i64;
        let day = input[1];
        let ytd = &input[2..];
        let n = ytd.len();
        let out = state.out.data_mut();

        let (is_new_year, days_elapsed) = if !state.initialized || year != state.prev_year {
            (true, day)
        } else {
            (false, day - state.prev_day)
        };

        if days_elapsed <= 0.0 {
            out.fill(f64::NAN);
        } else {
            let scale = 365.0 / days_elapsed;
            for i in 0..n {
                let delta = if is_new_year {
                    ytd[i]
                } else {
                    ytd[i] - state.prev_ytd[i]
                };
                out[i] = delta * scale;
            }
        }

        state.prev_ytd.copy_from_slice(ytd);
        state.prev_year = year;
        state.prev_day = day;
        state.initialized = true;
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, f64, 1>),
        state: &'b mut AnnualizeState,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        (false, state.out.view())
    }
}

/// Annualize a periodic (report-cadence) flow.
pub fn annualize() -> Annualize {
    Annualize::new()
}
