//! Annualize operator for YTD financial report data.
//!
//! [`Annualize`] converts year-to-date cumulative values into annualized
//! values using days-based scaling.  It handles any reporting frequency
//! (quarterly, semi-annual, annual) uniformly.

use crate::{Array, Notify, Operator};

/// Convert year-to-date (YTD) financial values into annualized values.
///
/// # Input
///
/// A single `Array<f64>` of shape `[2 + N]`, laid out as
/// `[year, day_of_year, ytd_1, …, ytd_N]`.  The first two elements are
/// report-date metadata produced by
/// [`FinancialReportSource`](crate::sources::FinancialReportSource) with
/// `with_report_date = true`.
///
/// # Output
///
/// `Array<f64>` of shape `[N]` containing the annualized values.
///
/// # Algorithm
///
/// On each report event:
///
/// 1. Extract `year: i64` and `day_of_year: f64` from the input.
/// 2. If this is the first report or a new calendar year (`year ≠
///    prev_year`), let `delta = ytd` and `days_elapsed = day_of_year`.
/// 3. Otherwise, `delta = ytd − prev_ytd` and
///    `days_elapsed = day_of_year − prev_day_of_year`.
/// 4. `annualized = delta × 365 / days_elapsed`.
///
/// For a standard Q2 report: `days_elapsed = 181 − 90 = 91`, so
/// `annualized ≈ delta × 4.01`.
pub struct Annualize;

impl Annualize {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Annualize {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for the annualize operator.
pub struct AnnualizeState {
    prev_ytd: Vec<f64>,
    prev_year: i64,
    prev_day: f64,
    initialized: bool,
}

impl Operator for Annualize {
    type State = AnnualizeState;
    type Inputs = (Array<f64>,);
    type Output = Array<f64>;

    fn init(self, inputs: (&Array<f64>,), _timestamp: i64) -> (AnnualizeState, Array<f64>) {
        let input_len = inputs.0.as_slice().len();
        assert!(
            input_len >= 3,
            "Annualize: input must have shape [2 + N] with N >= 1, got length {input_len}"
        );
        let n = input_len - 2;
        let state = AnnualizeState {
            prev_ytd: vec![0.0; n],
            prev_year: 0,
            prev_day: 0.0,
            initialized: false,
        };
        (state, Array::zeros(&[n]))
    }

    fn compute(
        state: &mut AnnualizeState,
        inputs: (&Array<f64>,),
        output: &mut Array<f64>,
        _timestamp: i64,
        _notify: &Notify<'_>,
    ) -> bool {
        let input = inputs.0.as_slice();
        let year = input[0].floor() as i64;
        let day = input[1];
        let ytd = &input[2..];
        let n = ytd.len();
        let out = output.as_mut_slice();

        // Check if this is the first report for the year.
        let (is_new_year, days_elapsed) = if !state.initialized || year != state.prev_year {
            (true, day)
        } else {
            (false, day - state.prev_day)
        };

        // Guard against zero / negative elapsed days (data error).
        if days_elapsed <= 0.0 {
            for o in out.iter_mut() {
                *o = f64::NAN;
            }
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

        // Update state.
        state.prev_ytd.copy_from_slice(ytd);
        state.prev_year = year;
        state.prev_day = day;
        state.initialized = true;

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build an input array `[year, day_of_year, v1, v2, …]`.
    fn input(year: f64, day: f64, values: &[f64]) -> Array<f64> {
        let mut data = vec![year, day];
        data.extend_from_slice(values);
        Array::from_vec(&[data.len()], data)
    }

    fn notify() -> Notify<'static> {
        Notify::new(&[true], &[0])
    }

    #[test]
    fn q1_annualizes_from_zero() {
        // Q1 2024: report date = 2024-03-31, day 91.
        // YTD revenue = 100, net_income = 20.
        let inp = input(2024.0, 91.0, &[100.0, 20.0]);
        let (mut s, mut o) = Annualize::new().init((&inp,), 0);

        Annualize::compute(&mut s, (&inp,), &mut o, 1, &notify());
        // annualized = ytd * 365 / 91
        let expected_rev = 100.0 * 365.0 / 91.0;
        let expected_ni = 20.0 * 365.0 / 91.0;
        assert!((o.as_slice()[0] - expected_rev).abs() < 1e-10);
        assert!((o.as_slice()[1] - expected_ni).abs() < 1e-10);
    }

    #[test]
    fn q2_differences_with_q1() {
        // Q1: day 90, YTD = [100, 20].
        let q1 = input(2024.0, 90.0, &[100.0, 20.0]);
        let (mut s, mut o) = Annualize::new().init((&q1,), 0);
        Annualize::compute(&mut s, (&q1,), &mut o, 1, &notify());

        // H1: day 181, YTD = [250, 55].
        let h1 = input(2024.0, 181.0, &[250.0, 55.0]);
        Annualize::compute(&mut s, (&h1,), &mut o, 2, &notify());

        // delta = [150, 35], days_elapsed = 91.
        let expected_rev = 150.0 * 365.0 / 91.0;
        let expected_ni = 35.0 * 365.0 / 91.0;
        assert!((o.as_slice()[0] - expected_rev).abs() < 1e-10);
        assert!((o.as_slice()[1] - expected_ni).abs() < 1e-10);
    }

    #[test]
    fn year_boundary_resets() {
        // Annual 2023: day 365, YTD = [1000].
        let annual = input(2023.0, 365.0, &[1000.0]);
        let (mut s, mut o) = Annualize::new().init((&annual,), 0);
        Annualize::compute(&mut s, (&annual,), &mut o, 1, &notify());

        // Q1 2024: day 91, YTD = [300].
        let q1 = input(2024.0, 91.0, &[300.0]);
        Annualize::compute(&mut s, (&q1,), &mut o, 2, &notify());

        // New year → delta = ytd = 300, days_elapsed = 91.
        let expected = 300.0 * 365.0 / 91.0;
        assert!((o.as_slice()[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn full_year_sequence() {
        let q1 = input(2024.0, 91.0, &[100.0]);
        let (mut s, mut o) = Annualize::new().init((&q1,), 0);
        Annualize::compute(&mut s, (&q1,), &mut o, 1, &notify());

        let h1 = input(2024.0, 182.0, &[210.0]);
        Annualize::compute(&mut s, (&h1,), &mut o, 2, &notify());
        // delta = 110, days = 91.
        assert!((o.as_slice()[0] - 110.0 * 365.0 / 91.0).abs() < 1e-10);

        let q3 = input(2024.0, 274.0, &[330.0]);
        Annualize::compute(&mut s, (&q3,), &mut o, 3, &notify());
        // delta = 120, days = 92.
        assert!((o.as_slice()[0] - 120.0 * 365.0 / 92.0).abs() < 1e-10);

        let annual = input(2024.0, 366.0, &[460.0]);
        Annualize::compute(&mut s, (&annual,), &mut o, 4, &notify());
        // delta = 130, days = 92.
        assert!((o.as_slice()[0] - 130.0 * 365.0 / 92.0).abs() < 1e-10);
    }

    #[test]
    fn nan_propagation() {
        let q1 = input(2024.0, 91.0, &[f64::NAN, 20.0]);
        let (mut s, mut o) = Annualize::new().init((&q1,), 0);
        Annualize::compute(&mut s, (&q1,), &mut o, 1, &notify());

        // NaN * scale = NaN.
        assert!(o.as_slice()[0].is_nan());
        // Non-NaN should be fine.
        assert!((o.as_slice()[1] - 20.0 * 365.0 / 91.0).abs() < 1e-10);
    }

    #[test]
    fn semi_annual_reporting() {
        // H1: day 182, YTD = [500].
        let h1 = input(2024.0, 182.0, &[500.0]);
        let (mut s, mut o) = Annualize::new().init((&h1,), 0);
        Annualize::compute(&mut s, (&h1,), &mut o, 1, &notify());
        // First report: annualized = 500 * 365 / 182.
        assert!((o.as_slice()[0] - 500.0 * 365.0 / 182.0).abs() < 1e-10);

        // Annual: day 366, YTD = [1100].
        let annual = input(2024.0, 366.0, &[1100.0]);
        Annualize::compute(&mut s, (&annual,), &mut o, 2, &notify());
        // delta = 600, days = 184.
        assert!((o.as_slice()[0] - 600.0 * 365.0 / 184.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Annualize: input must have shape")]
    fn rejects_too_small_input() {
        let inp = Array::from_vec(&[2], vec![2024.0, 91.0]); // no value columns
        Annualize::new().init((&inp,), 0);
    }
}
