//! Integration tests for `operators::rolling`: the windowed accumulators
//! (`sum`, `mean`, `var`, `std_dev`, `cov`, `mean_exp`) and the window-offset
//! readers (`lag`, `diff`, `pct_change`), each in both spellings — the
//! `series_*` form over a hoisted `SeriesPort`, and the self-recording form
//! that buffers a live `(signal, values)` stream behind a private
//! `series::buffer`, ingesting one sample per signal.
//!
//! Window semantics, as implemented by `rolling::base::Rolling` driving an
//! `Accumulator` over `Retention::trim_count`:
//!
//! * A `usize` window keeps the most recent `w` ticks. Before the window fills
//!   the operators emit over the partial window rather than gating on length.
//! * A `Duration` window keeps every tick stamped **strictly after**
//!   `now - duration`; a tick stamped exactly at the cutoff is evicted. The
//!   number of ticks in the window therefore depends on the cadence.
//! * `min_count` gates on the number of **finite** samples currently in the
//!   window, per element (per *pair* for `cov`) — not on the window length.
//!   The accumulators skip non-finite samples entirely, so a `NaN` contributes
//!   neither to the statistic nor to the count; below `min_count` the element
//!   is `NaN`. `min_count` must be positive (asserted at construction).
//! * `lag` takes no `min_count`: it reports the newest sample that has already
//!   been evicted from the window, and `NaN` until one has been.
//!
//! Reference values are recomputed from scratch in each test rather than
//! hand-typed, so they document the intended formula. Inputs come from
//! `quarter_path`, whose running sums are exact in binary floating point, which
//! makes equality against a freshly-summed reference sound.

use tradingflow::data::{Array, Duration, SeriesView};
use tradingflow::graph::Pool;
use tradingflow::graph::typed::Builder;
use tradingflow::operators::{elem, rolling, series};

use crate::harness::*;

// ---------------------------------------------------------------------------
// Reference models
// ---------------------------------------------------------------------------

/// The samples a `usize` retention of `w` keeps at tick `t`: the trailing
/// `min(t + 1, w)` samples.
fn count_window(path: &[f64], t: usize, w: usize) -> &[f64] {
    let k = (t + 1).min(w);
    &path[t + 1 - k..=t]
}

/// Index of the oldest sample a `Duration` retention of `span` days keeps at
/// tick `t` — the first one stamped strictly after `days[t] - span`.
fn first_in_duration_window(days: &[i64], t: usize, span: i64) -> usize {
    let cutoff = days[t] - span;
    days[..=t].partition_point(|&d| d <= cutoff)
}

/// The samples a `Duration` retention of `span` days keeps at tick `t`.
fn duration_window<'a>(path: &'a [f64], days: &[i64], t: usize, span: i64) -> &'a [f64] {
    &path[first_in_duration_window(days, t, span)..=t]
}

/// The finite samples of a window — the only ones the accumulators see.
fn finite(xs: &[f64]) -> impl Iterator<Item = f64> {
    xs.iter().copied().filter(|x| x.is_finite())
}

/// The count `min_count` gates on.
fn n_finite(xs: &[f64]) -> usize {
    finite(xs).count()
}

/// Rolling sum reference.
fn ref_sum(xs: &[f64], min_count: usize) -> f64 {
    if n_finite(xs) < min_count {
        return f64::NAN;
    }
    finite(xs).sum()
}

/// Rolling mean reference.
fn ref_mean(xs: &[f64], min_count: usize) -> f64 {
    let n = n_finite(xs);
    if n < min_count {
        return f64::NAN;
    }
    finite(xs).sum::<f64>() / n as f64
}

/// Rolling variance reference — the *population* variance (divide by `n`),
/// written two-pass so it is independent of the operator's `E[x²] - E[x]²`
/// form.
fn ref_var(xs: &[f64], min_count: usize) -> f64 {
    let n = n_finite(xs);
    if n < min_count {
        return f64::NAN;
    }
    let m = finite(xs).sum::<f64>() / n as f64;
    finite(xs).map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64
}

/// Rolling covariance reference for one matrix entry, computed over the ticks
/// where *both* components are finite ("pairwise complete").
fn ref_cov(xi: &[f64], xj: &[f64], min_count: usize) -> f64 {
    let pairs: Vec<(f64, f64)> = xi
        .iter()
        .zip(xj)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(&a, &b)| (a, b))
        .collect();
    if pairs.len() < min_count {
        return f64::NAN;
    }
    let n = pairs.len() as f64;
    let mi = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let mj = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    pairs.iter().map(|(a, b)| (a - mi) * (b - mj)).sum::<f64>() / n
}

/// Rolling EWMA reference: the newest in-window sample carries weight `alpha`,
/// each older one a further factor of `1 - alpha`, normalized by the weights
/// actually present in the window.
fn ref_mean_exp(xs: &[f64], alpha: f64, min_count: usize) -> f64 {
    if n_finite(xs) < min_count {
        return f64::NAN;
    }
    let (mut num, mut den) = (0.0, 0.0);
    for (age, &x) in xs.iter().rev().enumerate() {
        if !x.is_finite() {
            continue;
        }
        let w = alpha * (1.0 - alpha).powi(age as i32);
        num += w * x;
        den += w;
    }
    if den > 0.0 { num / den } else { f64::NAN }
}

/// The whole reference covariance matrix over the trailing count window, in the
/// row-major order `cov` writes it.
fn ref_cov_matrix(paths: &[Vec<f64>], t: usize, w: usize, min_count: usize) -> Vec<f64> {
    let k = paths.len();
    let mut out = Vec::with_capacity(k * k);
    for i in 0..k {
        for j in 0..k {
            out.push(ref_cov(
                count_window(&paths[i], t, w),
                count_window(&paths[j], t, w),
                min_count,
            ));
        }
    }
    out
}

/// The tick-`t` cross-section of a set of per-element paths.
fn cross(paths: &[Vec<f64>], t: usize) -> Array<f64, 1> {
    paths.iter().map(|p| p[t]).collect::<Vec<_>>().into()
}

// ---------------------------------------------------------------------------
// sum
// ---------------------------------------------------------------------------

/// `sum` emits over the partial window from the very first tick and drops the
/// evicted sample once the window is full, independently per cross-section
/// element. Quarter-valued inputs keep every running sum exact, so the
/// incremental accumulator must match a freshly-summed reference exactly.
#[test]
fn sum_tracks_the_trailing_count_window_exactly() {
    const W: usize = 4;
    let paths: Vec<Vec<f64>> = (0..3).map(|s| quarter_path(10 + s, 40)).collect();

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, vec![0.0_f64; 3].into());
    let s = b.segment(rolling::sum(W, 1), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for t in 0..40 {
        *g.state_mut(src) = cross(&paths, t);
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        let want: Vec<f64> = paths
            .iter()
            .map(|p| ref_sum(count_window(p, t, W), 1))
            .collect();
        assert_eq!(vals(g.view(s)), want, "tick {t}");
    }
}

/// `min_count` counts the *finite* samples in the window, not its length: the
/// output is `NaN` until enough finite samples are simultaneously in the
/// window, and it drops back to `NaN` when they age out again.
#[test]
fn sum_min_count_gates_on_the_finite_sample_count() {
    const W: usize = 3;
    const MIN: usize = 2;
    let nan = f64::NAN;
    let path = [1.0, nan, 2.0, nan, nan, 4.0, 5.0, 6.0];

    let mut b = Builder::new();
    // The NaN samples ride along a second, always-finite carrier element —
    // per-element independence is part of what the reference exercises.
    let (src, xv) = event_src(&mut b, vec![0.0_f64; 2].into());
    let rec = b.segment(series::record_all(), xv);
    let s = b.segment(rolling::series_sum(W, MIN), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let (mut gated, mut emitted) = (0, 0);
    for (t, &v) in path.iter().enumerate() {
        *g.state_mut(src) = [v, 0.0].into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        let want = ref_sum(count_window(&path, t, W), MIN);
        assert_close(&[vals(g.view(s))[0]], &[want], &format!("tick {t}"));
        if want.is_nan() {
            gated += 1;
        } else {
            emitted += 1;
        }
    }
    // The path must exercise both sides of the gate, including a re-gating
    // after an emitting tick (tick 2 emits, tick 3 gates again).
    assert!(
        gated >= 4 && emitted >= 2,
        "{gated} gated, {emitted} emitted"
    );
}

/// `min_count` must be positive: zero would make the "enough data" test
/// vacuous and let the accumulators divide by a zero count.
#[test]
#[should_panic(expected = "min_count must be positive")]
fn sum_rejects_a_zero_min_count() {
    let mut b = Builder::new();
    let (_src, xv) = event_src(&mut b, (0.0_f64).into());
    let rec = b.segment(series::record_all(), xv);
    let _ = b.segment(rolling::series_sum(3, 0), rec);
}

// ---------------------------------------------------------------------------
// mean
// ---------------------------------------------------------------------------

/// With `min_count = 1` the mean is emitted from the first tick over whatever
/// part of the window has filled — there is no warm-up gating on window length.
#[test]
fn mean_emits_partial_windows_before_filling() {
    const W: usize = 3;
    let path = [1.0, 2.0, 3.0, 6.0, 9.0];

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, (0.0_f64).into());
    let m = b.segment(rolling::mean(W, 1), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, &v) in path.iter().enumerate() {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        let window = count_window(&path, t, W);
        assert_eq!(window.len(), (t + 1).min(W), "tick {t}: window model");
        assert_close(
            &vals(g.view(m)),
            &[ref_mean(window, 1)],
            &format!("tick {t}"),
        );
    }
    // Spot-check the reference model itself against hand arithmetic.
    let modelled: Vec<f64> = (0..path.len())
        .map(|t| ref_mean(count_window(&path, t, W), 1))
        .collect();
    assert_close(
        &modelled,
        &[1.0, 1.5, 2.0, 11.0 / 3.0, 6.0],
        "reference model",
    );
}

/// Setting `min_count` equal to the window turns the partial window into a
/// warm-up: nothing is emitted until the window holds a full complement of
/// finite samples.
#[test]
fn mean_min_count_equal_to_the_window_forces_a_warm_up() {
    const W: usize = 4;
    let path = quarter_path(3, 20);

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, (0.0_f64).into());
    let m = b.segment(rolling::mean(W, W), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, &v) in path.iter().enumerate() {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        let got = vals(g.view(m));
        if t + 1 < W {
            assert!(
                got[0].is_nan(),
                "tick {t}: expected warm-up NaN, got {got:?}"
            );
        } else {
            assert_close(
                &got,
                &[ref_mean(count_window(&path, t, W), W)],
                &format!("tick {t}"),
            );
        }
    }
}

/// Each cross-section element accumulates independently: its own sum, its own
/// finite count, and therefore its own `min_count` gate. Element 1 alternates
/// `NaN`, element 2 goes permanently missing partway through.
#[test]
fn mean_accumulates_each_cross_section_element_independently() {
    const W: usize = 3;
    const MIN: usize = 2;
    let base = quarter_path(21, 24);
    let paths: Vec<Vec<f64>> = vec![
        base.clone(),
        base.iter()
            .enumerate()
            .map(|(t, &x)| if t % 2 == 0 { f64::NAN } else { x + 1.0 })
            .collect(),
        base.iter()
            .enumerate()
            .map(|(t, &x)| if t >= 8 { f64::NAN } else { x + 2.0 })
            .collect(),
    ];

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, vec![0.0_f64; 3].into());
    let m = b.segment(rolling::mean(W, MIN), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let mut gated = [0usize; 3];
    for t in 0..base.len() {
        *g.state_mut(src) = cross(&paths, t);
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        let want: Vec<f64> = paths
            .iter()
            .map(|p| ref_mean(count_window(p, t, W), MIN))
            .collect();
        assert_close(&vals(g.view(m)), &want, &format!("tick {t}"));
        for (e, v) in vals(g.view(m)).iter().enumerate() {
            gated[e] += usize::from(v.is_nan());
        }
    }
    // The three elements gate on entirely different schedules: element 0 only
    // for its one-tick warm-up, element 1 on every tick whose window holds
    // fewer than two finite samples, element 2 permanently once its NaNs fill
    // the window. If the accumulator shared state across elements these counts
    // could not differ.
    assert_eq!(gated[0], MIN - 1, "element 0 gated {} times", gated[0]);
    assert!(
        gated[0] < gated[1] && gated[1] < gated[2] && gated[2] < base.len(),
        "gating is not per-element: {gated:?}"
    );
}

/// A `Duration` window selects ticks by timestamp, not by count: on an
/// irregular cadence it holds however many ticks fall strictly after
/// `now - duration`, sometimes more and sometimes fewer than a same-sized
/// count window would. The self-recording spelling and the hoisted
/// `record` → `series_mean` spelling must agree bit for bit.
#[test]
fn mean_over_a_duration_window_selects_ticks_by_timestamp() {
    const SPAN: i64 = 3;
    #[rustfmt::skip]
    let days: [i64; 28] = [
        1, 2, 3, 4, 7, 8, 9, 12, 13, 14, 15, 16, 20, 21,
        25, 26, 27, 28, 29, 30, 40, 41, 42, 43, 44, 45, 46, 50,
    ];
    let path = quarter_path(7, days.len());

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, (0.0_f64).into());
    let fused = b.segment(rolling::mean(Duration::from_days(SPAN), 1), xv);
    let rec = b.segment(series::record_all(), xv);
    let hoisted = b.segment(rolling::series_mean(Duration::from_days(SPAN), 1), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let mut differs_from_count_window = 0;
    for (t, &v) in path.iter().enumerate() {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &day(days[t]));
        let window = duration_window(&path, &days, t, SPAN);
        assert_close(
            &vals(g.view(fused)),
            &[ref_mean(window, 1)],
            &format!("tick {t} (day {})", days[t]),
        );
        assert_same_bits(g.view(fused), g.view(hoisted), &format!("tick {t}"));
        if window.len() != (t + 1).min(SPAN as usize) {
            differs_from_count_window += 1;
        }
    }
    // The cadence must actually make the duration window differ from a 3-tick
    // count window, otherwise the test proves nothing about time.
    assert!(
        differs_from_count_window >= 8,
        "cadence too regular: only {differs_from_count_window} ticks differ"
    );
}

// ---------------------------------------------------------------------------
// var / std_dev
// ---------------------------------------------------------------------------

/// `var` is the population variance (divide by the finite count) of the
/// trailing window, computed independently per element.
#[test]
fn var_matches_the_population_variance_of_the_window() {
    const W: usize = 5;
    let paths: Vec<Vec<f64>> = (0..2).map(|s| quarter_path(31 + s, 40)).collect();

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, vec![0.0_f64; 2].into());
    let v = b.segment(rolling::var(W, 1), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for t in 0..40 {
        *g.state_mut(src) = cross(&paths, t);
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        let want: Vec<f64> = paths
            .iter()
            .map(|p| ref_var(count_window(p, t, W), 1))
            .collect();
        assert_close(&vals(g.view(v)), &want, &format!("tick {t}"));
    }
}

/// `std_dev` is the square root of that variance and shares its `min_count`
/// gate (`sqrt(NaN)` is `NaN`).
#[test]
fn std_dev_matches_the_square_root_of_the_window_variance() {
    const W: usize = 4;
    const MIN: usize = 3;
    let path = quarter_path(33, 30);

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, (0.0_f64).into());
    let s = b.segment(rolling::std_dev(W, MIN), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, &x) in path.iter().enumerate() {
        *g.state_mut(src) = x.into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        let want = ref_var(count_window(&path, t, W), MIN).sqrt();
        assert_close(&vals(g.view(s)), &[want], &format!("tick {t}"));
        assert_eq!(want.is_nan(), t + 1 < MIN, "tick {t}: gate boundary");
    }
}

/// The `E[x²] - E[x]²` form can round to a small *negative* number on a window
/// of identical samples. The accumulator clamps it at zero, which is what keeps
/// `std_dev` from turning a perfectly good finite window into `NaN`.
#[test]
fn var_clamps_the_rounding_error_of_a_constant_window_to_zero() {
    const W: usize = 7;
    let c = 1000.0 + 1.0 / 3.0;

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, (0.0_f64).into());
    let v = b.segment(rolling::var(W, 1), xv);
    let s = b.segment(rolling::std_dev(W, 1), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Mirror of the accumulator's own arithmetic — same values in the same
    // order — so the unclamped result it would have written is reproducible.
    let (mut sum, mut sum_sq, mut n) = (0.0_f64, 0.0_f64, 0_usize);
    let mut clamped = 0;
    for t in 0..40 {
        *g.state_mut(src) = c.into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));

        sum += c;
        sum_sq += c * c;
        n += 1;
        if n > W {
            sum -= c;
            sum_sq -= c * c;
            n -= 1;
        }
        let raw = sum_sq / n as f64 - (sum / n as f64) * (sum / n as f64);
        clamped += usize::from(raw < 0.0);

        let got_var = vals(g.view(v))[0];
        let got_std = vals(g.view(s))[0];
        assert_eq!(got_var, raw.max(0.0), "tick {t}: clamped variance");
        assert!(got_var >= 0.0, "tick {t}: negative variance {got_var}");
        assert!(got_std.is_finite(), "tick {t}: std_dev is {got_std}");
        assert!(got_var < 1e-6, "tick {t}: variance {got_var} is not ~0");
    }
    assert!(clamped > 0, "the clamp never fired — the test is vacuous");
}

// ---------------------------------------------------------------------------
// cov
// ---------------------------------------------------------------------------

/// `cov` maps a rank-1 cross-section of `K` components to the full `[K, K]`
/// covariance matrix. It is symmetric, its diagonal is the per-component
/// variance (the same value `var` reports), and a linear relation between two
/// components scales the corresponding entries by exactly that factor.
#[test]
fn cov_produces_a_symmetric_matrix_whose_diagonal_is_the_variance() {
    const W: usize = 5;
    const K: usize = 3;
    let x = quarter_path(41, 30);
    // Components 1 and 2 are exact linear images of component 0, so
    // Cov(x, a·x) == a·Var(x) must hold entry by entry.
    let paths: Vec<Vec<f64>> = vec![
        x.clone(),
        x.iter().map(|v| 2.0 * v).collect(),
        x.iter().map(|v| -v).collect(),
    ];

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, vec![0.0_f64; K].into());
    let c = b.segment(rolling::cov(W, 1), xv);
    let v = b.segment(rolling::var(W, 1), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for t in 0..x.len() {
        *g.state_mut(src) = cross(&paths, t);
        g.stabilize(&mut pool, &nano(t as i64 + 1));

        let got = vals(g.view(c));
        assert_eq!(got.len(), K * K, "tick {t}: covariance matrix shape");
        assert_close(&got, &ref_cov_matrix(&paths, t, W, 1), &format!("tick {t}"));

        for i in 0..K {
            for j in 0..K {
                assert_close(
                    &[got[i * K + j]],
                    &[got[j * K + i]],
                    &format!("tick {t}: symmetry at ({i}, {j})"),
                );
            }
        }
        let diag: Vec<f64> = (0..K).map(|i| got[i * K + i]).collect();
        assert_close(&diag, &vals(g.view(v)), &format!("tick {t}: diagonal"));
        let var_x = diag[0];
        assert_close(
            &[got[1], got[2], got[4], got[8]],
            &[2.0 * var_x, -var_x, 4.0 * var_x, var_x],
            &format!("tick {t}: linear scaling"),
        );
    }
}

/// Missing data is handled pairwise-complete: entry `[i, j]` is accumulated
/// only over the ticks where *both* components are finite, so different entries
/// of the same matrix reach `min_count` at different ticks.
#[test]
fn cov_gates_each_matrix_entry_on_its_own_pairwise_complete_count() {
    const W: usize = 4;
    const MIN: usize = 3;
    const K: usize = 2;
    let base = quarter_path(43, 24);
    let paths: Vec<Vec<f64>> = vec![
        base.iter()
            .enumerate()
            .map(|(t, &v)| if t % 5 == 1 { f64::NAN } else { v })
            .collect(),
        base.iter()
            .enumerate()
            .map(|(t, &v)| if t % 3 == 0 { f64::NAN } else { 0.5 * v + 1.0 })
            .collect(),
    ];

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, vec![0.0_f64; K].into());
    let rec = b.segment(series::record_all(), xv);
    let c = b.segment(rolling::series_cov(W, MIN), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let mut mixed_gates = 0;
    // Every poke is a pulse, so every cross row is recorded — the all-NaN rows
    // included, which occupy a window slot while contributing to no pair.
    for t in 0..base.len() {
        *g.state_mut(src) = cross(&paths, t);
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        let want = ref_cov_matrix(&paths, t, W, MIN);
        assert_close(&vals(g.view(c)), &want, &format!("tick {t}"));
        if want.iter().any(|v| v.is_nan()) && want.iter().any(|v| v.is_finite()) {
            mixed_gates += 1;
        }
    }
    // The point of the test: some ticks must have a partly-gated matrix.
    assert!(mixed_gates >= 3, "only {mixed_gates} partly-gated ticks");
}

// ---------------------------------------------------------------------------
// mean_exp
// ---------------------------------------------------------------------------

/// The EWMA weights the newest in-window sample by `alpha` and each older one
/// by a further factor of `1 - alpha`, normalizing by the weights present. As a
/// sample ages out of the window its weight leaves *both* the numerator and the
/// denominator, so the survivors are renormalized rather than left short.
#[test]
fn mean_exp_weights_the_window_geometrically() {
    let alpha = 0.5;
    let path = [10.0, 20.0, 30.0, 40.0];

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, (0.0_f64).into());
    let rec = b.segment(series::record_all(), xv);
    let e = b.segment(rolling::series_mean_exp(alpha, 2, 1), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Hand-computed: a 2-tick window keeps weights {alpha, alpha(1 - alpha)} =
    // {0.5, 0.25} on {newest, previous}, normalized by their sum 0.75.
    let want = [
        10.0,
        (0.5 * 20.0 + 0.25 * 10.0) / 0.75,
        (0.5 * 30.0 + 0.25 * 20.0) / 0.75,
        (0.5 * 40.0 + 0.25 * 30.0) / 0.75,
    ];
    for (t, &v) in path.iter().enumerate() {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        assert_close(&vals(g.view(e)), &want[t..t + 1], &format!("tick {t}"));
        // The same numbers, from the generic reference model.
        assert_close(
            &[ref_mean_exp(count_window(&path, t, 2), alpha, 1)],
            &want[t..t + 1],
            &format!("tick {t}: reference model"),
        );
    }
}

/// Over a longer path with a non-dyadic `alpha`, the incremental accumulator
/// (which scales its whole state by `1 - alpha` on every add and subtracts
/// `alpha (1 - alpha)^age` on every eviction) stays equal to a weighted sum
/// recomputed from scratch: the eviction weight must match the weight the
/// sample carried when it entered.
#[test]
fn mean_exp_matches_a_recomputed_weighted_window() {
    const W: usize = 6;
    let alpha = 0.3;
    let paths: Vec<Vec<f64>> = (0..2).map(|s| quarter_path(51 + s, 40)).collect();

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, vec![0.0_f64; 2].into());
    let e = b.segment(rolling::mean_exp(alpha, W, 1), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for t in 0..40 {
        *g.state_mut(src) = cross(&paths, t);
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        let want: Vec<f64> = paths
            .iter()
            .map(|p| ref_mean_exp(count_window(p, t, W), alpha, 1))
            .collect();
        assert_close(&vals(g.view(e)), &want, &format!("tick {t}"));
    }
}

/// `alpha == 1` is the degenerate end of the permitted range: the newest sample
/// takes all of the weight, so the EWMA collapses to the latest value whatever
/// the window is.
#[test]
fn mean_exp_with_alpha_one_collapses_to_the_latest_value() {
    let path = quarter_path(57, 20);

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, (0.0_f64).into());
    let e = b.segment(rolling::mean_exp(1.0, 4, 1), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, &v) in path.iter().enumerate() {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        assert_eq!(vals(g.view(e)), vec![v], "tick {t}");
    }
}

// ---------------------------------------------------------------------------
// lag / diff / pct_change
// ---------------------------------------------------------------------------

/// `lag(n)` reports the newest sample that has already left the window — the
/// value from `n` ticks ago for a count window — and `NaN` for the first `n`
/// ticks, while nothing has been evicted yet. Long enough to run past the point
/// where the private buffer compacts underneath the accumulator.
#[test]
fn lag_warms_up_to_nan_then_returns_the_value_n_ticks_ago() {
    const N: usize = 3;
    let path = quarter_path(61, 40);

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, (0.0_f64).into());
    let l = b.segment(rolling::lag(N), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, &v) in path.iter().enumerate() {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        let want = if t < N { f64::NAN } else { path[t - N] };
        assert_close(&vals(g.view(l)), &[want], &format!("tick {t}"));
    }
}

/// `diff(n)` is `x - x₋ₙ` and `pct_change(n)` is `(x - x₋ₙ) / x₋ₙ`, both `NaN`
/// while the lag is unavailable. Each is a fused chain over its own `lag`, so
/// this also pins that the fused spellings see the right lagged sample across
/// the buffer's compaction.
#[test]
fn diff_and_pct_change_warm_up_then_track_the_lagged_sample() {
    const N: usize = 2;
    let path: Vec<f64> = (1..=40).map(|i| (i * i) as f64).collect();

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, (0.0_f64).into());
    let d = b.segment(rolling::diff(N), xv);
    let p = b.segment(rolling::pct_change(N), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, &v) in path.iter().enumerate() {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        if t < N {
            assert!(
                vals(g.view(d))[0].is_nan() && vals(g.view(p))[0].is_nan(),
                "tick {t}: warm-up must be NaN"
            );
        } else {
            let base = path[t - N];
            assert_eq!(vals(g.view(d)), vec![v - base], "tick {t}: diff");
            assert_eq!(
                vals(g.view(p)),
                vec![(v - base) / base],
                "tick {t}: pct_change"
            );
        }
    }
}

/// Over a `Duration` window `lag` means "the last tick before the trailing time
/// window", which on an irregular cadence is not a fixed number of ticks back:
/// after a long gap every earlier tick is evicted at once and the lag jumps.
#[test]
fn lag_over_a_duration_window_returns_the_last_tick_before_it() {
    const SPAN: i64 = 3;
    #[rustfmt::skip]
    let days: [i64; 22] = [
        1, 2, 3, 4, 7, 8, 9, 12, 13, 14, 15,
        16, 20, 21, 25, 26, 27, 28, 29, 30, 40, 41,
    ];
    let path = quarter_path(63, days.len());

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, (0.0_f64).into());
    let l = b.segment(rolling::lag(Duration::from_days(SPAN)), xv);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let mut jumped = 0;
    let mut prev_first = 0usize;
    for (t, &v) in path.iter().enumerate() {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &day(days[t]));
        let first = first_in_duration_window(&days, t, SPAN);
        let want = if first == 0 {
            f64::NAN
        } else {
            path[first - 1]
        };
        assert_close(
            &vals(g.view(l)),
            &[want],
            &format!("tick {t} (day {})", days[t]),
        );
        // Count the ticks where a gap evicted more than one sample at once.
        if first > prev_first + 1 {
            jumped += 1;
        }
        prev_first = first;
    }
    assert!(jumped >= 3, "cadence never made the lag jump ({jumped})");
}

// ---------------------------------------------------------------------------
// Front compaction
// ---------------------------------------------------------------------------

/// A retention-bounded `Record` feeding a rolling accumulator and a `lag` keeps
/// producing correct values while its front is dropped underneath them: both
/// address the `SeriesView` by absolute element index, which survives
/// compaction. The record's physical storage really is bounded — it does not
/// silently keep everything.
#[test]
fn a_bounded_record_compacts_underneath_rolling_and_lag() {
    const W: usize = 5;
    const N: usize = 3;
    const TICKS: usize = 60;
    let path = quarter_path(71, TICKS);

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, (0.0_f64).into());
    // The rolling mean reads back W elements and the lag N + 1; a retention of
    // 8 covers both with margin. See `a_record_trimmed_to_the_window_is_rejected`
    // for what happens without that margin.
    let rec = b.segment(series::record_on(8, false), xv);
    let m = b.segment(rolling::series_mean(W, 1), rec);
    let l = b.segment(rolling::series_lag(N), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let mut compacted_at = None;
    for (t, &v) in path.iter().enumerate() {
        *g.state_mut(src) = v.into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));

        assert_close(
            &vals(g.view(m)),
            &[ref_mean(count_window(&path, t, W), 1)],
            &format!("tick {t}: mean"),
        );
        let want_lag = if t < N { f64::NAN } else { path[t - N] };
        assert_close(&vals(g.view(l)), &[want_lag], &format!("tick {t}: lag"));

        let s: SeriesView<f64, 0> = g.view(rec);
        assert!(
            s.len() <= 16,
            "tick {t}: physical storage unbounded ({} rows)",
            s.len()
        );
        if s.range().start > 0 && compacted_at.is_none() {
            compacted_at = Some(t);
        }
    }

    let s: SeriesView<f64, 0> = g.view(rec);
    assert!(
        compacted_at.is_some_and(|t| t < TICKS / 2),
        "expected front compaction well before the end, got {compacted_at:?}"
    );
    assert!(
        s.len() < TICKS,
        "no compaction: {} of {TICKS} rows",
        s.len()
    );
    let (last_ts, last_v) = s.at(s.range().end - 1);
    assert_eq!(
        &*last_v.to_contiguous(),
        &[path[TICKS - 1]],
        "latest value intact"
    );
    assert_eq!(last_ts, nano(TICKS as i64), "latest timestamp intact");
}

/// The safety contract behind that margin: a hoisted record must not trim to
/// exactly the rolling window, because the accumulator has to *read* an element
/// on the tick it evicts it. `series::record_on(w, false)` trims eagerly with
/// the same cutoff the rolling operator is about to use, so it eventually drops
/// a row out from under it and the driver's assertion fires. The self-recording
/// forms sidestep this by buffering with `delayed = true`, which defers
/// trimming by one tick.
#[test]
#[should_panic(expected = "input series dropped elements")]
fn a_record_trimmed_to_the_window_is_rejected() {
    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, (0.0_f64).into());
    let rec = b.segment(series::record_on(2, false), xv);
    let _ = b.segment(rolling::series_sum(2, 1), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for t in 0..8 {
        *g.state_mut(src) = (t as f64).into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));
    }
}

// ---------------------------------------------------------------------------
// series_* / self-recording equivalence
// ---------------------------------------------------------------------------

/// The headline equivalence: `rolling::std_dev(w, c)` over a live stream is
/// tick-for-tick *bit*-identical to the hoisted spelling
/// `series::record_on(..)` → `rolling::series_var(w, c)` → `elem::sqrt()`,
/// including which `NaN` the warm-up produces.
#[test]
fn std_dev_is_bit_identical_to_the_hoisted_var_then_sqrt() {
    const W: usize = 4;
    const MIN: usize = 2;
    let paths: Vec<Vec<f64>> = (0..2).map(|s| quarter_path(81 + s, 40)).collect();

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, vec![0.0_f64; 2].into());
    let fused = b.segment(rolling::std_dev(W, MIN), xv);
    let rec = b.segment(series::record_on(16, false), xv);
    let var = b.segment(rolling::series_var(W, MIN), rec);
    let hoisted = b.segment(elem::sqrt(), var);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let mut warm_up_ticks = 0;
    for t in 0..40 {
        *g.state_mut(src) = cross(&paths, t);
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        assert_same_bits(g.view(fused), g.view(hoisted), &format!("tick {t}"));
        if vals(g.view(fused)).iter().all(|v| v.is_nan()) {
            warm_up_ticks += 1;
        }
    }
    // `min_count = 2` must have produced a NaN warm-up, so the bit comparison
    // above covered the NaN case as well as the numeric one.
    assert_eq!(
        warm_up_ticks,
        MIN - 1,
        "expected a {}-tick warm-up",
        MIN - 1
    );
}

/// The same equivalence for every other statistic. The self-recording form is
/// exactly `series::buffer(window)` followed by the `series_*` form, so both
/// spellings must feed their accumulator the identical add/remove sequence and
/// produce bit-identical output — warm-up and missing-data `NaN`s included.
#[test]
fn every_statistic_is_bit_identical_to_its_hoisted_spelling() {
    const W: usize = 4;
    let alpha = 0.3;
    let paths: Vec<Vec<f64>> = (0..2)
        .map(|s| {
            quarter_path(91 + s as u64, 40)
                .into_iter()
                .enumerate()
                // Sprinkle in missing data so the min_count gates and the
                // finite-count bookkeeping are covered by the comparison too.
                .map(|(t, v)| if (t + s) % 7 == 3 { f64::NAN } else { v })
                .collect()
        })
        .collect();

    let mut b = Builder::new();
    let (src, xv) = event_src(&mut b, vec![0.0_f64; 2].into());
    let rec = b.segment(series::record_all(), xv);

    let sum_f = b.segment(rolling::sum(W, 1), xv);
    let sum_h = b.segment(rolling::series_sum(W, 1), rec);
    let mean_f = b.segment(rolling::mean(W, 3), xv);
    let mean_h = b.segment(rolling::series_mean(W, 3), rec);
    let var_f = b.segment(rolling::var(W, 2), xv);
    let var_h = b.segment(rolling::series_var(W, 2), rec);
    let std_f = b.segment(rolling::std_dev(W, 2), xv);
    let std_h = b.segment(rolling::series_std_dev(W, 2), rec);
    let exp_f = b.segment(rolling::mean_exp(alpha, W, 1), xv);
    let exp_h = b.segment(rolling::series_mean_exp(alpha, W, 1), rec);
    let lag_f = b.segment(rolling::lag(W), xv);
    let lag_h = b.segment(rolling::series_lag(W), rec);
    let pairs = [
        (sum_f, sum_h, "sum"),
        (mean_f, mean_h, "mean"),
        (var_f, var_h, "var"),
        (std_f, std_h, "std_dev"),
        (exp_f, exp_h, "mean_exp"),
        (lag_f, lag_h, "lag"),
    ];
    let cov_fused = b.segment(rolling::cov(W, 2), xv);
    let cov_hoisted = b.segment(rolling::series_cov(W, 2), rec);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for t in 0..40 {
        *g.state_mut(src) = cross(&paths, t);
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        for (fused, hoisted, name) in &pairs {
            assert_same_bits(
                g.view(*fused),
                g.view(*hoisted),
                &format!("tick {t}: {name}"),
            );
        }
        assert_same_bits(
            g.view(cov_fused),
            g.view(cov_hoisted),
            &format!("tick {t}: cov"),
        );
    }
}

/// `diff` and `pct_change` are the fused spellings of `x - lag(x)` and
/// `(x - lag(x)) / lag(x)`; each must be bit-identical to the same arithmetic
/// wired by hand over a hoisted `series_lag`, `NaN` warm-up included.
#[test]
fn diff_and_pct_change_are_bit_identical_to_their_hoisted_spellings() {
    const N: usize = 3;
    let paths: Vec<Vec<f64>> = (0..2).map(|s| quarter_path(101 + s, 30)).collect();

    let mut b = Builder::new();
    // The raw state wire (`rawv`) drives the hand-wired arithmetic: `elem` is
    // stateless, and the current sample is exactly the source's state.
    let (src, (xc, rawv)) = b.source(cell([0.0_f64; 2]));
    let fused_diff = b.segment(rolling::diff(N), (xc, rawv));
    let fused_pct = b.segment(rolling::pct_change(N), (xc, rawv));
    let rec = b.segment(series::record_all(), (xc, rawv));
    let prev = b.segment(rolling::series_lag(N), rec);
    let hoisted_diff = b.segment(elem::sub(), (rawv, prev));
    let hoisted_pct = b.segment(elem::div(), (hoisted_diff, prev));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    for t in 0..30 {
        *g.state_mut(src) = cross(&paths, t);
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        assert_same_bits(
            g.view(fused_diff),
            g.view(hoisted_diff),
            &format!("tick {t}: diff"),
        );
        assert_same_bits(
            g.view(fused_pct),
            g.view(hoisted_pct),
            &format!("tick {t}: pct_change"),
        );
    }
}
