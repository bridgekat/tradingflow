//! Integration tests for `expr`: the parser (precedence, desugaring, error
//! positions), canonicalization and hash-consing (CSE), and the lowered graphs'
//! semantics against hand-computed references.

use tradingflow::data::{Array, ArrayView, Duration, Instant, Scalar};
use tradingflow::expr::{Context, Error, Expr, parse};
use tradingflow::graph::typed::Builder;
use tradingflow::graph::{Operator, Pool};
use tradingflow::ports::{ArrayPort, SignalPort};

/// Default tolerance for [`assert_close`]. Loose enough for the accumulators'
/// incremental arithmetic, tight enough that a wrong formula never passes.
pub const EPS: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Instants
// ---------------------------------------------------------------------------

/// An instant `n` nanoseconds after the epoch — the default stamp for tests
/// that only need distinct, ordered ticks.
pub fn nano(n: i64) -> Instant {
    Instant::from_offset(Duration::from_nanos(n))
}

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------

/// Asserts elementwise equality within [`EPS`].
///
/// `NaN` matches only `NaN` and each infinity matches only itself with the
/// same sign — both are ordinary results here (`NaN` is the missing-data
/// marker, and infinities fall out of `recip(0)`, `ln(0)`, division by zero
/// and the like), so they are values to assert on rather than errors. Note a
/// tolerance test cannot express them: `inf - inf` is `NaN`, which compares
/// false against any bound.
#[track_caller]
pub fn assert_close(actual: &[f64], expected: &[f64], ctx: &str) {
    assert_close_tol(actual, expected, EPS, ctx);
}

/// [`assert_close`] with an explicit tolerance.
#[track_caller]
pub fn assert_close_tol(actual: &[f64], expected: &[f64], tol: f64, ctx: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{ctx}: length {} != expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let ok = if e.is_nan() {
            a.is_nan()
        } else if e.is_infinite() {
            a == e
        } else {
            (a - e).abs() <= tol
        };
        assert!(
            ok,
            "{ctx}: element {i} is {a}, expected {e}\n  actual: {actual:?}\n  expected: {expected:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Deterministic data
// ---------------------------------------------------------------------------

/// A deterministic pseudo-random path of quarter-valued samples. Quarters keep
/// every running sum exactly representable in binary floating point, so an
/// incremental accumulator and a freshly-summed reference agree bit-for-bit
/// and exact equality is a sound assertion.
pub fn quarter_path(seed: u64, len: usize) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % 1000) as f64 / 4.0
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Probe operators
// ---------------------------------------------------------------------------

/// A pokeable array cell paired with its accompanying signal: the signal
/// pulses on exactly the generations the cell is poked (a root with no inputs
/// is scheduled only when its own state is touched) and is `false` otherwise.
/// The manually-stepped counterpart of
/// [`sources::sync::array_series`](tradingflow::sources::sync::array_series).
pub struct Cell<T: Scalar, const N: usize> {
    value: Array<T, N>,
}

impl<T: Scalar, const N: usize> Operator for Cell<T, N> {
    type Inputs = ();
    type Outputs = (SignalPort<0>, ArrayPort<T, N>);
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self, _: ()) -> Self::State {
        self.value
    }

    fn reset<'a, 'b: 'a>(
        _: (),
        state: &'b mut Array<T, N>,
    ) -> (ArrayView<'a, bool, 0>, ArrayView<'a, T, N>) {
        (ArrayView::scalar(&false), state.view())
    }

    fn compute<'a, 'b: 'a>(
        _: (),
        state: &'b mut Array<T, N>,
        _: &Instant,
    ) -> (ArrayView<'a, bool, 0>, ArrayView<'a, T, N>) {
        (ArrayView::scalar(&true), state.view())
    }
}

/// See [`Cell`].
pub fn cell<T: Scalar, const N: usize>(
    value: impl Into<Array<T, N>>,
) -> impl Operator<
    Inputs = (),
    Outputs = (SignalPort<0>, ArrayPort<T, N>),
    Context = Instant,
    State = Array<T, N>,
> {
    Cell {
        value: value.into(),
    }
}

/// Shorthand for a call node.
fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::call(name, args)
}

fn field(name: &str) -> Expr {
    Expr::Field(name.to_string())
}

fn num(c: f64) -> Expr {
    Expr::Const(c)
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Infix operators desugar to canonical calls with conventional precedence.
#[test]
fn parses_with_standard_precedence() {
    assert_eq!(
        parse("$a + $b * 2").unwrap(),
        call(
            "Add",
            vec![field("a"), call("Mul", vec![field("b"), num(2.0)])]
        ),
    );
    assert_eq!(
        parse("($a + $b) * 2").unwrap(),
        call(
            "Mul",
            vec![call("Add", vec![field("a"), field("b")]), num(2.0)]
        ),
    );
    assert_eq!(
        parse("-$a * $b").unwrap(),
        call("Mul", vec![call("Neg", vec![field("a")]), field("b")]),
    );
    assert_eq!(
        parse("$a > 0 && !($b <= 1)").unwrap(),
        call(
            "And",
            vec![
                call("Gt", vec![field("a"), num(0.0)]),
                call("Not", vec![call("Le", vec![field("b"), num(1.0)])]),
            ]
        ),
    );
    // Negated literals fold in the parser, so window checks see constants.
    assert_eq!(
        parse("Ref($x, -1)").unwrap(),
        call("Ref", vec![field("x"), num(-1.0)]),
    );
    // Nested calls and scientific notation.
    assert_eq!(
        parse("Mean(Abs($x), 5) / 1e2").unwrap(),
        call(
            "Div",
            vec![
                call("Mean", vec![call("Abs", vec![field("x")]), num(5.0)]),
                num(100.0),
            ]
        ),
    );
}

/// Syntax errors carry the offending byte offset.
#[test]
fn rejects_malformed_expressions() {
    for (src, bad_pos) in [
        ("$x +", 4),           // missing right operand
        ("$", 0),              // bare '$'
        ("Mean($x 3)", 8),     // missing comma
        ("foo", 3),            // bare identifier without '('
        (")$x", 0),            // stray ')'
        ("$x $y", 3),          // trailing input
        ("$x = 1", 3),         // '=' is not an operator
        ("1.5.2", 3),          // trailing input after a number
        ("Mean($x, 3) @", 12), // unexpected character
    ] {
        match parse(src) {
            Err(Error::Parse { pos, .. }) => {
                assert_eq!(pos, bad_pos, "wrong position for {src:?}")
            }
            other => panic!("{src:?} parsed as {other:?}"),
        }
    }
}

/// `Display` round-trips through the parser to an equal tree.
#[test]
fn display_round_trips() {
    for src in [
        "$a + $b * 2",
        "Mean($close, 12) - Mean($close, 26)",
        "If($x > Mean($x, 3), 1, 0)",
        "!($a > 0) || $b < 1",
        "Corr($x, $y, 20)",
    ] {
        let expr = parse(src).unwrap();
        let printed = expr.to_string();
        assert_eq!(parse(&printed).unwrap(), expr, "via {printed:?}");
    }
}

// ---------------------------------------------------------------------------
// Lowering: hash-consing
// ---------------------------------------------------------------------------

/// Semantically equal expressions lower to the *same wire*: repeated lowering,
/// commutative reorderings, and shared subexpressions of larger factors all
/// hit the cache instead of creating new nodes.
#[test]
fn cse_dedupes_semantically_equal_wires() {
    let mut b = Builder::new();
    let (_src, (sig, x)) = b.source(cell([0.0_f64; 1]));
    let (_src2, (_, y)) = b.source(cell([0.0_f64; 1]));
    let mut ctx = Context::new(sig, [1]);
    ctx.add_field("x", x);
    ctx.add_field("y", y);

    let lower = |ctx: &mut Context<1>, b: &mut Builder<_>, src: &str| {
        ctx.lower_str(b, src).unwrap().index()
    };

    // Identical source: same wire.
    let a1 = lower(&mut ctx, &mut b, "Mean($x, 5)");
    let a2 = lower(&mut ctx, &mut b, "Mean($x, 5)");
    assert_eq!(a1, a2);

    // Commutative reordering: same wire.
    let m1 = lower(&mut ctx, &mut b, "$x * $y + 1");
    let m2 = lower(&mut ctx, &mut b, "1 + $y * $x");
    assert_eq!(m1, m2);

    // A subsequent factor reuses the shared rolling subexpression: lowering
    // the MACD line after its EMAs already exist must not re-create them.
    let e1 = lower(&mut ctx, &mut b, "EMA($x, 12)");
    let macd = ctx
        .lower_str(&mut b, "EMA($x, 12) - EMA($x, 26)")
        .unwrap()
        .index();
    let e2 = lower(&mut ctx, &mut b, "EMA($x, 12)");
    assert_eq!(e1, e2);
    assert_ne!(macd, e1);

    // Different windows are different wires.
    let w5 = lower(&mut ctx, &mut b, "Mean($x, 5)");
    let w6 = lower(&mut ctx, &mut b, "Mean($x, 6)");
    assert_ne!(w5, w6);
}

// ---------------------------------------------------------------------------
// Lowering: semantics
// ---------------------------------------------------------------------------

/// Reference rolling mean over the trailing `min(t + 1, w)` samples.
fn ref_mean(path: &[f64], t: usize, w: usize) -> f64 {
    let k = (t + 1).min(w);
    let xs = &path[t + 1 - k..=t];
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Arithmetic, lag, rolling and conditional expressions match hand-computed
/// references tick by tick, including NaN warmups and constant folding.
#[test]
fn lowered_graphs_match_reference() {
    let path = quarter_path(7, 40);
    let nan = f64::NAN;

    let mut b = Builder::new();
    let (src, (sig, x)) = b.source(cell([0.0_f64; 1]));
    let mut ctx = Context::new(sig, [1]).with_field("x", x);

    let delta = ctx.lower_str(&mut b, "$x - Ref($x, 1)").unwrap();
    let mean3 = ctx.lower_str(&mut b, "Mean($x, 3)").unwrap();
    let cum = ctx.lower_str(&mut b, "Sum($x, 0)").unwrap(); // expanding
    let folded = ctx.lower_str(&mut b, "$x * (2 + 3)").unwrap();
    let cond = ctx.lower_str(&mut b, "If($x > Mean($x, 3), 1, 0)").unwrap();
    let branches = ctx.lower_str(&mut b, "If($x > 0, $x, Ref($x, 1))").unwrap();
    let ref0 = ctx.lower_str(&mut b, "Ref($x, 0)").unwrap();
    assert_eq!(ref0.index(), x.index(), "Ref(x, 0) is x itself");

    // Constants materialize as correctly-shaped wires inside rolling
    // operators: the expanding sum of `1` counts ticks. Materialized
    // constants are deduplicated by bit pattern.
    let ticks = ctx.lower_str(&mut b, "Sum(1, 0)").unwrap();
    let mean_const = ctx.lower_str(&mut b, "Mean(1, 5)").unwrap();

    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, &v) in path.iter().enumerate() {
        *g.state_mut(src) = [v].into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));

        let prev = if t == 0 { nan } else { path[t - 1] };
        let mean = ref_mean(&path, t, 3);
        let want_delta = v - prev;
        let want_cum: f64 = path[..=t].iter().sum();
        let want_cond = if v > mean { 1.0 } else { 0.0 };
        let want_branches = if v > 0.0 { v } else { prev };

        assert_close(
            &g.view(delta).to_contiguous(),
            &[want_delta],
            &format!("delta, tick {t}"),
        );
        assert_close(
            &g.view(mean3).to_contiguous(),
            &[mean],
            &format!("mean3, tick {t}"),
        );
        assert_close(
            &g.view(cum).to_contiguous(),
            &[want_cum],
            &format!("cum, tick {t}"),
        );
        assert_close(
            &g.view(folded).to_contiguous(),
            &[v * 5.0],
            &format!("folded, tick {t}"),
        );
        assert_close(
            &g.view(cond).to_contiguous(),
            &[want_cond],
            &format!("cond, tick {t}"),
        );
        assert_close(
            &g.view(branches).to_contiguous(),
            &[want_branches],
            &format!("branches, tick {t}"),
        );
        assert_close(
            &g.view(ticks).to_contiguous(),
            &[(t + 1) as f64],
            &format!("ticks, tick {t}"),
        );
        assert_close(
            &g.view(mean_const).to_contiguous(),
            &[1.0],
            &format!("mean_const, tick {t}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Lowering: window-scanning operators
// ---------------------------------------------------------------------------

/// The finite `(position, value)` pairs and total row count of the trailing
/// window of `w` at tick `t` (`w = 0` is expanding), mirroring the documented
/// scanning semantics: positions are 0-based from the window start and count
/// every row, so non-finite samples leave gaps.
fn window_pairs(path: &[f64], t: usize, w: usize) -> (Vec<(usize, f64)>, usize) {
    let k = if w == 0 { t + 1 } else { (t + 1).min(w) };
    let pairs = path[t + 1 - k..=t]
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (i, v))
        .collect();
    (pairs, k)
}

fn sref_extreme(pairs: &[(usize, f64)], is_max: bool, emit_index: bool) -> f64 {
    let Some(&first) = pairs.first() else {
        return f64::NAN;
    };
    let best = pairs.iter().skip(1).fold(first, |best, &p| {
        if (is_max && p.1 > best.1) || (!is_max && p.1 < best.1) {
            p
        } else {
            best
        }
    });
    if emit_index {
        (best.0 + 1) as f64
    } else {
        best.1
    }
}

fn sref_quantile(pairs: &[(usize, f64)], q: f64) -> f64 {
    if pairs.is_empty() {
        return f64::NAN;
    }
    let mut v: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let h = q * (v.len() - 1) as f64;
    let lo = h.floor() as usize;
    if lo + 1 < v.len() {
        v[lo] + (h - lo as f64) * (v[lo + 1] - v[lo])
    } else {
        v[lo]
    }
}

fn sref_rank(pairs: &[(usize, f64)], len: usize) -> f64 {
    let Some(&(pos, v)) = pairs.last() else {
        return f64::NAN;
    };
    if pos + 1 != len {
        return f64::NAN; // The newest sample is not finite.
    }
    let strict = pairs.iter().filter(|p| p.1 < v).count();
    let weak = pairs.iter().filter(|p| p.1 <= v).count();
    (strict + weak) as f64 / (2 * pairs.len()) as f64
}

fn sref_mad(pairs: &[(usize, f64)]) -> f64 {
    if pairs.is_empty() {
        return f64::NAN;
    }
    let n = pairs.len() as f64;
    let mean = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    pairs.iter().map(|p| (p.1 - mean).abs()).sum::<f64>() / n
}

fn sref_wma(pairs: &[(usize, f64)]) -> f64 {
    if pairs.is_empty() {
        return f64::NAN;
    }
    let (mut num, mut den) = (0.0, 0.0);
    for &(pos, x) in pairs {
        num += (pos + 1) as f64 * x;
        den += (pos + 1) as f64;
    }
    num / den
}

/// Population central moments `(n, m2, m3, m4)`.
fn sref_moments(pairs: &[(usize, f64)]) -> (f64, f64, f64, f64) {
    let n = pairs.len() as f64;
    let mean = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
    for &(_, x) in pairs {
        let d = x - mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
    }
    (n, m2 / n, m3 / n, m4 / n)
}

fn sref_skew(pairs: &[(usize, f64)]) -> f64 {
    if pairs.len() < 3 {
        return f64::NAN;
    }
    let (n, m2, m3, _) = sref_moments(pairs);
    if m2 <= 0.0 {
        return f64::NAN;
    }
    (m3 / m2.powf(1.5)) * (n * (n - 1.0)).sqrt() / (n - 2.0)
}

fn sref_kurt(pairs: &[(usize, f64)]) -> f64 {
    if pairs.len() < 4 {
        return f64::NAN;
    }
    let (n, m2, _, m4) = sref_moments(pairs);
    if m2 <= 0.0 {
        return f64::NAN;
    }
    let g2 = m4 / (m2 * m2) - 3.0;
    ((n + 1.0) * g2 + 6.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0))
}

/// OLS sufficient statistics `(mx, my, sxx, sxy, syy)` against positions.
fn sref_ols(pairs: &[(usize, f64)]) -> (f64, f64, f64, f64, f64) {
    let n = pairs.len() as f64;
    let mx = pairs.iter().map(|p| p.0 as f64).sum::<f64>() / n;
    let my = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for &(pos, y) in pairs {
        let (dx, dy) = (pos as f64 - mx, y - my);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    (mx, my, sxx, sxy, syy)
}

/// The window-scanning operators (extremes, order statistics, positional
/// weights) match their references tick by tick, including `NaN` gaps that
/// shift finite samples off window positions.
#[test]
fn scan_ops_match_reference() {
    const W: usize = 5;
    let mut path = quarter_path(11, 40);
    path[3] = f64::NAN;
    path[7] = f64::NAN;
    path[15] = f64::NAN;

    let mut b = Builder::new();
    let (src, (sig, x)) = b.source(cell([0.0_f64; 1]));
    let mut ctx = Context::new(sig, [1]).with_field("x", x);

    let max = ctx.lower_str(&mut b, "Max($x, 5)").unwrap();
    let min = ctx.lower_str(&mut b, "Min($x, 5)").unwrap();
    let idx_max = ctx.lower_str(&mut b, "IdxMax($x, 5)").unwrap();
    let idx_min = ctx.lower_str(&mut b, "IdxMin($x, 5)").unwrap();
    let med = ctx.lower_str(&mut b, "Med($x, 5)").unwrap();
    let quartile = ctx.lower_str(&mut b, "Quantile($x, 5, 0.25)").unwrap();
    let rank = ctx.lower_str(&mut b, "Rank($x, 5)").unwrap();
    let mad = ctx.lower_str(&mut b, "Mad($x, 5)").unwrap();
    let count = ctx.lower_str(&mut b, "Count($x, 5)").unwrap();
    let wma = ctx.lower_str(&mut b, "WMA($x, 5)").unwrap();

    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, &v) in path.iter().enumerate() {
        *g.state_mut(src) = [v].into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        let (pairs, len) = window_pairs(&path, t, W);

        let checks = [
            (max, sref_extreme(&pairs, true, false), "max"),
            (min, sref_extreme(&pairs, false, false), "min"),
            (idx_max, sref_extreme(&pairs, true, true), "idx_max"),
            (idx_min, sref_extreme(&pairs, false, true), "idx_min"),
            (med, sref_quantile(&pairs, 0.5), "med"),
            (quartile, sref_quantile(&pairs, 0.25), "quartile"),
            (rank, sref_rank(&pairs, len), "rank"),
            (mad, sref_mad(&pairs), "mad"),
            (count, pairs.len() as f64, "count"),
            (wma, sref_wma(&pairs), "wma"),
        ];
        for (wire, want, what) in checks {
            assert_close(
                &g.view(wire).to_contiguous(),
                &[want],
                &format!("{what}, tick {t}"),
            );
        }
    }
}

/// The moment and regression operators match their references tick by tick,
/// including their minimum-sample gates.
#[test]
fn moment_and_regression_ops_match_reference() {
    const W: usize = 6;
    let mut path = quarter_path(13, 40);
    path[2] = f64::NAN;
    path[9] = f64::NAN;

    let mut b = Builder::new();
    let (src, (sig, x)) = b.source(cell([0.0_f64; 1]));
    let mut ctx = Context::new(sig, [1]).with_field("x", x);

    let skew = ctx.lower_str(&mut b, "Skew($x, 6)").unwrap();
    let kurt = ctx.lower_str(&mut b, "Kurt($x, 6)").unwrap();
    let slope = ctx.lower_str(&mut b, "Slope($x, 6)").unwrap();
    let rsquare = ctx.lower_str(&mut b, "Rsquare($x, 6)").unwrap();
    let resi = ctx.lower_str(&mut b, "Resi($x, 6)").unwrap();

    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, &v) in path.iter().enumerate() {
        *g.state_mut(src) = [v].into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));
        let (pairs, _len) = window_pairs(&path, t, W);

        let want_slope = if pairs.len() < 2 {
            f64::NAN
        } else {
            let (_, _, sxx, sxy, _) = sref_ols(&pairs);
            sxy / sxx
        };
        let want_rsquare = if pairs.len() < 2 {
            f64::NAN
        } else {
            let (_, _, sxx, sxy, syy) = sref_ols(&pairs);
            if syy <= 0.0 {
                f64::NAN
            } else {
                sxy * sxy / (sxx * syy)
            }
        };
        let want_resi = if pairs.len() < 2 {
            f64::NAN
        } else {
            let (mx, my, sxx, sxy, _) = sref_ols(&pairs);
            let &(pos, y) = pairs.last().unwrap();
            y - (my + sxy / sxx * (pos as f64 - mx))
        };

        let checks = [
            (skew, sref_skew(&pairs), "skew"),
            (kurt, sref_kurt(&pairs), "kurt"),
            (slope, want_slope, "slope"),
            (rsquare, want_rsquare, "rsquare"),
            (resi, want_resi, "resi"),
        ];
        for (wire, want, what) in checks {
            assert_close(
                &g.view(wire).to_contiguous(),
                &[want],
                &format!("{what}, tick {t}"),
            );
        }
    }
}

/// `Corr(x, x, w)` is 1 once `Std` has two samples (and `NaN` before), and
/// `Cov(x, y, w)` matches the sample covariance reference (`NaN` below two
/// complete ticks).
#[test]
fn pair_statistics_match_reference() {
    // Strictly increasing so the window variance is never zero.
    let path: Vec<f64> = (0..30).map(|t| 1.0 + t as f64 * 0.25).collect();
    const W: usize = 3;

    let mut b = Builder::new();
    let (src, (sig, x)) = b.source(cell([0.0_f64; 1]));
    let mut ctx = Context::new(sig, [1]).with_field("x", x);

    let corr = ctx.lower_str(&mut b, "Corr($x, $x, 3)").unwrap();
    let cov = ctx.lower_str(&mut b, "Cov($x, $x, 3)").unwrap();

    let mut g = b.build();
    let mut pool = Pool::new(0);

    for (t, &v) in path.iter().enumerate() {
        *g.state_mut(src) = [v].into();
        g.stabilize(&mut pool, &nano(t as i64 + 1));

        let k = (t + 1).min(W);
        let xs = &path[t + 1 - k..=t];
        let mean = xs.iter().sum::<f64>() / k as f64;
        let want_cov = if k < 2 {
            f64::NAN
        } else {
            xs.iter().map(|&a| (a - mean) * (a - mean)).sum::<f64>() / (k - 1) as f64
        };
        let want_corr = if k < 2 { f64::NAN } else { 1.0 };

        assert_close_tol(
            &g.view(cov).to_contiguous(),
            &[want_cov],
            1e-9,
            &format!("cov, tick {t}"),
        );
        assert_close_tol(
            &g.view(corr).to_contiguous(),
            &[want_corr],
            1e-9,
            &format!("corr, tick {t}"),
        );
    }
}

/// Cross-sectional operators act across the array at each tick.
#[test]
fn cross_sectional_operators_match_reference() {
    let mut b = Builder::new();
    let (src, (sig, x)) = b.source(cell([0.0_f64; 3]));
    let mut ctx = Context::new(sig, [3]).with_field("x", x);

    let z = ctx.lower_str(&mut b, "CSZscore($x)").unwrap();

    let mut g = b.build();
    let mut pool = Pool::new(0);

    let xs = [3.0, 1.0, 2.0];
    *g.state_mut(src) = xs.into();
    g.stabilize(&mut pool, &nano(1));

    let mean = xs.iter().sum::<f64>() / 3.0;
    let std = (xs.iter().map(|&a| (a - mean) * (a - mean)).sum::<f64>() / 3.0).sqrt();
    let want: Vec<f64> = xs.iter().map(|&a| (a - mean) / std).collect();
    assert_close(&g.view(z).to_contiguous(), &want, "zscore");
}

/// `Scale` renormalizes the cross-section to a target absolute sum, and
/// `IndNeutralize` demeans within the groups of a registered tag wire —
/// regrouping when the tags change; non-finite entries stay NaN and are
/// excluded from the sums and means.
#[test]
fn scale_and_group_neutralize_match_reference() {
    let nan = f64::NAN;

    let mut b = Builder::new();
    let (src, (sig, x)) = b.source(cell([0.0_f64; 4]));
    let (tag_src, (_tag_sig, tags)) = b.source(cell([0_u32, 0, 1, 1]));
    let mut ctx = Context::new(sig, [4])
        .with_field("x", x)
        .with_grouping("g", tags);

    let scaled = ctx.lower_str(&mut b, "Scale($x)").unwrap();
    let scaled2 = ctx.lower_str(&mut b, "Scale($x, 2)").unwrap();
    let neut = ctx.lower_str(&mut b, "IndNeutralize($x, $g)").unwrap();

    // Grouping misuse is rejected: unknown name, and a non-`$name` argument.
    assert!(matches!(
        ctx.lower_str(&mut b, "IndNeutralize($x, $h)"),
        Err(Error::UnknownGrouping(name)) if name == "h"
    ));
    assert!(matches!(
        ctx.lower_str(&mut b, "IndNeutralize($x, 1)"),
        Err(Error::Type { .. })
    ));
    assert!(matches!(
        ctx.lower_str(&mut b, "Scale($x, 0)"),
        Err(Error::Window { .. })
    ));

    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src) = [1.0, -3.0, 2.0, nan].into();
    g.stabilize(&mut pool, &nano(1));

    // Σ|x| over the finite entries is 6; NaN stays NaN.
    assert_close(
        &g.view(scaled).to_contiguous(),
        &[1.0 / 6.0, -3.0 / 6.0, 2.0 / 6.0, nan],
        "scale",
    );
    assert_close(
        &g.view(scaled2).to_contiguous(),
        &[2.0 / 6.0, -6.0 / 6.0, 4.0 / 6.0, nan],
        "scale to 2",
    );
    // Group 0 has mean -1; group 1's only finite member is its own mean.
    assert_close(
        &g.view(neut).to_contiguous(),
        &[2.0, -2.0, 0.0, nan],
        "group demean",
    );

    // An all-zero (or all-missing) cross-section cannot reach the target.
    *g.state_mut(src) = [0.0, 0.0, 0.0, 0.0].into();
    g.stabilize(&mut pool, &nano(2));
    assert_close(
        &g.view(scaled).to_contiguous(),
        &[nan, nan, nan, nan],
        "scale of zero",
    );

    // Tags are a live wire: retagging regroups the demean.
    *g.state_mut(src) = [1.0, -3.0, 2.0, nan].into();
    *g.state_mut(tag_src) = [0_u32, 0, 0, 1].into();
    g.stabilize(&mut pool, &nano(3));
    assert_close(
        &g.view(neut).to_contiguous(),
        &[1.0, -3.0, 2.0, nan], // Group 0 now has mean 0; group 1 is all-NaN.
        "group demean after retagging",
    );
}

// ---------------------------------------------------------------------------
// Lowering: errors
// ---------------------------------------------------------------------------

/// Lowering rejects unknown names, bad arity, inadmissible windows (including
/// future references), type mismatches and field-free expressions — each with
/// its own error variant.
#[test]
fn lowering_errors_are_reported() {
    let mut b = Builder::new();
    let (_src, (sig, x)) = b.source(cell([0.0_f64; 1]));
    let mut ctx = Context::new(sig, [1]).with_field("x", x);

    let lower =
        |b: &mut Builder<_>, ctx: &mut Context<1>, src: &str| ctx.lower_str(b, src).err().unwrap();

    assert!(matches!(
        lower(&mut b, &mut ctx, "$unknown"),
        Error::UnknownField(name) if name == "unknown"
    ));
    assert!(matches!(
        lower(&mut b, &mut ctx, "Foo($x)"),
        Error::UnknownFunction(name) if name == "Foo"
    ));
    assert!(matches!(
        lower(&mut b, &mut ctx, "Mean($x)"),
        Error::Arity {
            expected: 2,
            got: 1,
            ..
        }
    ));
    // Future references are inexpressible.
    assert!(matches!(
        lower(&mut b, &mut ctx, "Ref($x, -1)"),
        Error::Window { .. }
    ));
    assert!(matches!(
        lower(&mut b, &mut ctx, "Mean($x, 1.5)"),
        Error::Window { .. }
    ));
    assert!(matches!(
        lower(&mut b, &mut ctx, "Mean($x, $x)"),
        Error::Window { .. }
    ));
    // Booleans are not numbers (and vice versa).
    assert!(matches!(
        lower(&mut b, &mut ctx, "$x + ($x > 0)"),
        Error::Type { .. }
    ));
    assert!(matches!(
        lower(&mut b, &mut ctx, "$x && $x"),
        Error::Type { .. }
    ));
    assert!(matches!(
        lower(&mut b, &mut ctx, "$x > 0"),
        Error::Type { .. }
    ));
    // A factor must reference a field: constants only materialize as wires
    // *inside* rolling or cross-sectional operators, not as whole factors.
    assert!(matches!(lower(&mut b, &mut ctx, "1 + 2"), Error::Constant));
}
