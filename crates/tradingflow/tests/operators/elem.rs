//! Elementwise operators: `elem::{ops, cmp, float, boolean, cast}`.
//!
//! The surface is wide but regular — every constructor is a thin wrapper over
//! `array::{map, binary_map, ternary_map}` around one scalar function — so the
//! tests are grouped one per *family*, each running several members of the
//! family over a single shared input vector so their differences are legible
//! side by side. What is asserted is therefore mostly the scalar contract the
//! wrapper inherits (IEEE ordering, `round` half-away-from-zero, `as`
//! saturation, ...) plus, in the last section, the strided and broadcast paths
//! of the shared elementwise core that every one of them routes through.

use std::cmp::Ordering;
use std::f64::consts::{FRAC_PI_4, PI};

use tradingflow::data::Instant;
use tradingflow::graph::Pool;
use tradingflow::graph::typed::Builder;
use tradingflow::operators::{array, elem};

use crate::harness::*;

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

/// The reference for a transcendental family: `f` applied elementwise with the
/// same `f64` std-library call the operator itself makes, so the expectation
/// is a computation rather than a hand-typed decimal constant.
fn each(x: &[f64], f: impl Fn(f64) -> f64) -> Vec<f64> {
    x.iter().map(|&v| f(v)).collect()
}

// ---------------------------------------------------------------------------
// ops: arithmetic, bitwise, shifts
// ---------------------------------------------------------------------------

/// `add`/`sub`/`mul`/`div`/`rem`/`neg` over one `f64` pair. Pins the float
/// division edges (`x / 0` is `±inf`, `0 / 0` is `NaN`, and so is `x % 0`),
/// the remainder taking the sign of the *dividend*, and `neg` flipping the
/// sign bit of zero rather than being a no-op there.
#[test]
fn ops_arithmetic_family() {
    let mut b = Builder::new();
    let (xs, x) = b.source(array::from_parts([4], vec![0.0_f64; 4].into()));
    let y = b.value(array::from_parts([4], vec![2.0_f64, 2.0, 0.0, 0.0].into()));
    let sum = b.segment(elem::add(), (x, y));
    let diff = b.segment(elem::sub(), (x, y));
    let prod = b.segment(elem::mul(), (x, y));
    let quot = b.segment(elem::div(), (x, y));
    let rest = b.segment(elem::rem(), (x, y));
    let negated = b.segment(elem::neg(), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(xs) = arr([4], vec![7.5, -7.5, 3.0, 0.0]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(vals(g.view(sum)), vec![9.5, -5.5, 3.0, 0.0]);
    assert_eq!(vals(g.view(diff)), vec![5.5, -9.5, 3.0, 0.0]);
    assert_eq!(vals(g.view(prod)), vec![15.0, -15.0, 0.0, 0.0]);
    assert_close(
        &vals(g.view(quot)),
        &[3.75, -3.75, f64::INFINITY, f64::NAN],
        "div",
    );
    assert_close(&vals(g.view(rest)), &[1.5, -1.5, f64::NAN, f64::NAN], "rem");
    assert_eq!(
        bits(g.view(negated)),
        [-7.5_f64, 7.5, -3.0, -0.0].map(f64::to_bits).to_vec(),
        "neg must flip the sign bit of zero too"
    );
}

/// The same operators over `i32` — they are generic over the scalar type, not
/// specialised to `f64`. Integer division truncates toward zero and the
/// remainder takes the sign of the dividend, so `-7 / 2 == -3` (not `-4`) and
/// `-7 % 2 == -1` (not `+1`).
#[test]
fn ops_integer_arithmetic_truncates_toward_zero() {
    let mut b = Builder::new();
    let (xs, x) = b.source(array::from_parts([4], vec![0_i32; 4].into()));
    let y = b.value(array::from_parts([4], vec![2_i32, 2, -2, -2].into()));
    let sum = b.segment(elem::add::<i32, i32, 1>(), (x, y));
    let diff = b.segment(elem::sub::<i32, i32, 1>(), (x, y));
    let prod = b.segment(elem::mul::<i32, i32, 1>(), (x, y));
    let quot = b.segment(elem::div::<i32, i32, 1>(), (x, y));
    let rest = b.segment(elem::rem::<i32, i32, 1>(), (x, y));
    let negated = b.segment(elem::neg::<i32, 1>(), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(xs) = arr([4], vec![7_i32, -7, 7, -7]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(vals(g.view(sum)), vec![9, -5, 5, -9]);
    assert_eq!(vals(g.view(diff)), vec![5, -9, 9, -5]);
    assert_eq!(vals(g.view(prod)), vec![14, -14, -14, 14]);
    assert_eq!(vals(g.view(quot)), vec![3, -3, -3, 3]);
    assert_eq!(vals(g.view(rest)), vec![1, -1, 1, -1]);
    assert_eq!(vals(g.view(negated)), vec![-7, 7, -7, 7]);
}

/// `not`/`bitand`/`bitor`/`bitxor`/`shl`/`shr` over `i32`. The shifts take
/// their count as a `u32`, which is the *mixed-type* path of these operators:
/// they are generic over two scalar types (`T: Shl<U>`), not one. `shr` on a
/// signed left operand is arithmetic (sign-extending), so `-1 >> 4 == -1`.
#[test]
fn ops_bitwise_and_shift_family() {
    let mut b = Builder::new();
    let (xs, x) = b.source(array::from_parts([4], vec![0_i32; 4].into()));
    let y = b.value(array::from_parts([4], vec![10_i32, 3, 0, 255].into()));
    let n = b.value(array::from_parts([4], vec![1_u32, 2, 3, 4].into()));
    let and = b.segment(elem::bitand::<i32, i32, 1>(), (x, y));
    let or = b.segment(elem::bitor::<i32, i32, 1>(), (x, y));
    let xor = b.segment(elem::bitxor::<i32, i32, 1>(), (x, y));
    let complement = b.segment(elem::not::<i32, 1>(), x);
    let left = b.segment(elem::shl::<i32, u32, 1>(), (x, n));
    let right = b.segment(elem::shr::<i32, u32, 1>(), (x, n));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(xs) = arr([4], vec![12_i32, -8, 5, -1]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(vals(g.view(and)), vec![8, 0, 0, 255]);
    assert_eq!(vals(g.view(or)), vec![14, -5, 5, -1]);
    assert_eq!(vals(g.view(xor)), vec![6, -5, 5, -256]);
    assert_eq!(vals(g.view(complement)), vec![-13, 7, -6, 0]);
    assert_eq!(vals(g.view(left)), vec![24, -32, 40, -16]);
    assert_eq!(vals(g.view(right)), vec![6, -2, 0, -1]);
}

// ---------------------------------------------------------------------------
// cmp: predicates, ordering, Ord min/max/clamp
// ---------------------------------------------------------------------------

/// `eq`/`ne`/`lt`/`le`/`gt`/`ge` over one pair, covering the less-than, equal
/// and greater-than case of each. The last element pins that `-0.0` and `0.0`
/// compare *equal* — they are two bit patterns for one value.
#[test]
fn cmp_predicate_family() {
    let mut b = Builder::new();
    let (xs, x) = b.source(array::from_parts([4], vec![0.0_f64; 4].into()));
    let y = b.value(array::from_parts([4], vec![2.0_f64, 2.0, 1.0, 0.0].into()));
    let equal = b.segment(elem::eq(), (x, y));
    let unequal = b.segment(elem::ne(), (x, y));
    let less = b.segment(elem::lt(), (x, y));
    let less_eq = b.segment(elem::le(), (x, y));
    let greater = b.segment(elem::gt(), (x, y));
    let greater_eq = b.segment(elem::ge(), (x, y));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(xs) = arr([4], vec![1.0, 2.0, 3.0, -0.0]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(vals(g.view(equal)), vec![false, true, false, true]);
    assert_eq!(vals(g.view(unequal)), vec![true, false, true, false]);
    assert_eq!(vals(g.view(less)), vec![true, false, false, false]);
    assert_eq!(vals(g.view(less_eq)), vec![true, true, false, true]);
    assert_eq!(vals(g.view(greater)), vec![false, false, true, false]);
    assert_eq!(vals(g.view(greater_eq)), vec![false, true, true, true]);
}

/// IEEE `NaN` semantics, the rule the whole missing-data convention rests on:
/// `NaN` is *false* under every ordering predicate and under `eq`, but *true*
/// under `ne` — so `!(x > 0)` and `x <= 0` are not interchangeable on data
/// that can be missing. `partial_cmp` reports the incomparability explicitly
/// as `None`, and `is_finite` is the explicit missing-data test.
#[test]
fn cmp_follows_ieee_nan_semantics() {
    let mut b = Builder::new();
    let (xs, x) = b.source(array::from_parts([4], vec![0.0_f64; 4].into()));
    let zeros = b.value(array::from_parts([4], vec![0.0_f64; 4].into()));
    let equal = b.segment(elem::eq(), (x, zeros));
    let unequal = b.segment(elem::ne(), (x, zeros));
    let less = b.segment(elem::lt(), (x, zeros));
    let less_eq = b.segment(elem::le(), (x, zeros));
    let greater = b.segment(elem::gt(), (x, zeros));
    let greater_eq = b.segment(elem::ge(), (x, zeros));
    let order = b.segment(elem::partial_cmp(), (x, zeros));
    let finite = b.segment(elem::is_finite(), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(xs) = arr([4], vec![1.0, -1.0, 0.0, f64::NAN]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(vals(g.view(equal)), vec![false, false, true, false]);
    assert_eq!(vals(g.view(unequal)), vec![true, true, false, true]);
    assert_eq!(vals(g.view(less)), vec![false, true, false, false]);
    assert_eq!(vals(g.view(less_eq)), vec![false, true, true, false]);
    assert_eq!(vals(g.view(greater)), vec![true, false, false, false]);
    assert_eq!(vals(g.view(greater_eq)), vec![true, false, true, false]);
    assert_eq!(
        vals(g.view(order)),
        vec![
            Some(Ordering::Greater),
            Some(Ordering::Less),
            Some(Ordering::Equal),
            None,
        ]
    );
    assert_eq!(vals(g.view(finite)), vec![true, true, true, false]);
}

/// `min`/`max`/`clamp` are the `Ord` family, so they are the *integer* pick;
/// the float spellings are `minf`/`maxf`/`clampf`, which have different `NaN`
/// rules (see `float_min_max_clamp_nan_handling`).
#[test]
fn cmp_ord_min_max_clamp_family() {
    let mut b = Builder::new();
    let (xs, x) = b.source(array::from_parts([4], vec![0_i32; 4].into()));
    let y = b.value(array::from_parts([4], vec![4_i32, 2, -8, 7].into()));
    let smaller = b.segment(elem::min(), (x, y));
    let larger = b.segment(elem::max(), (x, y));
    let clamped = b.segment(elem::clamp(-2_i32, 5), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(xs) = arr([4], vec![1_i32, 5, -3, 7]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(vals(g.view(smaller)), vec![1, 2, -8, 7]);
    assert_eq!(vals(g.view(larger)), vec![4, 5, -3, 7]);
    assert_eq!(vals(g.view(clamped)), vec![1, 5, -2, 5]);
}

// ---------------------------------------------------------------------------
// float: rounding, sign, powers, trig, classification, missing data
// ---------------------------------------------------------------------------

/// `floor`/`ceil`/`round`/`trunc`/`fract` over the halves, where the five
/// differ maximally. `round` is half-*away-from-zero* (not banker's rounding:
/// `2.5` goes to `3`, not `2`), `trunc` rounds toward zero rather than down,
/// and `fract` keeps the sign of its input (`fract(-1.5) == -0.5`).
#[test]
fn float_rounding_family() {
    let mut b = Builder::new();
    let (xs, x) = b.source(array::from_parts([5], vec![0.0_f64; 5].into()));
    let down = b.segment(elem::floor(), x);
    let up = b.segment(elem::ceil(), x);
    let nearest = b.segment(elem::round(), x);
    let toward_zero = b.segment(elem::trunc(), x);
    let frac = b.segment(elem::fract(), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(xs) = arr([5], vec![-1.5, -0.5, 0.5, 1.5, 2.5]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(vals(g.view(down)), vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    assert_eq!(vals(g.view(up)), vec![-1.0, -0.0, 1.0, 2.0, 3.0]);
    assert_eq!(vals(g.view(nearest)), vec![-2.0, -1.0, 1.0, 2.0, 3.0]);
    assert_eq!(vals(g.view(toward_zero)), vec![-1.0, -0.0, 0.0, 1.0, 2.0]);
    assert_eq!(vals(g.view(frac)), vec![-0.5, -0.5, 0.5, 0.5, 0.5]);
}

/// `abs`/`signum`/`recip`. `signum` reads the *sign bit*, so `signum(-0.0)` is
/// `-1.0` while `signum(0.0)` is `+1.0` — it never returns zero — and it
/// propagates `NaN`. `recip` of a signed zero is the correspondingly signed
/// infinity.
#[test]
fn float_sign_and_magnitude_family() {
    let mut b = Builder::new();
    let (xs, x) = b.source(array::from_parts([5], vec![0.0_f64; 5].into()));
    let magnitude = b.segment(elem::abs(), x);
    let sign = b.segment(elem::signum(), x);
    let inverse = b.segment(elem::recip(), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(xs) = arr([5], vec![-2.0, -0.0, 0.0, 4.0, f64::NAN]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_close(
        &vals(g.view(magnitude)),
        &[2.0, 0.0, 0.0, 4.0, f64::NAN],
        "abs",
    );
    assert_close(
        &vals(g.view(sign)),
        &[-1.0, -1.0, 1.0, 1.0, f64::NAN],
        "signum",
    );
    assert_close(
        &vals(g.view(inverse)),
        &[-0.5, f64::NEG_INFINITY, f64::INFINITY, 0.25, f64::NAN],
        "recip",
    );
}

/// `powi`/`powf`/`sqrt`/`cbrt`/`exp`/`exp2`/`ln`/`log`/`log2`/`log10` over one
/// positive vector, against the `f64` std-library results. The powers of two
/// give exact anchors (`log2` is integral, `powi(3)` is exactly representable)
/// that a wrong base or a swapped operand could not accidentally satisfy.
/// A second input pins the domain edges: `sqrt` of a negative is `NaN` where
/// `cbrt` is defined, and `ln(0)` is `-inf` rather than `NaN`.
#[test]
fn float_powers_roots_and_logs_family() {
    let x = [1.0_f64, 2.0, 8.0, 0.5];
    let mut b = Builder::new();
    let (xs, xp) = b.source(array::from_parts([4], vec![1.0_f64; 4].into()));
    let cube = b.segment(elem::powi(3), xp);
    let half_power = b.segment(elem::powf(0.5), xp);
    let root = b.segment(elem::sqrt(), xp);
    let cube_root = b.segment(elem::cbrt(), xp);
    let e_pow = b.segment(elem::exp(), xp);
    let two_pow = b.segment(elem::exp2(), xp);
    let natural = b.segment(elem::ln(), xp);
    let base3 = b.segment(elem::log(3.0), xp);
    let binary = b.segment(elem::log2(), xp);
    let decimal = b.segment(elem::log10(), xp);

    let (ds, dp) = b.source(array::from_parts([2], vec![1.0_f64; 2].into()));
    let edge_root = b.segment(elem::sqrt(), dp);
    let edge_cube_root = b.segment(elem::cbrt(), dp);
    let edge_ln = b.segment(elem::ln(), dp);

    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(xs) = arr([4], x.to_vec());
    *g.state_mut(ds) = arr([2], vec![-1.0_f64, 0.0]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(vals(g.view(cube)), vec![1.0, 8.0, 512.0, 0.125]);
    assert_close(
        &vals(g.view(half_power)),
        &each(&x, |v| v.powf(0.5)),
        "powf",
    );
    assert_close(&vals(g.view(root)), &each(&x, f64::sqrt), "sqrt");
    assert_close(&vals(g.view(cube_root)), &each(&x, f64::cbrt), "cbrt");
    assert_close(&vals(g.view(e_pow)), &each(&x, f64::exp), "exp");
    assert_close(&vals(g.view(two_pow)), &each(&x, f64::exp2), "exp2");
    assert_close(&vals(g.view(natural)), &each(&x, f64::ln), "ln");
    assert_close(&vals(g.view(base3)), &each(&x, |v| v.log(3.0)), "log(3)");
    assert_eq!(vals(g.view(binary)), vec![0.0, 1.0, 3.0, -1.0]);
    assert_close(&vals(g.view(decimal)), &each(&x, f64::log10), "log10");

    assert_close(&vals(g.view(edge_root)), &[f64::NAN, 0.0], "sqrt domain");
    assert_eq!(vals(g.view(edge_cube_root)), vec![-1.0, 0.0]);
    assert_close(
        &vals(g.view(edge_ln)),
        &[f64::NAN, f64::NEG_INFINITY],
        "ln domain",
    );
}

/// `sin`/`cos`/`tan`/`asin`/`acos`/`atan` against the `f64` std-library
/// results, over one vector inside the inverse functions' `[-1, 1]` domain.
#[test]
fn float_trig_family() {
    let x = [0.0_f64, 0.5, -0.75, 1.0];
    let mut b = Builder::new();
    let (xs, xp) = b.source(array::from_parts([4], vec![0.0_f64; 4].into()));
    let sine = b.segment(elem::sin(), xp);
    let cosine = b.segment(elem::cos(), xp);
    let tangent = b.segment(elem::tan(), xp);
    let arcsine = b.segment(elem::asin(), xp);
    let arccosine = b.segment(elem::acos(), xp);
    let arctangent = b.segment(elem::atan(), xp);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(xs) = arr([4], x.to_vec());
    g.stabilize(&mut pool, &Instant::MIN);

    assert_close(&vals(g.view(sine)), &each(&x, f64::sin), "sin");
    assert_close(&vals(g.view(cosine)), &each(&x, f64::cos), "cos");
    assert_close(&vals(g.view(tangent)), &each(&x, f64::tan), "tan");
    assert_close(&vals(g.view(arcsine)), &each(&x, f64::asin), "asin");
    assert_close(&vals(g.view(arccosine)), &each(&x, f64::acos), "acos");
    assert_close(&vals(g.view(arctangent)), &each(&x, f64::atan), "atan");
    // Anchors that a wrong-function wiring could not satisfy.
    assert_eq!(vals(g.view(sine))[0], 0.0);
    assert_eq!(vals(g.view(cosine))[0], 1.0);
    assert_close(&[vals(g.view(arcsine))[3]], &[PI / 2.0], "asin(1)");
}

/// `sinh`/`cosh`/`tanh`/`asinh`/`acosh`/`atanh` against the `f64`
/// std-library results. The input straddles both inverse domains, which is
/// the point: `acosh` is `NaN` below `1`, and `atanh` is `±inf` *at* `±1`
/// and `NaN` outside — a boundary that is easy to get wrong by clamping.
#[test]
fn float_hyperbolic_family() {
    let x = [0.0_f64, 0.5, -0.75, 1.0, 2.0];
    let mut b = Builder::new();
    let (xs, xp) = b.source(array::from_parts([5], vec![0.0_f64; 5].into()));
    let hsine = b.segment(elem::sinh(), xp);
    let hcosine = b.segment(elem::cosh(), xp);
    let htangent = b.segment(elem::tanh(), xp);
    let harcsine = b.segment(elem::asinh(), xp);
    let harccosine = b.segment(elem::acosh(), xp);
    let harctangent = b.segment(elem::atanh(), xp);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(xs) = arr([5], x.to_vec());
    g.stabilize(&mut pool, &Instant::MIN);

    assert_close(&vals(g.view(hsine)), &each(&x, f64::sinh), "sinh");
    assert_close(&vals(g.view(hcosine)), &each(&x, f64::cosh), "cosh");
    assert_close(&vals(g.view(htangent)), &each(&x, f64::tanh), "tanh");
    assert_close(&vals(g.view(harcsine)), &each(&x, f64::asinh), "asinh");
    assert_close(
        &vals(g.view(harccosine)),
        &[f64::NAN, f64::NAN, f64::NAN, 0.0, 2.0_f64.acosh()],
        "acosh",
    );
    assert_close(
        &vals(g.view(harctangent)),
        &[
            0.0,
            0.5_f64.atanh(),
            (-0.75_f64).atanh(),
            f64::INFINITY,
            f64::NAN,
        ],
        "atanh",
    );
}

/// `atan2` is the one two-argument member of the family, and the argument
/// order is the thing to pin: the *first* port is `y` and the second is `x`,
/// so the operator resolves the full circle rather than the half-plane
/// `atan(y / x)` gives. All four quadrants plus the negative-`x` axis.
#[test]
fn float_atan2_resolves_all_quadrants() {
    let mut b = Builder::new();
    let (ys, y) = b.source(array::from_parts([5], vec![0.0_f64; 5].into()));
    let x = b.value(array::from_parts(
        [5],
        vec![1.0_f64, -1.0, -1.0, 1.0, -1.0].into(),
    ));
    let angle = b.segment(elem::atan2(), (y, x));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(ys) = arr([5], vec![1.0, 1.0, -1.0, -1.0, 0.0]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_close(
        &vals(g.view(angle)),
        &[FRAC_PI_4, 3.0 * FRAC_PI_4, -3.0 * FRAC_PI_4, -FRAC_PI_4, PI],
        "atan2(y, x)",
    );
}

/// `is_nan`/`is_infinite`/`is_finite`/`is_normal` over one vector holding each
/// float class. The classes are not nested the obvious way: zero and
/// subnormals are finite but *not* normal, so `is_normal` is strictly stronger
/// than `is_finite` and not a spelling of it.
#[test]
fn float_classification_family() {
    let mut b = Builder::new();
    let (xs, x) = b.source(array::from_parts([6], vec![0.0_f64; 6].into()));
    let nan = b.segment(elem::is_nan(), x);
    let infinite = b.segment(elem::is_infinite(), x);
    let finite = b.segment(elem::is_finite(), x);
    let normal = b.segment(elem::is_normal(), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    let subnormal = f64::MIN_POSITIVE / 2.0;
    assert!(subnormal > 0.0 && !subnormal.is_normal(), "input setup");
    *g.state_mut(xs) = arr(
        [6],
        vec![
            1.0,
            0.0,
            subnormal,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ],
    );
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(
        vals(g.view(nan)),
        vec![false, false, false, false, false, true]
    );
    assert_eq!(
        vals(g.view(infinite)),
        vec![false, false, false, true, true, false]
    );
    assert_eq!(
        vals(g.view(finite)),
        vec![true, true, true, false, false, false]
    );
    assert_eq!(
        vals(g.view(normal)),
        vec![true, false, false, false, false, false]
    );
}

/// `minf`/`maxf` do *not* propagate `NaN` the way arithmetic does: per
/// [`Float::min`]/[`Float::max`] they return the other operand, so a missing
/// value silently loses to any present one and only `NaN` against `NaN`
/// survives. `clampf` deliberately breaks with them and passes `NaN` through,
/// since clamping a missing value to a bound would fabricate an observation.
#[test]
fn float_min_max_clamp_nan_handling() {
    let mut b = Builder::new();
    let (xs, x) = b.source(array::from_parts([5], vec![0.0_f64; 5].into()));
    let y = b.value(array::from_parts(
        [5],
        vec![2.0_f64, 4.0, f64::NAN, 7.0, f64::NAN].into(),
    ));
    let smaller = b.segment(elem::minf(), (x, y));
    let larger = b.segment(elem::maxf(), (x, y));
    let clamped = b.segment(elem::clampf(2.0, 5.0), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(xs) = arr([5], vec![1.0, 5.0, 3.0, f64::NAN, f64::NAN]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_close(
        &vals(g.view(smaller)),
        &[1.0, 4.0, 3.0, 7.0, f64::NAN],
        "minf",
    );
    assert_close(
        &vals(g.view(larger)),
        &[2.0, 5.0, 3.0, 7.0, f64::NAN],
        "maxf",
    );
    assert_close(
        &vals(g.view(clamped)),
        &[2.0, 5.0, 3.0, f64::NAN, f64::NAN],
        "clampf",
    );
}

/// `fill_nan` and `fill_where`. Despite its name `fill_nan` replaces every
/// **non-finite** value, infinities included. `fill_where` selects by an
/// arbitrary predicate, and that predicate is false for `NaN` under any
/// ordering comparison — so `fill_where(|v| v <= 1.0, _)` leaves `NaN` alone
/// while replacing `-inf`.
#[test]
fn float_missing_data_fill_family() {
    let mut b = Builder::new();
    let (xs, x) = b.source(array::from_parts([5], vec![0.0_f64; 5].into()));
    let filled = b.segment(elem::fill_nan(0.0), x);
    let replaced = b.segment(elem::fill_where(|v: f64| v <= 1.0, -1.0), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(xs) = arr(
        [5],
        vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 3.0],
    );
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(vals(g.view(filled)), vec![1.0, 0.0, 0.0, 0.0, 3.0]);
    assert_close(
        &vals(g.view(replaced)),
        &[-1.0, f64::NAN, f64::INFINITY, -1.0, 3.0],
        "fill_where",
    );
}

/// `forward_fill` is the one stateful operator in `elem`: it carries the last
/// finite value *per element* across generations, so it needs a multi-tick
/// test. An element that has never been finite stays `NaN` (there is nothing
/// to carry), and "finite" is again the criterion — an infinity counts as
/// missing and does not overwrite the carried value.
#[test]
fn float_forward_fill_carries_last_finite_across_generations() {
    let nan = f64::NAN;
    let mut b = Builder::new();
    let (xs, x) = b.source(array::from_parts([3], vec![nan; 3].into()));
    let filled = b.segment(elem::forward_fill(), x);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // The carry is seeded from the build-time value: nothing filled yet.
    assert_close(&vals(g.view(filled)), &[nan, nan, nan], "gen 0");

    *g.state_mut(xs) = arr([3], vec![1.0, nan, 3.0]);
    g.stabilize(&mut pool, &Instant::MIN);
    assert_close(&vals(g.view(filled)), &[1.0, nan, 3.0], "gen 1");

    *g.state_mut(xs) = arr([3], vec![nan, 2.0, nan]);
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(vals(g.view(filled)), vec![1.0, 2.0, 3.0], "gen 2: carried");

    *g.state_mut(xs) = arr([3], vec![f64::INFINITY, nan, 30.0]);
    g.stabilize(&mut pool, &Instant::MIN);
    assert_eq!(
        vals(g.view(filled)),
        vec![1.0, 2.0, 30.0],
        "gen 3: an infinity counts as missing"
    );
}

// ---------------------------------------------------------------------------
// boolean: connectives and selection
// ---------------------------------------------------------------------------

/// `and`/`or` and the two selectors, plus the `ops` bitwise operators over
/// `bool` (where `not` is logical negation and `bitxor` is exclusive-or).
/// `choose` picks elementwise between two *arrays* under a mask; `indicator`
/// picks between two *constants*, i.e. it is `choose` with the operands baked
/// in.
#[test]
fn boolean_connectives_and_selection() {
    let mut b = Builder::new();
    let (ps, p) = b.source(array::from_parts([4], vec![false; 4].into()));
    let q = b.value(array::from_parts(
        [4],
        vec![true, false, true, false].into(),
    ));
    let both = b.segment(elem::and(), (p, q));
    let either = b.segment(elem::or(), (p, q));
    let neither = b.segment(elem::not::<bool, 1>(), either);
    let exactly_one = b.segment(elem::bitxor::<bool, bool, 1>(), (p, q));
    let lhs = b.value(array::from_parts([4], vec![1.0_f64, 2.0, 3.0, 4.0].into()));
    let rhs = b.value(array::from_parts(
        [4],
        vec![10.0_f64, 20.0, 30.0, 40.0].into(),
    ));
    let picked = b.segment(elem::choose(), (exactly_one, lhs, rhs));
    let signed = b.segment(elem::indicator(1.0_f64, -1.0), p);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(ps) = arr([4], vec![true, true, false, false]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(vals(g.view(both)), vec![true, false, false, false]);
    assert_eq!(vals(g.view(either)), vec![true, true, true, false]);
    assert_eq!(vals(g.view(neither)), vec![false, false, false, true]);
    assert_eq!(vals(g.view(exactly_one)), vec![false, true, true, false]);
    // Where exactly one of `p`/`q` holds take `lhs`, else `rhs`.
    assert_eq!(vals(g.view(picked)), vec![10.0, 2.0, 3.0, 40.0]);
    assert_eq!(vals(g.view(signed)), vec![1.0, 1.0, -1.0, -1.0]);
}

// ---------------------------------------------------------------------------
// cast: scalar type conversion
// ---------------------------------------------------------------------------

/// `into` is the *lossless* conversion — it only compiles where `From` does,
/// so `i32 -> f64` is available but `f64 -> i32` is not — while `as_` is the
/// `as` cast and will happily lose information: it truncates toward zero,
/// saturates at the target's bounds, and sends `NaN` to `0`.
#[test]
fn cast_into_is_lossless_where_as_truncates() {
    let mut b = Builder::new();
    let (is, i) = b.source(array::from_parts([3], vec![0_i32; 3].into()));
    let widened = b.segment(elem::into::<i32, f64, 1>(), i);
    let (fs, f) = b.source(array::from_parts([6], vec![0.0_f64; 6].into()));
    let narrowed = b.segment(elem::as_::<f64, i32, 1>(), f);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(is) = arr([3], vec![1_i32, -2, 3]);
    *g.state_mut(fs) = arr(
        [6],
        vec![1.9_f64, -1.9, 2.5, f64::NAN, 1e18, f64::NEG_INFINITY],
    );
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(vals(g.view(widened)), vec![1.0, -2.0, 3.0]);
    assert_eq!(
        vals(g.view(narrowed)),
        vec![1, -1, 2, 0, i32::MAX, i32::MIN]
    );
}

// ---------------------------------------------------------------------------
// The shared elementwise core: strided and broadcast inputs
// ---------------------------------------------------------------------------

/// Every operator above routes through `array::{map, binary_map}`, which has a
/// contiguous fast path and a strided fallback. Feeding it views that are
/// *not* row-major contiguous — a transposed panel, and two columns picked out
/// of a `[2, 2]` — exercises the fallback, at both a `f64` and a `bool` output
/// type. Asserted over two generations, because the operators build their
/// output with a different code path on the first one (`map`, allocating) than
/// on later ones (`map_into`, writing in place).
#[test]
fn core_accepts_strided_view_inputs() {
    let mut b = Builder::new();
    // A `[2, 3]` panel read transposed: extents `[3, 2]`, strides `[1, 3]`.
    let (ps, p) = b.source(array::from_parts([2, 3], vec![0.0_f64; 6].into()));
    let flipped = b.segment(array::transpose([1, 0]), p);
    let magnitude = b.segment(elem::abs(), flipped);
    // Two columns of a `[2, 2]`: rank-1 views of stride 2.
    let (qs, q) = b.source(array::from_parts([2, 2], vec![0.0_f64; 4].into()));
    let col0 = b.segment(array::select_at::<_, 2, 1>(0, 1), q);
    let col1 = b.segment(array::select_at::<_, 2, 1>(1, 1), q);
    let rising = b.segment(elem::lt(), (col0, col1));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(ps) = arr([2, 3], vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
    *g.state_mut(qs) = arr([2, 2], vec![1.0, 5.0, 4.0, 2.0]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(magnitude).extents(), [3, 2]);
    assert_eq!(vals(g.view(magnitude)), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(vals(g.view(rising)), vec![true, false]);

    // A second generation takes the in-place update path over the same views.
    *g.state_mut(ps) = arr([2, 3], vec![-10.0, 20.0, -30.0, 40.0, -50.0, 60.0]);
    *g.state_mut(qs) = arr([2, 2], vec![9.0, 1.0, 0.0, 7.0]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(
        vals(g.view(magnitude)),
        vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0]
    );
    assert_eq!(vals(g.view(rising)), vec![false, true]);
}

/// The binary core stretches extent-1 axes rather than requiring equal
/// extents: a `[2, 1]` column times a `[1, 3]` row is the `[2, 3]` outer
/// product. This is how a whole-cross-section constant is applied without
/// materializing it.
#[test]
fn core_binary_op_broadcasts_extent_one_axes() {
    let mut b = Builder::new();
    let (cs, col) = b.source(array::from_parts([2, 1], vec![0.0_f64; 2].into()));
    let row = b.value(array::from_parts([1, 3], vec![10.0_f64, 20.0, 30.0].into()));
    let outer = b.segment(elem::mul(), (col, row));
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(cs) = arr([2, 1], vec![1.0, 2.0]);
    g.stabilize(&mut pool, &Instant::MIN);

    assert_eq!(g.view(outer).extents(), [2, 3]);
    assert_eq!(
        vals(g.view(outer)),
        vec![10.0, 20.0, 30.0, 20.0, 40.0, 60.0]
    );
}
