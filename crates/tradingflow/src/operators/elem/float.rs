use num_traits::Float;

use crate::data::{Instant, Scalar};
use crate::graph::Operator;
use crate::operators::array;
use crate::ports::ArrayPort;

/// Elementwise NaN flag: [`Float::is_nan`].
pub fn is_nan<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<bool, N>, Context = Instant> {
    array::map(|x: &T| x.is_nan())
}

/// Elementwise infinite flag: [`Float::is_infinite`].
pub fn is_infinite<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<bool, N>, Context = Instant> {
    array::map(|x: &T| x.is_infinite())
}

/// Elementwise finite flag: [`Float::is_finite`].
pub fn is_finite<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<bool, N>, Context = Instant> {
    array::map(|x: &T| x.is_finite())
}

/// Elementwise normalized flag: [`Float::is_normal`].
pub fn is_normal<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<bool, N>, Context = Instant> {
    array::map(|x: &T| x.is_normal())
}

/// Elementwise floor: [`Float::floor`].
pub fn floor<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.floor())
}

/// Elementwise ceiling: [`Float::ceil`].
pub fn ceil<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.ceil())
}

/// Elementwise rounding: [`Float::round`].
pub fn round<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.round())
}

/// Elementwise integer part: [`Float::trunc`].
pub fn trunc<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.trunc())
}

/// Elementwise fractional part: [`Float::fract`].
pub fn fract<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.fract())
}

/// Elementwise absolute value: [`Float::abs`].
pub fn abs<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.abs())
}

/// Elementwise signum: [`Float::signum`].
pub fn signum<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.signum())
}

/// Elementwise reciprocal: [`Float::recip`].
pub fn recip<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.recip())
}

/// Elementwise integer power: [`Float::powi`].
pub fn powi<T: Scalar + Float, const N: usize>(
    n: i32,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(move |x: &T| x.powi(n))
}

/// Elementwise floating-point power: [`Float::powf`].
pub fn powf<T: Scalar + Float, const N: usize>(
    n: T,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(move |x: &T| x.powf(n))
}

/// Elementwise square root: [`Float::sqrt`].
pub fn sqrt<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.sqrt())
}

/// Elementwise cube root: [`Float::cbrt`].
pub fn cbrt<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.cbrt())
}

/// Elementwise exponential: [`Float::exp`].
pub fn exp<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.exp())
}

/// Elementwise base-2 exponential: [`Float::exp2`].
pub fn exp2<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.exp2())
}

/// Elementwise natural logarithm: [`Float::ln`].
pub fn ln<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.ln())
}

/// Elementwise logarithm: [`Float::log`].
pub fn log<T: Scalar + Float, const N: usize>(
    base: T,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(move |x: &T| x.log(base))
}

/// Elementwise base-2 logarithm: [`Float::log2`].
pub fn log2<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.log2())
}

/// Elementwise base-10 logarithm: [`Float::log10`].
pub fn log10<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.log10())
}

/// Elementwise sine: [`Float::sin`].
pub fn sin<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.sin())
}

/// Elementwise cosine: [`Float::cos`].
pub fn cos<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.cos())
}

/// Elementwise tangent: [`Float::tan`].
pub fn tan<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.tan())
}

/// Elementwise arcsine: [`Float::asin`].
pub fn asin<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.asin())
}

/// Elementwise arccosine: [`Float::acos`].
pub fn acos<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.acos())
}

/// Elementwise arctangent: [`Float::atan`].
pub fn atan<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.atan())
}

/// Elementwise two-argument arctangent: [`Float::atan2`].
pub fn atan2<T: Scalar + Float, const N: usize>() -> impl Operator<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    array::binary_map(|&x: &T, &y: &T| x.atan2(y))
}

/// Elementwise hyperbolic sine: [`Float::sinh`].
pub fn sinh<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.sinh())
}

/// Elementwise hyperbolic cosine: [`Float::cosh`].
pub fn cosh<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.cosh())
}

/// Elementwise hyperbolic tangent: [`Float::tanh`].
pub fn tanh<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.tanh())
}

/// Elementwise hyperbolic arcsine: [`Float::asinh`].
pub fn asinh<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.asinh())
}

/// Elementwise hyperbolic arccosine: [`Float::acosh`].
pub fn acosh<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.acosh())
}

/// Elementwise hyperbolic arctangent: [`Float::atanh`].
pub fn atanh<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(|x: &T| x.atanh())
}

/// Elementwise floating-point minimum: [`Float::min`].
pub fn minf<T: Scalar + Float, const N: usize>() -> impl Operator<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    array::binary_map(|&x: &T, &y: &T| x.min(y))
}

/// Elementwise floating-point maximum: [`Float::max`].
pub fn maxf<T: Scalar + Float, const N: usize>() -> impl Operator<
    Inputs = (ArrayPort<T, N>, ArrayPort<T, N>),
    Outputs = ArrayPort<T, N>,
    Context = Instant,
> {
    array::binary_map(|&x: &T, &y: &T| x.max(y))
}

/// Elementwise floating-point clamping, propagating `NaN`.
pub fn clampf<T: Scalar + Float, const N: usize>(
    min: T,
    max: T,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    array::map(move |&x: &T| if x.is_nan() { x } else { x.min(max).max(min) })
}
