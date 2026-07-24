//! Elementwise operators on arrays.

mod boolean;
mod cast;
mod cmp;
mod float;
mod ops;

pub use boolean::{and, choose, indicator, or};
pub use cast::{as_, into};
pub use cmp::{clamp, eq, ge, gt, le, lt, max, min, ne, partial_cmp};
pub use float::{
    abs, acos, acosh, asin, asinh, atan, atan2, atanh, cbrt, ceil, clampf, cos, cosh, exp, exp2,
    fill_nan, fill_where, floor, forward_fill, fract, is_finite, is_infinite, is_nan, is_normal,
    ln, log, log2, log10, maxf, minf, powf, powi, recip, round, signum, sin, sinh, sqrt, tan, tanh,
    trunc,
};
pub use ops::{add, bitand, bitor, bitxor, div, mul, neg, not, rem, shl, shr, sub};
