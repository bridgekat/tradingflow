//! Axis reduction operators on arrays.

mod boolean;
mod cmp;
mod float;
mod ops;

pub use boolean::{all, any, count};
pub use cmp::{max, min};
pub use float::{count_finite, maxf, minf, product_finite, sum_finite};
pub use ops::{product, sum};
