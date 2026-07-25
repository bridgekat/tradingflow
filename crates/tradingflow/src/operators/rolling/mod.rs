//! Basic rolling window operators on arrays or recorded series.

mod base;
mod cov;
mod lag;
mod mean;
mod mean_exp;
mod std_dev;
mod sum;
mod var;

pub use base::{Accumulator, Rolling, RollingState};
pub use cov::{cov, series_cov};
pub use lag::{diff, lag, pct_change, series_lag};
pub use mean::{mean, series_mean};
pub use mean_exp::{mean_exp, series_mean_exp};
pub use std_dev::{series_std_dev, std_dev};
pub use sum::{series_sum, sum};
pub use var::{series_var, var};
