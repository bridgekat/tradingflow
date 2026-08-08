//! Rolling window operators on arrays or recorded series.

mod base;
mod count;
mod cov;
mod kurt;
mod lag;
mod mean;
mod mean_exp;
mod mean_linear;
mod min_max;
mod order;
mod regression;
mod skew;
mod std_dev;
mod sum;
mod var;

pub use base::{Accumulator, Rolling, RollingState, Scanning, rolling, series_rolling};
pub use count::{count, series_count};
pub use cov::{cov, series_cov};
pub use kurt::{kurt, series_kurt};
pub use lag::{diff, lag, pct_change, series_lag};
pub use mean::{mean, series_mean};
pub use mean_exp::{mean_exp, series_mean_exp};
pub use mean_linear::{mean_linear, series_mean_linear};
pub use min_max::{
    idx_max, idx_min, max, min, series_idx_max, series_idx_min, series_max, series_min,
};
pub use order::{
    mad, median, quantile, rank, series_mad, series_median, series_quantile, series_rank,
};
pub use regression::{resi, rsquare, series_resi, series_rsquare, series_slope, slope};
pub use skew::{series_skew, skew};
pub use std_dev::{series_std_dev, std_dev};
pub use sum::{series_sum, sum};
pub use var::{series_var, var};
