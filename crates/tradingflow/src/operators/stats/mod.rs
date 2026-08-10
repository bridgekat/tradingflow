//! Statistical operators on (cross-sectional) arrays.

mod common;
mod corr;
mod demean;
mod gaussianize;
mod group_demean;
mod percentile;
mod scale;
mod standardize;
mod winsorize;

pub use corr::{corr, corr_or_zero};
pub use demean::demean;
pub use gaussianize::gaussianize;
pub use group_demean::group_demean;
pub use percentile::percentile;
pub use scale::{scale, scale_down, scale_down_or_zero, scale_or_zero, scale_up, scale_up_or_zero};
pub use standardize::{standardize, standardize_or_zero};
pub use winsorize::winsorize;
