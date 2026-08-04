//! Statistical operators on (cross-sectional) arrays.

mod demean;
mod gaussianize;
mod group_demean;
mod percentile;
mod rank;
mod scale;
mod standardize;
mod winsorize;

pub use demean::demean;
pub use gaussianize::gaussianize;
pub use group_demean::group_demean;
pub use percentile::percentile;
pub use scale::{scale, scale_down, scale_up};
pub use standardize::standardize;
pub use winsorize::winsorize;
