//! Statistical operators on (cross-sectional) arrays.

mod demean;
mod gaussianize;
mod percentile;
mod rank;
mod standardize;
mod winsorize;

pub use demean::demean;
pub use gaussianize::gaussianize;
pub use percentile::percentile;
pub use standardize::standardize;
pub use winsorize::winsorize;
