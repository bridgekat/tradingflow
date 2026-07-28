//! Statistical operators on (cross-sectional) arrays.

mod gaussianize;
mod percentile;
mod rank;
mod standardize;
mod winsorize;

pub use gaussianize::gaussianize;
pub use percentile::percentile;
pub use standardize::standardize;
pub use winsorize::winsorize;
