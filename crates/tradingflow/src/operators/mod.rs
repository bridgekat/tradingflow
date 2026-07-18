//! Built-in operator segments.

pub mod constant;
pub mod formula;
pub mod metrics;
pub mod num;
pub mod rolling;
pub mod stocks;
pub mod structural;
pub mod traders;
pub mod transform;

#[cfg(feature = "python")]
mod pyhost;

#[cfg(feature = "python")]
pub use pyhost::{
    NativeArrayView, NativeSeriesView, PyArgs, PyClassOperator, PyParams, py_class_operator,
    py_class_operator_file, py_class_operator_source,
};
