//! Helper functions for conversion between [`Array`](crate::Array) and
//! [`arrow::array::Array`], with support for long-format tables with numeric
//! or named axis schemas.

mod flat;
mod indexed;
mod scalar;

pub use flat::{build_column, read_column};
pub use indexed::{
    build_index_columns, build_value_column, read_index_columns, read_value_column, true_indices,
};
pub use scalar::{ArrowScalar, as_primitive_array};
