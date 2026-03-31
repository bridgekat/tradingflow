//! Native source dispatch for the Python bridge.
//!
//! [`dispatch_native_source`] maps a `(kind, dtype)` pair to a concrete
//! Rust source registration call.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::sources::ArraySource;
use crate::Scenario;

use super::dispatch::{ContiguousArrayInfo, dispatch_dtype};

/// Register a Rust-native source by `(kind, dtype)` and return the output
/// node index.
pub fn dispatch_native_source(
    sc: &mut Scenario,
    kind: &str,
    dtype: &str,
    params: &Bound<'_, PyDict>,
) -> PyResult<usize> {
    match kind {
        "array" => {
            let timestamps: Vec<i64> = params
                .get_item("timestamps")?
                .ok_or_else(|| PyTypeError::new_err("array source requires 'timestamps'"))?
                .extract()?;
            let values = params
                .get_item("values")?
                .ok_or_else(|| PyTypeError::new_err("array source requires 'values'"))?;
            let stride: usize = params
                .get_item("stride")?
                .ok_or_else(|| PyTypeError::new_err("array source requires 'stride'"))?
                .extract()?;
            register_array_source(sc, dtype, timestamps, &values, stride)
        }
        "csv" => {
            let path: String = params
                .get_item("path")?
                .ok_or_else(|| PyTypeError::new_err("csv source requires 'path'"))?
                .extract()?;
            let time_column: String = params
                .get_item("time_column")?
                .ok_or_else(|| PyTypeError::new_err("csv source requires 'time_column'"))?
                .extract()?;
            let value_columns: Vec<String> = params
                .get_item("value_columns")?
                .ok_or_else(|| PyTypeError::new_err("csv source requires 'value_columns'"))?
                .extract()?;
            use crate::sources::CsvSource;
            let source = CsvSource::new(path, time_column, value_columns);
            Ok(sc.add_source(source).index())
        }
        "clock" => {
            let timestamps: Vec<i64> = params
                .get_item("timestamps")?
                .ok_or_else(|| PyTypeError::new_err("clock source requires 'timestamps'"))?
                .extract()?;
            use crate::sources::clock;
            Ok(sc.add_source(clock(timestamps)).index())
        }
        "daily_clock" => {
            let start_ns: i64 = params
                .get_item("start_ns")?
                .ok_or_else(|| PyTypeError::new_err("daily_clock requires 'start_ns'"))?
                .extract()?;
            let end_ns: i64 = params
                .get_item("end_ns")?
                .ok_or_else(|| PyTypeError::new_err("daily_clock requires 'end_ns'"))?
                .extract()?;
            let tz: String = params
                .get_item("tz")?
                .ok_or_else(|| PyTypeError::new_err("daily_clock requires 'tz'"))?
                .extract()?;
            use crate::sources::daily_clock;
            Ok(sc.add_source(daily_clock(start_ns, end_ns, &tz)).index())
        }
        "monthly_clock" => {
            let start_ns: i64 = params
                .get_item("start_ns")?
                .ok_or_else(|| PyTypeError::new_err("monthly_clock requires 'start_ns'"))?
                .extract()?;
            let end_ns: i64 = params
                .get_item("end_ns")?
                .ok_or_else(|| PyTypeError::new_err("monthly_clock requires 'end_ns'"))?
                .extract()?;
            let tz: String = params
                .get_item("tz")?
                .ok_or_else(|| PyTypeError::new_err("monthly_clock requires 'tz'"))?
                .extract()?;
            use crate::sources::monthly_clock;
            Ok(sc.add_source(monthly_clock(start_ns, end_ns, &tz)).index())
        }
        other => Err(PyTypeError::new_err(format!(
            "unknown native source kind: {other}"
        ))),
    }
}

/// Create a node and register an `ArraySource` in one step.
fn register_array_source(
    sc: &mut Scenario,
    dtype: &str,
    timestamps: Vec<i64>,
    values: &Bound<'_, pyo3::types::PyAny>,
    stride: usize,
) -> PyResult<usize> {
    macro_rules! register {
        ($T:ty) => {{
            let info = ContiguousArrayInfo::try_from(values)?;
            // SAFETY: dispatch_dtype ensures $T is a numeric type.
            let data = unsafe { info.to_vec::<$T>() };
            let source = ArraySource::new(timestamps, data, stride);
            sc.add_source(source).index()
        }};
    }
    Ok(dispatch_dtype!(dtype, register, numeric))
}
