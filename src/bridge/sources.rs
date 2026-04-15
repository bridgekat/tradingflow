//! Native source dispatch for the Python bridge.
//!
//! [`dispatch_native_source`] maps a `(kind, dtype)` pair to a concrete
//! Rust source registration call.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::sources::ArraySource;
use crate::time::{Duration, Instant};
use crate::{Array, Scenario, Series};

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
            let timestamps = params
                .get_item("timestamps")?
                .ok_or_else(|| PyTypeError::new_err("array source requires 'timestamps'"))?;
            let values = params
                .get_item("values")?
                .ok_or_else(|| PyTypeError::new_err("array source requires 'values'"))?;
            let shape: Vec<usize> = params
                .get_item("shape")?
                .ok_or_else(|| PyTypeError::new_err("array source requires 'shape'"))?
                .extract()?;
            register_array_source(sc, dtype, &timestamps, &values, &shape)
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
            let timestamp_offset_ns: i64 = params
                .get_item("timestamp_offset_ns")?
                .map(|v| v.extract::<i64>())
                .transpose()?
                .unwrap_or(0);
            let start_ns: Option<i64> = params
                .get_item("start_ns")?
                .map(|v| v.extract::<i64>())
                .transpose()?;
            let end_ns: Option<i64> = params
                .get_item("end_ns")?
                .map(|v| v.extract::<i64>())
                .transpose()?;
            let is_utc: bool = params
                .get_item("is_utc")?
                .map(|v| v.extract::<bool>())
                .transpose()?
                .unwrap_or(true);
            let tz_offset_ns: i64 = params
                .get_item("tz_offset_ns")?
                .map(|v| v.extract::<i64>())
                .transpose()?
                .unwrap_or(0);
            use crate::sources::CsvSource;
            let source = CsvSource::new(
                path,
                time_column,
                value_columns,
                Duration::from_nanos(timestamp_offset_ns),
            )
            .with_timescale(is_utc, Duration::from_nanos(tz_offset_ns))
            .with_time_range(
                start_ns.map(Instant::from_nanos),
                end_ns.map(Instant::from_nanos),
            );
            Ok(sc.add_source(source).index())
        }
        "financial_report" => {
            let path: String = params
                .get_item("path")?
                .ok_or_else(|| PyTypeError::new_err("financial_report source requires 'path'"))?
                .extract()?;
            let report_date_column: String = params
                .get_item("report_date_column")?
                .ok_or_else(|| {
                    PyTypeError::new_err("financial_report source requires 'report_date_column'")
                })?
                .extract()?;
            let notice_date_column: String = params
                .get_item("notice_date_column")?
                .ok_or_else(|| {
                    PyTypeError::new_err("financial_report source requires 'notice_date_column'")
                })?
                .extract()?;
            let value_columns: Vec<String> = params
                .get_item("value_columns")?
                .ok_or_else(|| {
                    PyTypeError::new_err("financial_report source requires 'value_columns'")
                })?
                .extract()?;
            let with_report_date: bool = params
                .get_item("with_report_date")?
                .ok_or_else(|| {
                    PyTypeError::new_err("financial_report source requires 'with_report_date'")
                })?
                .extract()?;
            let use_effective_date: bool = params
                .get_item("use_effective_date")?
                .ok_or_else(|| {
                    PyTypeError::new_err(
                        "financial_report source requires 'use_effective_date'",
                    )
                })?
                .extract()?;
            let notice_date_fallback_ns: i64 = params
                .get_item("notice_date_fallback_ns")?
                .ok_or_else(|| {
                    PyTypeError::new_err(
                        "financial_report source requires 'notice_date_fallback_ns'",
                    )
                })?
                .extract()?;
            let start_ns: Option<i64> = params
                .get_item("start_ns")?
                .map(|v| v.extract::<i64>())
                .transpose()?;
            let end_ns: Option<i64> = params
                .get_item("end_ns")?
                .map(|v| v.extract::<i64>())
                .transpose()?;
            let is_utc: bool = params
                .get_item("is_utc")?
                .map(|v| v.extract::<bool>())
                .transpose()?
                .unwrap_or(true);
            let tz_offset_ns: i64 = params
                .get_item("tz_offset_ns")?
                .map(|v| v.extract::<i64>())
                .transpose()?
                .unwrap_or(0);
            use crate::sources::stocks::FinancialReportSource;
            let source = FinancialReportSource::new(
                path,
                report_date_column,
                notice_date_column,
                value_columns,
                with_report_date,
                use_effective_date,
                Duration::from_nanos(notice_date_fallback_ns),
            )
            .with_timescale(is_utc, Duration::from_nanos(tz_offset_ns))
            .with_time_range(
                start_ns.map(Instant::from_nanos),
                end_ns.map(Instant::from_nanos),
            );
            Ok(sc.add_source(source).index())
        }
        "clock" => {
            let timestamps: Vec<i64> = params
                .get_item("timestamps")?
                .ok_or_else(|| PyTypeError::new_err("clock source requires 'timestamps'"))?
                .extract()?;
            let timestamps: Vec<Instant> = timestamps
                .into_iter()
                .map(Instant::from_nanos)
                .collect();
            use crate::sources::clock;
            Ok(sc.add_source(clock(timestamps)).index())
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
    timestamps: &Bound<'_, pyo3::types::PyAny>,
    values: &Bound<'_, pyo3::types::PyAny>,
    shape: &[usize],
) -> PyResult<usize> {
    macro_rules! register {
        ($T:ty) => {{
            let info = ContiguousArrayInfo::try_from(timestamps)?;
            let unix_ns = unsafe { info.to_vec::<i64>() };
            let timestamps: Vec<Instant> =
                unix_ns.into_iter().map(Instant::from_nanos).collect();
            let info = ContiguousArrayInfo::try_from(values)?;
            let data = unsafe { info.to_vec::<$T>() };
            let source = ArraySource::new(
                Series::from_vec(shape, timestamps, data),
                Array::<$T>::zeros(shape),
            );
            sc.add_source(source).index()
        }};
    }
    Ok(dispatch_dtype!(dtype, register, numeric))
}
