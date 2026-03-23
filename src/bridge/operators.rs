//! Native operator dispatch for the Python bridge.
//!
//! [`dispatch_native_operator`] maps a `(kind, dtype)` pair to a
//! monomorphised `Scenario::register_operator_from_indices` call.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::operators;
use crate::scenario::Scenario;

use super::ViewKind;
use super::dispatch::{dispatch_dtype, normalise_dtype};

/// Register a Rust-native operator by `(kind, dtype)` and return the output
/// node index together with the output node kind.
///
/// If `clock` is `Some(idx)`, only the clock node triggers.
/// If `clock` is `None`, all inputs are trigger edges.
pub fn dispatch_native_operator(
    sc: &mut Scenario,
    kind: &str,
    dtype: &str,
    input_indices: &[usize],
    clock: Option<usize>,
    params: &Bound<'_, PyDict>,
) -> PyResult<(usize, ViewKind)> {
    let dtype = normalise_dtype(dtype);

    match kind {
        // -- Binary element-wise operators ----------------------------------
        "add" => {
            macro_rules! go {
                ($T:ty) => {
                    sc.register_operator_from_indices(operators::add::<$T>(), input_indices, clock)
                };
            }
            Ok((dispatch_dtype!(dtype, go, numeric), ViewKind::Array))
        }
        "subtract" => {
            macro_rules! go {
                ($T:ty) => {
                    sc.register_operator_from_indices(
                        operators::subtract::<$T>(),
                        input_indices,
                        clock,
                    )
                };
            }
            Ok((dispatch_dtype!(dtype, go, numeric), ViewKind::Array))
        }
        "multiply" => {
            macro_rules! go {
                ($T:ty) => {
                    sc.register_operator_from_indices(
                        operators::multiply::<$T>(),
                        input_indices,
                        clock,
                    )
                };
            }
            Ok((dispatch_dtype!(dtype, go, numeric), ViewKind::Array))
        }
        "divide" => {
            macro_rules! go {
                ($T:ty) => {
                    sc.register_operator_from_indices(
                        operators::divide::<$T>(),
                        input_indices,
                        clock,
                    )
                };
            }
            Ok((dispatch_dtype!(dtype, go, numeric), ViewKind::Array))
        }

        // -- Unary element-wise operators -----------------------------------
        "negate" => {
            macro_rules! go {
                ($T:ty) => {
                    sc.register_operator_from_indices(
                        operators::negate::<$T>(),
                        input_indices,
                        clock,
                    )
                };
            }
            Ok((dispatch_dtype!(dtype, go, signed), ViewKind::Array))
        }

        // -- Parameterised operators ----------------------------------------
        "select" => {
            let indices: Vec<usize> = params
                .get_item("indices")?
                .ok_or_else(|| PyTypeError::new_err("select requires 'indices' param"))?
                .extract()?;
            macro_rules! go {
                ($T:ty) => {
                    sc.register_operator_from_indices(
                        operators::Select::<$T>::flat(indices.clone()),
                        input_indices,
                        clock,
                    )
                };
            }
            Ok((dispatch_dtype!(dtype, go), ViewKind::Array))
        }

        // -- Variadic (homogeneous) operators -------------------------------
        "concat" => {
            let axis: usize = params
                .get_item("axis")?
                .ok_or_else(|| PyTypeError::new_err("concat requires 'axis' param"))?
                .extract()?;
            macro_rules! go {
                ($T:ty) => {
                    sc.register_operator_from_indices(
                        operators::Concat::<$T>::new(axis),
                        input_indices,
                        clock,
                    )
                };
            }
            Ok((dispatch_dtype!(dtype, go), ViewKind::Array))
        }
        "stack" => {
            let axis: usize = params
                .get_item("axis")?
                .ok_or_else(|| PyTypeError::new_err("stack requires 'axis' param"))?
                .extract()?;
            macro_rules! go {
                ($T:ty) => {
                    sc.register_operator_from_indices(
                        operators::Stack::<$T>::new(axis),
                        input_indices,
                        clock,
                    )
                };
            }
            Ok((dispatch_dtype!(dtype, go), ViewKind::Array))
        }

        // -- Record (Array → Series) ----------------------------------------
        "record" => {
            macro_rules! go {
                ($T:ty) => {{
                    use crate::operators::Record;
                    sc.register_operator_from_indices(Record::<$T>::new(), input_indices, clock)
                }};
            }
            Ok((dispatch_dtype!(dtype, go), ViewKind::Series))
        }

        // -- Last (Series → Array) ------------------------------------------
        "last" => {
            macro_rules! go {
                ($T:ty) => {{
                    let fill: $T = params
                        .get_item("fill")?
                        .map(|v| v.extract::<$T>())
                        .transpose()?
                        .unwrap_or_default();
                    sc.register_operator_from_indices(
                        operators::Last::<$T>::new(fill),
                        input_indices,
                        clock,
                    )
                }};
            }
            Ok((dispatch_dtype!(dtype, go), ViewKind::Array))
        }

        // -- Lag (Series → Series) ------------------------------------------
        "lag" => {
            let offset: usize = params
                .get_item("offset")?
                .ok_or_else(|| PyTypeError::new_err("lag requires 'offset' param"))?
                .extract()?;
            macro_rules! go {
                ($T:ty) => {{
                    let fill: $T = params
                        .get_item("fill")?
                        .map(|v| v.extract::<$T>())
                        .transpose()?
                        .unwrap_or_default();
                    sc.register_operator_from_indices(
                        operators::Lag::<$T>::new(offset, fill),
                        input_indices,
                        clock,
                    )
                }};
            }
            Ok((dispatch_dtype!(dtype, go), ViewKind::Series))
        }

        // -- Rolling operators (Series → Series, float only) ----------------
        //
        // All four windowed rolling operators share the same dispatch
        // pattern: extract `window`, monomorphise over float dtypes.
        "rolling_sum" | "rolling_mean" | "rolling_variance" | "rolling_covariance" => {
            let window: usize = params
                .get_item("window")?
                .ok_or_else(|| {
                    PyTypeError::new_err(format!("{kind} requires 'window' param"))
                })?
                .extract()?;
            macro_rules! go {
                ($T:ty) => {
                    match kind {
                        "rolling_sum" => sc.register_operator_from_indices(
                            operators::rolling::RollingSum::<$T>::new(window),
                            input_indices, clock,
                        ),
                        "rolling_mean" => sc.register_operator_from_indices(
                            operators::rolling::RollingMean::<$T>::new(window),
                            input_indices, clock,
                        ),
                        "rolling_variance" => sc.register_operator_from_indices(
                            operators::rolling::RollingVariance::<$T>::new(window),
                            input_indices, clock,
                        ),
                        "rolling_covariance" => sc.register_operator_from_indices(
                            operators::rolling::RollingCovariance::<$T>::new(window),
                            input_indices, clock,
                        ),
                        _ => unreachable!(),
                    }
                };
            }
            Ok((dispatch_dtype!(dtype, go, float), ViewKind::Series))
        }
        "ema" => {
            let window: usize = params
                .get_item("window")?
                .ok_or_else(|| PyTypeError::new_err("ema requires 'window' param"))?
                .extract()?;
            macro_rules! go {
                ($T:ty) => {{
                    let op = if let Some(v) = params.get_item("alpha")? {
                        operators::rolling::Ema::<$T>::new(v.extract::<$T>()?, window)
                    } else if let Some(v) = params.get_item("span")? {
                        operators::rolling::Ema::<$T>::with_span(v.extract::<usize>()?, window)
                    } else if let Some(v) = params.get_item("half_life")? {
                        operators::rolling::Ema::<$T>::with_half_life(v.extract::<$T>()?, window)
                    } else {
                        return Err(PyTypeError::new_err(
                            "ema requires one of 'alpha', 'span', or 'half_life'",
                        ));
                    };
                    sc.register_operator_from_indices(op, input_indices, clock)
                }};
            }
            Ok((dispatch_dtype!(dtype, go, float), ViewKind::Series))
        }
        "forward_fill" => {
            macro_rules! go {
                ($T:ty) => {
                    sc.register_operator_from_indices(
                        operators::rolling::ForwardFill::<$T>::default(),
                        input_indices,
                        clock,
                    )
                };
            }
            Ok((dispatch_dtype!(dtype, go, float), ViewKind::Series))
        }

        other => Err(PyTypeError::new_err(format!(
            "unknown native operator kind: {other}"
        ))),
    }
}
