//! Native operator dispatch for the Python bridge.
//!
//! [`dispatch_native_operator`] maps a `(kind, dtype)` pair to a
//! monomorphised `add_operator_from_indices` call.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::operators;
use crate::scenario::Scenario;
use crate::{ErasedOperator, Operator};

use super::dispatch::{dispatch_dtype, normalize_dtype};
use super::views::ViewKind;

fn add_operator_from_indices(
    sc: &mut Scenario,
    operator: impl Operator,
    input_indices: &[usize],
    trigger_index: Option<usize>,
) -> usize {
    let erased = ErasedOperator::from_operator(operator, input_indices.len());
    sc.add_erased_operator(erased, input_indices, trigger_index)
}

/// Dispatch macro for zero-parameter operators.
///
/// Usage: `dispatch_op!(dtype, num::Add, subset, sc, inputs, trigger)`
/// Expands to `dispatch_dtype!` + `Operator::<$T>::new()` + registration.
macro_rules! dispatch_op {
    ($dtype:expr, $Op:ident, $subset:ident, $sc:expr, $inputs:expr, $trigger:expr) => {{
        macro_rules! go {
            ($T:ty) => {
                add_operator_from_indices($sc, operators::$Op::<$T>::new(), $inputs, $trigger)
            };
        }
        dispatch_dtype!($dtype, go, $subset)
    }};
    // Two-level path: operators::sub::Xxx
    ($dtype:expr, $sub:ident :: $Op:ident, $subset:ident, $sc:expr, $inputs:expr, $trigger:expr) => {{
        macro_rules! go {
            ($T:ty) => {
                add_operator_from_indices($sc, operators::$sub::$Op::<$T>::new(), $inputs, $trigger)
            };
        }
        dispatch_dtype!($dtype, go, $subset)
    }};
}

/// Register a Rust-native operator by `(kind, dtype)` and return the output
/// node index together with the output [`ViewKind`].
///
/// `trigger_index`: if `Some`, only that node triggers the operator;
/// if `None`, all inputs are trigger edges.
pub fn dispatch_native_operator(
    sc: &mut Scenario,
    kind: &str,
    dtype: &str,
    input_indices: &[usize],
    trigger_index: Option<usize>,
    params: &Bound<'_, PyDict>,
) -> PyResult<(usize, ViewKind)> {
    let dtype = normalize_dtype(dtype);

    match kind {
        // -- Binary arithmetic -----------------------------------------------
        "add" => Ok((dispatch_op!(dtype, num::Add, numeric, sc, input_indices, trigger_index), ViewKind::Array)),
        "subtract" => Ok((dispatch_op!(dtype, num::Subtract, numeric, sc, input_indices, trigger_index), ViewKind::Array)),
        "multiply" => Ok((dispatch_op!(dtype, num::Multiply, numeric, sc, input_indices, trigger_index), ViewKind::Array)),
        "divide" => Ok((dispatch_op!(dtype, num::Divide, numeric, sc, input_indices, trigger_index), ViewKind::Array)),

        // -- Unary arithmetic ------------------------------------------------
        "negate" => Ok((dispatch_op!(dtype, num::Negate, signed, sc, input_indices, trigger_index), ViewKind::Array)),

        // -- Float unary math ------------------------------------------------
        "log" => Ok((dispatch_op!(dtype, num::Log, float, sc, input_indices, trigger_index), ViewKind::Array)),
        "log2" => Ok((dispatch_op!(dtype, num::Log2, float, sc, input_indices, trigger_index), ViewKind::Array)),
        "log10" => Ok((dispatch_op!(dtype, num::Log10, float, sc, input_indices, trigger_index), ViewKind::Array)),
        "exp" => Ok((dispatch_op!(dtype, num::Exp, float, sc, input_indices, trigger_index), ViewKind::Array)),
        "exp2" => Ok((dispatch_op!(dtype, num::Exp2, float, sc, input_indices, trigger_index), ViewKind::Array)),
        "sqrt" => Ok((dispatch_op!(dtype, num::Sqrt, float, sc, input_indices, trigger_index), ViewKind::Array)),
        "ceil" => Ok((dispatch_op!(dtype, num::Ceil, float, sc, input_indices, trigger_index), ViewKind::Array)),
        "floor" => Ok((dispatch_op!(dtype, num::Floor, float, sc, input_indices, trigger_index), ViewKind::Array)),
        "round" => Ok((dispatch_op!(dtype, num::Round, float, sc, input_indices, trigger_index), ViewKind::Array)),
        "recip" => Ok((dispatch_op!(dtype, num::Recip, float, sc, input_indices, trigger_index), ViewKind::Array)),

        // -- Signed unary math -----------------------------------------------
        "abs" => Ok((dispatch_op!(dtype, num::Abs, signed, sc, input_indices, trigger_index), ViewKind::Array)),
        "sign" => Ok((dispatch_op!(dtype, num::Sign, signed, sc, input_indices, trigger_index), ViewKind::Array)),

        // -- Float binary math -----------------------------------------------
        "min" => Ok((dispatch_op!(dtype, num::Min, float, sc, input_indices, trigger_index), ViewKind::Array)),
        "max" => Ok((dispatch_op!(dtype, num::Max, float, sc, input_indices, trigger_index), ViewKind::Array)),

        // -- Record (Array → Series) -----------------------------------------
        "record" => Ok((dispatch_op!(dtype, Record, numeric, sc, input_indices, trigger_index), ViewKind::Series)),

        // -- Forward-fill (Series → Array, float only) ------------------------
        "forward_fill" => Ok((dispatch_op!(dtype, num::ForwardFill, float, sc, input_indices, trigger_index), ViewKind::Array)),

        // -- Identity (Array → Array) ----------------------------------------
        "id" => {
            macro_rules! go {
                ($T:ty) => {
                    add_operator_from_indices(
                        sc,
                        operators::Id::<crate::Array<$T>>::new(),
                        input_indices,
                        trigger_index,
                    )
                };
            }
            Ok((dispatch_dtype!(dtype, go), ViewKind::Array))
        }

        // -- Parameterized unary: pow ----------------------------------------
        "pow" => {
            macro_rules! go {
                ($T:ty) => {{
                    let n: $T = params
                        .get_item("n")?
                        .ok_or_else(|| PyTypeError::new_err("pow requires 'n' param"))?
                        .extract()?;
                    add_operator_from_indices(sc, operators::num::Pow::<$T>::new(n), input_indices, trigger_index)
                }};
            }
            Ok((dispatch_dtype!(dtype, go, float), ViewKind::Array))
        }

        // -- Parameterized unary: scale --------------------------------------
        "scale" => {
            macro_rules! go {
                ($T:ty) => {{
                    let c: $T = params
                        .get_item("c")?
                        .ok_or_else(|| PyTypeError::new_err("scale requires 'c' param"))?
                        .extract()?;
                    add_operator_from_indices(sc, operators::num::Scale::<$T>::new(c), input_indices, trigger_index)
                }};
            }
            Ok((dispatch_dtype!(dtype, go, numeric), ViewKind::Array))
        }

        // -- Parameterized unary: shift --------------------------------------
        "shift" => {
            macro_rules! go {
                ($T:ty) => {{
                    let c: $T = params
                        .get_item("c")?
                        .ok_or_else(|| PyTypeError::new_err("shift requires 'c' param"))?
                        .extract()?;
                    add_operator_from_indices(sc, operators::num::Shift::<$T>::new(c), input_indices, trigger_index)
                }};
            }
            Ok((dispatch_dtype!(dtype, go, numeric), ViewKind::Array))
        }

        // -- Parameterized unary: clamp --------------------------------------
        "clamp" => {
            macro_rules! go {
                ($T:ty) => {{
                    let lo: $T = params
                        .get_item("lo")?
                        .ok_or_else(|| PyTypeError::new_err("clamp requires 'lo' param"))?
                        .extract()?;
                    let hi: $T = params
                        .get_item("hi")?
                        .ok_or_else(|| PyTypeError::new_err("clamp requires 'hi' param"))?
                        .extract()?;
                    add_operator_from_indices(sc, operators::num::Clamp::<$T>::new(lo, hi), input_indices, trigger_index)
                }};
            }
            Ok((dispatch_dtype!(dtype, go, float), ViewKind::Array))
        }

        // -- Parameterized unary: fillna (nan_to_num) ------------------------
        "nan_to_num" => {
            macro_rules! go {
                ($T:ty) => {{
                    let val: $T = params
                        .get_item("val")?
                        .ok_or_else(|| PyTypeError::new_err("nan_to_num requires 'val' param"))?
                        .extract()?;
                    add_operator_from_indices(sc, operators::num::Fillna::<$T>::new(val), input_indices, trigger_index)
                }};
            }
            Ok((dispatch_dtype!(dtype, go, float), ViewKind::Array))
        }

        // -- Selection -------------------------------------------------------
        "select" => {
            let indices: Vec<usize> = params
                .get_item("indices")?
                .ok_or_else(|| PyTypeError::new_err("select requires 'indices' param"))?
                .extract()?;
            let axis: usize = params
                .get_item("axis")?
                .map(|v| v.extract::<usize>())
                .transpose()?
                .unwrap_or(0);
            let squeeze: bool = params
                .get_item("squeeze")?
                .map(|v| v.extract::<bool>())
                .transpose()?
                .unwrap_or(false);
            macro_rules! go {
                ($T:ty) => {
                    add_operator_from_indices(sc, operators::Select::<$T>::new(indices.clone(), axis, squeeze), input_indices, trigger_index)
                };
            }
            Ok((dispatch_dtype!(dtype, go), ViewKind::Array))
        }

        // -- Variadic (homogeneous) ------------------------------------------
        "concat" => {
            let axis: usize = params
                .get_item("axis")?
                .ok_or_else(|| PyTypeError::new_err("concat requires 'axis' param"))?
                .extract()?;
            macro_rules! go {
                ($T:ty) => {
                    add_operator_from_indices(sc, operators::Concat::<$T>::new(axis), input_indices, trigger_index)
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
                    add_operator_from_indices(sc, operators::Stack::<$T>::new(axis), input_indices, trigger_index)
                };
            }
            Ok((dispatch_dtype!(dtype, go), ViewKind::Array))
        }

        // -- Last (Series → Array) -------------------------------------------
        "last" => {
            macro_rules! go {
                ($T:ty) => {{
                    let fill: $T = params
                        .get_item("fill")?
                        .map(|v| v.extract::<$T>())
                        .transpose()?
                        .unwrap_or_default();
                    add_operator_from_indices(sc, operators::Last::<$T>::new(fill), input_indices, trigger_index)
                }};
            }
            Ok((dispatch_dtype!(dtype, go), ViewKind::Array))
        }

        // -- Lag (Series → Series) -------------------------------------------
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
                    add_operator_from_indices(sc, operators::Lag::<$T>::new(offset, fill), input_indices, trigger_index)
                }};
            }
            Ok((dispatch_dtype!(dtype, go), ViewKind::Series))
        }

        // -- Rolling operators (Series → Array, float only) ------------------
        "rolling_sum" | "rolling_mean" | "rolling_variance" | "rolling_covariance" => {
            let window = params.get_item("window")?.map(|v| v.extract::<usize>()).transpose()?;
            let window_ns = params.get_item("window_ns")?.map(|v| v.extract::<i64>()).transpose()?;
            macro_rules! go {
                ($T:ty) => {{
                    macro_rules! make_op {
                        ($Op:ty) => {
                            match (window, window_ns) {
                                (Some(w), None) => add_operator_from_indices(sc, <$Op>::count(w), input_indices, trigger_index),
                                (None, Some(w)) => add_operator_from_indices(sc, <$Op>::time_delta(w), input_indices, trigger_index),
                                _ => return Err(PyTypeError::new_err(format!("{kind} requires exactly one of 'window' or 'window_ns'"))),
                            }
                        };
                    }
                    match kind {
                        "rolling_sum" => make_op!(operators::rolling::RollingSum::<$T>),
                        "rolling_mean" => make_op!(operators::rolling::RollingMean::<$T>),
                        "rolling_variance" => make_op!(operators::rolling::RollingVariance::<$T>),
                        "rolling_covariance" => make_op!(operators::rolling::RollingCovariance::<$T>),
                        _ => unreachable!(),
                    }
                }};
            }
            Ok((dispatch_dtype!(dtype, go, float), ViewKind::Array))
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
                    add_operator_from_indices(sc, op, input_indices, trigger_index)
                }};
            }
            Ok((dispatch_dtype!(dtype, go, float), ViewKind::Array))
        }

        // -- Cast (Array<S> → Array<T>) --------------------------------------
        "cast" => {
            let from_dtype: String = params
                .get_item("from_dtype")?
                .ok_or_else(|| PyTypeError::new_err("cast requires 'from_dtype' param"))?
                .extract()?;
            macro_rules! go_from {
                ($S:ty) => {{
                    macro_rules! go_to {
                        ($T:ty) => {
                            add_operator_from_indices(sc, operators::Cast::<$S, $T>::new(), input_indices, trigger_index)
                        };
                    }
                    dispatch_dtype!(dtype, go_to, numeric)
                }};
            }
            Ok((
                dispatch_dtype!(&from_dtype, go_from, numeric),
                ViewKind::Array,
            ))
        }

        // -- Forward adjust (Array × Array → Array, f64 only) ----------------
        "forward_adjust" => {
            let output_prices: bool = params
                .get_item("output_prices")?
                .map(|v| v.extract::<bool>())
                .transpose()?
                .unwrap_or(true);
            Ok((
                add_operator_from_indices(
                    sc,
                    operators::stocks::ForwardAdjust::new().with_output_prices(output_prices),
                    input_indices,
                    trigger_index,
                ),
                ViewKind::Array,
            ))
        }

        // -- Annualize (Array → Array, f64 only) ----------------------------
        "annualize" => Ok((
            add_operator_from_indices(
                sc,
                operators::stocks::Annualize::new(),
                input_indices,
                trigger_index,
            ),
            ViewKind::Array,
        )),

        // -- Const (0-input → Array) -----------------------------------------
        "const" => {
            let shape: Vec<usize> = params
                .get_item("shape")?
                .ok_or_else(|| PyTypeError::new_err("const requires 'shape' param"))?
                .extract()?;
            let value = params.get_item("value")?;
            macro_rules! go {
                ($T:ty) => {{
                    let arr = match value {
                        Some(ref v) => {
                            let info = super::dispatch::ContiguousArrayInfo::try_from(v)?;
                            // SAFETY: dispatch_dtype ensures $T is a numeric type.
                            let data = unsafe { info.to_vec::<$T>() };
                            crate::Array::from_vec(&shape, data)
                        }
                        None => crate::Array::<$T>::zeros(&shape),
                    };
                    add_operator_from_indices(sc, operators::Const::new(arr), input_indices, trigger_index)
                }};
            }
            Ok((dispatch_dtype!(dtype, go), ViewKind::Array))
        }

        // -- Metrics --------------------------------------------------------
        "compound_return" | "average_return" | "volatility" | "sharpe_ratio" | "drawdown" => {
            macro_rules! go {
                ($T:ty) => {
                    match kind {
                        "compound_return" => add_operator_from_indices(sc, operators::metrics::CompoundReturn::<$T>::new(), input_indices, trigger_index),
                        "average_return" => add_operator_from_indices(sc, operators::metrics::AverageReturn::<$T>::new(), input_indices, trigger_index),
                        "volatility" => add_operator_from_indices(sc, operators::metrics::Volatility::<$T>::new(), input_indices, trigger_index),
                        "sharpe_ratio" => add_operator_from_indices(sc, operators::metrics::SharpeRatio::<$T>::new(), input_indices, trigger_index),
                        "drawdown" => add_operator_from_indices(sc, operators::metrics::Drawdown::<$T>::new(), input_indices, trigger_index),
                        _ => unreachable!(),
                    }
                };
            }
            Ok((dispatch_dtype!(dtype, go, float), ViewKind::Array))
        }

        other => Err(PyTypeError::new_err(format!(
            "unknown native operator kind: {other}"
        ))),
    }
}
