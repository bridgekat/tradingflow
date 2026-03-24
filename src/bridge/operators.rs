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

    // Each non-parameterized op gets its own inline `go!` macro.
    // A helper macro can't forward $T:ty through nested macro_rules.
    macro_rules! go_add { ($T:ty) => { sc.register_operator_from_indices(operators::add::<$T>(), input_indices, clock) }; }
    macro_rules! go_sub { ($T:ty) => { sc.register_operator_from_indices(operators::subtract::<$T>(), input_indices, clock) }; }
    macro_rules! go_mul { ($T:ty) => { sc.register_operator_from_indices(operators::multiply::<$T>(), input_indices, clock) }; }
    macro_rules! go_div { ($T:ty) => { sc.register_operator_from_indices(operators::divide::<$T>(), input_indices, clock) }; }
    macro_rules! go_neg { ($T:ty) => { sc.register_operator_from_indices(operators::negate::<$T>(), input_indices, clock) }; }
    macro_rules! go_log { ($T:ty) => { sc.register_operator_from_indices(operators::log::<$T>(), input_indices, clock) }; }
    macro_rules! go_log2 { ($T:ty) => { sc.register_operator_from_indices(operators::log2::<$T>(), input_indices, clock) }; }
    macro_rules! go_log10 { ($T:ty) => { sc.register_operator_from_indices(operators::log10::<$T>(), input_indices, clock) }; }
    macro_rules! go_exp { ($T:ty) => { sc.register_operator_from_indices(operators::exp::<$T>(), input_indices, clock) }; }
    macro_rules! go_exp2 { ($T:ty) => { sc.register_operator_from_indices(operators::exp2::<$T>(), input_indices, clock) }; }
    macro_rules! go_sqrt { ($T:ty) => { sc.register_operator_from_indices(operators::sqrt::<$T>(), input_indices, clock) }; }
    macro_rules! go_ceil { ($T:ty) => { sc.register_operator_from_indices(operators::ceil::<$T>(), input_indices, clock) }; }
    macro_rules! go_floor { ($T:ty) => { sc.register_operator_from_indices(operators::floor::<$T>(), input_indices, clock) }; }
    macro_rules! go_round { ($T:ty) => { sc.register_operator_from_indices(operators::round::<$T>(), input_indices, clock) }; }
    macro_rules! go_recip { ($T:ty) => { sc.register_operator_from_indices(operators::recip::<$T>(), input_indices, clock) }; }
    macro_rules! go_abs { ($T:ty) => { sc.register_operator_from_indices(operators::abs::<$T>(), input_indices, clock) }; }
    macro_rules! go_sign { ($T:ty) => { sc.register_operator_from_indices(operators::sign::<$T>(), input_indices, clock) }; }
    macro_rules! go_min { ($T:ty) => { sc.register_operator_from_indices(operators::min::<$T>(), input_indices, clock) }; }
    macro_rules! go_max { ($T:ty) => { sc.register_operator_from_indices(operators::max::<$T>(), input_indices, clock) }; }

    match kind {
        // -- Binary arithmetic -----------------------------------------------
        "add"      => Ok((dispatch_dtype!(dtype, go_add, numeric), ViewKind::Array)),
        "subtract" => Ok((dispatch_dtype!(dtype, go_sub, numeric), ViewKind::Array)),
        "multiply" => Ok((dispatch_dtype!(dtype, go_mul, numeric), ViewKind::Array)),
        "divide"   => Ok((dispatch_dtype!(dtype, go_div, numeric), ViewKind::Array)),

        // -- Unary arithmetic ------------------------------------------------
        "negate" => Ok((dispatch_dtype!(dtype, go_neg, signed), ViewKind::Array)),

        // -- Float unary math ------------------------------------------------
        "log"   => Ok((dispatch_dtype!(dtype, go_log, float), ViewKind::Array)),
        "log2"  => Ok((dispatch_dtype!(dtype, go_log2, float), ViewKind::Array)),
        "log10" => Ok((dispatch_dtype!(dtype, go_log10, float), ViewKind::Array)),
        "exp"   => Ok((dispatch_dtype!(dtype, go_exp, float), ViewKind::Array)),
        "exp2"  => Ok((dispatch_dtype!(dtype, go_exp2, float), ViewKind::Array)),
        "sqrt"  => Ok((dispatch_dtype!(dtype, go_sqrt, float), ViewKind::Array)),
        "ceil"  => Ok((dispatch_dtype!(dtype, go_ceil, float), ViewKind::Array)),
        "floor" => Ok((dispatch_dtype!(dtype, go_floor, float), ViewKind::Array)),
        "round" => Ok((dispatch_dtype!(dtype, go_round, float), ViewKind::Array)),
        "recip" => Ok((dispatch_dtype!(dtype, go_recip, float), ViewKind::Array)),

        // -- Signed unary math -----------------------------------------------
        "abs"  => Ok((dispatch_dtype!(dtype, go_abs, signed), ViewKind::Array)),
        "sign" => Ok((dispatch_dtype!(dtype, go_sign, signed), ViewKind::Array)),

        // -- Float binary math -----------------------------------------------
        "min" => Ok((dispatch_dtype!(dtype, go_min, float), ViewKind::Array)),
        "max" => Ok((dispatch_dtype!(dtype, go_max, float), ViewKind::Array)),

        // -- Parameterized unary: pow ----------------------------------------
        "pow" => {
            macro_rules! go {
                ($T:ty) => {{
                    let n: $T = params
                        .get_item("n")?
                        .ok_or_else(|| PyTypeError::new_err("pow requires 'n' param"))?
                        .extract()?;
                    sc.register_operator_from_indices(
                        operators::pow::<$T>(n),
                        input_indices,
                        clock,
                    )
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
                    sc.register_operator_from_indices(
                        operators::scale::<$T>(c),
                        input_indices,
                        clock,
                    )
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
                    sc.register_operator_from_indices(
                        operators::shift::<$T>(c),
                        input_indices,
                        clock,
                    )
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
                    sc.register_operator_from_indices(
                        operators::clamp::<$T>(lo, hi),
                        input_indices,
                        clock,
                    )
                }};
            }
            Ok((dispatch_dtype!(dtype, go, float), ViewKind::Array))
        }

        // -- Parameterized unary: nan_to_num ---------------------------------
        "nan_to_num" => {
            macro_rules! go {
                ($T:ty) => {{
                    let val: $T = params
                        .get_item("val")?
                        .ok_or_else(|| PyTypeError::new_err("nan_to_num requires 'val' param"))?
                        .extract()?;
                    sc.register_operator_from_indices(
                        operators::nan_to_num::<$T>(val),
                        input_indices,
                        clock,
                    )
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

        // -- Variadic (homogeneous) ------------------------------------------
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

        // -- Record (Array → Series) -----------------------------------------
        "record" => {
            macro_rules! go {
                ($T:ty) => {{
                    use crate::operators::Record;
                    sc.register_operator_from_indices(Record::<$T>::new(), input_indices, clock)
                }};
            }
            Ok((dispatch_dtype!(dtype, go), ViewKind::Series))
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
                    sc.register_operator_from_indices(
                        operators::Last::<$T>::new(fill),
                        input_indices,
                        clock,
                    )
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
                    sc.register_operator_from_indices(
                        operators::Lag::<$T>::new(offset, fill),
                        input_indices,
                        clock,
                    )
                }};
            }
            Ok((dispatch_dtype!(dtype, go), ViewKind::Series))
        }

        // -- Rolling operators (Series → Series, float only) -----------------
        "rolling_sum" | "rolling_mean" | "rolling_variance" | "rolling_covariance" => {
            let window: usize = params
                .get_item("window")?
                .ok_or_else(|| PyTypeError::new_err(format!("{kind} requires 'window' param")))?
                .extract()?;
            macro_rules! go {
                ($T:ty) => {
                    match kind {
                        "rolling_sum" => sc.register_operator_from_indices(
                            operators::rolling::RollingSum::<$T>::new(window),
                            input_indices,
                            clock,
                        ),
                        "rolling_mean" => sc.register_operator_from_indices(
                            operators::rolling::RollingMean::<$T>::new(window),
                            input_indices,
                            clock,
                        ),
                        "rolling_variance" => sc.register_operator_from_indices(
                            operators::rolling::RollingVariance::<$T>::new(window),
                            input_indices,
                            clock,
                        ),
                        "rolling_covariance" => sc.register_operator_from_indices(
                            operators::rolling::RollingCovariance::<$T>::new(window),
                            input_indices,
                            clock,
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
