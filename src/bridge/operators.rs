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

use super::ViewKind;
use super::dispatch::{dispatch_dtype, normalise_dtype};

fn add_operator_from_indices(
    sc: &mut Scenario,
    operator: impl Operator,
    input_indices: &[usize],
    trigger_index: Option<usize>,
) -> usize {
    let erased = ErasedOperator::from_operator(operator, input_indices.len());
    sc.add_erased_operator(erased, input_indices, trigger_index)
}

/// Register a Rust-native operator by `(kind, dtype)` and return the output
/// node index together with the output node kind.
///
/// If `trigger_index` is `Some(idx)`, only the trigger node triggers.
/// If `trigger_index` is `None`, all inputs are trigger edges.
pub fn dispatch_native_operator(
    sc: &mut Scenario,
    kind: &str,
    dtype: &str,
    input_indices: &[usize],
    trigger_index: Option<usize>,
    params: &Bound<'_, PyDict>,
) -> PyResult<(usize, ViewKind)> {
    let dtype = normalise_dtype(dtype);

    // Each non-parameterized op gets its own inline `go!` macro.
    // A helper macro can't forward $T:ty through nested macro_rules.
    macro_rules! go_add {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::add::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_sub {
        ($T:ty) => {
            add_operator_from_indices(
                sc,
                operators::subtract::<$T>(),
                input_indices,
                trigger_index,
            )
        };
    }
    macro_rules! go_mul {
        ($T:ty) => {
            add_operator_from_indices(
                sc,
                operators::multiply::<$T>(),
                input_indices,
                trigger_index,
            )
        };
    }
    macro_rules! go_div {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::divide::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_neg {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::negate::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_log {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::log::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_log2 {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::log2::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_log10 {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::log10::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_exp {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::exp::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_exp2 {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::exp2::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_sqrt {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::sqrt::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_ceil {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::ceil::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_floor {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::floor::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_round {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::round::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_recip {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::recip::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_abs {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::abs::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_sign {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::sign::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_min {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::min::<$T>(), input_indices, trigger_index)
        };
    }
    macro_rules! go_max {
        ($T:ty) => {
            add_operator_from_indices(sc, operators::max::<$T>(), input_indices, trigger_index)
        };
    }

    match kind {
        // -- Binary arithmetic -----------------------------------------------
        "add" => Ok((dispatch_dtype!(dtype, go_add, numeric), ViewKind::Array)),
        "subtract" => Ok((dispatch_dtype!(dtype, go_sub, numeric), ViewKind::Array)),
        "multiply" => Ok((dispatch_dtype!(dtype, go_mul, numeric), ViewKind::Array)),
        "divide" => Ok((dispatch_dtype!(dtype, go_div, numeric), ViewKind::Array)),

        // -- Unary arithmetic ------------------------------------------------
        "negate" => Ok((dispatch_dtype!(dtype, go_neg, signed), ViewKind::Array)),

        // -- Float unary math ------------------------------------------------
        "log" => Ok((dispatch_dtype!(dtype, go_log, float), ViewKind::Array)),
        "log2" => Ok((dispatch_dtype!(dtype, go_log2, float), ViewKind::Array)),
        "log10" => Ok((dispatch_dtype!(dtype, go_log10, float), ViewKind::Array)),
        "exp" => Ok((dispatch_dtype!(dtype, go_exp, float), ViewKind::Array)),
        "exp2" => Ok((dispatch_dtype!(dtype, go_exp2, float), ViewKind::Array)),
        "sqrt" => Ok((dispatch_dtype!(dtype, go_sqrt, float), ViewKind::Array)),
        "ceil" => Ok((dispatch_dtype!(dtype, go_ceil, float), ViewKind::Array)),
        "floor" => Ok((dispatch_dtype!(dtype, go_floor, float), ViewKind::Array)),
        "round" => Ok((dispatch_dtype!(dtype, go_round, float), ViewKind::Array)),
        "recip" => Ok((dispatch_dtype!(dtype, go_recip, float), ViewKind::Array)),

        // -- Signed unary math -----------------------------------------------
        "abs" => Ok((dispatch_dtype!(dtype, go_abs, signed), ViewKind::Array)),
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
                    add_operator_from_indices(
                        sc,
                        operators::pow::<$T>(n),
                        input_indices,
                        trigger_index,
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
                    add_operator_from_indices(
                        sc,
                        operators::scale::<$T>(c),
                        input_indices,
                        trigger_index,
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
                    add_operator_from_indices(
                        sc,
                        operators::shift::<$T>(c),
                        input_indices,
                        trigger_index,
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
                    add_operator_from_indices(
                        sc,
                        operators::clamp::<$T>(lo, hi),
                        input_indices,
                        trigger_index,
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
                    add_operator_from_indices(
                        sc,
                        operators::nan_to_num::<$T>(val),
                        input_indices,
                        trigger_index,
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
                    add_operator_from_indices(
                        sc,
                        operators::Select::<$T>::flat(indices.clone()),
                        input_indices,
                        trigger_index,
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
                    add_operator_from_indices(
                        sc,
                        operators::Concat::<$T>::new(axis),
                        input_indices,
                        trigger_index,
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
                    add_operator_from_indices(
                        sc,
                        operators::Stack::<$T>::new(axis),
                        input_indices,
                        trigger_index,
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
                    add_operator_from_indices(sc, Record::<$T>::new(), input_indices, trigger_index)
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
                    add_operator_from_indices(
                        sc,
                        operators::Last::<$T>::new(fill),
                        input_indices,
                        trigger_index,
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
                    add_operator_from_indices(
                        sc,
                        operators::Lag::<$T>::new(offset, fill),
                        input_indices,
                        trigger_index,
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
                        "rolling_sum" => add_operator_from_indices(
                            sc,
                            operators::rolling::RollingSum::<$T>::new(window),
                            input_indices,
                            trigger_index,
                        ),
                        "rolling_mean" => add_operator_from_indices(
                            sc,
                            operators::rolling::RollingMean::<$T>::new(window),
                            input_indices,
                            trigger_index,
                        ),
                        "rolling_variance" => add_operator_from_indices(
                            sc,
                            operators::rolling::RollingVariance::<$T>::new(window),
                            input_indices,
                            trigger_index,
                        ),
                        "rolling_covariance" => add_operator_from_indices(
                            sc,
                            operators::rolling::RollingCovariance::<$T>::new(window),
                            input_indices,
                            trigger_index,
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
                    add_operator_from_indices(sc, op, input_indices, trigger_index)
                }};
            }
            Ok((dispatch_dtype!(dtype, go, float), ViewKind::Series))
        }
        "forward_fill" => {
            macro_rules! go {
                ($T:ty) => {
                    add_operator_from_indices(
                        sc,
                        operators::rolling::ForwardFill::<$T>::default(),
                        input_indices,
                        trigger_index,
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
