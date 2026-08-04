use std::collections::HashMap;

use super::Error;
use super::ast::Expr;
use crate::data::{Array, Instant, Retention, Scalar};
use crate::graph::typed::{HandlesInterface, InterfaceHandles};
use crate::graph::{Operator, Time};
use crate::operators::{array, rolling, stats};
use crate::ports::{ArrayPortHandle, SignalPortHandle};

/// Graph-construction access for expression lowering, implemented by both the
/// typed-layer builder ([`typed::Builder`](crate::graph::typed::Builder)) and
/// the driver-layer builder ([`graph::Builder`](crate::graph::Builder)), so a
/// [`Context`] works against either.
pub trait Builder {
    /// Adds an operator node to the graph.
    fn add_op<T, H>(&mut self, op: T, inputs: H) -> <T::Outputs as InterfaceHandles>::HandlesOwned
    where
        H: HandlesInterface,
        T: Operator<Inputs = H::Interface, Context = Instant>,
        T::Outputs: InterfaceHandles;
}

impl Builder for crate::graph::typed::Builder<Instant> {
    fn add_op<T, H>(&mut self, op: T, inputs: H) -> <T::Outputs as InterfaceHandles>::HandlesOwned
    where
        H: HandlesInterface,
        T: Operator<Inputs = H::Interface, Context = Instant>,
        T::Outputs: InterfaceHandles,
    {
        self.op(op, inputs)
    }
}

impl<S: Time<Instant>> Builder for crate::graph::Builder<Instant, S> {
    fn add_op<T, H>(&mut self, op: T, inputs: H) -> <T::Outputs as InterfaceHandles>::HandlesOwned
    where
        H: HandlesInterface,
        T: Operator<Inputs = H::Interface, Context = Instant>,
        T::Outputs: InterfaceHandles,
    {
        self.op(op, inputs)
    }
}

/// A lowered (sub)expression: a wire in the graph, or a constant that has not
/// been materialized. Constants stay symbolic as long as possible — they fold
/// into the closures of adjacent elementwise nodes — and only become constant
/// array nodes (shaped by the context extents) where an operator needs an
/// actual wire.
#[derive(Clone, Copy)]
enum Value<const N: usize> {
    Num(ArrayPortHandle<f64, N>),
    Bool(ArrayPortHandle<bool, N>),
    ConstNum(f64),
    ConstBool(bool),
}

impl<const N: usize> Value<N> {
    fn describe(&self) -> &'static str {
        match self {
            Value::Num(_) => "a numeric wire",
            Value::Bool(_) => "a boolean wire",
            Value::ConstNum(_) => "a numeric constant",
            Value::ConstBool(_) => "a boolean constant",
        }
    }
}

/// Lowering context: the clock signal that paces the rolling operators, the
/// shared cross-section shape, the `$field` environment, and the hash-consing
/// cache.
///
/// Every wire in one context — every registered field and everything lowered
/// from them — carries the same `extents`: the elementwise operators require
/// matching shapes and everything else preserves them. This is what allows
/// constants to materialize as correctly-shaped constant array nodes where an
/// operator needs a wire rather than a closure-foldable scalar.
///
/// The cache persists across [`lower`](Context::lower) calls, so lowering a
/// whole factor zoo through one `Context` automatically shares every common
/// subexpression (including the ring buffers of shared rolling windows) as a
/// single graph node.
pub struct Context<const N: usize> {
    clock: SignalPortHandle<0>,
    extents: [usize; N],
    fields: HashMap<String, ArrayPortHandle<f64, N>>,
    groupings: HashMap<String, ArrayPortHandle<u32, N>>,
    cache: HashMap<Expr, Value<N>>,
    const_wires: HashMap<u64, ArrayPortHandle<f64, N>>,
}

impl<const N: usize> Context<N> {
    /// Creates a context. `clock` paces every rolling operator: one sample is
    /// ingested per pulse (e.g. the `daily` signal of a panel source).
    /// `extents` is the shape shared by every field wire registered on this
    /// context.
    pub fn new(clock: SignalPortHandle<0>, extents: [usize; N]) -> Self {
        Self {
            clock,
            extents,
            fields: HashMap::new(),
            groupings: HashMap::new(),
            cache: HashMap::new(),
            const_wires: HashMap::new(),
        }
    }

    /// Registers a `$name` field wire.
    pub fn add_field(&mut self, name: impl Into<String>, wire: ArrayPortHandle<f64, N>) {
        self.fields.insert(name.into(), wire);
    }

    /// Chaining form of [`add_field`](Context::add_field).
    pub fn with_field(mut self, name: impl Into<String>, wire: ArrayPortHandle<f64, N>) -> Self {
        self.add_field(name, wire);
        self
    }

    /// Registers a grouping for `IndNeutralize(x, $name)`: a wire tagging
    /// each cross-section element with a group id (e.g. industry tags encoded
    /// as small dense integers), shaped like the context extents. The tags
    /// may change over time — the demean regroups every tick. A static
    /// tagging is a constant array node
    /// ([`array::constant`](crate::operators::array::constant)).
    pub fn add_grouping(&mut self, name: impl Into<String>, tags: ArrayPortHandle<u32, N>) {
        self.groupings.insert(name.into(), tags);
    }

    /// Chaining form of [`add_grouping`](Context::add_grouping).
    pub fn with_grouping(mut self, name: impl Into<String>, tags: ArrayPortHandle<u32, N>) -> Self {
        self.add_grouping(name, tags);
        self
    }

    /// Parses and lowers a factor expression, returning its output wire.
    pub fn lower_str(
        &mut self,
        b: &mut impl Builder,
        src: &str,
    ) -> Result<ArrayPortHandle<f64, N>, Error> {
        let expr = super::parse(src)?;
        self.lower(b, &expr)
    }

    /// Lowers a parsed factor expression, returning its output wire.
    ///
    /// The expression is canonicalized first, then translated bottom-up into
    /// graph nodes with hash-consing: a subexpression already lowered by this
    /// context (in this call or any earlier one) reuses its existing wire.
    pub fn lower(
        &mut self,
        b: &mut impl Builder,
        expr: &Expr,
    ) -> Result<ArrayPortHandle<f64, N>, Error> {
        let expr = expr.clone().canonicalize();
        match self.eval(b, &expr)? {
            Value::Num(wire) => Ok(wire),
            Value::Bool(_) => Err(Error::Type {
                name: "<result>".into(),
                message: "expression yields a boolean; wrap it in If(cond, a, b) \
                          or compare it into a number"
                    .into(),
            }),
            Value::ConstNum(_) | Value::ConstBool(_) => Err(Error::Constant),
        }
    }

    /// Recursively lowers a canonical expression to a [`Value`].
    fn eval<B: Builder>(&mut self, b: &mut B, expr: &Expr) -> Result<Value<N>, Error> {
        match expr {
            Expr::Field(name) => match self.fields.get(name) {
                Some(&wire) => Ok(Value::Num(wire)),
                None => Err(Error::UnknownField(name.clone())),
            },
            Expr::Const(c) => Ok(Value::ConstNum(*c)),
            Expr::Call(name, args) => {
                if let Some(&cached) = self.cache.get(expr) {
                    return Ok(cached);
                }
                let value = self.eval_call(b, name, args)?;
                self.cache.insert(expr.clone(), value);
                Ok(value)
            }
        }
    }

    fn eval_call<B: Builder>(
        &mut self,
        b: &mut B,
        name: &str,
        args: &[Expr],
    ) -> Result<Value<N>, Error> {
        // Elementwise arithmetic (binary).
        if let Some(f) = binary_fn(name) {
            let [x, y] = check_arity(name, args)?;
            let x = self.eval(b, x)?;
            let y = self.eval(b, y)?;
            return match (x, y) {
                (Value::ConstNum(p), Value::ConstNum(q)) => Ok(Value::ConstNum(f(p, q))),
                (Value::Num(x), Value::Num(y)) => Ok(Value::Num(map2(b, x, y, f))),
                (Value::Num(x), Value::ConstNum(q)) => Ok(Value::Num(map1(b, x, move |p| f(p, q)))),
                (Value::ConstNum(p), Value::Num(y)) => Ok(Value::Num(map1(b, y, move |q| f(p, q)))),
                (x, y) => Err(type_error(name, "numeric operands", &x, Some(&y))),
            };
        }

        // Elementwise arithmetic (unary).
        if let Some(f) = unary_fn(name) {
            let [x] = check_arity(name, args)?;
            return match self.eval(b, x)? {
                Value::ConstNum(p) => Ok(Value::ConstNum(f(p))),
                Value::Num(x) => Ok(Value::Num(map1(b, x, f))),
                x => Err(type_error(name, "a numeric operand", &x, None)),
            };
        }

        // Comparisons (numeric → boolean, IEEE semantics for NaN).
        if let Some(f) = cmp_fn(name) {
            let [x, y] = check_arity(name, args)?;
            let x = self.eval(b, x)?;
            let y = self.eval(b, y)?;
            return match (x, y) {
                (Value::ConstNum(p), Value::ConstNum(q)) => Ok(Value::ConstBool(f(p, q))),
                (Value::Num(x), Value::Num(y)) => Ok(Value::Bool(map2(b, x, y, f))),
                (Value::Num(x), Value::ConstNum(q)) => {
                    Ok(Value::Bool(map1(b, x, move |p| f(p, q))))
                }
                (Value::ConstNum(p), Value::Num(y)) => {
                    Ok(Value::Bool(map1(b, y, move |q| f(p, q))))
                }
                (x, y) => Err(type_error(name, "numeric operands", &x, Some(&y))),
            };
        }

        // Boolean connectives.
        if let Some(f) = bool_fn(name) {
            let [x, y] = check_arity(name, args)?;
            let x = self.eval(b, x)?;
            let y = self.eval(b, y)?;
            return match (x, y) {
                (Value::ConstBool(p), Value::ConstBool(q)) => Ok(Value::ConstBool(f(p, q))),
                (Value::Bool(x), Value::Bool(y)) => Ok(Value::Bool(map2(b, x, y, f))),
                (Value::Bool(x), Value::ConstBool(q)) => {
                    Ok(Value::Bool(map1(b, x, move |p| f(p, q))))
                }
                (Value::ConstBool(p), Value::Bool(y)) => {
                    Ok(Value::Bool(map1(b, y, move |q| f(p, q))))
                }
                (x, y) => Err(type_error(name, "boolean operands", &x, Some(&y))),
            };
        }

        match name {
            "Not" => {
                let [x] = check_arity(name, args)?;
                match self.eval(b, x)? {
                    Value::ConstBool(p) => Ok(Value::ConstBool(!p)),
                    Value::Bool(x) => Ok(Value::Bool(map1(b, x, |p: bool| !p))),
                    x => Err(type_error(name, "a boolean operand", &x, None)),
                }
            }

            "If" => {
                let [c, x, y] = check_arity(name, args)?;
                let cond = self.eval(b, c)?;
                // A constant condition selects a branch without lowering the
                // other one.
                if let Value::ConstBool(p) = cond {
                    let taken = self.eval(b, if p { x } else { y })?;
                    return match taken {
                        Value::Num(_) | Value::ConstNum(_) => Ok(taken),
                        v => Err(type_error(name, "numeric branches", &v, None)),
                    };
                }
                let Value::Bool(cond) = cond else {
                    return Err(type_error(name, "a boolean condition", &cond, None));
                };
                let x = self.eval(b, x)?;
                let y = self.eval(b, y)?;
                match (x, y) {
                    (Value::Num(x), Value::Num(y)) => Ok(Value::Num(map3(
                        b,
                        cond,
                        x,
                        y,
                        |c, p, q| if c { p } else { q },
                    ))),
                    (Value::Num(x), Value::ConstNum(q)) => Ok(Value::Num(map2(
                        b,
                        cond,
                        x,
                        move |c, p| if c { p } else { q },
                    ))),
                    (Value::ConstNum(p), Value::Num(y)) => Ok(Value::Num(map2(
                        b,
                        cond,
                        y,
                        move |c, q| if c { p } else { q },
                    ))),
                    (Value::ConstNum(p), Value::ConstNum(q)) => {
                        Ok(Value::Num(map1(b, cond, move |c| if c { p } else { q })))
                    }
                    (x, y) => Err(type_error(name, "numeric branches", &x, Some(&y))),
                }
            }

            // Time-series operators, paced by the context clock. `min_count`
            // policy: 1 for Mean/Sum/EMA, 2 for Std/Var (see module docs).
            "Ref" => {
                let [x, w] = check_arity(name, args)?;
                let w = window(name, w, 0)?;
                if w == 0 {
                    return self.eval(b, x); // Qlib: Ref(x, 0) is x itself.
                }
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(
                    rolling::lag(Retention::count(w)),
                    (self.clock, x),
                )))
            }
            "Delta" => {
                let [x, w] = check_arity(name, args)?;
                let w = window(name, w, 1)?;
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(
                    rolling::diff(Retention::count(w)),
                    (self.clock, x),
                )))
            }
            "Mean" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(rolling::mean(w, 1), (self.clock, x))))
            }
            "Sum" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(rolling::sum(w, 1), (self.clock, x))))
            }
            "Std" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(
                    b.add_op(rolling::std_dev(w, 2), (self.clock, x)),
                ))
            }
            "Var" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(rolling::var(w, 2), (self.clock, x))))
            }
            "EMA" => {
                let [x, span] = check_arity(name, args)?;
                let span = window(name, span, 1)?;
                let alpha = 2.0 / (span as f64 + 1.0);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(
                    rolling::mean_exp(alpha, Retention::unbounded(), 1),
                    (self.clock, x),
                )))
            }
            "WMA" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(
                    b.add_op(rolling::mean_linear(w, 1), (self.clock, x)),
                ))
            }
            "Count" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(rolling::count(w), (self.clock, x))))
            }
            "Max" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(rolling::max(w, 1), (self.clock, x))))
            }
            "Min" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(rolling::min(w, 1), (self.clock, x))))
            }
            "IdxMax" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(
                    b.add_op(rolling::idx_max(w, 1), (self.clock, x)),
                ))
            }
            "IdxMin" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(
                    b.add_op(rolling::idx_min(w, 1), (self.clock, x)),
                ))
            }
            "Med" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(rolling::median(w, 1), (self.clock, x))))
            }
            "Quantile" => {
                let [x, w, q] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let q = fraction(name, q)?;
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(
                    b.add_op(rolling::quantile(q, w, 1), (self.clock, x)),
                ))
            }
            "Rank" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(rolling::rank(w, 1), (self.clock, x))))
            }
            "Mad" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(rolling::mad(w, 1), (self.clock, x))))
            }
            "Skew" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(rolling::skew(w, 3), (self.clock, x))))
            }
            "Kurt" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(rolling::kurt(w, 4), (self.clock, x))))
            }
            "Slope" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(rolling::slope(w, 2), (self.clock, x))))
            }
            "Rsquare" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(
                    b.add_op(rolling::rsquare(w, 2), (self.clock, x)),
                ))
            }
            "Resi" => {
                let [x, w] = check_arity(name, args)?;
                let w = retention(window(name, w, 0)?);
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(rolling::resi(w, 2), (self.clock, x))))
            }

            // Pair statistics, lowered as rewrites over Mean/Std so they share
            // subexpressions with everything else (see module docs for the
            // numerical caveat).
            "Cov" => {
                let [x, y, w] = check_arity(name, args)?;
                window(name, w, 0)?; // Validate here for a clean error.
                let mean = |e: Expr| Expr::call("Mean", vec![e, w.clone()]);
                // The population form `E[xy] − E[x]E[y]`, Bessel-corrected to
                // the sample estimator by `n / (n − 1)`, where `n` counts the
                // pairwise-complete ticks (`x·y` is finite iff both are), and
                // gated to `NaN` below two complete ticks like the sample
                // variance.
                let pop = Expr::call(
                    "Sub",
                    vec![
                        mean(Expr::call("Mul", vec![x.clone(), y.clone()])),
                        Expr::call("Mul", vec![mean(x.clone()), mean(y.clone())]),
                    ],
                );
                let n = Expr::call(
                    "Count",
                    vec![Expr::call("Mul", vec![x.clone(), y.clone()]), w.clone()],
                );
                let bessel = Expr::call(
                    "Div",
                    vec![
                        n.clone(),
                        Expr::call("Sub", vec![n.clone(), Expr::Const(1.0)]),
                    ],
                );
                let rewrite = Expr::call(
                    "If",
                    vec![
                        Expr::call("Gt", vec![n, Expr::Const(1.0)]),
                        Expr::call("Mul", vec![pop, bessel]),
                        Expr::Const(f64::NAN),
                    ],
                )
                .canonicalize();
                self.eval(b, &rewrite)
            }
            "Corr" => {
                let [x, y, w] = check_arity(name, args)?;
                window(name, w, 0)?;
                let std = |e: Expr| Expr::call("Std", vec![e, w.clone()]);
                let rewrite = Expr::call(
                    "Div",
                    vec![
                        Expr::call("Cov", vec![x.clone(), y.clone(), w.clone()]),
                        Expr::call("Mul", vec![std(x.clone()), std(y.clone())]),
                    ],
                )
                .canonicalize();
                self.eval(b, &rewrite)
            }

            // Cross-sectional operators (beyond Qlib's vocabulary, whose
            // expressions are per-instrument only).
            "CSRank" => {
                let [x] = check_arity(name, args)?;
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(stats::percentile(), x)))
            }
            "CSZscore" => {
                let [x] = check_arity(name, args)?;
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(stats::standardize(), x)))
            }
            "Scale" => {
                // `Scale(x)` targets 1; `Scale(x, a)` targets `a`.
                let (x, target) = match args {
                    [x] => (x, 1.0),
                    [x, a] => (x, scale_target(name, a)?),
                    _ => {
                        return Err(Error::Arity {
                            name: name.to_string(),
                            expected: 2,
                            got: args.len(),
                        });
                    }
                };
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(stats::scale(target), x)))
            }
            "IndNeutralize" => {
                let [x, g] = check_arity(name, args)?;
                let Expr::Field(group) = g else {
                    return Err(Error::Type {
                        name: name.to_string(),
                        message: "the grouping must be a bare `$name` registered \
                                  via `Context::add_grouping`"
                            .into(),
                    });
                };
                let Some(&tags) = self.groupings.get(group) else {
                    return Err(Error::UnknownGrouping(group.clone()));
                };
                let x = self.num_wire(b, name, x)?;
                Ok(Value::Num(b.add_op(stats::group_demean(), (x, tags))))
            }

            _ => Err(Error::UnknownFunction(name.to_string())),
        }
    }

    /// Evaluates an argument that must come out as a numeric wire. A numeric
    /// constant materializes as a constant array node of the context extents
    /// (deduplicated by bit pattern), so e.g. `Sum(1, 0)` counts ticks.
    fn num_wire<B: Builder>(
        &mut self,
        b: &mut B,
        name: &str,
        expr: &Expr,
    ) -> Result<ArrayPortHandle<f64, N>, Error> {
        match self.eval(b, expr)? {
            Value::Num(wire) => Ok(wire),
            Value::ConstNum(c) => Ok(*self
                .const_wires
                .entry(c.to_bits())
                .or_insert_with(|| b.add_op(array::constant(Array::full(self.extents, c)), ()))),
            v => Err(type_error(name, "a numeric operand", &v, None)),
        }
    }
}

/// Checks the argument count and destructures the slice.
fn check_arity<'a, const K: usize>(name: &str, args: &'a [Expr]) -> Result<&'a [Expr; K], Error> {
    args.try_into().map_err(|_| Error::Arity {
        name: name.to_string(),
        expected: K,
        got: args.len(),
    })
}

/// Extracts a window argument: a non-negative integer constant no smaller
/// than `min`.
fn window(name: &str, expr: &Expr, min: usize) -> Result<usize, Error> {
    let fail = |message: String| Error::Window {
        name: name.to_string(),
        message,
    };
    let &Expr::Const(c) = expr else {
        return Err(fail("window must be an integer literal".into()));
    };
    if c < 0.0 {
        return Err(fail(format!(
            "window is {c}; negative windows (future references) are not supported"
        )));
    }
    if c.fract() != 0.0 || !c.is_finite() || c > usize::MAX as f64 {
        return Err(fail(format!("window is {c}; expected an integer")));
    }
    let w = c as usize;
    if w < min {
        return Err(fail(format!("window is {w}; must be at least {min}")));
    }
    Ok(w)
}

/// Extracts a `Scale` target: a positive finite constant literal.
fn scale_target(name: &str, expr: &Expr) -> Result<f64, Error> {
    let fail = |message: String| Error::Window {
        name: name.to_string(),
        message,
    };
    let &Expr::Const(c) = expr else {
        return Err(fail("scale target must be a numeric literal".into()));
    };
    if !(c.is_finite() && c > 0.0) {
        return Err(fail(format!("scale target is {c}; must be positive")));
    }
    Ok(c)
}

/// Extracts a fraction argument: a constant in `[0, 1]`.
fn fraction(name: &str, expr: &Expr) -> Result<f64, Error> {
    let fail = |message: String| Error::Window {
        name: name.to_string(),
        message,
    };
    let &Expr::Const(c) = expr else {
        return Err(fail("quantile must be a numeric literal".into()));
    };
    if !(0.0..=1.0).contains(&c) {
        return Err(fail(format!("quantile is {c}; must be in [0, 1]")));
    }
    Ok(c)
}

/// A `usize` window as a retention: `0` means expanding (since inception).
fn retention(w: usize) -> Retention {
    if w == 0 {
        Retention::unbounded()
    } else {
        Retention::count(w)
    }
}

fn type_error<const N: usize>(
    name: &str,
    expected: &str,
    got: &Value<N>,
    got2: Option<&Value<N>>,
) -> Error {
    let message = match got2 {
        Some(second) => format!(
            "expected {expected}, got {} and {}",
            got.describe(),
            second.describe()
        ),
        None => format!("expected {expected}, got {}", got.describe()),
    };
    Error::Type {
        name: name.to_string(),
        message,
    }
}

// ---------------------------------------------------------------------------
// Elementwise node constructors
// ---------------------------------------------------------------------------

fn map1<A: Scalar, T: Scalar, const N: usize>(
    b: &mut impl Builder,
    x: ArrayPortHandle<A, N>,
    f: impl Fn(A) -> T + Clone + Send + 'static,
) -> ArrayPortHandle<T, N> {
    b.add_op(array::map(move |p: &A| f(p.clone())), x)
}

fn map2<A: Scalar, B: Scalar, T: Scalar, const N: usize>(
    b: &mut impl Builder,
    x: ArrayPortHandle<A, N>,
    y: ArrayPortHandle<B, N>,
    f: impl Fn(A, B) -> T + Clone + Send + 'static,
) -> ArrayPortHandle<T, N> {
    b.add_op(
        array::binary_map(move |p: &A, q: &B| f(p.clone(), q.clone())),
        (x, y),
    )
}

fn map3<A: Scalar, B: Scalar, C: Scalar, T: Scalar, const N: usize>(
    b: &mut impl Builder,
    x: ArrayPortHandle<A, N>,
    y: ArrayPortHandle<B, N>,
    z: ArrayPortHandle<C, N>,
    f: impl Fn(A, B, C) -> T + Clone + Send + 'static,
) -> ArrayPortHandle<T, N> {
    b.add_op(
        array::ternary_map(move |p: &A, q: &B, r: &C| f(p.clone(), q.clone(), r.clone())),
        (x, y, z),
    )
}

// ---------------------------------------------------------------------------
// Scalar function tables
// ---------------------------------------------------------------------------

fn unary_fn(name: &str) -> Option<fn(f64) -> f64> {
    Some(match name {
        "Neg" => |p| -p,
        "Abs" => f64::abs,
        "Sign" => f64::signum,
        "Log" => f64::ln,
        "Log2" => f64::log2,
        "Log10" => f64::log10,
        "Sqrt" => f64::sqrt,
        "Exp" => f64::exp,
        _ => return None,
    })
}

fn binary_fn(name: &str) -> Option<fn(f64, f64) -> f64> {
    Some(match name {
        "Add" => |p, q| p + q,
        "Sub" => |p, q| p - q,
        "Mul" => |p, q| p * q,
        "Div" => |p, q| p / q,
        "Power" => f64::powf,
        "Greater" => f64::max,
        "Less" => f64::min,
        _ => return None,
    })
}

fn cmp_fn(name: &str) -> Option<fn(f64, f64) -> bool> {
    Some(match name {
        "Gt" => |p, q| p > q,
        "Ge" => |p, q| p >= q,
        "Lt" => |p, q| p < q,
        "Le" => |p, q| p <= q,
        "Eq" => |p, q| p == q,
        "Ne" => |p, q| p != q,
        _ => return None,
    })
}

fn bool_fn(name: &str) -> Option<fn(bool, bool) -> bool> {
    Some(match name {
        "And" => |p, q| p && q,
        "Or" => |p, q| p || q,
        _ => return None,
    })
}
