//! Factor expressions: a Qlib-compatible mini-language that lowers into graph
//! nodes.
//!
//! A factor is *data*, not code: a string such as
//! `"Mean($close, 12) - Mean($close, 26)"` parses into an [`Expr`] and lowers
//! into a subgraph of the existing `rolling`/`array`/`stats` operators. This
//! keeps the whole authoring loop free of compilation: expressions can be
//! generated at runtime, hashed and deduplicated by their canonical form, and
//! evaluated incrementally by a prebuilt binary.
//!
//! # Syntax
//!
//! - **Fields**: `$close`, `$volume` — wires registered on the [`Context`].
//! - **Literals**: `2`, `0.5`, `1e-3`.
//! - **Infix**: `+ - * /`, comparisons `> >= < <= == !=`, boolean `&& ||` and
//!   prefix `- !`, with conventional precedence. Infix forms are sugar for the
//!   canonical calls (`a + b` ≡ `Add(a, b)`).
//! - **Calls**: `Name(arg, ...)` — see the vocabulary below.
//!
//! # Vocabulary
//!
//! Time-series operators ingest one sample per pulse of the context clock;
//! `w` is an integer literal window in ticks, and `w = 0` means an expanding
//! (since-inception) window. Negative windows — future references — do not
//! exist, by construction.
//!
//! | Call | Meaning |
//! | --- | --- |
//! | `Ref(x, w)` | value `w` ticks ago (`Ref(x, 0)` is `x`); `NaN` until seen |
//! | `Delta(x, w)` | `x - Ref(x, w)`, `w ≥ 1` |
//! | `Count(x, w)` | number of finite samples in the window |
//! | `Mean(x, w)`, `Sum(x, w)` | rolling mean / sum of finite samples |
//! | `WMA(x, w)` | linearly-weighted rolling mean, newest heaviest |
//! | `EMA(x, span)` | exponentially-weighted rolling mean, `α = 2 / (span + 1)` |
//! | `Std(x, w)`, `Var(x, w)` | rolling sample std / variance (`n − 1`) |
//! | `Skew(x, w)`, `Kurt(x, w)` | adjusted sample skewness / excess kurtosis |
//! | `Cov(x, y, w)`, `Corr(x, y, w)` | pair moments, lowered as `Mean`/`Std` rewrites |
//! | `Max(x, w)`, `Min(x, w)` | rolling extremes |
//! | `IdxMax(x, w)`, `IdxMin(x, w)` | 1-based window position of the extreme (oldest = 1) |
//! | `Med(x, w)`, `Quantile(x, w, q)` | rolling median / interpolated quantile, `q ∈ [0, 1]` |
//! | `Rank(x, w)` | percentile rank in `[0, 1]` of the newest value in the window |
//! | `Mad(x, w)` | rolling mean absolute deviation around the mean |
//! | `Slope(x, w)`, `Rsquare(x, w)`, `Resi(x, w)` | OLS of `x` against window position: slope / R² / newest residual |
//! | `Abs, Sign, Log, Log2, Log10, Sqrt, Exp, Neg` | elementwise maps |
//! | `Power(a, b)`, `Greater(a, b)`, `Less(a, b)` | elementwise `powf` / max / min |
//! | `If(c, a, b)`, `And, Or, Not` | elementwise choice and boolean logic |
//! | `CSRank(x)`, `CSZscore(x)` | **cross-sectional** percentile rank / z-score |
//!
//! The cross-sectional operators go beyond Qlib's expression engine (whose
//! expressions are strictly per-instrument): here every wire is a
//! cross-section, so `CSRank` is just [`stats::percentile`](crate::operators::stats::percentile)
//! on the current tick.
//!
//! # Semantics notes
//!
//! - Comparisons follow IEEE: any comparison with `NaN` is `false`, so an
//!   `If(x > y, ...)` takes its else-branch while either side is missing.
//! - Every wire in one [`Context`] shares a single shape — the `extents`
//!   given at construction. Elementwise operators require it; everything else
//!   preserves it.
//! - Constants fold: `2 + 3` never becomes a node, and a constant operand
//!   folds into the adjacent elementwise closure (`$x * 5` is one node).
//!   Where an operator needs an actual wire — rolling and cross-sectional
//!   inputs — a constant materializes as a constant array node of the context
//!   extents, deduplicated by bit pattern (so `Sum(1, 0)` counts ticks). An
//!   expression that folds to a constant outright is still rejected
//!   ([`Error::Constant`]) — a factor must reference at least one field.
//! - `Cov`/`Corr` lower to `Mean(x*y, w) - Mean(x, w) * Mean(y, w)` (and its
//!   `Std`-normalized quotient), the one-pass textbook forms,
//!   Bessel-corrected to the sample estimator by `n / (n − 1)` over the
//!   pairwise-complete tick count and `NaN` below two such ticks. They share
//!   subexpressions with the rest of the zoo, at the cost of some precision
//!   when values are large with tiny variance.
//! - Lowering hash-conses: within one [`Context`], every distinct canonical
//!   subexpression becomes exactly one graph node, across all expressions
//!   lowered through it. `$a + $b` and `$b + $a` are the same node.
//! - The sum-family operators (`Mean`, `Sum`, `Std`, `Var`, `EMA`, `WMA`,
//!   `Count`, `Skew`, `Kurt`) update incrementally in `O(1)` per tick; the
//!   window-scanning ones (`Max`/`Min`/`Idx*`, `Med`/`Quantile`/`Rank`/`Mad`,
//!   `Slope`/`Rsquare`/`Resi`) rescan the window each tick — `O(w)` per lane,
//!   and `O(history)` when used with the expanding window `w = 0`.
//!
//! # Example
//!
//! ```
//! use tradingflow::data::{Array, Duration, Instant, Series};
//! use tradingflow::graph::{Builder, Pool};
//! use tradingflow::operators::feature::expr;
//! use tradingflow::sources::sync;
//! use tradingflow::time::UnixTime;
//!
//! #[tokio::main]
//! async fn main() {
//!     // A one-instrument close series (any rank works; wires are cross-sections).
//!     let timestamps = (0..100)
//!         .map(|i| Instant::from_offset(Duration::from_days(i)))
//!         .collect::<Vec<_>>();
//!     let values = (0..100).map(|i| Array::from([100.0 + i as f64])).collect::<Vec<_>>();
//!
//!     let mut b = Builder::new(UnixTime);
//!     let (daily, close) = b.source(sync::array_series(Series::from((timestamps, values))));
//!
//!     let mut ctx = expr::Context::new(daily, [1]).with_field("close", close);
//!     let macd = ctx
//!         .lower_str(&mut b, "EMA($close, 12) - EMA($close, 26)")
//!         .unwrap();
//!     let signal = ctx.lower_str(&mut b, "EMA(EMA($close, 12) - EMA($close, 26), 9)").unwrap();
//!     // The shared `EMA - EMA` subgraph is lowered once (hash-consing).
//! #   let _ = (macd, signal);
//! }
//! ```

mod ast;
mod error;
mod lower;
mod parse;

pub use ast::Expr;
pub use error::Error;
pub use lower::{Builder, Context};
pub use parse::parse;
