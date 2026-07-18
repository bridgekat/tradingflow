//! Procedural macros for composing subgraphs.
//!
//! This crate is the macro half of the `tradingflow` computation-graph
//! engine, kept separate because a `proc-macro` crate can export nothing else.
//! It has no dependency on the engine (the engine depends on *it*), so the
//! names below — `Segment`, `Port`, the combinators — are documented there
//! rather than linked from here.
//!
//! # The [`segment!`](macro@segment) macro
//!
//! A [`segment!`](macro@segment) expression creates a new `Segment` by wiring
//! together smaller `Segment`s. It reads like a closure:
//!
//! ```rust,ignore
//! let seg = segment!(|a: Port<i64>, b: Port<i64>| -> (Port<i64>, Port<i64>) {
//!     let c = add() @ (a, b);  // c = a + b
//!     let d = inc(1) @ c;      // d = c + 1
//!     (d, c)                   // two outputs
//! });
//! ```
//!
//! The header names each input wire with its interface type, and then to the
//! right of `->` the output wire interface type. The body is a sequence of `let`
//! bindings, ending in a result wire expression.
//!
//! Segments apply to wires prefix-style with `@` — the expression left of `@` is
//! the segment, the wires to its right are its inputs.
//!
//! Applications nest inside any wire expression and chain right-associatively,
//! each nesting desugaring to a fresh intermediate wire:
//!
//! ```rust,ignore
//! segment!(|x: Port<i64>| -> Port<i64> {
//!     add() @ (x, inc(1) @ x)  // let t = inc(1) @ x; add() @ (x, t)
//! });
//! ```
//!
//! # Module path override
//!
//! Expansions prefix the combinators by `::tradingflow::graph::cb`, so callers
//! need `tradingflow` among their dependencies. A leading `@[path]`
//! overrides that path:
//!
//! ```rust,ignore
//! segment!(@[::some::path::to::module] |a: Port<i64>| -> Port<i64> { /* ... */ });
//! ```
//!
//! will resolve combinators in `::some::path::to::module` instead of
//! `::tradingflow::graph::cb`.

use proc_macro2::{Delimiter, TokenStream, TokenTree};
use quote::quote;

mod ast;
mod lower;

/// Arrow notation for fused `Segment` composition: named wires instead of
/// point-free `Comp`/`Fork` plumbing. Segments apply to wires prefix-style
/// (`let c = Add @ (a, b);`); applications nest inside any wire expression
/// (`Add @ (a, Add @ (a, b))`, chaining right-associatively), each nesting
/// desugaring to a fresh intermediate wire. The whole expression becomes a
/// single scheduled node; all plumbing is zero-copy ref routing.
///
/// Expansions name the combinator namespace as `::tradingflow::graph::cb` so
/// callers need `tradingflow` among their dependencies. A facade crate
/// re-exporting this macro overrides that with a leading `@[path]`:
/// a `macro_rules!` wrapper forwarding `@[$crate::path::to::cb] $($input)*`
/// makes expansions resolve through the facade itself (the override is spliced
/// verbatim, so `$crate` survives with its hygiene intact).
#[proc_macro]
pub fn segment(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let (rt, input) = runtime_override(input.into());
    let segment = match syn::parse2::<ast::Flow>(input) {
        Ok(flow) => flow,
        Err(e) => return e.into_compile_error().into(),
    };
    lower::lower(segment, rt)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Split an optional leading `@[path]` off `input`, returning the runtime
/// path to emit (defaulting to `::tradingflow::graph::cb`) and the remaining
/// tokens. Unambiguous: a flow always starts with `|`.
fn runtime_override(input: TokenStream) -> (TokenStream, TokenStream) {
    let mut iter = input.clone().into_iter();
    if let (Some(TokenTree::Punct(p)), Some(TokenTree::Group(g))) = (iter.next(), iter.next())
        && p.as_char() == '@'
        && g.delimiter() == Delimiter::Bracket
    {
        return (g.stream(), iter.collect());
    }
    (quote!(::tradingflow::graph::cb), input)
}
