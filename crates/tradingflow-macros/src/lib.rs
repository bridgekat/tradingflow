#![doc = include_str!("../README.md")]

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
/// Expansions name the typed layer as `::tradingflow_graph::typed`, so callers need
/// `tradingflow-graph` among their dependencies. A facade crate re-exporting this
/// macro overrides that with a leading `@[path]`: a `macro_rules!` wrapper
/// forwarding `@[$crate::path::to::typed] $($input)*` makes expansions
/// resolve through the facade itself (the override is spliced verbatim, so
/// `$crate` survives with its hygiene intact).
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
/// path to emit (defaulting to `::tradingflow_graph::typed`) and the remaining
/// tokens. Unambiguous: a flow always starts with `|`.
fn runtime_override(input: TokenStream) -> (TokenStream, TokenStream) {
    let mut iter = input.clone().into_iter();
    if let (Some(TokenTree::Punct(p)), Some(TokenTree::Group(g))) = (iter.next(), iter.next())
        && p.as_char() == '@'
        && g.delimiter() == Delimiter::Bracket
    {
        return (g.stream(), iter.collect());
    }
    (quote!(::tradingflow_graph::typed), input)
}
