//! Procedural macros for the `tradingflow` crate.

use proc_macro2::{Delimiter, TokenStream, TokenTree};
use quote::quote;

mod ast;
mod lower;

/// Macro for operator composition.
///
/// Expansions name the combinator namespace as `::tradingflow::graph::cb` by
/// default, which can be override using the syntax
/// `fuse!(@[::some::path::to::module] |a: Port<i64>| -> Port<i64> { /* ... */ })`.
#[proc_macro]
pub fn fuse(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let (rt, input) = runtime_override(input.into());
    let fuse = match syn::parse2::<ast::Operator>(input) {
        Ok(flow) => flow,
        Err(e) => return e.into_compile_error().into(),
    };
    lower::lower(fuse, rt)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

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
