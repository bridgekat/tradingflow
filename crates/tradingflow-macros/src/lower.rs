//! Lowering: from AST to a `bind` and `route` chain.
//!
//! This is Paterson's arrow-notation translation specialized to a cartesian
//! category: the *environment* of live wires is threaded as a left-nested pair
//! tree `((params, out_1), out_2)...`, each statement extends it via `bind`,
//! and plumbing closures are pure ref shuffles. The translation is entirely
//! positional: no type information is needed. Closure patterns bind whatever
//! `Values` the wires turn out to have. Wires the closure does not use
//! lower to `_`; shadowed slots also lower to `_`.
//!
//! Inline applications (`seg @ wires`) are desugared first: each one becomes a
//! fresh `__flowN` statement (post-order, so nested applications come first),
//! and the surrounding expression keeps its tree shape over the fresh wires.
//! After this pass, the body is the flat statement list the translation above
//! consumes, followed by a result wire tree that the closing `route` projects
//! out of the environment.

use std::collections::{HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;

use crate::ast::{Segment, WireExpr, WirePat};

impl WirePat {
    fn leaves(&self, f: &mut impl FnMut(&syn::Ident)) {
        match self {
            WirePat::Var(v) => f(v),
            WirePat::Tuple(ts) => ts.iter().for_each(|t| t.leaves(f)),
        }
    }
}

/// A fully desugared statement: one segment applied to tree-shaped args.
struct FlatStmt {
    pat: WirePat,
    seg: syn::Expr,
    args: WirePat,
}

/// Desugar inline applications out of a wire expression. Each `seg @ wires`
/// leaf is emitted as a fresh `__flowN` statement (its span kept on the
/// segment expression, so type errors point at the right application) and
/// replaced by the fresh wire; `__flow` is a reserved name prefix.
fn flatten(e: WireExpr, out: &mut Vec<FlatStmt>, n: &mut usize) -> WirePat {
    match e {
        WireExpr::Var(v) => WirePat::Var(v),
        WireExpr::Tuple(es) => WirePat::Tuple(es.into_iter().map(|e| flatten(e, out, n)).collect()),
        WireExpr::Apply(seg, args) => {
            let args = flatten(*args, out, n);
            let v = syn::Ident::new(&format!("__flow{n}"), seg.span());
            *n += 1;
            out.push(FlatStmt {
                pat: WirePat::Var(v.clone()),
                seg,
                args,
            });
            WirePat::Var(v)
        }
    }
}

/// Render one env slot as a closure sub-pattern; `counter` walks the global
/// leaf index, `live` flags innermost bindings, `used` the names this closure
/// references. Fully-unused subtrees collapse to a single `_`.
fn slot_pattern(
    slot: &WirePat,
    counter: &mut usize,
    live: &[bool],
    used: &HashSet<String>,
) -> TokenStream {
    match slot {
        WirePat::Var(v) => {
            let i = *counter;
            *counter += 1;
            if live[i] && used.contains(&v.to_string()) {
                quote!(#v)
            } else {
                quote!(_)
            }
        }
        WirePat::Tuple(ts) => {
            let subs: Vec<_> = ts
                .iter()
                .map(|t| slot_pattern(t, counter, live, used))
                .collect();
            if subs.iter().all(|s| s.to_string() == "_") {
                quote!(_)
            } else {
                quote!(( #(#subs),* ))
            }
        }
    }
}

/// Wire expression: a name's last occurrence moves; earlier occurrences
/// `.clone()` (`Interface::Values` is `Copy`, so the clone is free).
fn wires_expr(t: &WirePat, remaining: &mut HashMap<String, usize>) -> TokenStream {
    match t {
        WirePat::Var(v) => {
            let r = remaining.get_mut(&v.to_string()).unwrap();
            *r -= 1;
            if *r > 0 {
                quote!(#v.clone())
            } else {
                quote!(#v)
            }
        }
        WirePat::Tuple(ts) => {
            let es: Vec<_> = ts.iter().map(|t| wires_expr(t, remaining)).collect();
            quote!(( #(#es),* ))
        }
    }
}

/// The shuffle closure `|env_pattern, _| wires` over the current environment.
fn closure(env: &[WirePat], wires: &WirePat) -> syn::Result<TokenStream> {
    // Innermost-binding flags by leaf index (later binding of a name wins).
    let mut last = HashMap::new();
    let mut n = 0;
    for s in env {
        s.leaves(&mut |v| {
            last.insert(v.to_string(), n);
            n += 1;
        });
    }
    let mut live = vec![false; n];
    for &i in last.values() {
        live[i] = true;
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut undef = None;
    wires.leaves(&mut |v| {
        let k = v.to_string();
        if last.contains_key(&k) {
            *counts.entry(k).or_insert(0) += 1;
        } else if undef.is_none() {
            undef = Some(syn::Error::new(v.span(), format!("unbound wire `{k}`")));
        }
    });
    if let Some(e) = undef {
        return Err(e);
    }

    let used: HashSet<_> = counts.keys().cloned().collect();
    let mut counter = 0;
    let mut pat = slot_pattern(&env[0], &mut counter, &live, &used);
    for s in &env[1..] {
        let p = slot_pattern(s, &mut counter, &live, &used);
        pat = quote!((#pat, #p));
    }
    let expr = wires_expr(wires, &mut counts);
    Ok(quote!(|#pat, _| #expr))
}

pub fn lower(flow: Segment, rt: TokenStream) -> syn::Result<TokenStream> {
    let tys: Vec<_> = flow.params.iter().map(|p| &p.1).collect();
    let in_ty = match &tys[..] {
        [t] => quote!(#t),
        _ => quote!(( #(#tys),* )),
    };
    let mut env = vec![match flow.params.len() {
        1 => flow.params[0].0.clone(),
        _ => WirePat::Tuple(flow.params.iter().map(|p| p.0.clone()).collect()),
    }];

    // Desugar inline `@` applications into the flat statement list.
    let mut n = 0;
    let mut stmts = Vec::new();
    for s in flow.stmts {
        let args = flatten(s.args, &mut stmts, &mut n);
        stmts.push(FlatStmt {
            pat: s.pat,
            seg: s.seg,
            args,
        });
    }
    let result = flatten(flow.result, &mut stmts, &mut n);

    // The seed's context is an inference hole `_`: it unifies with any
    // context-pinned segment in the body, or ultimately with the builder's
    // context at the push site.
    let mut cur = quote!(#rt::Id::<#in_ty, _>::default());
    for s in &stmts {
        let (seg, f) = (&s.seg, closure(&env, &s.args)?);
        cur = quote!(#rt::SegmentExt::bind(#cur, #seg, #f));
        env.push(s.pat.clone());
    }

    // The result is a pure projection of the accumulated environment; the
    // required `-> OutInterface` annotation pins the routed output type.
    let out = &flow.output;
    let f = closure(&env, &result)?;
    Ok(quote!(#rt::SegmentExt::route::<#out, _>(#cur, #f)))
}
