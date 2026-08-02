//! The surface AST for the `fuse!` macro.

use syn::parse::{Parse, ParseStream};
use syn::{Expr, Ident, Token, Type, token};

/// A wire pattern: a single named wire or a tuple of subtrees. The same shape
/// serves as binder pattern (`let (x, y) = ...`) and wire expression
/// (`(a, (b, c))`), mirroring how a `RefPorts` tree is both produced and
/// consumed.
#[derive(Clone)]
pub enum WirePat {
    Var(Ident),
    Tuple(Vec<WirePat>),
}

/// A wire expression: a wire tree whose leaves may also be inline operator
/// applications `op @ wires`. Applications nest (`Add @ (a, Add @ (a, b))`);
/// lowering desugars each one into a fresh intermediate wire.
#[derive(Clone)]
pub enum WireExpr {
    Var(Ident),
    Tuple(Vec<WireExpr>),
    Apply(Expr, Box<WireExpr>),
}

#[derive(Clone)]
pub struct Stmt {
    pub pat: WirePat,
    pub op: Expr,
    pub args: WireExpr,
}

#[derive(Clone)]
pub struct Operator {
    pub params: Vec<(WirePat, Type)>,
    pub output: Type,
    pub stmts: Vec<Stmt>,
    pub result: WireExpr,
}

impl Parse for WirePat {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(token::Paren) {
            let inner;
            syn::parenthesized!(inner in input);
            let elems = inner.parse_terminated(WirePat::parse, Token![,])?;
            // `(x)` is grouping, not a 1-tuple (mirroring Rust expressions).
            if elems.len() == 1 && !elems.trailing_punct() {
                Ok(elems.into_iter().next().unwrap())
            } else {
                Ok(WirePat::Tuple(elems.into_iter().collect()))
            }
        } else {
            Ok(WirePat::Var(input.parse()?))
        }
    }
}

/// True iff an operator application `Expr @ wires` is ahead. `@` is not a Rust
/// expression operator, so a speculative `Expr` parse consumes exactly the
/// operator expression and stops right before the `@`; if what follows is
/// anything else (`,`, `)`, `;`), the tokens were not a operator.
fn peek_apply(input: ParseStream) -> bool {
    let fork = input.fork();
    fork.parse::<Expr>().is_ok() && fork.peek(Token![@])
}

impl Parse for WireExpr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if peek_apply(input) {
            let op = input.parse()?;
            input.parse::<Token![@]>()?;
            let args = input.parse()?;
            return Ok(WireExpr::Apply(op, Box::new(args)));
        }
        if input.peek(token::Paren) {
            let inner;
            syn::parenthesized!(inner in input);
            let elems = inner.parse_terminated(WireExpr::parse, Token![,])?;
            // `(x)` is grouping, not a 1-tuple (mirroring Rust expressions).
            if elems.len() == 1 && !elems.trailing_punct() {
                Ok(elems.into_iter().next().unwrap())
            } else {
                Ok(WireExpr::Tuple(elems.into_iter().collect()))
            }
        } else {
            Ok(WireExpr::Var(input.parse()?))
        }
    }
}

impl Parse for Operator {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        input.parse::<Token![|]>()?;
        let mut params = Vec::new();
        while !input.peek(Token![|]) {
            let pat = input.parse()?;
            input.parse::<Token![:]>()?;
            let ty = input.parse()?;
            params.push((pat, ty));
            if !input.peek(Token![|]) {
                input.parse::<Token![,]>()?;
            }
        }
        input.parse::<Token![|]>()?;
        input.parse::<Token![->]>()?;
        let output = input.parse()?;
        let body;
        syn::braced!(body in input);
        let mut stmts = Vec::new();
        while body.peek(Token![let]) {
            body.parse::<Token![let]>()?;
            let pat = body.parse()?;
            body.parse::<Token![=]>()?;
            let rhs: WireExpr = body.parse()?;
            let WireExpr::Apply(op, args) = rhs else {
                return Err(body.error("expected an operator application `op @ wires`"));
            };
            body.parse::<Token![;]>()?;
            stmts.push(Stmt {
                pat,
                op,
                args: *args,
            });
        }
        let result = body.parse()?;
        if !body.is_empty() {
            return Err(body.error("expected end of operator body after the result wires"));
        }
        Ok(Operator {
            params,
            output,
            stmts,
            result,
        })
    }
}
