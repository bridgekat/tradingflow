use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

/// A parsed factor expression.
///
/// Infix operators are desugared by the parser into canonical [`Expr::Call`]
/// nodes (`a + b` becomes `Add(a, b)`), so a call is the only compound node
/// and the lowering dispatches on function names alone.
///
/// Equality, ordering and hashing are structural, with constants compared by
/// their bit patterns (so `NaN == NaN` holds and expressions are usable as
/// cache keys). [`Expr::canonicalize`] additionally sorts the operands of
/// commutative operators, so semantically equal spellings such as
/// `$a + $b` and `$b + $a` become identical trees — this is what makes
/// hash-consing during lowering (and deduplication across a factor zoo)
/// effective.
#[derive(Debug, Clone)]
pub enum Expr {
    /// A data field reference: `$close`.
    Field(String),
    /// A numeric literal.
    Const(f64),
    /// A function application: `Mean($close, 12)`.
    Call(String, Vec<Expr>),
}

impl Expr {
    /// Convenience constructor for a call node.
    pub fn call(name: impl Into<String>, args: Vec<Expr>) -> Self {
        Expr::Call(name.into(), args)
    }

    /// Recursively sorts the operands of commutative operators into a
    /// canonical order. Floating-point addition and multiplication are
    /// exactly commutative (unlike associative), so a two-operand swap never
    /// changes results bit-for-bit.
    pub fn canonicalize(self) -> Self {
        match self {
            Expr::Call(name, args) => {
                let mut args: Vec<Expr> = args.into_iter().map(Expr::canonicalize).collect();
                if args.len() == 2 && is_commutative(&name) && args[0] > args[1] {
                    args.swap(0, 1);
                }
                Expr::Call(name, args)
            }
            other => other,
        }
    }

    fn rank(&self) -> u8 {
        match self {
            Expr::Const(_) => 0,
            Expr::Field(_) => 1,
            Expr::Call(_, _) => 2,
        }
    }
}

/// Whether swapping the two operands of `name` is semantics-preserving.
fn is_commutative(name: &str) -> bool {
    matches!(
        name,
        "Add" | "Mul" | "Eq" | "Ne" | "And" | "Or" | "Greater" | "Less"
    )
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Expr::Field(a), Expr::Field(b)) => a == b,
            (Expr::Const(a), Expr::Const(b)) => a.to_bits() == b.to_bits(),
            (Expr::Call(f, a), Expr::Call(g, b)) => f == g && a == b,
            _ => false,
        }
    }
}

impl Eq for Expr {}

impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.rank().hash(state);
        match self {
            Expr::Field(name) => name.hash(state),
            Expr::Const(c) => c.to_bits().hash(state),
            Expr::Call(name, args) => {
                name.hash(state);
                args.hash(state);
            }
        }
    }
}

impl PartialOrd for Expr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Expr {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Expr::Const(a), Expr::Const(b)) => a.total_cmp(b),
            (Expr::Field(a), Expr::Field(b)) => a.cmp(b),
            (Expr::Call(f, a), Expr::Call(g, b)) => f.cmp(g).then_with(|| a.cmp(b)),
            _ => self.rank().cmp(&other.rank()),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Field(name) => write!(f, "${name}"),
            Expr::Const(c) => write!(f, "{c}"),
            Expr::Call(name, args) => {
                if let Some(op) = infix_symbol(name)
                    && let [a, b] = args.as_slice()
                {
                    return write!(f, "({a} {op} {b})");
                }
                if let [a] = args.as_slice() {
                    match name.as_str() {
                        "Neg" => return write!(f, "(-{a})"),
                        "Not" => return write!(f, "(!{a})"),
                        _ => {}
                    }
                }
                write!(f, "{name}(")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// The infix spelling of a canonical call name, if it has one.
fn infix_symbol(name: &str) -> Option<&'static str> {
    Some(match name {
        "Add" => "+",
        "Sub" => "-",
        "Mul" => "*",
        "Div" => "/",
        "Gt" => ">",
        "Ge" => ">=",
        "Lt" => "<",
        "Le" => "<=",
        "Eq" => "==",
        "Ne" => "!=",
        "And" => "&&",
        "Or" => "||",
        _ => return None,
    })
}
