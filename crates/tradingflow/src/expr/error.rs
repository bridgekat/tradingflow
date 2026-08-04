use std::fmt;

/// Errors from parsing or lowering a factor expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Syntax error at a byte offset of the source string.
    Parse { pos: usize, message: String },
    /// A `$field` not registered on the [`Context`](super::Context).
    UnknownField(String),
    /// A grouping not registered on the [`Context`](super::Context).
    UnknownGrouping(String),
    /// A function name outside the vocabulary.
    UnknownFunction(String),
    /// Wrong number of arguments to a function.
    Arity {
        name: String,
        expected: usize,
        got: usize,
    },
    /// A window or quantile argument that is not an admissible literal.
    Window { name: String, message: String },
    /// An operand of the wrong kind (numeric vs boolean, wire vs constant).
    Type { name: String, message: String },
    /// The whole expression folds to a constant: it references no field.
    Constant,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse { pos, message } => write!(f, "parse error at offset {pos}: {message}"),
            Error::UnknownField(name) => write!(f, "unknown field `${name}`"),
            Error::UnknownGrouping(name) => write!(
                f,
                "unknown grouping `${name}`; register it with `Context::add_grouping`"
            ),
            Error::UnknownFunction(name) => write!(f, "unknown function `{name}`"),
            Error::Arity {
                name,
                expected,
                got,
            } => write!(f, "`{name}` expects {expected} argument(s), got {got}"),
            Error::Window { name, message } => write!(f, "`{name}`: {message}"),
            Error::Type { name, message } => write!(f, "`{name}`: {message}"),
            Error::Constant => write!(
                f,
                "expression folds to a constant; a factor must reference at least one `$field`"
            ),
        }
    }
}

impl std::error::Error for Error {}
