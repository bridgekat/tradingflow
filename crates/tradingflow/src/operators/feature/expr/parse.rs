use super::Error;
use super::ast::Expr;

/// A lexed token with its starting byte offset.
#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Field(String),
    Ident(String),
    LParen,
    RParen,
    Comma,
    Plus,
    Minus,
    Star,
    Slash,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    Ne,
    And,
    Or,
    Bang,
}

fn err(pos: usize, message: impl Into<String>) -> Error {
    Error::Parse {
        pos,
        message: message.into(),
    }
}

/// Lexes `src` into a token list. Whitespace separates tokens; `&`/`&&` and
/// `|`/`||` are accepted interchangeably.
fn lex(src: &str) -> Result<Vec<(usize, Tok)>, Error> {
    let bytes = src.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        let c = bytes[i];
        match c {
            b' ' | b'\t' | b'\r' | b'\n' => i += 1,
            b'(' | b')' | b',' | b'+' | b'-' | b'*' | b'/' => {
                let tok = match c {
                    b'(' => Tok::LParen,
                    b')' => Tok::RParen,
                    b',' => Tok::Comma,
                    b'+' => Tok::Plus,
                    b'-' => Tok::Minus,
                    b'*' => Tok::Star,
                    _ => Tok::Slash,
                };
                toks.push((start, tok));
                i += 1;
            }
            b'<' | b'>' | b'=' | b'!' => {
                let eq = i + 1 < bytes.len() && bytes[i + 1] == b'=';
                let tok = match (c, eq) {
                    (b'<', true) => Tok::Le,
                    (b'<', false) => Tok::Lt,
                    (b'>', true) => Tok::Ge,
                    (b'>', false) => Tok::Gt,
                    (b'=', true) => Tok::EqEq,
                    (b'=', false) => return Err(err(start, "'=' is not an operator; use '=='")),
                    (b'!', true) => Tok::Ne,
                    (b'!', false) => Tok::Bang,
                    _ => unreachable!(),
                };
                i += if eq { 2 } else { 1 };
                toks.push((start, tok));
            }
            b'&' | b'|' => {
                // Accept both the single and doubled spellings.
                if i + 1 < bytes.len() && bytes[i + 1] == c {
                    i += 2;
                } else {
                    i += 1;
                }
                toks.push((start, if c == b'&' { Tok::And } else { Tok::Or }));
            }
            b'$' => {
                i += 1;
                let ident_start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                if i == ident_start {
                    return Err(err(start, "expected a field name after '$'"));
                }
                toks.push((start, Tok::Field(src[ident_start..i].to_string())));
            }
            b'0'..=b'9' => {
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i < bytes.len() && bytes[i] == b'.' {
                    i += 1;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                    i += 1;
                    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                        i += 1;
                    }
                    let exp_start = i;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                    if i == exp_start {
                        return Err(err(start, "expected digits in exponent"));
                    }
                }
                let text = &src[start..i];
                let value = text
                    .parse::<f64>()
                    .map_err(|_| err(start, format!("invalid number literal `{text}`")))?;
                toks.push((start, Tok::Num(value)));
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                toks.push((start, Tok::Ident(src[start..i].to_string())));
            }
            _ => {
                let ch = src[start..].chars().next().unwrap();
                return Err(err(start, format!("unexpected character `{ch}`")));
            }
        }
    }
    Ok(toks)
}

/// Recursive-descent parser with conventional precedence:
/// `||` < `&&` < comparisons < `+ -` < `* /` < unary `- !` < atoms.
struct Parser {
    toks: Vec<(usize, Tok)>,
    i: usize,
    end: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i).map(|(_, t)| t)
    }

    fn pos(&self) -> usize {
        self.toks.get(self.i).map_or(self.end, |&(p, _)| p)
    }

    fn bump(&mut self) -> Tok {
        let tok = self.toks[self.i].1.clone();
        self.i += 1;
        tok
    }

    fn expect(&mut self, tok: Tok, what: &str) -> Result<(), Error> {
        if self.peek() == Some(&tok) {
            self.i += 1;
            Ok(())
        } else {
            Err(err(self.pos(), format!("expected {what}")))
        }
    }

    fn or_expr(&mut self) -> Result<Expr, Error> {
        let mut lhs = self.and_expr()?;
        while self.peek() == Some(&Tok::Or) {
            self.bump();
            let rhs = self.and_expr()?;
            lhs = Expr::call("Or", vec![lhs, rhs]);
        }
        Ok(lhs)
    }

    fn and_expr(&mut self) -> Result<Expr, Error> {
        let mut lhs = self.cmp_expr()?;
        while self.peek() == Some(&Tok::And) {
            self.bump();
            let rhs = self.cmp_expr()?;
            lhs = Expr::call("And", vec![lhs, rhs]);
        }
        Ok(lhs)
    }

    fn cmp_expr(&mut self) -> Result<Expr, Error> {
        let mut lhs = self.add_expr()?;
        while let Some(name) = match self.peek() {
            Some(Tok::Lt) => Some("Lt"),
            Some(Tok::Le) => Some("Le"),
            Some(Tok::Gt) => Some("Gt"),
            Some(Tok::Ge) => Some("Ge"),
            Some(Tok::EqEq) => Some("Eq"),
            Some(Tok::Ne) => Some("Ne"),
            _ => None,
        } {
            self.bump();
            let rhs = self.add_expr()?;
            lhs = Expr::call(name, vec![lhs, rhs]);
        }
        Ok(lhs)
    }

    fn add_expr(&mut self) -> Result<Expr, Error> {
        let mut lhs = self.mul_expr()?;
        while let Some(name) = match self.peek() {
            Some(Tok::Plus) => Some("Add"),
            Some(Tok::Minus) => Some("Sub"),
            _ => None,
        } {
            self.bump();
            let rhs = self.mul_expr()?;
            lhs = Expr::call(name, vec![lhs, rhs]);
        }
        Ok(lhs)
    }

    fn mul_expr(&mut self) -> Result<Expr, Error> {
        let mut lhs = self.unary_expr()?;
        while let Some(name) = match self.peek() {
            Some(Tok::Star) => Some("Mul"),
            Some(Tok::Slash) => Some("Div"),
            _ => None,
        } {
            self.bump();
            let rhs = self.unary_expr()?;
            lhs = Expr::call(name, vec![lhs, rhs]);
        }
        Ok(lhs)
    }

    fn unary_expr(&mut self) -> Result<Expr, Error> {
        match self.peek() {
            Some(Tok::Minus) => {
                self.bump();
                let inner = self.unary_expr()?;
                // Fold a negated literal so window arguments like `-1` reach
                // the lowering as constants (and get a proper error there).
                Ok(match inner {
                    Expr::Const(c) => Expr::Const(-c),
                    other => Expr::call("Neg", vec![other]),
                })
            }
            Some(Tok::Bang) => {
                self.bump();
                let inner = self.unary_expr()?;
                Ok(Expr::call("Not", vec![inner]))
            }
            _ => self.primary_expr(),
        }
    }

    fn primary_expr(&mut self) -> Result<Expr, Error> {
        let pos = self.pos();
        match self.peek() {
            Some(Tok::Num(_)) => {
                let Tok::Num(c) = self.bump() else {
                    unreachable!()
                };
                Ok(Expr::Const(c))
            }
            Some(Tok::Field(_)) => {
                let Tok::Field(name) = self.bump() else {
                    unreachable!()
                };
                Ok(Expr::Field(name))
            }
            Some(Tok::Ident(_)) => {
                let Tok::Ident(name) = self.bump() else {
                    unreachable!()
                };
                self.expect(
                    Tok::LParen,
                    "'(' after function name (fields are spelled `$name`)",
                )?;
                let mut args = Vec::new();
                if self.peek() != Some(&Tok::RParen) {
                    loop {
                        args.push(self.or_expr()?);
                        if self.peek() == Some(&Tok::Comma) {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(Tok::RParen, "')' to close the argument list")?;
                Ok(Expr::Call(name, args))
            }
            Some(Tok::LParen) => {
                self.bump();
                let inner = self.or_expr()?;
                self.expect(Tok::RParen, "')'")?;
                Ok(inner)
            }
            _ => Err(err(
                pos,
                "expected a number, `$field`, function call or '('",
            )),
        }
    }
}

/// Parses a factor expression such as `"Mean($close, 12) - Mean($close, 26)"`.
///
/// See the [module-level docs](super) for the full grammar and vocabulary.
pub fn parse(src: &str) -> Result<Expr, Error> {
    let toks = lex(src)?;
    let mut parser = Parser {
        toks,
        i: 0,
        end: src.len(),
    };
    let expr = parser.or_expr()?;
    if parser.i != parser.toks.len() {
        return Err(err(parser.pos(), "unexpected trailing input"));
    }
    Ok(expr)
}
