//! Schema — bidirectional name↔position mapping for array axes.
//!
//! A [`Schema`] maps string labels to integer positions along a single
//! array axis.  It is a construction-time helper — not embedded in the
//! DAG or carried by arrays at runtime.
//!
//! # Examples
//!
//! ```
//! use tradingflow::utils::Schema;
//!
//! let symbols = Schema::new(["000001.SZ", "000002.SZ", "600519.SH"]);
//! assert_eq!(symbols.index("600519.SH"), 2);
//! assert_eq!(symbols.name(0), "000001.SZ");
//! assert_eq!(symbols.indices(["600519.SH", "000001.SZ"]), vec![2, 0]);
//! ```

use std::collections::HashMap;

/// Bidirectional name↔position mapping for a single array axis.
///
/// Used at graph construction time to resolve named columns/symbols into
/// the integer indices that operators like [`Select`](crate::operators::Select)
/// and [`Concat`](crate::operators::Concat) expect.
#[derive(Clone, Debug)]
pub struct Schema {
    names: Vec<String>,
    lookup: HashMap<String, usize>,
}

impl Schema {
    /// Create a schema from an ordered list of names.
    ///
    /// # Panics
    ///
    /// Panics if any name appears more than once.
    pub fn new(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let names: Vec<String> = names.into_iter().map(Into::into).collect();
        let mut lookup = HashMap::with_capacity(names.len());
        for (i, name) in names.iter().enumerate() {
            if lookup.insert(name.clone(), i).is_some() {
                panic!("duplicate name in schema: {name}");
            }
        }
        Self { names, lookup }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether the schema is empty.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Look up the position of a name.
    ///
    /// # Panics
    ///
    /// Panics if the name is not in the schema.
    pub fn index(&self, name: &str) -> usize {
        self.lookup[name]
    }

    /// Resolve multiple names to positions.
    ///
    /// # Panics
    ///
    /// Panics if any name is not in the schema.
    pub fn indices(&self, names: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<usize> {
        names.into_iter().map(|n| self.index(n.as_ref())).collect()
    }

    /// Look up the position of a name, returning `None` if absent.
    pub fn try_index(&self, name: &str) -> Option<usize> {
        self.lookup.get(name).copied()
    }

    /// Look up the name at a position.
    ///
    /// # Panics
    ///
    /// Panics if the position is out of bounds.
    pub fn name(&self, index: usize) -> &str {
        &self.names[index]
    }

    /// All names in order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Whether the schema contains a name.
    pub fn contains(&self, name: &str) -> bool {
        self.lookup.contains_key(name)
    }

    /// Create a sub-schema by selecting names at the given positions.
    pub fn select(&self, indices: &[usize]) -> Self {
        let names: Vec<String> = indices.iter().map(|&i| self.names[i].clone()).collect();
        Self::new(names)
    }

    /// Create a schema by concatenating this schema with another.
    ///
    /// # Panics
    ///
    /// Panics if any name appears in both schemas.
    pub fn concat(&self, other: &Schema) -> Self {
        let mut names = self.names.clone();
        names.extend(other.names.iter().cloned());
        Self::new(names)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_basic() {
        let s = Schema::new(["a", "b", "c"]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.index("a"), 0);
        assert_eq!(s.index("c"), 2);
        assert_eq!(s.name(1), "b");
        assert!(s.contains("b"));
        assert!(!s.contains("d"));
    }

    #[test]
    fn schema_indices() {
        let s = Schema::new(["x", "y", "z"]);
        assert_eq!(s.indices(["z", "x"]), vec![2, 0]);
    }

    #[test]
    fn schema_try_index() {
        let s = Schema::new(["a", "b"]);
        assert_eq!(s.try_index("a"), Some(0));
        assert_eq!(s.try_index("missing"), None);
    }

    #[test]
    fn schema_select() {
        let s = Schema::new(["a", "b", "c", "d"]);
        let sub = s.select(&[1, 3]);
        assert_eq!(sub.names(), &["b", "d"]);
        assert_eq!(sub.index("d"), 1);
    }

    #[test]
    fn schema_concat() {
        let s1 = Schema::new(["a", "b"]);
        let s2 = Schema::new(["c", "d"]);
        let merged = s1.concat(&s2);
        assert_eq!(merged.len(), 4);
        assert_eq!(merged.index("c"), 2);
    }

    #[test]
    #[should_panic(expected = "duplicate name")]
    fn schema_duplicate_panics() {
        Schema::new(["a", "b", "a"]);
    }

    #[test]
    #[should_panic(expected = "duplicate name")]
    fn schema_concat_overlap_panics() {
        let s1 = Schema::new(["a", "b"]);
        let s2 = Schema::new(["b", "c"]);
        s1.concat(&s2);
    }
}
