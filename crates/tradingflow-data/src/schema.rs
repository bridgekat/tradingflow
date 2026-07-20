//! The [`Schema`] helper for bookkeeping array axes.

use std::collections::HashMap;

/// A bidirectional map between string labels and integer indices.
#[derive(Debug, Clone)]
pub struct Schema {
    names: Vec<String>,
    lookup: HashMap<String, usize>,
}

impl Schema {
    /// Creates a schema from an ordered list of names.
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

    /// Looks up the name at a position.
    ///
    /// # Panics
    ///
    /// Panics if the position is out of bounds.
    pub fn name(&self, index: usize) -> &str {
        &self.names[index]
    }

    /// Looks up the position of a name.
    ///
    /// # Panics
    ///
    /// Panics if the name is not in the schema.
    pub fn index(&self, name: &str) -> usize {
        self.lookup[name]
    }

    /// Resolves multiple names to positions.
    ///
    /// # Panics
    ///
    /// Panics if any name is not in the schema.
    pub fn indices(&self, names: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<usize> {
        names.into_iter().map(|n| self.index(n.as_ref())).collect()
    }

    /// Whether the schema contains a name.
    pub fn contains(&self, name: &str) -> bool {
        self.lookup.contains_key(name)
    }

    /// All names in order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Creates a sub-schema by selecting names at the given positions.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds or appears more than once.
    pub fn select(&self, indices: &[usize]) -> Self {
        let names: Vec<String> = indices.iter().map(|&i| self.names[i].clone()).collect();
        Self::new(names)
    }

    /// Creates a schema by concatenating this schema with another.
    ///
    /// # Panics
    ///
    /// Panics if any name appears in both schemas.
    pub fn union(&self, other: &Schema) -> Self {
        let mut names = self.names.clone();
        names.extend(other.names.iter().cloned());
        Self::new(names)
    }
}
