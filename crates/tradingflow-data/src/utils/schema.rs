use std::collections::HashMap;

/// A labeling scheme on an array axis.
#[derive(Debug, Clone)]
pub enum Axis {
    /// A labeled axis with the given labels.
    Labeled(Schema),
    /// An unlabelled axis with the given length.
    Fixed(usize),
    /// An unlabelled axis without a known length.
    None,
}

/// A bidirectional map between string labels and integer indices.
#[derive(Debug, Clone)]
pub struct Schema {
    labels: Vec<String>,
    lookup: HashMap<String, usize>,
}

impl Schema {
    /// Creates a schema from an ordered list of labels.
    ///
    /// # Panics
    ///
    /// Panics if any label appears more than once.
    pub fn new(labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let labels: Vec<String> = labels.into_iter().map(Into::into).collect();
        let mut lookup = HashMap::with_capacity(labels.len());
        for (i, label) in labels.iter().enumerate() {
            if lookup.insert(label.clone(), i).is_some() {
                panic!("duplicate label in schema: {label}");
            }
        }
        Self { labels, lookup }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether the schema is empty.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Looks up the label at a position.
    ///
    /// # Panics
    ///
    /// Panics if the position is out of bounds.
    pub fn label(&self, index: usize) -> &str {
        &self.labels[index]
    }

    /// Looks up the label at a position, or `None` if out of bounds.
    pub fn try_label(&self, index: usize) -> Option<&str> {
        self.labels.get(index).map(|s| s.as_str())
    }

    /// Looks up the position of a label.
    ///
    /// # Panics
    ///
    /// Panics if the label is not in the schema.
    pub fn index(&self, label: &str) -> usize {
        self.lookup[label]
    }

    /// Looks up the position of a label, or `None` if it is not in the schema.
    pub fn try_index(&self, label: &str) -> Option<usize> {
        self.lookup.get(label).copied()
    }

    /// Resolves multiple labels to positions.
    ///
    /// # Panics
    ///
    /// Panics if any label is not in the schema.
    pub fn indices(&self, labels: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<usize> {
        labels.into_iter().map(|n| self.index(n.as_ref())).collect()
    }

    /// Whether the schema contains a label.
    pub fn contains(&self, label: &str) -> bool {
        self.lookup.contains_key(label)
    }

    /// All labels in order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Creates a sub-schema by selecting labels at the given positions.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds or appears more than once.
    pub fn select(&self, indices: &[usize]) -> Self {
        let labels: Vec<String> = indices.iter().map(|&i| self.labels[i].clone()).collect();
        Self::new(labels)
    }

    /// Creates a schema by concatenating this schema with another.
    ///
    /// # Panics
    ///
    /// Panics if any label appears in both schemas.
    pub fn union(&self, other: &Schema) -> Self {
        let mut labels = self.labels.clone();
        labels.extend(other.labels.iter().cloned());
        Self::new(labels)
    }
}
