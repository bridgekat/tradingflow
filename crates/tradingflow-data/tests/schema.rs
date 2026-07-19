use tradingflow_data::schema::Schema;

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
    let merged = s1.union(&s2);
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
    s1.union(&s2);
}
