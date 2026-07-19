use tradingflow_data::layout::{ColMajor, RowMajor, Strided};
use tradingflow_data::{Array, Layout};

#[test]
fn row_major_strides_and_len() {
    let l = RowMajor::new([2, 3, 4]);
    assert_eq!(l.extents(), [2, 3, 4]);
    assert_eq!(l.strides(), [12, 4, 1]);
    assert_eq!(l.len(), 24);
    assert_eq!(l.ndim(), 3);
    assert!(l.is_contiguous());
    assert!(!l.is_empty());

    // The empty product: a rank-0 layout holds exactly one element.
    let l = RowMajor::new([]);
    assert_eq!(l.len(), 1);
    assert!(l.is_contiguous());

    // A zero extent makes the layout empty but keeps it contiguous.
    let l = RowMajor::new([2, 0]);
    assert_eq!(l.len(), 0);
    assert!(l.is_empty());
}

#[test]
fn col_major_strides_and_contiguity() {
    let l = ColMajor::new([2, 3, 4]);
    assert_eq!(l.strides(), [1, 2, 6]);
    assert_eq!(l.len(), 24);
    // Column-major is not *row-major* contiguous beyond rank 1.
    assert!(!l.is_contiguous());
    assert!(ColMajor::new([5]).is_contiguous());
    assert!(ColMajor::new([]).is_contiguous());

    // Offsets still walk in logical row-major order, through the strides.
    assert_eq!(
        ColMajor::new([2, 3]).offsets().collect::<Vec<_>>(),
        vec![0, 2, 4, 1, 3, 5],
    );
}

#[test]
fn strided_detects_row_major() {
    // Canonical row-major strides for [2, 3] are [3, 1].
    assert!(Strided::new([2, 3], [3, 1]).is_contiguous());
    assert!(!Strided::new([2, 3], [1, 2]).is_contiguous());

    // The policy conversions preserve extents and strides.
    let rm: Strided<2> = RowMajor::new([2, 3]).into();
    assert_eq!(rm, Strided::new([2, 3], [3, 1]));
    let cm: Strided<2> = ColMajor::new([2, 3]).into();
    assert_eq!(cm, Strided::new([2, 3], [1, 2]));
    assert!(!cm.is_contiguous());
}

#[test]
fn offset_is_the_stride_dot_product() {
    let l = RowMajor::new([2, 3]);
    assert_eq!(l.offset([0, 0]), 0);
    assert_eq!(l.offset([0, 2]), 2);
    assert_eq!(l.offset([1, 0]), 3);
    assert_eq!(l.offset([1, 2]), 5);

    // A rank-0 layout addresses its element at the empty index.
    assert_eq!(RowMajor::new([]).offset([]), 0);

    // Strided: a column of the [2, 3] panel above.
    let col = Strided::new([2], [3]);
    assert_eq!(col.offset([0]), 0);
    assert_eq!(col.offset([1]), 3);
}

#[test]
fn offset_bounds_are_per_axis() {
    let l = RowMajor::new([2, 3]);
    assert!(l.contains([1, 2]));
    // Flat offset 3 is in the buffer, but [0, 3] runs off the end of a row.
    assert!(!l.contains([0, 3]));
    assert_eq!(l.offset_checked([0, 3]), None);
    assert_eq!(l.offset_checked([2, 0]), None);
    assert_eq!(l.offset_checked([1, 2]), Some(5));

    // Nothing is in bounds in an empty layout; everything is in a rank-0 one.
    assert!(!RowMajor::new([0]).contains([0]));
    assert!(RowMajor::new([]).contains([]));
}

#[test]
#[should_panic(expected = "index [2, 0] out of bounds for extents [2, 3]")]
fn offset_panics_out_of_bounds() {
    let _ = RowMajor::new([2, 3]).offset([2, 0]);
}

#[test]
fn offsets_walk_row_major() {
    // Contiguous: offsets are just 0..len.
    let l = RowMajor::new([2, 3]);
    assert_eq!(l.offsets().collect::<Vec<_>>(), (0..6).collect::<Vec<_>>());
    assert_eq!(l.offsets().len(), 6);

    // Strided: a column, stride 3.
    let col = Strided::new([2], [3]);
    assert_eq!(col.offsets().collect::<Vec<_>>(), vec![0, 3]);

    // Transposed [2, 3] -> [3, 2]: row-major over swapped axes.
    let t = Strided::new([3, 2], [1, 3]);
    assert_eq!(t.offsets().collect::<Vec<_>>(), vec![0, 3, 1, 4, 2, 5]);

    // A rank-0 layout yields its single element; an empty one yields nothing.
    assert_eq!(RowMajor::new([]).offsets().collect::<Vec<_>>(), vec![0]);
    assert_eq!(
        RowMajor::new([0, 2]).offsets().collect::<Vec<_>>(),
        vec![] as Vec<usize>,
    );
}

#[test]
fn offsets_agree_with_offset() {
    // The traversal's incremental carry must match the direct dot product,
    // whatever the policy behind the trait.
    fn check(l: impl Layout<3>) {
        let e = l.extents();
        let direct: Vec<_> = (0..e[0])
            .flat_map(|i| (0..e[1]).flat_map(move |j| (0..e[2]).map(move |k| [i, j, k])))
            .map(|idx| l.offset(idx))
            .collect();
        assert_eq!(l.offsets().collect::<Vec<_>>(), direct);
    }
    check(RowMajor::new([2, 3, 2]));
    check(ColMajor::new([2, 3, 2]));
    check(Strided::new([2, 3, 2], [1, 8, 3]));
}

#[test]
fn offsets_are_exhausted_once() {
    let mut it = RowMajor::new([2]).offsets();
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next(), Some(1));
    assert_eq!(it.next(), None);
    assert_eq!(it.next(), None); // fused
}

#[test]
fn span_is_the_last_offset_plus_one() {
    // Contiguous: span == len.
    assert_eq!(RowMajor::new([2, 3]).span(), 6);
    assert_eq!(RowMajor::new([]).span(), 1);
    // Strided: a column of a [3, 4] panel touches offsets {0, 4, 8}.
    assert_eq!(Strided::new([3], [4]).span(), 9);
    // Empty: addresses nothing.
    assert_eq!(RowMajor::new([2, 0]).span(), 0);
    // The span is 1 + the largest offset `offsets` ever yields.
    let l = Strided::new([2, 3], [1, 2]);
    assert_eq!(l.span(), 1 + l.offsets().max().unwrap());
}

#[test]
fn containers_are_usable_behind_the_trait() {
    // `Array` and `ArrayView` implement `Layout` by delegation, so a generic
    // helper accepts policies and containers alike.
    fn total_len<const N: usize>(l: &impl Layout<N>) -> usize {
        l.len()
    }
    assert_eq!(total_len(&RowMajor::new([2, 3])), 6);
    let a = Array::<f64, 2>::zeros([2, 3]);
    assert_eq!(total_len(&a), 6);
    assert_eq!(total_len(&a.view()), 6);
}
