//! Integration tests for `Shape` and its row-major offset traversal.

use tradingflow_data::Shape;

#[test]
fn row_major_strides_and_len() {
    let s = Shape::row_major([2, 3, 4]);
    assert_eq!(s.strides(), [12, 4, 1]);
    assert_eq!(s.len(), 24);
    assert!(s.is_contiguous());
    assert!(!s.is_empty());

    // The empty product: a rank-0 shape holds exactly one element.
    let s = Shape::row_major([]);
    assert_eq!(s.len(), 1);
    assert!(s.is_contiguous());

    // A zero extent makes the shape empty but keeps it contiguous.
    let s = Shape::row_major([2, 0]);
    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
}

#[test]
fn offset_is_the_stride_dot_product() {
    let s = Shape::row_major([2, 3]);
    assert_eq!(s.offset([0, 0]), 0);
    assert_eq!(s.offset([0, 2]), 2);
    assert_eq!(s.offset([1, 0]), 3);
    assert_eq!(s.offset([1, 2]), 5);

    // A rank-0 shape addresses its element at the empty index.
    assert_eq!(Shape::row_major([]).offset([]), 0);

    // Strided: a column of the [2, 3] panel above.
    let col = Shape::strided([2], [3]);
    assert_eq!(col.offset([0]), 0);
    assert_eq!(col.offset([1]), 3);
}

#[test]
fn offset_bounds_are_per_axis() {
    let s = Shape::row_major([2, 3]);
    assert!(s.contains([1, 2]));
    // Flat offset 3 is in the buffer, but [0, 3] runs off the end of a row.
    assert!(!s.contains([0, 3]));
    assert_eq!(s.offset_checked([0, 3]), None);
    assert_eq!(s.offset_checked([2, 0]), None);
    assert_eq!(s.offset_checked([1, 2]), Some(5));

    // Nothing is in bounds in an empty shape; everything is in a rank-0 one.
    assert!(!Shape::row_major([0]).contains([0]));
    assert!(Shape::row_major([]).contains([]));
}

#[test]
#[should_panic(expected = "index [2, 0] out of bounds for extents [2, 3]")]
fn offset_panics_out_of_bounds() {
    let _ = Shape::row_major([2, 3]).offset([2, 0]);
}

#[test]
fn offsets_walk_row_major() {
    // Contiguous: offsets are just 0..len.
    let s = Shape::row_major([2, 3]);
    assert_eq!(s.offsets().collect::<Vec<_>>(), (0..6).collect::<Vec<_>>());
    assert_eq!(s.offsets().len(), 6);

    // Strided: a column, stride 3.
    let col = Shape::strided([2], [3]);
    assert_eq!(col.offsets().collect::<Vec<_>>(), vec![0, 3]);

    // Transposed [2, 3] -> [3, 2]: row-major over swapped axes.
    let t = Shape::strided([3, 2], [1, 3]);
    assert_eq!(t.offsets().collect::<Vec<_>>(), vec![0, 3, 1, 4, 2, 5]);

    // A rank-0 shape yields its single element; an empty one yields nothing.
    assert_eq!(Shape::row_major([]).offsets().collect::<Vec<_>>(), vec![0]);
    assert_eq!(
        Shape::row_major([0, 2]).offsets().collect::<Vec<_>>(),
        vec![] as Vec<usize>,
    );
}

#[test]
fn offsets_agree_with_offset() {
    // The traversal's incremental carry must match the direct dot product.
    let shapes = [
        Shape::row_major([2, 3, 2]),
        Shape::strided([2, 3, 2], [1, 8, 3]),
    ];
    for s in shapes {
        let direct: Vec<_> = (0..2)
            .flat_map(|i| (0..3).flat_map(move |j| (0..2).map(move |k| [i, j, k])))
            .map(|idx| s.offset(idx))
            .collect();
        assert_eq!(s.offsets().collect::<Vec<_>>(), direct);
    }
}

#[test]
fn offsets_are_exhausted_once() {
    let mut it = Shape::row_major([2]).offsets();
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next(), Some(1));
    assert_eq!(it.next(), None);
    assert_eq!(it.next(), None); // fused
}

#[test]
fn packed_block_ranges() {
    let s = Shape::row_major([2, 3]); // 6 scalars per block
    assert_eq!(s.blocks_len(0), 0);
    assert_eq!(s.blocks_len(4), 24);
    assert_eq!(s.block_range(0), 0..6);
    assert_eq!(s.block_range(2), 12..18);
    assert_eq!(s.blocks_range(1..3), 6..18);
    assert_eq!(s.blocks_range(2..2), 12..12);

    // Rank-0 blocks are one scalar each, so ranges are the indices.
    let s = Shape::row_major([]);
    assert_eq!(s.block_range(3), 3..4);
    assert_eq!(s.blocks_range(1..4), 1..4);
}

#[test]
fn span_is_the_last_offset_plus_one() {
    // Contiguous: span == len.
    assert_eq!(Shape::row_major([2, 3]).span(), 6);
    assert_eq!(Shape::row_major([]).span(), 1);
    // Strided: a column of a [3, 4] panel touches offsets {0, 4, 8}.
    assert_eq!(Shape::strided([3], [4]).span(), 9);
    // Empty: addresses nothing.
    assert_eq!(Shape::row_major([2, 0]).span(), 0);
    // The span is 1 + the largest offset `offsets` ever yields.
    let s = Shape::strided([2, 3], [1, 2]);
    assert_eq!(s.span(), 1 + s.offsets().max().unwrap());
}

#[test]
fn stacked_and_split_first_are_inverse() {
    let elem = Shape::row_major([2, 3]);
    let stacked = elem.stacked::<3>(4);
    assert_eq!(stacked.extents(), [4, 2, 3]);
    assert_eq!(stacked.strides(), [6, 3, 1]);
    assert!(stacked.is_contiguous());
    // Axis 0's step is the block size, so `block_range` and `stacked` agree.
    let (step, row) = stacked.split_first::<2>();
    assert_eq!(step, elem.len());
    assert_eq!(row, elem);
}

#[test]
fn split_first_keeps_inner_strides() {
    // A transposed-ish view: dropping axis 0 must preserve the rest as-is.
    let s = Shape::strided([4, 2, 3], [100, 1, 2]);
    let (step, row) = s.split_first::<2>();
    assert_eq!(step, 100);
    assert_eq!(row, Shape::strided([2, 3], [1, 2]));
    assert!(!row.is_contiguous());
}

#[test]
#[should_panic(expected = "split_first: rank N must be >= 1, got 0")]
fn split_first_needs_an_axis() {
    let _ = Shape::row_major([]).split_first::<0>();
}
