//! Integration tests for `core`'s untyped graph (pointer-graph API).
//!
//! Each node's output VALUE lives in its erased state; its output slot holds a
//! pointer to that value. A `ComputeFn` reads its input slots, writes the value
//! into state, and points the output slot at it. `source_fn` re-exposes the
//! state value (the value itself is poked via `state_mut`).

use std::any::TypeId;
use std::thread;

use tradingflow_graph::core::{Builder, ComputeFn, ErasedCell, Error, Graph, Pool, Segment};

fn pool() -> Pool {
    Pool::new(thread::available_parallelism().unwrap().get())
}

unsafe fn source_fn(
    _: *const [bool],
    _: *const [*const ()],
    out_flags: *mut [bool],
    out_ptrs: *mut [*const ()],
    _context: *const (),
    state: *mut (),
) {
    unsafe {
        out_flags.as_mut_unchecked()[0] = true;
        out_ptrs.as_mut_unchecked()[0] = state.cast_const();
    }
}

unsafe fn inc(
    _: *const [bool],
    in_ptrs: *const [*const ()],
    out_flags: *mut [bool],
    out_ptrs: *mut [*const ()],
    _context: *const (),
    state: *mut (),
) {
    let in_ptrs = unsafe { in_ptrs.as_ref_unchecked() };
    let s = unsafe { &mut *(state as *mut i64) };
    *s = unsafe { *(in_ptrs[0] as *const i64) } + 1;
    unsafe {
        out_flags.as_mut_unchecked()[0] = true;
        out_ptrs.as_mut_unchecked()[0] = (s as *const i64).cast();
    }
}

unsafe fn times10(
    _: *const [bool],
    in_ptrs: *const [*const ()],
    out_flags: *mut [bool],
    out_ptrs: *mut [*const ()],
    _context: *const (),
    state: *mut (),
) {
    let in_ptrs = unsafe { in_ptrs.as_ref_unchecked() };
    let s = unsafe { &mut *(state as *mut i64) };
    *s = unsafe { *(in_ptrs[0] as *const i64) } * 10;
    unsafe {
        out_flags.as_mut_unchecked()[0] = true;
        out_ptrs.as_mut_unchecked()[0] = (s as *const i64).cast();
    }
}

unsafe fn add(
    _: *const [bool],
    in_ptrs: *const [*const ()],
    out_flags: *mut [bool],
    out_ptrs: *mut [*const ()],
    _context: *const (),
    state: *mut (),
) {
    let in_ptrs = unsafe { in_ptrs.as_ref_unchecked() };
    let s = unsafe { &mut *(state as *mut i64) };
    *s = unsafe { *(in_ptrs[0] as *const i64) + *(in_ptrs[1] as *const i64) };
    unsafe {
        out_flags.as_mut_unchecked()[0] = true;
        out_ptrs.as_mut_unchecked()[0] = (s as *const i64).cast();
    }
}

unsafe fn sum_all(
    _: *const [bool],
    in_ptrs: *const [*const ()],
    out_flags: *mut [bool],
    out_ptrs: *mut [*const ()],
    _context: *const (),
    state: *mut (),
) {
    let in_ptrs = unsafe { in_ptrs.as_ref_unchecked() };
    let s = unsafe { &mut *(state as *mut i64) };
    *s = in_ptrs.iter().map(|&p| unsafe { *(p as *const i64) }).sum();
    unsafe {
        out_flags.as_mut_unchecked()[0] = true;
        out_ptrs.as_mut_unchecked()[0] = (s as *const i64).cast();
    }
}

/// Passes its input through, but panics on a negative value (for the
/// poison test).
unsafe fn panic_if_negative(
    _: *const [bool],
    in_ptrs: *const [*const ()],
    out_flags: *mut [bool],
    out_ptrs: *mut [*const ()],
    _context: *const (),
    state: *mut (),
) {
    let in_ptrs = unsafe { in_ptrs.as_ref_unchecked() };
    let x = unsafe { *(in_ptrs[0] as *const i64) };
    assert!(x >= 0, "negative input");
    let s = unsafe { &mut *(state as *mut i64) };
    *s = x;
    unsafe {
        out_flags.as_mut_unchecked()[0] = true;
        out_ptrs.as_mut_unchecked()[0] = (s as *const i64).cast();
    }
}

/// An `i64`-output segment: state holds the output value, slot points at it.
fn seg(input_types: Box<[TypeId]>, f: ComputeFn) -> Segment {
    let state = ErasedCell::new(0i64);
    let ptr = state.get().cast_const();
    unsafe {
        Segment::new(
            input_types,
            Box::new([TypeId::of::<i64>()]),
            f,
            state,
            Box::new([false]),
            Box::new([ptr]),
        )
    }
}

fn unary(f: ComputeFn) -> Segment {
    seg(Box::new([TypeId::of::<i64>()]), f)
}

fn binary(f: ComputeFn) -> Segment {
    seg(Box::new([TypeId::of::<i64>(), TypeId::of::<i64>()]), f)
}

fn source() -> Segment {
    seg(Box::new([]), source_fn)
}

/// Read an output slot's `i64` value (through its pointer).
fn read(g: &Graph, slot: usize) -> i64 {
    unsafe { *(g.slot_ptr(slot) as *const i64) }
}

/// Poke a source node's state value in place.
fn set(g: &mut Graph, slot: usize, v: i64) {
    unsafe { *(g.state_mut(slot).get() as *mut i64) = v };
}

#[test]
fn diamond_and_root_edge_are_glitch_free() {
    // S -> A=S+1, S -> B=S*10, (A,B) -> D=A+B; plus E=S+A.
    let mut b = Builder::new(ErasedCell::new(()));
    let s = b.push(source(), &[]).unwrap().start;
    let a = b.push(unary(inc), &[s]).unwrap().start;
    let bb = b.push(unary(times10), &[s]).unwrap().start;
    let d = b.push(binary(add), &[a, bb]).unwrap().start;
    let e = b.push(binary(add), &[s, a]).unwrap().start;
    let mut g = b.build();
    let mut pool = pool();

    set(&mut g, s, 1);
    g.stabilize(&mut pool);
    assert_eq!(read(&g, a), 2);
    assert_eq!(read(&g, bb), 10);
    assert_eq!(read(&g, d), 12); // (1+1) + (1*10)
    assert_eq!(read(&g, e), 3); //  1  + (1+1)

    set(&mut g, s, 4);
    g.stabilize(&mut pool);
    assert_eq!(read(&g, d), 45); // 5 + 40
    assert_eq!(read(&g, e), 9); //  4 + 5

    // Untouched source -> empty dirty cone -> stabilize is a no-op.
    g.stabilize(&mut pool);
    assert_eq!(read(&g, d), 45);
}

#[test]
fn wide_fan_in_runs_on_the_pool() {
    // S -> N independent `inc` nodes -> one aggregate that reads all N.
    const N: usize = 64;
    let agg = seg(vec![TypeId::of::<i64>(); N].into_boxed_slice(), sum_all);

    let mut b = Builder::new(ErasedCell::new(()));
    let s = b.push(source(), &[]).unwrap().start;
    let layer: Vec<usize> = (0..N)
        .map(|_| b.push(unary(inc), &[s]).unwrap().start)
        .collect();
    let total = b.push(agg, &layer).unwrap().start;
    let mut g = b.build();
    let mut pool = pool();

    set(&mut g, s, 1);
    g.stabilize(&mut pool);
    assert_eq!(read(&g, total), 2 * N as i64); // each layer node = 1 + 1

    set(&mut g, s, 9);
    g.stabilize(&mut pool);
    assert_eq!(read(&g, total), 10 * N as i64); // each = 9 + 1
}

#[test]
fn panic_in_compute_poisons_the_graph() {
    // S -> P (panics on a negative input) -> D. A handler panic means a node
    // failed to produce a valid output (downstream slots may hold dangling
    // forwarded pointers), so the graph is poisoned: no recovery is attempted,
    // and every later stabilize and slot read panics.
    let mut b = Builder::new(ErasedCell::new(()));
    let s = b.push(source(), &[]).unwrap().start;
    let p = b.push(unary(panic_if_negative), &[s]).unwrap().start;
    let d = b.push(unary(inc), &[p]).unwrap().start;
    let mut g = b.build();
    let mut pool = pool();

    // Poison value: P panics, unwinding out of `stabilize` (the panic message
    // is printed to stderr by the default hook -- expected, the test catches it).
    set(&mut g, s, -1);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.stabilize(&mut pool)));
    assert!(r.is_err());

    // Poisoned: a slot read panics rather than touching a possibly-stale output.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| read(&g, d)));
    assert!(r.is_err());

    // Poisoned: a node state write panics.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| set(&mut g, s, 5)));
    assert!(r.is_err());
}

// ===== wiring errors (push returns Err, not a panic) =====

#[test]
fn push_rejects_out_of_bounds_input() {
    let mut b = Builder::new(ErasedCell::new(()));
    // No slots exist yet, so slot 0 is out of bounds.
    let err = b.push(unary(inc), &[0]).unwrap_err();
    assert_eq!(
        err,
        Error::InputOutOfBounds {
            place: 0,
            slot: 0,
            num_slots: 0,
        }
    );
}

#[test]
fn push_rejects_arity_mismatch() {
    let mut b = Builder::new(ErasedCell::new(()));
    let s = b.push(source(), &[]).unwrap().start;
    // `binary` declares two inputs; wire only one.
    let err = b.push(binary(add), &[s]).unwrap_err();
    assert_eq!(
        err,
        Error::InputArity {
            expected: 2,
            actual: 1
        }
    );
}

#[test]
fn push_rejects_type_mismatch() {
    let mut b = Builder::new(ErasedCell::new(()));
    // An f64 source slot ...
    let f = {
        let state = ErasedCell::new(0.0f64);
        let ptr = state.get().cast_const();
        unsafe {
            Segment::new(
                Box::new([]),
                Box::new([TypeId::of::<f64>()]),
                source_fn,
                state,
                Box::new([false]),
                Box::new([ptr]),
            )
        }
    };
    let fc = b.push(f, &[]).unwrap().start;
    // ... wired into `inc`, which expects i64.
    let err = b.push(unary(inc), &[fc]).unwrap_err();
    assert_eq!(
        err,
        Error::InputType {
            place: 0,
            slot: fc,
            expected: TypeId::of::<i64>(),
            actual: TypeId::of::<f64>(),
        }
    );
}
