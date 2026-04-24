//! Wavefront scheduler over the 2D `(node, tick)` dependency grid.
//!
//! Uniform dependency tracking: both horizontal (upstream inputs at
//! same tick) and vertical (same node at previous tick, if stateful)
//! edges are satisfied by decrementing a per-cell dep counter.
//! When a counter reaches zero, the cell is enqueued as a ready task.
//!
//! For stateful operators, both `State` and `Output` carry across
//! ticks — tick `t+1` clones its starting (state, output) pair from
//! the values produced by tick `t`.  For stateless operators, each
//! tick clones fresh from the init values.

use std::collections::VecDeque;

use crate::data::Instant;

use super::graph::{WavefrontGraph, WavefrontNode};

/// Run the wavefront over all ticks.
///
/// `timestamps` must be sorted.  Source nodes must already have their
/// versioned rings pre-filled for all ticks.
pub(super) fn run_wavefront(graph: &mut WavefrontGraph, timestamps: &[Instant]) {
    let num_nodes = graph.nodes.len();
    let num_ticks = timestamps.len();

    if num_ticks == 0 {
        return;
    }

    // ---- dep_remaining[node][tick] ----
    let mut dep_remaining: Vec<Vec<usize>> = (0..num_nodes)
        .map(|_| vec![0; num_ticks])
        .collect();

    for node in &graph.nodes {
        if node.is_source {
            continue;
        }
        let base = node.input_indices.len();
        let is_stateful = node.is_stateful;
        for t in 0..num_ticks {
            let mut d = base;
            if is_stateful && t > 0 {
                d += 1; // vertical dependency
            }
            dep_remaining[node.index][t] = d;
        }
    }

    // ---- next_tick and nodes_at_tick ----
    let mut next_tick: Vec<usize> = vec![0; num_nodes];
    let mut nodes_at_tick: Vec<usize> = vec![0; num_ticks + 1];
    for node in &graph.nodes {
        if node.is_source {
            next_tick[node.index] = num_ticks;
            nodes_at_tick[num_ticks] += 1;
        } else {
            next_tick[node.index] = 0;
            nodes_at_tick[0] += 1;
        }
    }
    let mut low_water: usize = 0;

    // ---- ready_queue ----
    let mut ready_queue: VecDeque<(usize, usize)> = VecDeque::new();

    // Source nodes notify their downstream for all ticks.
    for &src_idx in &graph.source_indices {
        let node = &graph.nodes[src_idx];
        for t in 0..num_ticks {
            for &(down, _input_pos) in &node.trigger_edges {
                dep_remaining[down][t] -= 1;
                if dep_remaining[down][t] == 0 {
                    ready_queue.push_back((t, down));
                }
            }
        }
    }

    // ---- main loop ----
    while let Some((t, node_idx)) = ready_queue.pop_front() {
        // Advance next_tick
        {
            let old = next_tick[node_idx];
            let new = t + 1;
            if old != new {
                nodes_at_tick[old] -= 1;
                nodes_at_tick[new] += 1;
                next_tick[node_idx] = new;
            }
        }

        // Resolve input pointers before mutably borrowing the node.
        let resolved: Vec<*const u8> = {
            let node = &graph.nodes[node_idx];
            let n = node.input_indices.len();
            let mut v = Vec::with_capacity(n);
            for pos in 0..n {
                let upstream = &graph.nodes[node.input_indices[pos]];
                v.push(
                    upstream
                        .versioned
                        .get(t)
                        .unwrap_or_else(|| {
                            panic!(
                                "input {} at tick {} not found in upstream node {}",
                                pos, t, upstream.index
                            )
                        }),
                );
            }
            v
        };

        // Execute
        execute_node(&mut graph.nodes[node_idx], t, timestamps[t], &resolved);

        // Notify horizontal: each downstream node at same tick
        for &(down, _input_pos) in &graph.nodes[node_idx].trigger_edges {
            dep_remaining[down][t] -= 1;
            if dep_remaining[down][t] == 0 {
                ready_queue.push_back((t, down));
            }
        }

        // Notify vertical: if stateful, self at t+1
        if graph.nodes[node_idx].is_stateful && t + 1 < num_ticks {
            dep_remaining[node_idx][t + 1] -= 1;
            if dep_remaining[node_idx][t + 1] == 0 {
                ready_queue.push_back((t + 1, node_idx));
            }
        }

        // Advance low_water (only for deadlock detection / progress tracking).
        // GC is deferred to after the wavefront completes, since entries
        // are needed for value_at() inspection.
        while low_water < nodes_at_tick.len() && nodes_at_tick[low_water] == 0 {
            low_water += 1;
        }
    }

    debug_assert!(next_tick.iter().all(|&t| t >= num_ticks));
}

/// Execute one `(node, tick)` cell.
fn execute_node(
    node: &mut WavefrontNode,
    tick: usize,
    timestamp: Instant,
    resolved_inputs: &[*const u8],
) {
    if node.is_source {
        return;
    }

    let num_inputs = resolved_inputs.len();
    let input_ptrs: Box<[*const u8]> = resolved_inputs.into();

    // 2. Clone (state, output) pair for this tick.
    let (tick_state, tick_output) = if node.is_stateful {
        // Chain from previous tick's values.
        unsafe {
            (
                (node.state_clone_fn)(node.current_state_ptr),
                (node.output_clone_fn)(node.current_output_ptr),
            )
        }
    } else {
        // Fresh from init values.
        unsafe {
            (
                (node.state_clone_fn)(node.init_state_ptr),
                (node.output_clone_fn)(node.init_output_ptr),
            )
        }
    };

    // 3. Build produced bitset (all true for PoC).
    let produced_words: Vec<u64> = vec![u64::MAX; num_inputs.div_ceil(64).max(1)];

    // 4. Call erased compute.
    unsafe {
        (node.compute_fn)(
            tick_state,
            &input_ptrs,
            tick_output,
            timestamp,
            &produced_words,
            0,
            num_inputs,
        );
    }

    // 5. Push output to versioned ring (ring takes ownership).
    node.versioned
        .push(tick, tick_output, node.output_drop_fn);

    // 6. Update current (state, output) pointers.
    if node.is_stateful {
        // Carry forward: clone output for current (ring owns tick_output).
        let next_output = unsafe { (node.output_clone_fn)(tick_output) };
        unsafe { (node.state_drop_fn)(node.current_state_ptr) };
        unsafe { (node.output_drop_fn)(node.current_output_ptr) };
        node.current_state_ptr = tick_state;
        node.current_output_ptr = next_output;
    } else {
        // Stateless: one-shot clones, drop state (output is in ring).
        unsafe { (node.state_drop_fn)(tick_state) };
    }
}
