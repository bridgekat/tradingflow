//! Layer 1: typed, passive graph and builder.

mod flat;
mod graph;
mod handles;
mod interface;
mod segment;

pub use flat::{FlatRead, FlatWrite};
pub use graph::{Builder, Graph};
pub use handles::{HandlesInterface, InterfaceHandles, NodeHandle, PortHandle};
pub use interface::{Interface, Pass, Port, Ports, Ref, Slice, Val};
pub use segment::{Operator, Segment};
