mod flat;
mod graph;
mod handles;
mod interface;
mod operator;
mod segment;
mod source;

pub use flat::{FlatRead, FlatWrite};
pub use graph::{Builder, Graph};
pub use handles::{HandlesInterface, InterfaceHandles, NodeHandle, PortHandle};
pub use interface::{
    Interface, Port, Ports, Ref, RefPort, RefPorts, Scalar, Slice, Value, ViewPort, ViewPorts,
};
pub use operator::Operator;
pub use segment::{Arr, Comp, Fork, Id, Left, Par, Right, Segment, SegmentExt};
pub use source::{RefSource, Source, ViewSource};
