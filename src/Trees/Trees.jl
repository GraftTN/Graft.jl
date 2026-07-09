"""
L1 — Tree topology and traversal (architecture §3).

Topology is *data*: an immutable `TreeTopology` value object separated from any
tensors (PyTreeNet: core/graph_node.py + core/tree_structure.py). Geometry
constructors (`Geometries`) return plain topologies; sweep engines never branch
on which geometry they run (design principle §0.1).
"""
module Trees

export TreeTopology, nnodes, nodeid, nodeindex, isroot, isleaf, leaves,
    nchildren, neighbors, childslot, edges, postorder, preorder, path_to_root,
    path_between, subtree_nodes,
    tdvp_update_path, orth_path_segments, sweep_order,
    mps_topology, star_topology, binary_topology, fork_topology, mount_chain,
    is_t3ns

include("topology.jl")
include("paths.jl")
include("geometries.jl")

end # module Trees
