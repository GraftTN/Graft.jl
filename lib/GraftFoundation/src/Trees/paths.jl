# Sweep / update path finding (PyTreeNet: core/tree_structure.py + time_evolution path finders).
#
# An *update path* visits every node exactly once; between consecutive update
# sites the orthogonality center is moved along the unique connecting path
# (for TDVP, evolving link tensors backward on every crossed edge).

"""
    tdvp_update_path(t::TreeTopology) -> Vector{Int}

Update path for single-site sweeps (TDVP/DMRG): visits every node exactly once,
children before parents, ending at the root. Consecutive entries are generally
adjacent; where they are not, the caller moves the center along
`path_between`.

PyTreeNet reference: `TDVPUpdatePathFinder.find_path` — it starts from the leaf
deepest in the tree and traverses so that each subtree is finished before its
root is updated. Post-order has the same property; branch-visit order follows
the child leg order.
"""
tdvp_update_path(t::TreeTopology) = postorder(t)

"""
    orth_path_segments(t, path) -> Vector{Vector{Int}}

For each consecutive pair `(path[i], path[i+1])` of an update path, the node
sequence the orthogonality center walks through (inclusive of both endpoints).
"""
orth_path_segments(t::TreeTopology, path::Vector{Int}) =
    [path_between(t, path[i], path[i + 1]) for i in 1:(length(path) - 1)]

"""
    sweep_order(t; reverse=false) -> Vector{Int}

Leaf→root (post-order) or root→leaf (reversed) full sweep order for DMRG-style
back-and-forth sweeps.
"""
sweep_order(t::TreeTopology; reverse::Bool=false) =
    reverse ? Base.reverse(postorder(t)) : postorder(t)
