# TreeTopology: immutable tree structure (architecture §3, global constraint §9.4).
#
# Topology and tensors are separated: `TreeTopology` is an immutable value object
# (adjacency, parent/child relations, traversal orders); networks (TTNS/TTNO) hold
# a reference to it plus their tensors. Changing geometry means building a new
# `TreeTopology`. Warm-start validation, checkpointing and future MPI subtree
# ownership all rely on its stable `hash`/`==`.
#
# Node identity: `Symbol` at the API boundary, dense `Int` (1:nnodes) internally
# (architecture §10.10 — concrete containers, no Dict{String,Any}).

"""
    TreeTopology(root::Symbol, edges::Vector{Pair{Symbol,Symbol}})

Immutable tree topology. Each edge is given as `parent => child`; the order in
which children appear defines the child (and hence tensor leg) ordering of the
parent node.

Every edge is directed towards the root (architecture §2): the *parent leg* of a
node is its unique outgoing edge.
"""
struct TreeTopology
    ids::Vector{Symbol}               # internal index -> node id
    index::Dict{Symbol,Int}           # node id -> internal index
    parent::Vector{Int}               # parent index, 0 for the root
    children::Vector{Vector{Int}}     # children indices, in leg order
    root::Int
    depth::Vector{Int}                # root has depth 0
end

function TreeTopology(root::Symbol, edges::Vector{Pair{Symbol,Symbol}})
    ids = Symbol[root]
    index = Dict{Symbol,Int}(root => 1)
    parent = Int[0]
    children = Vector{Int}[Int[]]
    for (p, c) in edges
        haskey(index, p) || throw(ArgumentError("parent $p unknown when adding edge $p => $c (edges must be given top-down)"))
        haskey(index, c) && throw(ArgumentError("node $c added twice"))
        push!(ids, c)
        index[c] = length(ids)
        push!(parent, index[p])
        push!(children, Int[])
        push!(children[index[p]], index[c])
    end
    depth = zeros(Int, length(ids))
    for i in 2:length(ids)              # ids are added top-down, so parent[i] < i
        depth[i] = depth[parent[i]] + 1
    end
    return TreeTopology(ids, index, parent, children, 1, depth)
end

# Value semantics: two topologies are equal iff they describe the same labelled tree.
Base.:(==)(a::TreeTopology, b::TreeTopology) =
    a.ids == b.ids && a.parent == b.parent && a.children == b.children
Base.hash(t::TreeTopology, h::UInt) = hash(t.children, hash(t.parent, hash(t.ids, hash(:TreeTopology, h))))

nnodes(t::TreeTopology) = length(t.ids)
nodeid(t::TreeTopology, i::Int) = t.ids[i]
nodeindex(t::TreeTopology, s::Symbol) = t.index[s]
nodeindex(t::TreeTopology, i::Int) = i
isroot(t::TreeTopology, i::Int) = i == t.root
isleaf(t::TreeTopology, i::Int) = isempty(t.children[i])
leaves(t::TreeTopology) = [i for i in 1:nnodes(t) if isleaf(t, i)]
nchildren(t::TreeTopology, i::Int) = length(t.children[i])

"""All neighbours of node `i`: children in leg order, then the parent (if any)."""
function neighbors(t::TreeTopology, i::Int)
    ns = copy(t.children[i])
    t.parent[i] == 0 || push!(ns, t.parent[i])
    return ns
end

"""Position of child `c` in the children list of `p` (leg slot of that edge on `p`)."""
function childslot(t::TreeTopology, p::Int, c::Int)
    k = findfirst(==(c), t.children[p])
    k === nothing && throw(ArgumentError("$(t.ids[c]) is not a child of $(t.ids[p])"))
    return k
end

"""
    edges(t) -> Vector{Tuple{Int,Int}}

All edges as `(child, parent)` tuples, i.e. oriented towards the root.
"""
edges(t::TreeTopology) = [(i, t.parent[i]) for i in 1:nnodes(t) if t.parent[i] != 0]

"""Post-order traversal (children before parents; root last). Valid leaf→root sweep order."""
function postorder(t::TreeTopology)
    order = Int[]
    sizehint!(order, nnodes(t))
    _postorder!(order, t, t.root)
    return order
end
function _postorder!(order::Vector{Int}, t::TreeTopology, i::Int)
    for c in t.children[i]
        _postorder!(order, t, c)
    end
    push!(order, i)
    return order
end

"""Pre-order traversal (parents before children; root first)."""
function preorder(t::TreeTopology)
    order = Int[]
    sizehint!(order, nnodes(t))
    _preorder!(order, t, t.root)
    return order
end
function _preorder!(order::Vector{Int}, t::TreeTopology, i::Int)
    push!(order, i)
    for c in t.children[i]
        _preorder!(order, t, c)
    end
    return order
end

"""Path from `a` to the root, inclusive on both ends."""
function path_to_root(t::TreeTopology, a::Int)
    p = Int[a]
    while t.parent[p[end]] != 0
        push!(p, t.parent[p[end]])
    end
    return p
end

"""
    path_between(t, a, b) -> Vector{Int}

Unique path from node `a` to node `b`, inclusive on both ends (via the lowest
common ancestor). PyTreeNet: `TreeStructure.path_from_to`.
"""
function path_between(t::TreeTopology, a::Int, b::Int)
    pa = path_to_root(t, a)
    pb = path_to_root(t, b)
    inb = Set(pb)
    lca_pos = findfirst(in(inb), pa)::Int
    lca = pa[lca_pos]
    up = pa[1:lca_pos]
    down = reverse!(pb[1:(findfirst(==(lca), pb)::Int - 1)])
    return append!(up, down)
end

"""
    subtree_nodes(t, n, avoiding) -> Vector{Int}

All nodes of the connected component containing `n` when the edge between `n`
and its neighbour `avoiding` is cut. Used for environment ownership/invalidation.
"""
function subtree_nodes(t::TreeTopology, n::Int, avoiding::Int)
    acc = Int[]
    stack = Tuple{Int,Int}[(n, avoiding)]   # (node, came-from)
    while !isempty(stack)
        (i, from) = pop!(stack)
        push!(acc, i)
        for m in neighbors(t, i)
            m == from && continue
            push!(stack, (m, i))
        end
    end
    return acc
end
