# Geometry constructors (architecture §6.1): tree layouts are runtime data, not code.
# All constructors return a plain `TreeTopology`; the sweep engines never know
# which geometry they run on (design principle §0.1).
#
# PyTreeNet counterparts: special_ttn/{mps,binary,star,fttn}.py.

"""
    mps_topology(n; prefix=:site) -> TreeTopology

Linear chain of `n` nodes rooted at the *last* site: `site1 - site2 - … - siteN(root)`.
An MPS is a TTNS on this topology.
"""
function mps_topology(n::Int; prefix::Symbol=:site)
    ids = [Symbol(prefix, i) for i in 1:n]
    edges = [ids[i + 1] => ids[i] for i in (n - 1):-1:1]
    return TreeTopology(ids[n], edges)
end

"""
    star_topology(nbranches, branchlength; center=:center, prefix=:site) -> TreeTopology

Star: a central node with `nbranches` chains of length `branchlength` attached.
The standard single-impurity layout (impurity at the center, bath chains/stars
as branches).
"""
function star_topology(nbranches::Int, branchlength::Int; center::Symbol=:center, prefix::Symbol=:b)
    edges = Pair{Symbol,Symbol}[]
    for b in 1:nbranches
        prev = center
        for l in 1:branchlength
            node = Symbol(prefix, b, :_, l)
            push!(edges, prev => node)
            prev = node
        end
    end
    return TreeTopology(center, edges)
end

"""
    binary_topology(depth; prefix=:n) -> TreeTopology

Perfect binary tree of the given depth (root at depth 0).
PyTreeNet: `special_ttn/binary.py`.
"""
function binary_topology(depth::Int; prefix::Symbol=:n)
    root = Symbol(prefix, "0_1")
    edges = Pair{Symbol,Symbol}[]
    for d in 1:depth, k in 1:(2^d)
        child = Symbol(prefix, d, :_, k)
        parent = Symbol(prefix, d - 1, :_, (k + 1) ÷ 2)
        push!(edges, parent => child)
    end
    return TreeTopology(root, edges)
end

"""
    fork_topology(nteeth, toothlength; prefix=:spine) -> TreeTopology

Fork / comb / FTPS layout: a spine of `nteeth` nodes, each carrying one chain
("tooth") of `toothlength` nodes. Multi-orbital impurity layout: one spine node
per orbital, bath chain as its tooth (Hund coupling runs along the spine).
"""
function fork_topology(nteeth::Int, toothlength::Int; prefix::Symbol=:spine)
    root = Symbol(prefix, 1)
    edges = Pair{Symbol,Symbol}[]
    for s in 2:nteeth
        push!(edges, Symbol(prefix, s - 1) => Symbol(prefix, s))
    end
    for s in 1:nteeth
        prev = Symbol(prefix, s)
        for l in 1:toothlength
            node = Symbol(:tooth, s, :_, l)
            push!(edges, prev => node)
            prev = node
        end
    end
    return TreeTopology(root, edges)
end

"""
    mount_chain(topo, at, len; prefix) -> TreeTopology

Return a new topology with a length-`len` chain mounted below node `at`.
`prefix` names the new nodes as `Symbol(prefix, i)`. A boson leaf is
`len == 1`. The input topology is unchanged (§9.4).
"""
mount_chain(topo::TreeTopology, at::Symbol, len::Int; prefix::Symbol) =
    mount_chain(topo, nodeindex(topo, at), len; prefix)
function mount_chain(topo::TreeTopology, at::Int, len::Int; prefix::Symbol)
    len >= 0 || throw(ArgumentError("mounted chain length must be nonnegative"))
    1 <= at <= nnodes(topo) || throw(ArgumentError("invalid mount node index $at"))
    len == 0 && return topo

    new_edges = Pair{Symbol,Symbol}[
        nodeid(topo, topo.parent[i]) => nodeid(topo, i)
        for i in 2:nnodes(topo)
    ]
    existing = Set(topo.ids)
    prev = nodeid(topo, at)
    for i in 1:len
        node = Symbol(prefix, i)
        node in existing && throw(ArgumentError("mounted node id $node already exists"))
        push!(existing, node)
        push!(new_edges, prev => node)
        prev = node
    end
    return TreeTopology(nodeid(topo, topo.root), new_edges)
end

"""
    is_t3ns(t::TreeTopology; physical=Symbol[]) -> Bool

T3NS constraint predicate (architecture §6.1): every tensor has at most 3 legs.
`physical` lists the nodes carrying a physical leg (T3NS separates physical
and branching tensors, so a degree-3 junction must be physless). Implemented
as a predicate, not a type, so sweep engines stay geometry-blind.
"""
function is_t3ns(t::TreeTopology; physical::AbstractVector{Symbol}=Symbol[])
    phys = Set(physical)
    return all(1:nnodes(t)) do i
        nlegs = nchildren(t, i) + (isroot(t, i) ? 0 : 1) + (nodeid(t, i) in phys ? 1 : 0)
        nlegs <= 3
    end
end

# TODO(M5+): cayley_topology (Bethe lattice, coordination z), interleaved orderings,
# impurity-specific geometry builders from Partition (§6.2).
