# TTNO assembly from an OpSum — tree finite-state-machine construction.
#
# Channel structure per edge e = (child c, parent p) — cutting the tree at e,
# every term t is classified by its restriction to the subtree below e:
#   * IDLE:  t has no site below e            (identity flows below)
#   * DONE:  t lies entirely below e          (identity flows above; all such
#            terms share one channel, coefficients applied at completion)
#   * ACTIVE(r): proper nonempty restriction r (terms with identical
#            restriction share the channel — the "state diagram" merge)
# A term's coefficient is applied exactly once, at its completion node (the
# lowest node whose subtree contains all its sites). This reproduces the
# channels of PyTreeNet's state-diagram pipeline (single-term diagrams +
# hyperedge/vertex merging) for product-term Hamiltonians.
#
# TODO(port, §4b): PyTreeNet's edge-cut compression pass can beat the channel
#   construction when term combinations are linearly dependent (not just
#   identical). Port order (per state_diagram.py::from_hamiltonian_modified):
#   per edge cut, build the exact-rational Γ bond matrix (rows/cols = child-/
#   parent-side hyperedges), symbolic Gaussian elimination over Rational
#   (symbolic_gaussian_elimination_fraction.py), then Kőnig minimum vertex
#   cover via Hopcroft–Karp (bipartite_graph.py, generic and portable
#   verbatim); keep whichever of raw-Γ / eliminated-Γ gives the smaller cover
#   (TTNOFinder.SGE semantics). Run bottom-up level by level.
# Sector-graded virtual legs: each ACTIVE channel carries the fused charge flux
# of its restriction, while IDLE/DONE channels are trivial. The trivial-sector
# path below still uses dense `ComplexSpace(χ)` channels.

# subtree membership via Euler (DFS in/out) intervals
struct _Euler
    tin::Vector{Int}
    tout::Vector{Int}
end
function _Euler(t::TreeTopology)
    tin = zeros(Int, nnodes(t))
    tout = zeros(Int, nnodes(t))
    clock = Ref(0)
    function dfs(n)
        tin[n] = (clock[] += 1)
        for c in t.children[n]
            dfs(c)
        end
        tout[n] = (clock[] += 1)
    end
    dfs(t.root)
    return _Euler(tin, tout)
end
_insub(E::_Euler, x::Int, c::Int) = E.tin[c] <= E.tin[x] <= E.tout[c]

const _ChannelKey = Union{Symbol,Tuple}   # :idle, :done, or sorted ((node, opname), …)

_rkey(pairs::Vector{Tuple{Int,Symbol}}) = Tuple(sort(pairs))

"""
    ttno_from_opsum(H::OpSum, topo, phys; elt=ComplexF64, hermitian=false) -> TTNO

Assemble a TTNO from a sum of product terms. `phys :: Dict{Symbol,<:ElementarySpace}`
gives the physical space of every site-carrying node (others become branching
tensors). For graded abelian spaces, charged [`SiteOp`](@ref) factors thread
their fused restriction charge through TTNO virtual channels. `hermitian` sets
the `ishermitian` trait on the result — a wrong `true` is a caller bug (§9.8).
"""
function ttno_from_opsum(H::OpSum, topo::TreeTopology, phys::Dict{Symbol,S};
                         elt::Type{<:Number}=ComplexF64,
                         hermitian::Bool=false) where {S<:ElementarySpace}
    t = topo
    E = _Euler(t)
    N = nnodes(t)
    graded = S !== ComplexSpace
    unit_sector = graded ? one(sectortype(first(values(phys)))) : nothing

    # preprocess terms: node-indexed factor maps
    terms = map(H.terms) do term
        ops = Dict{Int,SiteOp}()
        for so in term.ops
            n = nodeindex(t, so.site)
            haskey(phys, so.site) || throw(ArgumentError("term factor on $(so.site), which has no physical space"))
            spacetype(codomain(so.op)[1]) == S ||
                throw(ArgumentError("term factor on $(so.site) uses a physical-space symmetry incompatible with `phys`"))
            ops[n] = so
        end
        (; coeff=term.coeff, ops, nodes=sort!(collect(keys(ops))))
    end
    isempty(terms) && throw(ArgumentError("empty OpSum"))

    opmats = Dict{Tuple{Int,Symbol},AbstractTensorMap}()   # (node, opname) -> P ← P
    used = [Set{_ChannelKey}() for _ in 1:N]               # channel usage, edge keyed by child
    chsector = [Dict{_ChannelKey,Any}() for _ in 1:N]       # channel sector, edge keyed by child
    # entry accumulator: (node, αkeys, βkey, opname) => coefficient
    entries = Dict{Tuple{Int,Tuple,_ChannelKey,Symbol},Any}()

    restrict(nodes, ops, c) = _rkey([(x, ops[x].name) for x in nodes if _insub(E, x, c)])
    restriction_charge(nodes, ops, c) =
        _fuse_charges((charge(ops[x]) for x in nodes if _insub(E, x, c)), unit_sector)
    function register!(edge::Int, key::_ChannelKey, q)
        push!(used[edge], key)
        if graded
            old = get(chsector[edge], key, nothing)
            if old === nothing
                chsector[edge][key] = q
            elseif old != q
                throw(ArgumentError("state-diagram channel $key on edge $(nodeid(t, edge)) has inconsistent charges $old and $q"))
            end
        end
        return key
    end

    for tm in terms
        for so in values(tm.ops)
            get!(opmats, (nodeindex(t, so.site), so.name), so.op)
        end
        lca = tm.nodes[1]
        while !all(x -> _insub(E, x, lca), tm.nodes)
            lca = t.parent[lca]
        end
        # per-term entries exist at nodes n in subtree(lca) where the term is
        # not confined to a single child branch: exactly lca itself and the
        # "spine" nodes below it where the term is partially present.
        active = Set{Int}(tm.nodes)
        # climb from every site towards lca, collecting pass-through nodes
        for s in tm.nodes
            x = s
            while x != lca
                x = t.parent[x]
                push!(active, x)
            end
        end
        for n in sort!(collect(active))
            # skip nodes where the whole term sits inside one child branch
            any(c -> all(x -> _insub(E, x, c), tm.nodes), t.children[n]) && continue
            αkeys = _ChannelKey[]
            αcharges = Any[]
            for c in t.children[n]
                r = restrict(tm.nodes, tm.ops, c)
                if isempty(r)
                    push!(αkeys, :idle)
                    graded && push!(αcharges, unit_sector)
                else
                    push!(αkeys, r)
                    graded && push!(αcharges, restriction_charge(tm.nodes, tm.ops, c))
                end
            end
            complete = all(x -> _insub(E, x, n), tm.nodes)
            βkey::_ChannelKey = complete ? :done :
                _rkey([(x, tm.ops[x].name) for x in tm.nodes if _insub(E, x, n)])
            βcharge = graded ? (complete ? unit_sector : restriction_charge(tm.nodes, tm.ops, n)) : nothing
            opname = haskey(tm.ops, n) ? tm.ops[n].name : :I
            if graded
                localq = haskey(tm.ops, n) ? charge(tm.ops[n]) : unit_sector
                lhs = _fuse_charges(αcharges, unit_sector)
                lhs = _fuse_charge(lhs, localq)
                lhs == βcharge ||
                    throw(ArgumentError("state-diagram charge mismatch at $(nodeid(t, n)): children/local fuse to $lhs but parent channel has $βcharge"))
            end
            for (c, k, q) in zip(t.children[n], αkeys, graded ? αcharges : fill(nothing, length(αkeys)))
                register!(c, k, q)
            end
            t.parent[n] == 0 ? (@assert complete) : register!(n, βkey, βcharge)
            key = (n, Tuple(αkeys), βkey, opname)
            if complete
                entries[key] = get(entries, key, zero(tm.coeff)) + tm.coeff
            else
                entries[key] = 1                # pass-through/merge: identical for all sharers
            end
        end
    end

    # DONE propagation upward: a completed term must ride the done channel on
    # every edge up to the root; done-pass entries reference idle on siblings.
    for n in postorder(t)
        t.parent[n] == 0 && continue
        if :done in used[n] && t.parent[t.parent[n]] != 0
            register!(t.parent[n], :done, unit_sector)
        end
    end
    for p in 1:N, c in t.children[p]
        if :done in used[c]
            for c2 in t.children[p]
                c2 == c || register!(c2, :idle, unit_sector)
            end
        end
    end
    # IDLE propagation downward: an idle edge needs identity flowing through
    # every node below it.
    for n in preorder(t)
        t.parent[n] == 0 && continue
        if :idle in used[n]
            for c in t.children[n]
                register!(c, :idle, unit_sector)
            end
        end
    end
    # transport entries
    for p in 1:N
        for c in t.children[p]
            if :done in used[c]
                αkeys = _ChannelKey[c2 == c ? :done : :idle for c2 in t.children[p]]
                entries[(p, Tuple(αkeys), :done, :I)] = 1
            end
        end
    end
    for n in 1:N
        t.parent[n] == 0 && continue
        if :idle in used[n]
            αkeys = ntuple(_ -> :idle, nchildren(t, n))
            entries[(n, αkeys, :idle, :I)] = 1
        end
    end

    # channel index assignment (deterministic: idle, done, then sorted actives)
    chindex = [Dict{_ChannelKey,Int}() for _ in 1:N]
    chcoord = [Dict{_ChannelKey,Int}() for _ in 1:N]
    vspaces = Vector{S}(undef, N)
    for c in 1:N
        if t.parent[c] == 0
            vspaces[c] = oneunit(S)
            continue
        end
        ordered = _ChannelKey[]
        :idle in used[c] && push!(ordered, :idle)
        :done in used[c] && push!(ordered, :done)
        for r in sort!([k for k in used[c] if k isa Tuple]; by=string)
            push!(ordered, r)
        end
        for (i, k) in enumerate(ordered)
            chindex[c][k] = i
        end
        if graded
            vspaces[c], chcoord[c] = _channel_layout(S, ordered, chsector[c], unit_sector)
        else
            vspaces[c] = ComplexSpace(max(length(ordered), 1))
            for (i, k) in enumerate(ordered)
                chcoord[c][k] = i
            end
        end
    end

    # assemble dense per-node tensors
    unit = oneunit(S)
    tensors = map(1:N) do n
        K = nchildren(t, n)
        hp = haskey(phys, nodeid(t, n))
        P = hp ? phys[nodeid(t, n)] : nothing
        d = hp ? dim(P) : 1
        χp = t.parent[n] == 0 ? 1 : dim(vspaces[n])
        dims = (ntuple(k -> dim(vspaces[t.children[n][k]]), K)..., (hp ? (d, d) : ())..., χp)
        W = zeros(elt, dims)
        for ((m, αkeys, βkey, opname), coeff) in entries
            m == n || continue
            αidx = ntuple(k -> chcoord[t.children[n][k]][αkeys[k]], K)
            βidx = t.parent[n] == 0 ? 1 : chcoord[n][βkey]
            if hp
                mat = _siteop_matrix(opmats, n, opname, elt, d)
                view(W, αidx..., :, :, βidx) .+= elt(coeff) .* mat
            else
                opname == :I || throw(ArgumentError("operator factor on branching node $(nodeid(t, n))"))
                W[αidx..., βidx] += elt(coeff)
            end
        end
        cods = S[]
        for c in t.children[n]
            push!(cods, vspaces[c])
        end
        hp && push!(cods, P)
        cod = isempty(cods) ? one(unit) : reduce(⊗, cods)
        Vp = t.parent[n] == 0 ? oneunit(S) : vspaces[n]
        TensorMap(W, cod ← (hp ? P ⊗ Vp : ProductSpace(Vp)))
    end

    return TTNO(t, tensors; ishermitian=hermitian)
end

_eye(::Type{T}, d::Int) where {T} = T[i == j ? one(T) : zero(T) for i in 1:d, j in 1:d]

function _siteop_matrix(opmats, n::Int, opname::Symbol, ::Type{T}, d::Int) where {T}
    opname == :I && return _eye(T, d)
    op = opmats[(n, opname)]
    if numout(op) == 1 && numin(op) == 1
        return T.(reshape(convert(Array, op), d, d))
    elseif numout(op) == 1 && numin(op) == 2 && dim(domain(op)[2]) == 1
        arr = reshape(convert(Array, op), d, d, :)
        size(arr, 3) == 1 || throw(ArgumentError("charged SiteOp charge leg must be one-dimensional"))
        return T.(arr[:, :, 1])
    else
        throw(ArgumentError("SiteOp tensor for `$opname` must be `P ← P` or charged `P ← P ⊗ C`"))
    end
end

function _fuse_charge(a, b)
    a === nothing && return nothing
    fused = a ⊗ b
    length(fused) == 1 ||
        throw(ArgumentError("non-abelian TTNO charge bookkeeping needs SU2Reduce/graded fusion-tree support (TODO(M3))"))
    return only(fused)
end

function _fuse_charges(qs, unit_sector)
    unit_sector === nothing && return nothing
    qtot = unit_sector
    for q in qs
        qtot = _fuse_charge(qtot, q)
    end
    return qtot
end

function _channel_layout(::Type{S}, ordered::Vector{_ChannelKey},
                         sector_of::Dict{_ChannelKey,Any}, unit_sector) where {S<:ElementarySpace}
    if isempty(ordered)
        return oneunit(S), Dict{_ChannelKey,Int}()
    end
    Q = typeof(unit_sector)
    groups = Dict{Q,Vector{_ChannelKey}}()
    for key in ordered
        q = sector_of[key]
        dim(Vect[Q](q => 1)) == 1 ||
            throw(ArgumentError("TTNO virtual channels currently require abelian one-dimensional sectors (TODO(M3))"))
        push!(get!(groups, q, _ChannelKey[]), key)
    end
    pairs = [q => length(groups[q]) for q in sort!(collect(keys(groups)); by=string)]
    V = Vect[Q](pairs...)
    coord = Dict{_ChannelKey,Int}()
    offset = 0
    for q in sectors(V)
        for (j, key) in enumerate(groups[q])
            coord[key] = offset + j
        end
        offset += length(groups[q])
    end
    return V, coord
end
