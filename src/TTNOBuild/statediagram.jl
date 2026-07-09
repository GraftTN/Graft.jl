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
# TODO(port, §4b): PyTreeNet's bipartite-graph optimization + symbolic Gaussian
#   elimination can compress further (shared prefixes/suffixes); add as a
#   `compress!` pass on the assembled TTNO or on the diagram.
# TODO(M0 fermion path): sector-graded virtual legs — each ACTIVE channel
#   carries the charge flux of its restriction; requires charged SiteOps
#   (Symbolic TODO) and GradedSpace bond assembly. Everything below assumes
#   trivial-sector (dense) physical spaces, exactly like PyTreeNet.

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
tensors). Trivial-sector spaces only for now (see TODOs above). `hermitian`
sets the `ishermitian` trait on the result — a wrong `true` is a caller bug
(§9.8).
"""
function ttno_from_opsum(H::OpSum, topo::TreeTopology, phys::Dict{Symbol,S};
                         elt::Type{<:Number}=ComplexF64,
                         hermitian::Bool=false) where {S<:ElementarySpace}
    S === ComplexSpace ||
        throw(ArgumentError("TODO(M0 fermion path): graded TTNO assembly — only trivial-sector spaces supported yet"))
    t = topo
    E = _Euler(t)
    N = nnodes(t)

    # preprocess terms: node-indexed factor maps
    terms = map(H.terms) do term
        ops = Dict{Int,SiteOp}()
        for so in term.ops
            n = nodeindex(t, so.site)
            haskey(phys, so.site) || throw(ArgumentError("term factor on $(so.site), which has no physical space"))
            ops[n] = so
        end
        (; coeff=term.coeff, ops, nodes=sort!(collect(keys(ops))))
    end
    isempty(terms) && throw(ArgumentError("empty OpSum"))

    opmats = Dict{Tuple{Int,Symbol},AbstractTensorMap}()   # (node, opname) -> P ← P
    used = [Set{_ChannelKey}() for _ in 1:N]               # channel usage, edge keyed by child
    # entry accumulator: (node, αkeys, βkey, opname) => coefficient
    entries = Dict{Tuple{Int,Tuple,_ChannelKey,Symbol},Any}()

    restrict(nodes, ops, c) = _rkey([(x, ops[x].name) for x in nodes if _insub(E, x, c)])

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
            αkeys = map(t.children[n]) do c
                r = restrict(tm.nodes, tm.ops, c)
                isempty(r) ? :idle : r
            end
            complete = all(x -> _insub(E, x, n), tm.nodes)
            βkey::_ChannelKey = complete ? :done :
                _rkey([(x, tm.ops[x].name) for x in tm.nodes if _insub(E, x, n)])
            opname = haskey(tm.ops, n) ? tm.ops[n].name : :I
            for (c, k) in zip(t.children[n], αkeys)
                push!(used[c], k)
            end
            t.parent[n] == 0 ? (@assert complete) : push!(used[n], βkey)
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
            push!(used[t.parent[n]], :done)
        end
    end
    for p in 1:N, c in t.children[p]
        if :done in used[c]
            for c2 in t.children[p]
                c2 == c || push!(used[c2], :idle)
            end
        end
    end
    # IDLE propagation downward: an idle edge needs identity flowing through
    # every node below it.
    for n in preorder(t)
        t.parent[n] == 0 && continue
        if :idle in used[n]
            for c in t.children[n]
                push!(used[c], :idle)
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
    for c in 1:N
        t.parent[c] == 0 && continue
        i = 0
        :idle in used[c] && (chindex[c][:idle] = (i += 1))
        :done in used[c] && (chindex[c][:done] = (i += 1))
        for r in sort!([k for k in used[c] if k isa Tuple]; by=string)
            chindex[c][r] = (i += 1)
        end
    end
    χ(c) = max(length(chindex[c]), 1)

    # assemble dense per-node tensors
    unit = oneunit(S)
    tensors = map(1:N) do n
        K = nchildren(t, n)
        hp = haskey(phys, nodeid(t, n))
        P = hp ? phys[nodeid(t, n)] : nothing
        d = hp ? dim(P) : 1
        χp = t.parent[n] == 0 ? 1 : χ(n)
        dims = (ntuple(k -> χ(t.children[n][k]), K)..., (hp ? (d, d) : ())..., χp)
        W = zeros(elt, dims)
        for ((m, αkeys, βkey, opname), coeff) in entries
            m == n || continue
            αidx = ntuple(k -> chindex[t.children[n][k]][αkeys[k]], K)
            βidx = t.parent[n] == 0 ? 1 : chindex[n][βkey]
            if hp
                mat = opname == :I ? _eye(elt, d) :
                    elt.(reshape(convert(Array, opmats[(n, opname)]), d, d))
                view(W, αidx..., :, :, βidx) .+= elt(coeff) .* mat
            else
                opname == :I || throw(ArgumentError("operator factor on branching node $(nodeid(t, n))"))
                W[αidx..., βidx] += elt(coeff)
            end
        end
        cods = S[ComplexSpace(χ(c)) for c in t.children[n]]
        hp && push!(cods, P)
        cod = isempty(cods) ? one(unit) : reduce(⊗, cods)
        Vp = ComplexSpace(χp)
        TensorMap(W, cod ← (hp ? P ⊗ Vp : ProductSpace(Vp)))
    end

    return TTNO(t, tensors; ishermitian=hermitian)
end

_eye(::Type{T}, d::Int) where {T} = T[i == j ? one(T) : zero(T) for i in 1:d, j in 1:d]
