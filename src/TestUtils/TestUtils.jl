"""
Cross-cutting — test infrastructure (architecture §1: `pytreenet.random` +
`exact_*` counterparts). ED cross-validation is a merge requirement (§9.11):
every kernel ships a ≲16-site tree test against exact diagonalization plus a
gauge-invariance property test.

Site-ordering convention for dense references: physical sites appear in
*internal node-index order* (i.e. `TreeTopology` insertion order), first index
fastest (Julia column-major). `dense_hamiltonian` follows the same convention,
so `to_dense(ψ)' * dense_hamiltonian(H, ψ) * to_dense(ψ)` matches `expect`.
"""
module TestUtils

using Random
using LinearAlgebra: LinearAlgebra, eigen, Hermitian
using ..Backend
using ..Trees
using ..Networks
using ..Symbolic

export random_ttns, product_ttns, canonicalize!, to_dense, dense_hamiltonian,
    exact_groundstate, exact_evolve, physical_sites

"""
    canonicalize!(ψ; center=ψ.topo.root) -> ψ

Bring an arbitrary (un-gauged) TTNS into canonical form with the given center:
leaf→root QR sweep (PyTreeNet: core/canonical_form.py), then a center move.
"""
function canonicalize!(ψ::TTNS, center_::Int=ψ.topo.root)
    t = ψ.topo
    for n in postorder(t)
        n == t.root && continue
        Q, C = left_orth(ψ.tensors[n])
        ψ.tensors[n] = Q
        p = t.parent[n]
        ψ.tensors[p] = absorb_on_leg(ψ.tensors[p], C, childslot(t, p, n))
    end
    ψ.center = t.root
    return move_center!(ψ, center_)
end
canonicalize!(ψ::TTNS, center_::Symbol) = canonicalize!(ψ, nodeindex(ψ.topo, center_))

"""
    random_ttns(rng, T, topo, phys, bond; center=topo.root) -> TTNS

Random canonical TTNS. `phys`: `Dict{Symbol,<:ElementarySpace}` — nodes absent
from the Dict get no physical leg (pure branching tensors). `bond`: an
`ElementarySpace` used for every edge, or a `Dict{Symbol,<:ElementarySpace}`
keyed by the child node of each edge. RNG is explicit (§9.6).
"""
function random_ttns(rng::AbstractRNG, ::Type{T}, topo::TreeTopology,
                     phys::Dict{Symbol,S}, bond;
                     center=topo.root) where {T<:Number,S<:ElementarySpace}
    unit = oneunit(S)
    edgespace(n) = bond isa ElementarySpace ? bond : bond[nodeid(topo, n)]
    tensors = map(1:nnodes(topo)) do n
        cod = Vector{S}()
        for c in topo.children[n]
            push!(cod, edgespace(c))
        end
        p = get(phys, nodeid(topo, n), nothing)
        p === nothing || push!(cod, p)
        dom = topo.parent[n] == 0 ? unit : edgespace(n)
        isempty(cod) && throw(ArgumentError("node $(nodeid(topo, n)) has no legs besides its parent"))
        randn(rng, T, reduce(⊗, cod) ← dom)
    end
    ψ = TTNS(topo, tensors, topo.root)
    canonicalize!(ψ, center isa Symbol ? nodeindex(topo, center) : center)
    normalize!(ψ)
    return ψ
end

"""
    product_ttns(T, topo, states::Dict{Symbol,<:AbstractVector}) -> TTNS
    product_ttns(T, topo, phys, sectors) -> TTNS

Product state on trivial-sector spaces: `states[site]` is the local state
vector; nodes absent from the Dict are branching tensors. For graded spaces,
`phys` gives each physical space and `sectors[site]` gives the local basis
sector (or `sector => degeneracy_index`). Bond sectors carry the accumulated
subtree charge toward the root.
"""
function product_ttns(::Type{T}, topo::TreeTopology,
                      states::Dict{Symbol,<:AbstractVector}) where {T<:Number}
    unit = ℂ^1
    tensors = map(1:nnodes(topo)) do n
        K = nchildren(topo, n)
        v = get(states, nodeid(topo, n), nothing)
        cod = K == 0 ? one(unit) : reduce(⊗, ntuple(_ -> unit, K))
        if v === nothing
            arr = ones(T, ntuple(_ -> 1, K + 1))
            TensorMap(arr, cod ← unit)
        else
            arr = reshape(T.(v), ntuple(_ -> 1, K)..., length(v), 1)
            TensorMap(arr, cod ⊗ ℂ^length(v) ← unit)
        end
    end
    ψ = TTNS(topo, tensors, topo.root)
    return ψ
end

function product_ttns(::Type{T}, topo::TreeTopology,
                      phys::Dict{Symbol,S}, basis) where {T<:Number, S<:ElementarySpace}
    S === ComplexSpace && throw(ArgumentError("graded product_ttns expects graded physical spaces; use the vector-state method for ComplexSpace"))
    isempty(phys) && throw(ArgumentError("graded product_ttns needs at least one physical site"))
    Q = sectortype(first(values(phys)))
    unitq = one(Q)
    qlocal = fill(unitq, nnodes(topo))
    ilocal = fill(1, nnodes(topo))
    for (site, P) in phys
        spacetype(P) == S || throw(ArgumentError("all physical spaces must share one symmetry type"))
        n = nodeindex(topo, site)
        q, i = _sector_index(basis[site])
        q isa Q || throw(ArgumentError("local sector for $site has type $(typeof(q)); expected $Q"))
        1 <= i <= dim(P, q) || throw(ArgumentError("local sector index $i out of range for $site sector $q"))
        qlocal[n] = q
        ilocal[n] = i
    end

    qsub = fill(unitq, nnodes(topo))
    for n in postorder(topo)
        q = qlocal[n]
        for c in topo.children[n]
            q = _fuse_one(q, qsub[c])
        end
        qsub[n] = q
    end

    edgespace(n) = Vect[Q](qsub[n] => 1)
    rootspace = Vect[Q](qsub[topo.root] => 1)
    tensors = map(1:nnodes(topo)) do n
        cods = S[]
        for c in topo.children[n]
            push!(cods, edgespace(c))
        end
        hp = haskey(phys, nodeid(topo, n))
        hp && push!(cods, phys[nodeid(topo, n)])
        cod = isempty(cods) ? oneunit(S) : reduce(⊗, cods)
        dom = topo.parent[n] == 0 ? rootspace : edgespace(n)
        A = zeros(T, cod ← dom)
        _, codcoord = _basis_coord(cods, unitq)
        _, domcoord = _basis_coord(S[dom], unitq)
        codidx = hp ? (ntuple(_ -> 1, nchildren(topo, n))..., _sector_offset(phys[nodeid(topo, n)], qlocal[n], ilocal[n])) :
            ntuple(_ -> 1, nchildren(topo, n))
        cq, row = codcoord[codidx]
        dq, col = domcoord[(1,)]
        cq == dq || throw(ArgumentError("internal product_ttns charge mismatch at $(nodeid(topo, n)): codomain $cq vs domain $dq"))
        for (q, b) in blocks(A)
            q == cq && (b[row, col] = one(T))
        end
        A
    end
    return TTNS(topo, tensors, topo.root)
end

_sector_index(x::Pair) = x.first, x.second
_sector_index(x::Tuple{Any,Int}) = x[1], x[2]
_sector_index(q) = q, 1

function _fuse_one(a, b)
    fused = a ⊗ b
    length(fused) == 1 || throw(ArgumentError("graded product_ttns currently supports abelian one-channel fusion"))
    return only(fused)
end

function _sector_offset(V::ElementarySpace, q, i::Int)
    offset = 0
    for s in sectors(V)
        s == q && return offset + i
        offset += dim(V, s)
    end
    throw(ArgumentError("sector $q is absent from $V"))
end

function _basis_coord(legs::Vector{S}, unitq) where {S<:ElementarySpace}
    groups = Dict{typeof(unitq),Vector{Tuple}}()
    coord = Dict{Tuple,Tuple{typeof(unitq),Int}}()
    if isempty(legs)
        groups[unitq] = [()]
        coord[()] = (unitq, 1)
        return groups, coord
    end
    legqs = Vector{Vector{typeof(unitq)}}()
    for V in legs
        qs = typeof(unitq)[]
        for q in sectors(V), _ in 1:dim(V, q)
            push!(qs, q)
        end
        push!(legqs, qs)
    end
    for I in CartesianIndices(Tuple(length(qs) for qs in legqs))
        idx = Tuple(I)
        q = unitq
        for k in eachindex(idx)
            q = _fuse_one(q, legqs[k][idx[k]])
        end
        rows = get!(groups, q, Tuple[])
        push!(rows, idx)
        coord[idx] = (q, length(rows))
    end
    return groups, coord
end

"""Internal-order list of nodes that carry a physical leg."""
physical_sites(ψ::TTNS) = [n for n in 1:nnodes(ψ.topo) if hasphys(ψ, n)]
physical_sites(O::TTNO) = [n for n in 1:nnodes(O.topo) if hasphys(O, n)]

"""
    to_dense(ψ::TTNS) -> Vector

Full state vector by brute-force contraction (small trees only). Site order:
see module docstring. The root parent leg is left open before flattening, so
fixed nontrivial one-dimensional root sectors used by charged states are
represented instead of being capped to the trivial sector.
"""
function to_dense(ψ::TTNS)
    t = ψ.topo
    sites = physical_sites(ψ)
    nopen = 0
    openlabel = Dict(n => -(nopen += 1) for n in sites)
    rootlabel = -(nopen + 1)
    tensors = Any[]
    indices = Vector{Int}[]
    for n in 1:nnodes(t)
        A = ψ.tensors[n]
        idx = zeros(Int, numind(A))
        for (k, c) in enumerate(t.children[n])
            idx[k] = c                    # edge label = child index (unique, positive)
        end
        hasphys(ψ, n) && (idx[physleg(ψ, n)] = openlabel[n])
        if t.parent[n] == 0
            idx[end] = rootlabel
        else
            idx[end] = n
        end
        push!(tensors, A); push!(indices, idx)
    end
    res = ncon(tensors, indices, falses(length(tensors)))
    return vec(convert(Array, res))
end

"""
    to_dense(O::TTNO) -> Matrix

Dense matrix of a TTNO (small trees only), same site ordering as `to_dense(ψ)`.
"""
function to_dense(O::TTNO)
    t = O.topo
    sites = physical_sites(O)
    outlabel = Dict(n => -i for (i, n) in enumerate(sites))
    inlabel = Dict(n => -(length(sites) + i) for (i, n) in enumerate(sites))
    tensors = Any[]
    indices = Vector{Int}[]
    for n in 1:nnodes(t)
        W = O.tensors[n]
        idx = zeros(Int, numind(W))
        for (k, c) in enumerate(t.children[n])
            idx[k] = c
        end
        if hasphys(O, n)
            K = nchildren(t, n)
            idx[K + 1] = outlabel[n]
            idx[K + 2] = inlabel[n]
        end
        if t.parent[n] == 0
            cap = ones_tensor(eltype(O), ProductSpace(domain(W)[numin(W)]))
            lbl = nnodes(t) + 1
            idx[end] = lbl
            push!(tensors, cap); push!(indices, [lbl])
        else
            idx[end] = n
        end
        push!(tensors, W); push!(indices, idx)
    end
    res = ncon(tensors, indices, falses(length(tensors)))
    arr = convert(Array, res)
    d = Int(sqrt(length(arr)))
    return reshape(arr, d, d)
end

"""
    dense_hamiltonian(H::OpSum, ψ) -> Matrix
    dense_hamiltonian(H::OpSum, topo, phys) -> Matrix

Dense matrix of an `OpSum` on the physical sites of `ψ` (identity padding),
kron-ordered to match `to_dense`. Small systems only.
"""
function dense_hamiltonian(H::OpSum, ψ::TTNS)
    phys = Dict(nodeid(ψ.topo, n) => physspace(ψ, n) for n in physical_sites(ψ))
    return dense_hamiltonian(H, ψ.topo, phys)
end

function dense_hamiltonian(H::OpSum, t::TreeTopology, phys::Dict{Symbol,<:ElementarySpace})
    sites = [n for n in 1:nnodes(t) if haskey(phys, nodeid(t, n))]
    dims = [dim(phys[nodeid(t, n)]) for n in sites]
    D = prod(dims)
    Hd = zeros(ComplexF64, D, D)
    for term in H
        mats = [Matrix{ComplexF64}(LinearAlgebra.I, d, d) for d in dims]
        for so in term.ops
            i = findfirst(==(nodeindex(t, so.site)), sites)
            i === nothing && throw(ArgumentError("term site $(so.site) has no physical leg"))
            mats[i] = _dense_siteop_matrix(so.op, dims[i])
        end
        Hd .+= term.coeff .* reduce(kron, reverse(mats))
    end
    return Hd
end

function _dense_siteop_matrix(op::AbstractTensorMap, d::Int)
    if numout(op) == 1 && numin(op) == 1
        return reshape(convert(Array, op), d, d)
    elseif numout(op) == 1 && numin(op) == 2 && dim(domain(op)[2]) == 1
        arr = reshape(convert(Array, op), d, d, :)
        size(arr, 3) == 1 || throw(ArgumentError("charged SiteOp charge leg must be one-dimensional"))
        return arr[:, :, 1]
    else
        throw(ArgumentError("SiteOp tensor must be `P ← P` or charged `P ← P ⊗ C`"))
    end
end

"""Exact ground state (dense, hermitian): returns `(E0, v0)`."""
function exact_groundstate(Hd::AbstractMatrix)
    F = eigen(Hermitian((Hd + Hd') / 2))
    return F.values[1], F.vectors[:, 1]
end

"""Exact propagation `exp(dz·H)·v` (dense; complex dz welcome — §5b convention)."""
exact_evolve(Hd::AbstractMatrix, v::AbstractVector, dz::Number) =
    LinearAlgebra.exp(Matrix(dz * Hd)) * v

end # module TestUtils
