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

Product state on trivial-sector spaces: `states[site]` is the local state
vector; nodes absent from the Dict are branching tensors. All bonds are
one-dimensional. TODO(M0 fermion path): graded version (bond sectors must
carry the accumulated charge towards the root).
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

"""Internal-order list of nodes that carry a physical leg."""
physical_sites(ψ::TTNS) = [n for n in 1:nnodes(ψ.topo) if hasphys(ψ, n)]
physical_sites(O::TTNO) = [n for n in 1:nnodes(O.topo) if hasphys(O, n)]

"""
    to_dense(ψ::TTNS) -> Vector

Full state vector by brute-force contraction (small trees only). Site order:
see module docstring.
"""
function to_dense(ψ::TTNS)
    t = ψ.topo
    sites = physical_sites(ψ)
    nopen = 0
    openlabel = Dict(n => -(nopen += 1) for n in sites)
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
            cap = ones_tensor(eltype(ψ), ProductSpace(domain(A)[1]))
            lbl = nnodes(t) + 1
            idx[end] = lbl
            push!(tensors, cap); push!(indices, [lbl])
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
