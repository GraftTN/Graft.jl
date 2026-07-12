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
using LinearAlgebra: LinearAlgebra, eigen, Hermitian, eigvals, tr
using ..Backend
using ..Trees
using ..Networks
using ..Symbolic

export random_ttns, product_ttns, canonicalize!, to_dense, dense_hamiltonian,
    exact_groundstate, exact_evolve, physical_sites,
    exact_thermal_Z, exact_thermal_logZ, exact_thermal_expect,
    exact_thermal_correlator, pad_bonds!

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

# ---------------------------------------------------------------------------
# Dense thermal references (P0 of finite-temperature v1, §05 plan)
# ---------------------------------------------------------------------------

"""Exact partition function `Z = tr(exp(-beta*Hd))` (dense)."""
function exact_thermal_Z(Hd::AbstractMatrix, beta::Real)
    beta == 0 && return ComplexF64(size(Hd, 1))
    return tr(LinearAlgebra.exp(-Float64(beta) * Matrix{ComplexF64}(Hd)))
end

"""Exact `logZ = log(tr(exp(-beta*Hd)))` (dense, overflow-safe for large beta)."""
function exact_thermal_logZ(Hd::AbstractMatrix, beta::Real)
    vals = eigvals(Hermitian(Matrix{ComplexF64}(Hd)))
    ls = sort!(-Float64(beta) .* real.(vals))
    m = maximum(ls)
    return m + log(sum(exp.(ls .- m)))
end

"""
    exact_thermal_expect(Hd, Od, beta) -> Number

Dense thermal expectation `tr(O * exp(-beta*Hd)) / Z`. At `beta=0` this is
`tr(O)/dim(O)`, which pins the trace (not supertrace) convention.
"""
function exact_thermal_expect(Hd::AbstractMatrix, Od::AbstractMatrix, beta::Real)
    Z = exact_thermal_Z(Hd, beta)
    if beta == 0
        return tr(Matrix{ComplexF64}(Od)) / Z
    end
    rho = LinearAlgebra.exp(-Float64(beta) * Matrix{ComplexF64}(Hd))
    return tr(Matrix{ComplexF64}(Od) * rho) / Z
end

"""
    exact_thermal_correlator(Hd, Ad, Bd, beta, taus) -> Vector

Dense thermal correlator `C_AB(tau) = tr(exp(-(beta-tau)*Hd) * A * exp(-tau*Hd) * B) / Z`
for each `tau` in `taus`. Uses the stable β-τ preparation formula.
"""
function exact_thermal_correlator(Hd::AbstractMatrix, Ad::AbstractMatrix,
                                  Bd::AbstractMatrix, beta::Real, taus)
    Z = exact_thermal_Z(Hd, beta)
    Hd64 = Matrix{ComplexF64}(Hd)
    A64 = Matrix{ComplexF64}(Ad)
    B64 = Matrix{ComplexF64}(Bd)
    return [tr(LinearAlgebra.exp(-(Float64(beta) - Float64(tau)) * Hd64) * A64 *
               LinearAlgebra.exp(-Float64(tau) * Hd64) * B64) / Z for tau in taus]
end

"""
    pad_bonds!(ψ, bond) -> ψ

Enlarge every virtual bond of `ψ` to at least `dim(bond)` by zero-padding the
bond sector. The state vector is unchanged (padded sectors are zero-filled).
Used by fixed-manifold evolver fixtures (GlobalKrylov, ImplicitLogTime) that
need a bond-padded start from the minimal `|I⟩` (§05 plan §2.4).
"""
function pad_bonds!(ψ::TTNS, bond)
    t = ψ.topo
    for n in 1:nnodes(t)
        t.parent[n] == 0 && continue
        V = domain(ψ.tensors[n])[1]
        dim(V) >= dim(bond) && continue
        # Replace this bond with a padded space and zero-fill
        newV = _padded_space(V, bond)
        _pad_one_bond!(ψ, n, newV)
    end
    return ψ
end

function _padded_space(V::S, bond::S) where {S<:ElementarySpace}
    spacetype(V) === spacetype(bond) ||
        throw(ArgumentError("pad_bonds! space type mismatch"))
    if sectortype(V) === Trivial
        return ℂ^max(dim(V), dim(bond))
    end
    Q = sectortype(V)
    dims = Dict{Q,Int}()
    for q in sectors(V)
        dims[q] = dim(V, q)
    end
    for q in sectors(bond)
        dims[q] = max(get(dims, q, 0), dim(bond, q))
    end
    return Vect[Q](dims...)
end

function _pad_one_bond!(ψ::TTNS{S}, child::Int, newV::S) where {S<:ElementarySpace}
    t = ψ.topo
    p = t.parent[child]
    k = childslot(t, p, child)
    oldA = ψ.tensors[p]
    oldV = space(oldA, k)
    newcod = S[space(oldA, i) for i in 1:numout(oldA)]
    newcod[k] = newV
    newA = zeros(eltype(ψ), reduce(⊗, newcod) ← domain(oldA))
    _copy_block!(newA, oldA, oldV, newV, k)
    ψ.tensors[p] = newA
    oldC = ψ.tensors[child]
    newC = zeros(eltype(ψ), codomain(oldC) ← newV)
    _copy_block!(newC, oldC, oldV, newV, 0)
    return ψ
end

function _copy_block!(dst::AbstractTensorMap, src::AbstractTensorMap,
                      oldV::S, newV::S, k::Int) where {S<:ElementarySpace}
    Q = sectortype(oldV)
    if Q === Trivial
        od, nd = dim(oldV), dim(newV)
        for (_, db) in blocks(dst)
            for (_, sb) in blocks(src)
                sz = min(size(sb, 1), size(db, 1)), min(size(sb, 2), size(db, 2))
                db[1:sz[1], 1:sz[2]] .= sb[1:sz[1], 1:sz[2]]
            end
        end
        return dst
    end
    for q in sectors(oldV)
        for (qd, db) in blocks(dst)
            for (qs, sb) in blocks(src)
                _copy_sector_block!(db, sb, qd, qs, oldV, newV, k)
            end
        end
    end
    return dst
end

function _copy_sector_block!(db, sb, qd, qs, oldV, newV, k)
    Q = sectortype(oldV)
    if k == 0
        qd == qs || return nothing
        r = min(size(sb, 1), size(db, 1))
        c = min(size(sb, 2), size(db, 2))
        db[1:r, 1:c] .= sb[1:r, 1:c]
    else
        qd == qs || return nothing
        r = min(size(sb, 1), size(db, 1))
        c = min(size(sb, 2), size(db, 2))
        db[1:r, 1:c] .= sb[1:r, 1:c]
    end
end

end # module TestUtils
