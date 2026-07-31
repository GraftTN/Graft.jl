# SD5 — restricted exact symbolic Gaussian elimination and proof-backed
# cover selection (state-diagram-compiler plan v2, SD5 deliverables).
#
# Categorical data is formal-exact: floating Hamiltonian coefficients remain
# opaque atoms combined with exact scalar multipliers. The restricted
# elimination permits row and column swaps, multiplication by an exact
# nonzero scalar, exact-proportional row or column addition, and exact
# zero, duplicate, and proportional-row/column elimination. It never
# invents rational approximations to floating coefficients, never pivots on
# arbitrary rational functions of coefficient atoms, and never infers
# equality from floating values: proportionality exists only between formal
# expressions over the same atoms whose exact bookkeeping scalars reproduce
# each other exactly. Raw and eliminated covers are computed independently
# and the eliminated result is selected only when strictly smaller — raw
# wins every tie.

# ---------------------------------------------------------------------------
# Formal exact coefficient expressions
# ---------------------------------------------------------------------------

"""
    GammaExpr(terms)

Formal exact linear combination `Σ scale_k · atom_k` over opaque
coefficient atoms, with sorted unique atom indices and no zero scales. The
scales are the compiler's own exact bookkeeping scalars (slot scales times
certificate data), never Hamiltonian coefficient values.
"""
struct GammaExpr{C<:Number}
    terms::Vector{Tuple{Int,C}}
end

GammaExpr{C}() where {C} = GammaExpr(Tuple{Int,C}[])

function _gexpr(pairs::Vector{Tuple{Int,C}}) where {C}
    acc = Dict{Int,C}()
    for (atom, scale) in pairs
        acc[atom] = get(acc, atom, zero(C)) + scale
    end
    terms = Tuple{Int,C}[(atom, acc[atom])
                         for atom in sort!(collect(keys(acc)))
                         if !iszero(acc[atom])]
    return GammaExpr(terms)
end

_gexpr_iszero(e::GammaExpr) = isempty(e.terms)

function _gexpr_axpy(e::GammaExpr{C}, λ::C, f::GammaExpr{C}) where {C}
    return _gexpr(vcat(e.terms, Tuple{Int,C}[(a, λ * s) for (a, s) in f.terms]))
end

"""
Exact proportionality factor `λ` with `lhs == λ · rhs` in the formal
algebra, or `nothing`. The factor must reproduce every scale exactly under
floating multiplication (the supported exact scalar domain: ±1 and other
exactly representable products); atom values are never consulted.
"""
function _gexpr_proportional(lhs::GammaExpr{C}, rhs::GammaExpr{C}) where {C}
    (isempty(lhs.terms) || isempty(rhs.terms)) && return nothing
    length(lhs.terms) == length(rhs.terms) || return nothing
    all(lhs.terms[k][1] == rhs.terms[k][1] for k in eachindex(lhs.terms)) ||
        return nothing
    λ = lhs.terms[1][2] / rhs.terms[1][2]
    (isfinite(abs(λ)) && !iszero(λ)) || return nothing
    for k in eachindex(lhs.terms)
        lhs.terms[k][2] == λ * rhs.terms[k][2] || return nothing
    end
    return λ
end

for T in (:GammaExpr,)
    @eval begin
        Base.:(==)(a::$T, b::$T) = _ir_isequal(a, b)
        Base.isequal(a::$T, b::$T) = _ir_isequal(a, b)
        Base.hash(x::$T, h::UInt) = _ir_hash(x, h)
    end
end

# ---------------------------------------------------------------------------
# Typed restricted-elimination operations
# ---------------------------------------------------------------------------

"One typed exact operation of the restricted elimination."
abstract type AbstractSGEOperation end

"`target -= factor · source` zeroed an exactly proportional row."
struct SGERowElimination{C<:Number} <: AbstractSGEOperation
    target::Int
    source::Int
    factor::C
end

"`target -= factor · source` zeroed an exactly proportional column."
struct SGEColElimination{C<:Number} <: AbstractSGEOperation
    target::Int
    source::Int
    factor::C
end

for T in (:SGERowElimination, :SGEColElimination)
    @eval begin
        Base.:(==)(a::$T, b::$T) = _ir_isequal(a, b)
        Base.isequal(a::$T, b::$T) = _ir_isequal(a, b)
        Base.hash(x::$T, h::UInt) = _ir_hash(x, h)
    end
end

"Exactly representable elimination factors admitted for reconstruction."
_sge_exact_factor(λ) = λ == 1 || λ == -1

"""
Restricted exact elimination over a Gamma matrix of formal expressions.
Deterministically zeroes exact duplicate and exact-proportional rows, then
columns, through exact-proportional additions; returns the eliminated
matrix and the typed operation log. Only ±1 factors are admitted so every
reconstruction product is bitwise exact.
"""
function _restricted_sge(matrix::Matrix{GammaExpr{C}}) where {C}
    work = copy(matrix)
    nrows, ncols = size(work)
    ops = AbstractSGEOperation[]
    for j in 2:nrows
        _row_zero(work, j) && continue
        for i in 1:(j - 1)
            _row_zero(work, i) && continue
            λ = _rowvec_proportional(work, j, i)
            λ === nothing && continue
            _sge_exact_factor(λ) || continue
            for v in 1:ncols
                work[j, v] = _gexpr_axpy(work[j, v], -λ, work[i, v])
            end
            push!(ops, SGERowElimination(j, i, λ))
            break
        end
    end
    for v in 2:ncols
        _col_zero(work, v) && continue
        for w in 1:(v - 1)
            _col_zero(work, w) && continue
            λ = _colvec_proportional(work, v, w)
            λ === nothing && continue
            _sge_exact_factor(λ) || continue
            for u in 1:nrows
                work[u, v] = _gexpr_axpy(work[u, v], -λ, work[u, w])
            end
            push!(ops, SGEColElimination(v, w, λ))
            break
        end
    end
    return work, ops
end

_row_zero(m, u) = all(_gexpr_iszero(m[u, v]) for v in axes(m, 2))
_col_zero(m, v) = all(_gexpr_iszero(m[u, v]) for u in axes(m, 1))

function _rowvec_proportional(m::Matrix{GammaExpr{C}}, j::Int, i::Int) where {C}
    λ = nothing
    for v in axes(m, 2)
        zj = _gexpr_iszero(m[j, v])
        zi = _gexpr_iszero(m[i, v])
        zj != zi && return nothing
        zj && continue
        μ = _gexpr_proportional(m[j, v], m[i, v])
        μ === nothing && return nothing
        if λ === nothing
            λ = μ
        elseif λ != μ
            return nothing
        end
    end
    return λ
end

function _colvec_proportional(m::Matrix{GammaExpr{C}}, v::Int, w::Int) where {C}
    λ = nothing
    for u in axes(m, 1)
        zv = _gexpr_iszero(m[u, v])
        zw = _gexpr_iszero(m[u, w])
        zv != zw && return nothing
        zv && continue
        μ = _gexpr_proportional(m[u, v], m[u, w])
        μ === nothing && return nothing
        if λ === nothing
            λ = μ
        elseif λ != μ
            return nothing
        end
    end
    return λ
end

# ---------------------------------------------------------------------------
# Elimination-backed cover witness
# ---------------------------------------------------------------------------

"""
    EliminationCoverWitness

Exact reconstruction log of one restricted-elimination-backed cover merge:
the embedded raw-Gamma witness (rows, columns, support, matching, raw
cover, assignments, atom relocations, per-term previous labels), the typed
elimination operations, the eliminated support with its independent
matching and cover, both cover sizes (the eliminated cover was selected
because it is strictly smaller), and every ±1 exact scalar slot rewrite
performed so proportional-row terms share their partner's above context.
"""
struct EliminationCoverWitness{C<:Number} <: AbstractRedundancyWitness
    raw::GammaCoverWitness
    operations::Vector{AbstractSGEOperation}
    eliminated_support::Vector{Tuple{Int,Int}}
    eliminated_matching::Vector{Tuple{Int,Int}}
    eliminated_cover_rows::Vector{Int}
    eliminated_cover_cols::Vector{Int}
    raw_cover_size::Int
    eliminated_cover_size::Int
    slot_rewrites::Vector{Tuple{Int,Int,C,C}}
end

for T in (:EliminationCoverWitness,)
    @eval begin
        Base.:(==)(a::$T, b::$T) = _ir_isequal(a, b)
        Base.isequal(a::$T, b::$T) = _ir_isequal(a, b)
        Base.hash(x::$T, h::UInt) = _ir_hash(x, h)
    end
end

# ---------------------------------------------------------------------------
# Nontrivial span basis transport (forward/inverse, proof-validated)
# ---------------------------------------------------------------------------

"""
    SpanBasisTransport(lhs, rhs, forward, inverse)

A nontrivial span-level basis relation: an exact invertible transport
between two compatible channel-copy spans, with forward and inverse
matrices whose products are exactly the identity. A basis isomorphism only
transports an entire compatible span into a common basis — it never
reduces dimension, and no merge may cite a transport without a separate
exact redundancy witness.
"""
struct SpanBasisTransport{C<:Number} <: AbstractSpanRelation
    lhs::ChannelSpan
    rhs::ChannelSpan
    forward::Matrix{C}
    inverse::Matrix{C}
end

for T in (:SpanBasisTransport,)
    @eval begin
        Base.:(==)(a::$T, b::$T) = _ir_isequal(a, b)
        Base.isequal(a::$T, b::$T) = _ir_isequal(a, b)
        Base.hash(x::$T, h::UInt) = _ir_hash(x, h)
    end
end

function validate_basis_proof(::AbelianEquivalenceService,
                              relation::SpanBasisTransport)
    size(relation.forward) == size(relation.inverse) || return false
    size(relation.forward, 1) == size(relation.forward, 2) || return false
    n = size(relation.forward, 1)
    eye = [i == j ? one(eltype(relation.forward)) : zero(eltype(relation.forward))
           for i in 1:n, j in 1:n]
    return relation.forward * relation.inverse == eye &&
        relation.inverse * relation.forward == eye
end
