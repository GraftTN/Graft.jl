"""
    THCReport

Diagnostics for an ISDF-THC factorization. `pair_density_relative_error`
measures the interpolative decomposition before the interaction kernel is
applied. The interaction errors are `nothing` unless `assess=true` was
requested, because evaluating them materializes the dense `norb^2 × norb^2`
interaction that THC is intended to avoid.
"""
struct THCReport{R<:Real}
    grid_size::Int
    orbital_count::Int
    pair_density_rank::Int
    selected_rank::Int
    interpolation_points::Vector{Int}
    pair_density_relative_error::Union{Nothing,R}
    interaction_relative_error::Union{Nothing,R}
    interaction_max_abs_error::Union{Nothing,R}
end

"""
    THCFactorization

Tensor-hypercontraction factors for a four-index interaction. If `X` has shape
`(norb, nthc)`, the selected pair-density matrix is

`C[P, (i,j)] = conj(X[i,P]) * X[j,P]`

and the pair-flattened interaction is `C' * coupling * C`. For real orbitals
this is the architecture's `X_iP X_jP M_PQ X_kQ X_lQ` convention.
"""
struct THCFactorization{T<:Number,R<:Real}
    X::Matrix{T}
    coupling::Matrix{T}
    report::THCReport{R}
    function THCFactorization{T,R}(
            X::Matrix{T}, coupling::Matrix{T}, report::THCReport{R}) where {T<:Number,R<:Real}
        norb, nthc = size(X)
        norb > 0 || throw(ArgumentError("THC factors need at least one orbital"))
        nthc > 0 || throw(ArgumentError("THC factors need at least one column"))
        size(coupling) == (nthc, nthc) ||
            throw(DimensionMismatch("THC coupling must have size ($nthc, $nthc)"))
        report.orbital_count == norb ||
            throw(DimensionMismatch("THC report orbital count disagrees with X"))
        report.selected_rank == nthc ||
            throw(DimensionMismatch("THC report selected rank disagrees with X"))
        all(isfinite, X) || throw(ArgumentError("THC factors contain non-finite values"))
        all(isfinite, coupling) ||
            throw(ArgumentError("THC coupling contains non-finite values"))
        return new{T,R}(X, coupling, report)
    end
end

function THCFactorization(X::AbstractMatrix, coupling::AbstractMatrix,
                          report::THCReport{R}) where {R<:Real}
    T = promote_type(float(eltype(X)), float(eltype(coupling)))
    return THCFactorization{T,R}(Matrix{T}(X), Matrix{T}(coupling), report)
end

function _check_isdf_inputs(orbitals::AbstractMatrix, kernel::AbstractMatrix,
                            rtol::Real, rank)
    ngrid, norb = size(orbitals)
    ngrid > 0 || throw(ArgumentError("ISDF needs at least one grid point"))
    norb > 0 || throw(ArgumentError("ISDF needs at least one orbital"))
    size(kernel) == (ngrid, ngrid) ||
        throw(DimensionMismatch("kernel must have size ($ngrid, $ngrid)"))
    isfinite(rtol) && 0 <= rtol < 1 ||
        throw(ArgumentError("rtol must be finite and lie in [0, 1)"))
    rank === nothing || (rank isa Integer && 1 <= rank <= min(ngrid, norb^2)) ||
        throw(ArgumentError("rank must lie between 1 and min(ngrid, norb^2)"))
    all(isfinite, orbitals) ||
        throw(ArgumentError("orbitals contain non-finite values"))
    all(isfinite, kernel) ||
        throw(ArgumentError("kernel contains non-finite values"))

    T = promote_type(float(eltype(orbitals)), float(eltype(kernel)))
    Φ = Matrix{T}(orbitals)
    K = Matrix{T}(kernel)
    scale = max(norm(K), one(real(T)))
    norm(K - K') <= sqrt(eps(real(T))) * scale ||
        throw(ArgumentError("ISDF interaction kernel must be Hermitian"))
    return Φ, K
end

function _pair_densities(orbitals::AbstractMatrix{T}) where {T}
    ngrid, norb = size(orbitals)
    density = Matrix{T}(undef, ngrid, norb^2)
    for j in 1:norb, i in 1:norb
        col = (j - 1) * norb + i
        @views density[:, col] .= conj.(orbitals[:, i]) .* orbitals[:, j]
    end
    return density
end

function _relative_error(reference, approximation)
    denominator = norm(reference)
    denominator == 0 && return real(norm(approximation))
    return real(norm(reference - approximation) / denominator)
end

function _isdf_rank_and_points(density::AbstractMatrix, rtol::Real, requested_rank)
    factorization = qr(adjoint(density), ColumnNorm())
    diagonal = abs.(diag(factorization.R))
    isempty(diagonal) && throw(ArgumentError("pair-density matrix is empty"))
    leading = first(diagonal)
    leading > 0 || throw(ArgumentError("all orbital pair densities are zero"))
    numerical_rank = count(value -> value > rtol * leading, diagonal)
    numerical_rank = max(numerical_rank, 1)
    selected_rank = requested_rank === nothing ? numerical_rank : Int(requested_rank)
    return numerical_rank, selected_rank, factorization.p[1:selected_rank]
end

function _selected_pair_densities(X::AbstractMatrix{T}) where {T}
    norb, nthc = size(X)
    selected = Matrix{T}(undef, nthc, norb^2)
    for j in 1:norb, i in 1:norb
        col = (j - 1) * norb + i
        @views selected[:, col] .= conj.(X[i, :]) .* X[j, :]
    end
    return selected
end

"""
    isdf_thc(orbitals, kernel; rank=nothing, rtol=sqrt(eps()), assess=false)
        -> THCFactorization

Construct an interpolative-separable density-fitting THC representation.
`orbitals[r,i]` is orbital `i` evaluated at grid point `r`; quadrature weights
must already be included in the Hermitian `kernel`. Pivoted QR of the orbital
pair densities chooses deterministic interpolation points. With no explicit
`rank`, the QR numerical rank selected by `rtol` is used.

The default path scales without materializing the dense four-index
interaction. Set `assess=true` only for validation-sized problems.
"""
function isdf_thc(orbitals::AbstractMatrix, kernel::AbstractMatrix;
                  rank=nothing,
                  rtol::Real=sqrt(eps(real(float(promote_type(
                      eltype(orbitals), eltype(kernel)))))),
                  assess::Bool=false)
    Φ, K = _check_isdf_inputs(orbitals, kernel, rtol, rank)
    density = _pair_densities(Φ)
    numerical_rank, selected_rank, points =
        _isdf_rank_and_points(density, rtol, rank)
    selected = density[points, :]
    interpolation = density * pinv(selected; rtol=rtol)
    coupling = interpolation' * K * interpolation
    coupling = Matrix((coupling + coupling') / 2)

    approximated_density = interpolation * selected
    density_error = _relative_error(density, approximated_density)
    interaction_error = nothing
    max_abs_error = nothing
    if assess
        reference = density' * K * density
        approximation = selected' * coupling * selected
        interaction_error = _relative_error(reference, approximation)
        max_abs_error = real(maximum(abs, reference - approximation))
    end

    R = typeof(density_error)
    report = THCReport{R}(
        size(Φ, 1),
        size(Φ, 2),
        numerical_rank,
        selected_rank,
        collect(Int, points),
        density_error,
        interaction_error,
        max_abs_error,
    )
    X = Matrix(transpose(Φ[points, :]))
    return THCFactorization{eltype(X),R}(X, Matrix(coupling), report)
end

"""
    fit_thc(interaction, X; rtol=sqrt(eps()), assess=true)
        -> THCFactorization

Fit the central THC coupling for fixed orbital factors `X`. `interaction` must
have shape `(norb, norb, norb, norb)`. This is the dense-input fallback for
callers that already have interpolation factors but not a grid kernel.
"""
function fit_thc(interaction::AbstractArray{<:Number,4}, X::AbstractMatrix;
                 rtol::Real=sqrt(eps(real(float(promote_type(
                     eltype(interaction), eltype(X)))))),
                 assess::Bool=true)
    norb, nthc = size(X)
    size(interaction) == (norb, norb, norb, norb) ||
        throw(DimensionMismatch("interaction and X orbital dimensions disagree"))
    nthc > 0 || throw(ArgumentError("X needs at least one THC column"))
    isfinite(rtol) && 0 <= rtol < 1 ||
        throw(ArgumentError("rtol must be finite and lie in [0, 1)"))
    all(isfinite, interaction) ||
        throw(ArgumentError("interaction contains non-finite values"))
    all(isfinite, X) || throw(ArgumentError("X contains non-finite values"))

    T = promote_type(float(eltype(interaction)), float(eltype(X)))
    factors = Matrix{T}(X)
    selected = _selected_pair_densities(factors)
    pair_interaction = reshape(Array{T}(interaction), norb^2, norb^2)
    scale = max(norm(pair_interaction), one(real(T)))
    norm(pair_interaction - pair_interaction') <= sqrt(eps(real(T))) * scale ||
        throw(ArgumentError("THC interaction must be Hermitian when pair-flattened"))
    left = selected'
    coupling = pinv(left; rtol=rtol) * pair_interaction * pinv(selected; rtol=rtol)
    coupling = Matrix((coupling + coupling') / 2)
    approximation = left * coupling * selected
    relative_error = assess ? _relative_error(pair_interaction, approximation) : nothing
    max_abs_error = assess ? real(maximum(abs, pair_interaction - approximation)) : nothing
    report = THCReport{real(T)}(
        0,
        norb,
        nthc,
        nthc,
        Int[],
        nothing,
        relative_error,
        max_abs_error,
    )
    return THCFactorization{T,real(T)}(factors, coupling, report)
end

"""
    reconstruct_thc(factors; tensor=true)

Materialize the interaction represented by `factors`. The default result has
shape `(norb, norb, norb, norb)`; `tensor=false` returns its pair-flattened
`norb^2 × norb^2` matrix.
"""
function reconstruct_thc(factors::THCFactorization; tensor::Bool=true)
    selected = _selected_pair_densities(factors.X)
    interaction = selected' * factors.coupling * selected
    tensor || return interaction
    norb = size(factors.X, 1)
    return reshape(interaction, norb, norb, norb, norb)
end
