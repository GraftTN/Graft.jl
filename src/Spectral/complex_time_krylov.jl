"""
    ComplexTimeKrylovResult

Generalized Ritz solution in a nonorthogonal snapshot basis. Columns of
`coefficients` contain the retained-basis coefficients of each Ritz vector.
"""
struct ComplexTimeKrylovResult{T<:Number,D<:NamedTuple}
    values::Vector{T}
    weights::Vector{Float64}
    coefficients::Matrix{T}
    overlap::Matrix{T}
    hamiltonian::Matrix{T}
    residuals::Vector{Float64}
    diagnostics::D
end

"""
    complex_time_krylov(overlap, hamiltonian; atol=0, rtol=sqrt(eps()),
                        reference=1)

Solve the generalized Ritz problem `H c = energy * S c` after removing the
numerical null space of the Gram matrix `S`. `weights` are normalized spectral
weights with respect to the selected reference snapshot.
"""
function complex_time_krylov(overlap::AbstractMatrix,
                             hamiltonian::AbstractMatrix;
                             atol::Real=0,
                             rtol::Real=sqrt(eps(Float64)),
                             reference::Integer=1)
    size(overlap, 1) == size(overlap, 2) ||
        throw(ArgumentError("overlap matrix must be square"))
    size(hamiltonian) == size(overlap) ||
        throw(ArgumentError("hamiltonian and overlap matrices must have equal size"))
    n = size(overlap, 1)
    1 <= reference <= n || throw(ArgumentError("reference must lie in 1:$n"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))

    S = Matrix{ComplexF64}((overlap + overlap') / 2)
    H = Matrix{ComplexF64}((hamiltonian + hamiltonian') / 2)
    SF = eigen(Hermitian(S))
    threshold = max(Float64(atol), Float64(rtol) * maximum(abs, SF.values))
    minimum(SF.values) >= -threshold ||
        throw(ArgumentError(
            "snapshot Gram matrix is not positive semidefinite within tolerance"))
    keep = findall(x -> x > threshold, SF.values)
    isempty(keep) && throw(ArgumentError("snapshot Gram matrix has zero numerical rank"))
    X = SF.vectors[:, keep] * Diagonal(inv.(sqrt.(SF.values[keep])))
    Hred = Hermitian((X' * H * X + (X' * H * X)') / 2)
    HF = eigen(Hred)
    coefficients = X * HF.vectors
    values = complex.(HF.values)

    residuals = [
        norm(H * coefficients[:, k] - values[k] * S * coefficients[:, k]) /
        max((norm(H) + abs(values[k]) * norm(S)) * norm(coefficients[:, k]),
            eps(Float64))
        for k in eachindex(values)
    ]
    overlaps = vec(transpose(S[reference, :]) * coefficients)
    raw_weights = abs2.(overlaps)
    weight_scale = sum(raw_weights)
    weights = iszero(weight_scale) ? zeros(Float64, length(raw_weights)) :
              Float64.(raw_weights ./ weight_scale)
    diagnostics = (;
        method=:complex_time_krylov,
        snapshot_count=n,
        retained_rank=length(keep),
        discarded_rank=n - length(keep),
        gram_eigenvalues=collect(SF.values),
        gram_threshold=threshold,
        reference=Int(reference),
        max_residual=maximum(residuals),
    )
    return ComplexTimeKrylovResult(
        values, weights, coefficients, S, H, residuals, diagnostics)
end

"""
    complex_time_krylov(snapshots, H; kwargs...)

Assemble the snapshot Gram matrices with tree contractions and solve the dense
generalized Ritz problem. Each `H * snapshot[j]` is formed once and reused
across all bra snapshots.
"""
function complex_time_krylov(snapshots::AbstractVector{<:TTNS}, H::TTNO;
                             optimize::Bool=true, kwargs...)
    isempty(snapshots) && throw(ArgumentError("snapshots must be nonempty"))
    topo = topology(first(snapshots))
    topology(H) == topo ||
        throw(ArgumentError("Hamiltonian and snapshots have mismatched topologies"))
    Networks.ishermitian(H) ||
        throw(ArgumentError("complex_time_krylov requires a Hermitian TTNO"))
    all(psi -> topology(psi) == topo, snapshots) ||
        throw(ArgumentError("all snapshots must have the same topology"))

    n = length(snapshots)
    S = Matrix{ComplexF64}(undef, n, n)
    HM = Matrix{ComplexF64}(undef, n, n)
    acted = [apply(H, psi; center=center(psi), optimize) for psi in snapshots]
    for j in 1:n, i in 1:j
        S[i, j] = inner(snapshots[i], snapshots[j]; optimize)
        S[j, i] = conj(S[i, j])
        HM[i, j] = inner(snapshots[i], acted[j]; optimize)
        if i == j
            HM[j, i] = HM[i, j]
        else
            HM[j, i] = conj(HM[i, j])
        end
    end
    return complex_time_krylov(S, HM; kwargs...)
end
