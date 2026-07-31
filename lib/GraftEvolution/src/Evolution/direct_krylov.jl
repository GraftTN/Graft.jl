"""
    DirectKrylovInfo

Diagnostics from one [`DirectKrylovBootstrap`](@ref) step.
`projected_residual` is the absolute full-Hilbert-space norm of the exponential
Galerkin defect

`H * psi_K(dz) - d(psi_K(dz)) / d(dz)`.

`gram_condition` is the condition number of the retained Gram eigenspectrum.
Bond-dimension vectors use non-root node-index order.
"""
struct DirectKrylovInfo
    requested_dimension::Int
    raw_dimension::Int
    retained_dimension::Int
    discarded_dimension::Int
    gram_eigenvalues::Vector{Float64}
    gram_threshold::Float64
    gram_condition::Float64
    initial_projection_error::Float64
    projected_residual::Float64
    action_count::Int
    initial_bond_dimensions::Vector{Int}
    final_bond_dimensions::Vector{Int}
end

"""
    DirectKrylovBootstrap(; krylovdim=4, max_basis=16,
                          gram_atol=0, gram_rtol=sqrt(eps(Float64)),
                          max_exact_bond=4096,
                          max_exact_payload=100_000_000,
                          optimize=true)

Strict full-state Global-Krylov bootstrap. Raw basis vectors are constructed as
exact, normalized powers of `H` using `apply`; Gram and projected-Hamiltonian
matrices use exact TTNS contractions; and the propagated state is formed with
[`exact_linear_combination`](@ref). No `fit!`, truncation, or projection back
into the input bond manifold occurs.

Projected dense algebra, the converted time step, and diagnostics use
`Float64`/`ComplexF64`; input states must therefore use `Float32`, `Float64`,
`ComplexF32`, or `ComplexF64` scalars. The result retains the input TTNS
scalar type.

The exact reference is intentionally bounded. `max_basis` must explicitly
permit the requested Krylov dimension, while `max_exact_bond` and
`max_exact_payload` reject an exact action or final block sum before its
conservative size exceeds the declared limits.
"""
Base.@kwdef mutable struct DirectKrylovBootstrap <: Evolver
    krylovdim::Int = 4
    max_basis::Int = 16
    gram_atol::Float64 = 0.0
    gram_rtol::Float64 = sqrt(eps(Float64))
    max_exact_bond::Int = 4096
    max_exact_payload::Int = 100_000_000
    optimize::Bool = true
    last_info::Union{Nothing,DirectKrylovInfo} = nothing
end

function step!(
        ev::DirectKrylovBootstrap, psi::TTNS, H::TTNO, dz::Number)
    _check_direct_krylov(ev, psi, H, dz)
    dense_step = _direct_krylov_dense_step(dz)
    initial_bonds = _direct_krylov_bond_dimensions(psi)
    initial = copy(psi)
    basis, actions = _direct_krylov_raw_basis(ev, initial, H)
    S, A, C = _direct_krylov_matrices(
        basis, actions; optimize=ev.optimize)
    X, eigenvalues, threshold, condition =
        _direct_krylov_whitener(S, ev.gram_atol, ev.gram_rtol)
    retained = size(X, 2)

    projected = X' * A * X
    if ishermitian(H)
        projected = Matrix(
            LinearAlgebra.Hermitian((projected + projected') / 2))
    end
    initial_overlaps = ComplexF64[
        inner(v, initial; optimize=ev.optimize) for v in basis]
    q0 = X' * initial_overlaps
    initial_norm = norm(initial)
    initial_projection_error =
        _direct_krylov_projection_error(initial_norm, q0)
    q = LinearAlgebra.exp(dense_step * projected) * q0
    coefficients = X * q
    result_coefficients = _direct_krylov_coefficients(
        eltype(psi), coefficients)

    result = exact_linear_combination(
        basis, result_coefficients;
        max_bond=ev.max_exact_bond,
        max_payload=ev.max_exact_payload)
    residual = _direct_krylov_residual(S, A, C, X, projected, q)
    final_bonds = _direct_krylov_bond_dimensions(result)
    _replace_state!(psi, result)
    ev.last_info = DirectKrylovInfo(
        ev.krylovdim,
        length(basis),
        retained,
        length(basis) - retained,
        eigenvalues,
        threshold,
        condition,
        initial_projection_error,
        residual,
        length(actions),
        initial_bonds,
        final_bonds,
    )
    return psi
end

function _check_direct_krylov(
        ev::DirectKrylovBootstrap, psi::TTNS, H::TTNO, dz::Number)
    topology(psi) == topology(H) ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: H and psi have mismatched topologies"))
    psi.hasphys == H.hasphys ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: H and psi have mismatched physical layout"))
    spacetype(psi) == spacetype(H) ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: H and psi have mismatched spacetype"))
    promote_type(eltype(psi), eltype(H)) == eltype(psi) ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: exact H action would promote the TTNS " *
            "eltype; convert the input state explicitly"))
    eltype(psi) <: Union{Float32,Float64,ComplexF32,ComplexF64} ||
        throw(ArgumentError(
            "DirectKrylovBootstrap supports Float32, Float64, ComplexF32, " *
            "and ComplexF64 TTNS scalars"))
    ev.krylovdim >= 1 ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: krylovdim must be positive"))
    ev.max_basis >= 1 ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: max_basis must be positive"))
    ev.krylovdim <= ev.max_basis ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: krylovdim exceeds max_basis"))
    ev.gram_atol >= 0 ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: gram_atol must be nonnegative"))
    ev.gram_rtol >= 0 ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: gram_rtol must be nonnegative"))
    ev.max_exact_bond >= 1 ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: max_exact_bond must be positive"))
    ev.max_exact_payload >= 1 ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: max_exact_payload must be positive"))
    if !(eltype(psi) <: Complex) && dz isa Complex && !isreal(dz)
        throw(ArgumentError(
            "DirectKrylovBootstrap complex-step evolution requires a " *
            "complex-eltype TTNS; convert explicitly"))
    end
    return nothing
end

function _direct_krylov_dense_step(dz::Number)
    dense_step = try
        ComplexF64(dz)
    catch error
        error isa InexactError || error isa MethodError || rethrow()
        throw(ArgumentError(
            "DirectKrylovBootstrap: dz is not representable as ComplexF64"))
    end
    isfinite(real(dense_step)) && isfinite(imag(dense_step)) ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: dz must be finite after ComplexF64 conversion"))
    return dense_step
end

function _direct_krylov_raw_basis(
        ev::DirectKrylovBootstrap, initial::TTNS, H::TTNO)
    first_vector = _direct_krylov_normalized_copy(initial)
    basis = typeof(first_vector)[first_vector]
    actions = typeof(first_vector)[]
    while length(basis) <= ev.krylovdim
        current = basis[end]
        _check_direct_action_guard(ev, current, H)
        acted = apply(
            H, current; center=topology(current).root, optimize=ev.optimize)
        push!(actions, acted)
        length(basis) == ev.krylovdim && break
        action_norm = norm(acted)
        action_norm > 0 || break
        next_vector = copy(acted)
        update_tensor!(
            next_vector, center(next_vector),
            next_vector.tensors[center(next_vector)] / action_norm)
        push!(basis, next_vector)
    end
    return basis, actions
end

function _direct_krylov_normalized_copy(psi::TTNS)
    vector = copy(psi)
    target = topology(vector).root
    center(vector) == target || move_center!(vector, target)
    vector_norm = norm(vector)
    vector_norm > 0 ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: initial state has zero norm"))
    update_tensor!(
        vector, center(vector),
        vector.tensors[center(vector)] / vector_norm)
    return vector
end

function _direct_krylov_matrices(
        basis::Vector{<:TTNS}, actions::Vector{<:TTNS};
        optimize::Bool)
    dimension = length(basis)
    length(actions) == dimension ||
        throw(ArgumentError("DirectKrylovBootstrap: incomplete action basis"))
    S = zeros(ComplexF64, dimension, dimension)
    A = zeros(ComplexF64, dimension, dimension)
    C = zeros(ComplexF64, dimension, dimension)
    for j in 1:dimension, i in 1:dimension
        S[i, j] = inner(basis[i], basis[j]; optimize)
        A[i, j] = inner(basis[i], actions[j]; optimize)
        C[i, j] = inner(actions[i], actions[j]; optimize)
    end
    S = Matrix(LinearAlgebra.Hermitian((S + S') / 2))
    C = Matrix(LinearAlgebra.Hermitian((C + C') / 2))
    return S, A, C
end

function _direct_krylov_whitener(
        S::Matrix{ComplexF64}, atol::Float64, rtol::Float64)
    factorization = LinearAlgebra.eigen(LinearAlgebra.Hermitian(S))
    eigenvalues = Float64.(factorization.values)
    scale = maximum(abs, eigenvalues; init=0.0)
    threshold = max(atol, rtol * scale)
    minimum(eigenvalues; init=0.0) >= -threshold ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: Gram matrix is not positive semidefinite"))
    keep = findall(value -> value > threshold, eigenvalues)
    isempty(keep) &&
        throw(ArgumentError(
            "DirectKrylovBootstrap: Krylov Gram matrix has zero numerical rank"))
    retained_values = eigenvalues[keep]
    X = factorization.vectors[:, keep] *
        LinearAlgebra.Diagonal(inv.(sqrt.(retained_values)))
    condition = maximum(retained_values) / minimum(retained_values)
    return X, eigenvalues, threshold, condition
end

function _direct_krylov_residual(
        S::Matrix{ComplexF64},
        A::Matrix{ComplexF64},
        C::Matrix{ComplexF64},
        X::Matrix{ComplexF64},
        projected::Matrix{ComplexF64},
        q::Vector{ComplexF64})
    state_coefficients = X * q
    derivative_coefficients = X * (projected * q)
    residual2 =
        state_coefficients' * C * state_coefficients -
        2 * real(state_coefficients' * A' * derivative_coefficients) +
        derivative_coefficients' * S * derivative_coefficients
    real_residual2 = real(residual2)
    tolerance = 100 * eps(Float64) * max(
        real(state_coefficients' * C * state_coefficients), 1.0)
    real_residual2 >= -tolerance ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: negative residual norm beyond roundoff"))
    return sqrt(max(real_residual2, 0.0))
end

function _direct_krylov_projection_error(
        initial_norm::Real, coordinates::Vector{ComplexF64})
    norm_value = Float64(initial_norm)
    projected_norm = LinearAlgebra.norm(coordinates)
    tolerance = 100 * eps(Float64) * max(norm_value, 1.0)
    projected_norm <= norm_value + tolerance ||
        throw(ArgumentError(
            "DirectKrylovBootstrap: initial projection norm exceeds input norm"))
    norm_value - projected_norm <= tolerance && return 0.0
    return sqrt(max(
        (norm_value - projected_norm) * (norm_value + projected_norm),
        0.0))
end

function _direct_krylov_coefficients(
        ::Type{T}, coefficients::Vector{ComplexF64}) where {T<:Number}
    if T <: Real
        imaginary_scale = maximum(abs ∘ imag, coefficients; init=0.0)
        real_scale = maximum(abs ∘ real, coefficients; init=1.0)
        imaginary_scale <= 100 * eps(Float64) * real_scale ||
            throw(ArgumentError(
                "DirectKrylovBootstrap: real TTNS produced complex coefficients"))
        return T.(real.(coefficients))
    end
    return T.(coefficients)
end

function _check_direct_action_guard(
        ev::DirectKrylovBootstrap, psi::TTNS, H::TTNO)
    t = topology(psi)
    predicted_bonds = Dict{Int,Int}()
    for child in 1:nnodes(t)
        t.parent[child] == 0 && continue
        state_space = virtualspace(psi, child)
        operator_space = domain(H.tensors[child])[numin(H.tensors[child])]
        predicted = dim(state_space) * dim(operator_space)
        predicted <= ev.max_exact_bond ||
            throw(ArgumentError(
                "DirectKrylovBootstrap: exact H action on edge " *
                string(nodeid(t, child)) * " requires bond dimension " *
                string(predicted) * ", exceeding max_exact_bond=" *
                string(ev.max_exact_bond)))
        predicted_bonds[child] = predicted
    end
    for n in 1:nnodes(t)
        payload = BigInt(1)
        for child in t.children[n]
            payload *= predicted_bonds[child]
        end
        hasphys(psi, n) && (payload *= dim(physspace(psi, n)))
        if t.parent[n] == 0
            state_root = domain(psi.tensors[n])[1]
            operator_root = domain(H.tensors[n])[numin(H.tensors[n])]
            payload *= dim(state_root) * dim(operator_root)
        else
            payload *= predicted_bonds[n]
        end
        payload <= ev.max_exact_payload ||
            throw(ArgumentError(
                "DirectKrylovBootstrap: exact H action at node " *
                string(nodeid(t, n)) * " requires a conservative payload of " *
                string(payload) * " scalars, exceeding max_exact_payload=" *
                string(ev.max_exact_payload)))
    end
    return nothing
end

function _direct_krylov_bond_dimensions(psi::TTNS)
    t = topology(psi)
    return [
        dim(virtualspace(psi, child))
        for child in 1:nnodes(t) if t.parent[child] != 0
    ]
end
