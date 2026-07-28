using Test
using Graft
using Graft.TestUtils
using Graft.Backend
using Random
using LinearAlgebra: Diagonal, Hermitian, eigen, norm

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

function _direct_krylov_spin_hamiltonian(topo, sites)
    spin = spin_ops()
    hamiltonian = OpSum()
    for (index, site) in pairs(sites)
        hamiltonian += Term(
            -0.23 * index, SiteOp(site, :X, spin.X))
        hamiltonian += Term(
            0.11 * (index + 1), SiteOp(site, :Z, spin.Z))
    end
    for (left, right) in zip(sites[1:(end - 1)], sites[2:end])
        hamiltonian += Term(
            0.37, SiteOp(left, :Z, spin.Z), SiteOp(right, :Z, spin.Z))
    end
    return hamiltonian
end

function _dense_direct_krylov_reference(
        hamiltonian::AbstractMatrix,
        initial::AbstractVector,
        dz::Number,
        dimension::Int;
        gram_atol::Float64=0.0,
        gram_rtol::Float64=sqrt(eps(Float64)))
    basis = Vector{Vector{ComplexF64}}()
    actions = Vector{Vector{ComplexF64}}()
    push!(basis, ComplexF64.(initial) / norm(initial))
    while length(basis) <= dimension
        acted = ComplexF64.(hamiltonian * basis[end])
        push!(actions, acted)
        length(basis) == dimension && break
        action_norm = norm(acted)
        action_norm > sqrt(eps(Float64)) || break
        push!(basis, acted / action_norm)
    end

    raw_basis = reduce(hcat, basis)
    raw_actions = reduce(hcat, actions)
    gram = Matrix(Hermitian(raw_basis' * raw_basis))
    projected_action = raw_basis' * raw_actions
    decomposition = eigen(Hermitian(gram))
    threshold = max(
        gram_atol,
        gram_rtol * maximum(abs, decomposition.values; init=0.0))
    keep = findall(value -> value > threshold, decomposition.values)
    whitener = decomposition.vectors[:, keep] *
        Diagonal(inv.(sqrt.(decomposition.values[keep])))
    projected = Matrix(Hermitian(
        (whitener' * projected_action * whitener +
         whitener' * projected_action' * whitener) / 2))
    initial_coordinates =
        whitener' * (raw_basis' * ComplexF64.(initial))
    evolved_coordinates = exp(dz * projected) * initial_coordinates
    orthonormal_basis = raw_basis * whitener
    state = orthonormal_basis * evolved_coordinates
    residual = norm(
        ComplexF64.(hamiltonian) * state -
        orthonormal_basis * (projected * evolved_coordinates))
    return (;
        state,
        residual,
        raw_dimension=length(basis),
        retained_dimension=length(keep),
    )
end

function _direct_krylov_bonds(state)
    topo = topology(state)
    return [
        dim(virtualspace(state, child))
        for child in 1:nnodes(topo) if topo.parent[child] != 0
    ]
end

@graft_testset "exact TTNS linear combination" begin
    chain = mps_topology(3)
    chain_phys = Dict(
        nodeid(chain, node) => spin_ops().P for node in 1:nnodes(chain))
    chain_states = [
        random_ttns(
            Xoshiro(3101), ComplexF64, chain, chain_phys, ℂ^1),
        random_ttns(
            Xoshiro(3102), ComplexF64, chain, chain_phys, ℂ^2),
    ]
    move_center!(chain_states[2], 1)
    chain_sources = to_dense.(chain_states)
    coefficients = ComplexF64[0.37 - 0.11im, -0.29 + 0.41im]
    chain_result = exact_linear_combination(chain_states, coefficients)
    chain_reference = sum(
        coefficients[index] * to_dense(chain_states[index])
        for index in eachindex(chain_states))
    @test norm(to_dense(chain_result) - chain_reference) < 1e-12
    @test to_dense.(chain_states) == chain_sources
    @test check_arrows(chain_result)
    @test _direct_krylov_bonds(chain_result) == [3, 2]
    rank_chain = mps_topology(2)
    rank_state = product_ttns(
        ComplexF64,
        rank_chain,
        Dict(nodeid(rank_chain, node) => ComplexF64[1, 0]
             for node in 1:nnodes(rank_chain)))
    zero_weight_result = exact_linear_combination(
        [rank_state, rank_state, rank_state], ComplexF64[1, 0, 0])
    @test norm(to_dense(zero_weight_result) - to_dense(rank_state)) < 1e-12
    @test _direct_krylov_bonds(zero_weight_result) == [2]
    @test_throws ArgumentError exact_linear_combination(
        chain_states, coefficients; max_bond=2)
    @test_throws ArgumentError exact_linear_combination(
        chain_states, coefficients; max_payload=5)

    branch = binary_topology(2; prefix=:directsum)
    branch_sites = [nodeid(branch, node) for node in leaves(branch)]
    branch_phys = Dict(site => spin_ops().P for site in branch_sites)
    branch_states = [
        random_ttns(
            Xoshiro(3103), ComplexF64, branch, branch_phys, ℂ^1),
        random_ttns(
            Xoshiro(3104), ComplexF64, branch, branch_phys, ℂ^2),
    ]
    branch_coefficients = ComplexF64[0.63, -0.24im]
    branch_result =
        exact_linear_combination(branch_states, branch_coefficients)
    branch_reference = sum(
        branch_coefficients[index] * to_dense(branch_states[index])
        for index in eachindex(branch_states))
    @test norm(to_dense(branch_result) - branch_reference) < 1e-12
    @test check_arrows(branch_result)

    fermion = fermion_ops_z2()
    graded = mps_topology(2)
    graded_phys =
        Dict(nodeid(graded, node) => fermion.P for node in 1:nnodes(graded))
    occupied_left = product_ttns(
        ComplexF64, graded, graded_phys,
        Dict(:site1 => FermionParity(1), :site2 => FermionParity(0)))
    occupied_right = product_ttns(
        ComplexF64, graded, graded_phys,
        Dict(:site1 => FermionParity(0), :site2 => FermionParity(1)))
    graded_coefficients = ComplexF64[0.4 + 0.2im, -0.3im]
    graded_result = exact_linear_combination(
        [occupied_left, occupied_right], graded_coefficients)
    graded_reference =
        graded_coefficients[1] * categorical_coordinates(occupied_left) +
        graded_coefficients[2] * categorical_coordinates(occupied_right)
    @test norm(categorical_coordinates(graded_result) - graded_reference) <
        1e-12
    @test check_arrows(graded_result)
end

@graft_testset "strict direct Global-Krylov bootstrap" begin
    chain = mps_topology(2)
    sites = [nodeid(chain, node) for node in 1:nnodes(chain)]
    phys = Dict(site => spin_ops().P for site in sites)
    hamiltonian = _direct_krylov_spin_hamiltonian(chain, sites)
    operator =
        ttno_from_opsum(hamiltonian, chain, phys; hermitian=true)
    state = random_ttns(
        Xoshiro(3201), ComplexF64, chain, phys, ℂ^1)
    move_center!(state, 1)
    initial = to_dense(state)
    dense_h = dense_hamiltonian(hamiltonian, state)
    dz = -0.047im
    reference = _dense_direct_krylov_reference(
        dense_h, initial, dz, 3; gram_rtol=1e-12)
    initial_bonds = _direct_krylov_bonds(state)
    evolver = DirectKrylovBootstrap(
        krylovdim=3,
        max_basis=3,
        gram_rtol=1e-12,
        max_exact_bond=128,
        max_exact_payload=100_000)
    step!(evolver, state, operator, dz)

    @test norm(to_dense(state) - reference.state) < 1e-10
    @test evolver.last_info !== nothing
    @test evolver.last_info.requested_dimension == 3
    @test evolver.last_info.raw_dimension == 3
    @test evolver.last_info.retained_dimension == 3
    @test evolver.last_info.action_count == 3
    @test evolver.last_info.initial_projection_error == 0.0
    @test evolver.last_info.initial_bond_dimensions == initial_bonds
    @test evolver.last_info.final_bond_dimensions ==
        _direct_krylov_bonds(state)
    @test maximum(evolver.last_info.final_bond_dimensions) >
        maximum(evolver.last_info.initial_bond_dimensions)
    @test evolver.last_info.projected_residual ≈ reference.residual atol = 1e-10
    @test check_arrows(state)

    exact_state = random_ttns(
        Xoshiro(3202), ComplexF64, chain, phys, ℂ^1)
    exact_initial = to_dense(exact_state)
    exact_evolver = DirectKrylovBootstrap(
        krylovdim=4,
        max_basis=4,
        gram_rtol=1e-12,
        max_exact_bond=128,
        max_exact_payload=100_000)
    step!(exact_evolver, exact_state, operator, dz)
    @test norm(
        to_dense(exact_state) -
        exact_evolve(dense_h, exact_initial, dz)) < 1e-10

    unchanged = random_ttns(
        Xoshiro(3203), ComplexF64, chain, phys, ℂ^1)
    unchanged_dense = to_dense(unchanged)
    @test_throws ArgumentError step!(
        DirectKrylovBootstrap(krylovdim=3, max_basis=2),
        unchanged, operator, dz)
    @test to_dense(unchanged) == unchanged_dense
    @test_throws ArgumentError step!(
        DirectKrylovBootstrap(
            krylovdim=2, max_basis=2, max_exact_bond=1),
        unchanged, operator, dz)
    @test to_dense(unchanged) == unchanged_dense
    @test_throws ArgumentError step!(
        DirectKrylovBootstrap(
            krylovdim=2, max_basis=2, max_exact_payload=1),
        unchanged, operator, dz)
    @test to_dense(unchanged) == unchanged_dense

    real_state = random_ttns(
        Xoshiro(3204), Float64, chain, phys, ℂ^1)
    @test_throws ArgumentError step!(
        DirectKrylovBootstrap(krylovdim=2, max_basis=2),
        real_state, operator, dz)

    one_site = mps_topology(1)
    one_phys = Dict(:site1 => spin_ops().P)
    spin = spin_ops()
    tiny_hamiltonian = OpSum()
    tiny_hamiltonian +=
        Term(1e-200, SiteOp(:site1, :X, spin.X))
    tiny_operator = ttno_from_opsum(
        tiny_hamiltonian, one_site, one_phys; hermitian=true)
    tiny_state = product_ttns(
        ComplexF64, one_site, Dict(:site1 => ComplexF64[1, 0]))
    tiny_initial = to_dense(tiny_state)
    tiny_dense_h = dense_hamiltonian(
        tiny_hamiltonian, tiny_state)
    tiny_evolver = DirectKrylovBootstrap(
        krylovdim=2, max_basis=2, gram_rtol=1e-12)
    tiny_dz = -1e200im
    step!(tiny_evolver, tiny_state, tiny_operator, tiny_dz)
    @test norm(
        to_dense(tiny_state) -
        exact_evolve(tiny_dense_h, tiny_initial, tiny_dz)) < 1e-10
    @test tiny_evolver.last_info.raw_dimension == 2
    @test tiny_evolver.last_info.retained_dimension == 2

    eigen_hamiltonian = OpSum()
    eigen_hamiltonian += Term(1.0, SiteOp(:site1, :Z, spin.Z))
    eigen_operator = ttno_from_opsum(
        eigen_hamiltonian, one_site, one_phys; hermitian=true)
    converted_step_state = product_ttns(
        ComplexF64, one_site, Dict(:site1 => ComplexF64[1, 0]))
    converted_step = Complex{BigFloat}(0, BigFloat("-0.047"))
    step!(
        DirectKrylovBootstrap(krylovdim=2, max_basis=2),
        converted_step_state,
        eigen_operator,
        converted_step)
    @test norm(
        to_dense(converted_step_state) -
        exact_evolve(
            dense_hamiltonian(eigen_hamiltonian, converted_step_state),
            ComplexF64[1, 0],
            ComplexF64(converted_step))) < 1e-10

    eigen_state = product_ttns(
        ComplexF64, one_site, Dict(:site1 => ComplexF64[1, 0]))
    eigen_initial = to_dense(eigen_state)
    eigen_evolver =
        DirectKrylovBootstrap(krylovdim=3, max_basis=3)
    step!(eigen_evolver, eigen_state, eigen_operator, dz)
    @test norm(
        to_dense(eigen_state) -
        exact_evolve(
            dense_hamiltonian(eigen_hamiltonian, eigen_state),
            eigen_initial,
            dz)) < 1e-10
    @test eigen_evolver.last_info.raw_dimension == 3
    @test eigen_evolver.last_info.retained_dimension == 1
    @test eigen_evolver.last_info.discarded_dimension == 2
    @test eigen_evolver.last_info.action_count == 3
    @test eigen_evolver.last_info.projected_residual < 1e-10
end

@graft_testset "direct Global-Krylov on branching and graded trees" begin
    branch = binary_topology(2; prefix=:directkrylov)
    branch_sites = [nodeid(branch, node) for node in leaves(branch)]
    branch_phys = Dict(site => spin_ops().P for site in branch_sites)
    branch_hamiltonian =
        _direct_krylov_spin_hamiltonian(branch, branch_sites)
    branch_operator = ttno_from_opsum(
        branch_hamiltonian, branch, branch_phys; hermitian=true)
    branch_state = random_ttns(
        Xoshiro(3301), ComplexF64, branch, branch_phys, ℂ^1)
    branch_initial = to_dense(branch_state)
    branch_dense_h = dense_hamiltonian(
        branch_hamiltonian, branch_state)
    branch_dz = -0.019im
    branch_reference = _dense_direct_krylov_reference(
        branch_dense_h, branch_initial, branch_dz, 2;
        gram_rtol=1e-12)
    branch_evolver = DirectKrylovBootstrap(
        krylovdim=2,
        max_basis=2,
        gram_rtol=1e-12,
        max_exact_bond=256,
        max_exact_payload=1_000_000)
    step!(
        branch_evolver, branch_state, branch_operator, branch_dz)
    @test norm(to_dense(branch_state) - branch_reference.state) < 1e-10
    @test check_arrows(branch_state)

    fermion = fermion_ops_z2()
    graded = mps_topology(2)
    graded_phys =
        Dict(nodeid(graded, node) => fermion.P for node in 1:nnodes(graded))
    hopping = OpSum()
    hopping += Term(
        -1.0,
        SiteOp(:site1, :Cd, fermion.Cd),
        SiteOp(:site2, :C, fermion.C))
    hopping += Term(
        -1.0,
        SiteOp(:site1, :C, fermion.C),
        SiteOp(:site2, :Cd, fermion.Cd))
    hopping_operator =
        ttno_from_opsum(hopping, graded, graded_phys; hermitian=true)
    graded_state = product_ttns(
        ComplexF64, graded, graded_phys,
        Dict(:site1 => FermionParity(1), :site2 => FermionParity(0)))
    graded_initial = categorical_coordinates(graded_state)
    graded_dense_h = dense_hamiltonian(hopping, graded, graded_phys)
    graded_dz = -0.071im
    graded_evolver = DirectKrylovBootstrap(
        krylovdim=4,
        max_basis=4,
        gram_rtol=1e-12,
        max_exact_bond=128,
        max_exact_payload=100_000)
    step!(
        graded_evolver, graded_state, hopping_operator, graded_dz)
    @test norm(
        categorical_coordinates(graded_state) -
        exact_evolve(graded_dense_h, graded_initial, graded_dz)) < 1e-11
    @test graded_evolver.last_info.raw_dimension == 4
    @test graded_evolver.last_info.retained_dimension == 2
    @test graded_evolver.last_info.discarded_dimension == 2
    @test graded_evolver.last_info.action_count == 4
    @test graded_evolver.last_info.initial_bond_dimensions == [1]
    @test graded_evolver.last_info.final_bond_dimensions == [2]
    @test graded_evolver.last_info.projected_residual < 1e-10
    @test check_arrows(graded_state)
end
