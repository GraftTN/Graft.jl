using Test
using GraftFoundation: ℂ, TensorMap, mps_topology, norm, ←
using GraftNetworks: TTNS, center, check_arrows, exact_linear_combination,
    topology

function one_site_state(entries::AbstractVector{<:Number})
    length(entries) == 2 || throw(ArgumentError("expected a two-level state"))
    topo = mps_topology(1)
    physical = ℂ^2
    root = ℂ^1
    tensor = TensorMap(reshape(ComplexF64.(entries), 2, 1), physical ← root)
    return TTNS(topo, [tensor], topo.root)
end

@testset "GraftNetworks exact TTNS linear combination" begin
    first_state = one_site_state([1, 0])
    second_state = one_site_state([0, 1])
    first_before = copy(first_state[1])
    second_before = copy(second_state[1])
    coefficients = ComplexF64[0.4 - 0.2im, -0.3 + 0.7im]

    combined = exact_linear_combination(
        [first_state, second_state], coefficients)
    expected = coefficients[1] * first_before + coefficients[2] * second_before

    @test topology(combined) === topology(first_state)
    @test center(combined) == topology(combined).root
    @test check_arrows(combined)
    @test norm(combined[1] - expected) < 1.0e-14
    @test norm(first_state[1] - first_before) == 0
    @test norm(second_state[1] - second_before) == 0

    @test_throws ArgumentError exact_linear_combination(
        [first_state, second_state], [1.0])
    @test_throws ArgumentError exact_linear_combination(
        [first_state, second_state], coefficients; max_payload=1)
end

include("global_subspace_expansion.jl")
