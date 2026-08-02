import GraftFoundation
import GraftEvolution
import Random
using GraftEvolution: AbstractCBE, CBESelectionInfo, LGVDCBEInfo,
    PredictorCBE, PredictorLegacyCBE, NaiveCBE, LGVDCBE, TDVP1_CBE
using GraftFoundation: TensorMap, blocks, catdomain, codomain, domain, dim,
    FermionParity, Vect, left_null, mps_topology, norm, permute,
    star_topology, ℂ, ←, ⊗
using LinearAlgebra: adjoint, transpose
using Random: Xoshiro, randn!
using Test

const _CBE = GraftEvolution.Evolution

function _cbe_random_map(rng, cod, dom)
    tensor = zeros(ComplexF64, cod ← dom)
    for (_, block) in blocks(tensor)
        randn!(rng, block)
    end
    return tensor
end

@testset "typed TDVP-only CBE strategies" begin
    @test TDVP1_CBE().cbe isa PredictorCBE
    @test PredictorCBE() isa AbstractCBE
    @test PredictorLegacyCBE() isa AbstractCBE
    @test NaiveCBE(rng=Xoshiro(11)) isa AbstractCBE
    @test LGVDCBE() isa AbstractCBE
    @test_throws UndefKeywordError NaiveCBE()
    @test typeof(TDVP1_CBE(cbe=PredictorLegacyCBE()).cbe) === PredictorLegacyCBE
    @test TDVP1_CBE(cbe=NaiveCBE(rng=Xoshiro(12))).cbe.rng isa Xoshiro

    lgvd = LGVDCBE()
    @test lgvd.preselection_threshold == 1e-4
    @test lgvd.final_selection_threshold == 1e-6
    @test lgvd.trim_threshold == 1e-12
    @test fieldtype(CBESelectionInfo, :selected_rank) === Int
    @test fieldtype(LGVDCBEInfo, :strict_schedule_applied) === Bool
end

@testset "partner injection preserves explicit domain order" begin
    rng = Xoshiro(21)
    partner = _cbe_random_map(rng, ℂ^2 ⊗ ℂ^3, ℂ^1)
    Np = left_null(partner)
    injection = _CBE._cbe_right_injection(Np)
    Z = _cbe_random_map(rng, ℂ^4 ⊗ ℂ^2 ⊗ ℂ^3, one(ℂ^1))
    Zs = permute(Z, ((1,), (2, 3)))

    @test codomain(injection) == domain(Zs)
    @test_throws Exception Zs * transpose(adjoint(Np))
    @test Zs * injection isa TensorMap
end

@testset "zero-weight splice is exactly state preserving" begin
    rng = Xoshiro(31)
    tA = _cbe_random_map(rng, ℂ^5, ℂ^2)
    Q, _ = GraftFoundation.left_orth(tA)
    complement = left_null(Q)
    direction = _cbe_random_map(rng, domain(complement), ℂ^1)
    direction, _ = GraftFoundation.left_orth(direction)
    E = complement * direction

    U, R = _CBE._cbe_state_preserving_splice(tA, Q, E)

    @test U == catdomain(Q, E)
    @test dim(domain(U)) == dim(domain(Q)) + dim(domain(E))
    @test norm(U * R - tA) <= 100 * eps(Float64) * max(norm(tA), 1)
end

@testset "partial Predictor and implicit Naive range selection" begin
    rng = Xoshiro(2026031546)
    M = _cbe_random_map(rng, ℂ^8, ℂ^6)
    exact = _CBE._cbe_exact_directions(M, 2)
    exact_projector = exact * exact'

    predictor, predictor_info = _CBE._cbe_predictor_directions(
        PredictorCBE(max_add=2, spawn_threshold=0, neigs=2, krylovdim=4),
        (; M, Θ=M), -0.01im, 2)
    @test predictor_info.solver === :krylov_svd
    @test dim(domain(predictor)) == 2
    @test norm(predictor * predictor' - exact_projector) < 1e-10

    even, odd = FermionParity(0), FermionParity(1)
    graded_codomain = Vect[FermionParity](even => 4, odd => 3)
    graded_domain = Vect[FermionParity](even => 3, odd => 2)
    graded = _cbe_random_map(rng, graded_codomain, graded_domain)
    graded_exact = _CBE._cbe_exact_directions(graded, 2)
    graded_predictor, graded_info = _CBE._cbe_predictor_directions(
        PredictorCBE(max_add=2, spawn_threshold=0, neigs=2, krylovdim=4),
        (; M=graded, Θ=graded), -0.01im, 2)
    @test graded_info.solver === :exact_svd_sector_fallback
    @test norm(graded_predictor * graded_predictor' -
               graded_exact * graded_exact') < 1e-12

    factors = (
        Na=GraftFoundation.id(codomain(M)),
        Np=GraftFoundation.id(domain(M)),
        Zs=M,
        injection=GraftFoundation.id(domain(M)),
        Θ=M,
    )
    policy(seed) = NaiveCBE(
        rng=Xoshiro(seed), max_add=2, oversample=4, poweriter=1,
        enr_atol=0, enr_rtol=0)
    oracle, oracle_info = _CBE._cbe_naive_exact_oracle(
        policy(0x0ac1e), (; M, Θ=M), 2)
    @test oracle_info.solver === :exact_svd_oracle
    @test norm(oracle * oracle' - exact_projector) < 1e-12
    sampled1, _ = _CBE._cbe_implicit_rsvd_directions(
        policy(0x51eed), factors, 2)
    sampled2, _ = _CBE._cbe_implicit_rsvd_directions(
        policy(0x51eed), factors, 2)
    sampled_projector = sampled1 * sampled1'

    # Projector distance is the sine-principal-angle oracle and is invariant
    # under singular-vector phases and rotations inside degenerate subspaces.
    @test norm(sampled_projector - sampled2 * sampled2') < 1e-13
    @test norm(sampled_projector - exact_projector) < 1e-10

    global_rng = Random.default_rng()
    saved_global_rng = copy(global_rng)
    try
        expected = rand(global_rng, UInt64)
        copy!(global_rng, saved_global_rng)
        _CBE._cbe_implicit_rsvd_directions(
            policy(0x600d), factors, 2)
        @test rand(global_rng, UInt64) == expected
    finally
        copy!(global_rng, saved_global_rng)
    end
end

@testset "exact selector and strict LGVD topology guard" begin
    rng = Xoshiro(41)
    M = _cbe_random_map(rng, ℂ^5, ℂ^4)
    directions = _CBE._cbe_exact_directions(M, 2)
    @test dim(domain(directions)) == 2
    @test norm(directions' * directions -
               GraftFoundation.id(domain(directions))) < 1e-12

    @test _CBE._lgvd_require_chain(mps_topology(4)) === nothing
    @test_throws ArgumentError _CBE._lgvd_require_chain(star_topology(3, 1))
end
