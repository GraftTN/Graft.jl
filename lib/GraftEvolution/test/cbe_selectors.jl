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
    sampled1, sampled_exact1 = _CBE._cbe_implicit_rsvd_directions(
        policy(0x51eed), factors, 2)
    sampled2, sampled_exact2 = _CBE._cbe_implicit_rsvd_directions(
        policy(0x51eed), factors, 2)
    zero_sampled, zero_exact = _CBE._cbe_implicit_rsvd_directions(
        policy(0x51eed), factors, 0)
    sampled_projector = sampled1 * sampled1'

    # Projector distance is the sine-principal-angle oracle and is invariant
    # under singular-vector phases and rotations inside degenerate subspaces.
    @test sampled_exact1 isa _CBE.CBEExactLeftSVD
    @test sampled_exact1.values ≈ sampled_exact2.values rtol=1e-13 atol=1e-13
    @test norm(sampled_projector - sampled2 * sampled2') < 1e-13
    @test norm(sampled_projector - exact_projector) < 1e-10
    @test isnothing(zero_sampled)
    @test isnothing(zero_exact)

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

@testset "reusable exact CBE spectra and thresholded projectors" begin
    rng = Xoshiro(2026080303)
    even, odd = FermionParity(0), FermionParity(1)
    maps = (
        dense=_cbe_random_map(rng, ℂ^7, ℂ^5),
        fermion_parity=_cbe_random_map(
            rng,
            Vect[FermionParity](even => 4, odd => 3),
            Vect[FermionParity](even => 3, odd => 2)),
    )

    for (label, M) in pairs(maps)
        @testset "$label spectrum and projector" begin
            exact = _CBE.CBEExactLeftSVD(M)
            reference_values = sort!(
                Float64.(collect(GraftFoundation.svd_vals(M))); rev=true)
            @test exact.values ≈ reference_values rtol=1e-13 atol=1e-13

            cutoff = (reference_values[2] + reference_values[3]) / 2
            directions = _CBE._cbe_exact_directions(
                exact, length(reference_values); atol=cutoff)
            reference, _, _ = GraftFoundation.split_svd(
                M,
                GraftFoundation.TruncationScheme(
                    maxdim=length(reference_values), atol=cutoff))
            expected_rank = count(>(cutoff), reference_values)

            @test dim(domain(directions)) == expected_rank
            @test norm(directions * directions' - reference * reference') < 1e-12
        end
    end

    graded = maps.fermion_parity
    exact = _CBE.CBEExactLeftSVD(graded)
    cutoff = (exact.values[2] + exact.values[3]) / 2
    dz = -0.25im
    spawn_threshold = cutoff * abs(dz) / norm(graded)
    predictor, info = _CBE._cbe_predictor_directions(
        PredictorCBE(
            max_add=length(exact.values),
            spawn_threshold=spawn_threshold,
            neigs=length(exact.values),
            krylovdim=length(exact.values)),
        (; M=graded, Θ=graded), dz, length(exact.values))
    reference = _CBE._cbe_exact_directions(
        exact, length(exact.values); atol=cutoff)

    @test info.solver === :exact_svd_sector_fallback
    @test info.singular_threshold ≈ cutoff rtol=1e-13
    @test info.selected_rank == count(>(cutoff), exact.values)
    @test info.score_max ≈ abs(dz) * first(exact.values) / norm(graded)
    @test norm(predictor * predictor' - reference * reference') < 1e-12

    dense = maps.dense
    exact = _CBE.CBEExactLeftSVD(dense)
    final_cutoff = (exact.values[2] + exact.values[3]) / 2
    pre_cutoff = (exact.values[3] + exact.values[4]) / 2
    naive = NaiveCBE(
        rng=Xoshiro(2026080304), max_add=length(exact.values), rsvd=false,
        enr_atol=final_cutoff, enr_rtol=0)
    naive_directions, naive_info = _CBE._cbe_naive_exact_oracle(
        naive, (; M=dense, Θ=dense), length(exact.values))
    factors = (
        Na=GraftFoundation.id(codomain(dense)),
        Np=GraftFoundation.id(domain(dense)),
        Zs=dense,
        injection=GraftFoundation.id(domain(dense)),
        Θ=dense,
    )
    production_directions, production_info = _CBE._cbe_naive_directions(
        naive, factors, length(exact.values))
    reference = _CBE._cbe_exact_directions(
        exact, length(exact.values); atol=final_cutoff)

    @test naive_info.solver === :exact_svd_oracle
    @test naive_info.singular_threshold == final_cutoff
    @test naive_info.selected_rank == count(>(final_cutoff), exact.values)
    @test naive_info.score_max ≈ first(exact.values)
    @test norm(naive_directions * naive_directions' -
               reference * reference') < 1e-12
    @test production_info.solver === :exact_svd
    @test production_info.selected_rank == naive_info.selected_rank
    @test production_info.score_max ≈ naive_info.score_max
    @test norm(production_directions * production_directions' -
               reference * reference') < 1e-12

    lgvd = LGVDCBE(
        max_add=length(exact.values),
        preselection_threshold=pre_cutoff / first(exact.values),
        final_selection_threshold=final_cutoff / first(exact.values))
    lgvd_directions, lgvd_info = _CBE._lgvd_c1_directions(
        lgvd, (; M=dense), length(exact.values))

    @test lgvd_info.selector === :full_c1
    @test lgvd_info.preselected_rank == count(>(pre_cutoff), exact.values)
    @test lgvd_info.selected_rank == count(>(final_cutoff), exact.values)
    @test norm(lgvd_directions * lgvd_directions' -
               reference * reference') < 1e-12
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
