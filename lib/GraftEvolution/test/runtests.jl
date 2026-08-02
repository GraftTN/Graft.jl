import GraftEvolution
using GraftEvolution: TDVP1_CBE, TDVP1_GSE, TDVP1_LSE, ImplicitLogTime,
    LogGaussLegendre, LogTrapezoid, logarithmic_time_grid, step!,
    supports_complex_step
using GraftFoundation: mps_topology, ℂ
using GraftSymbolic: OpSum, SiteOp, Term, spin_ops
using GraftTestUtils: dense_hamiltonian, random_ttns, to_dense
using GraftTTNOBuild: ttno_from_opsum
using LinearAlgebra: I, norm
using Random: Xoshiro
using Test

@testset "TDVP1 expansion-family names" begin
    @test TDVP1_CBE() isa TDVP1_CBE
    @test TDVP1_GSE(ancillary_shift=0.01) isa TDVP1_GSE
    @test_throws UndefKeywordError TDVP1_GSE()
    @test TDVP1_LSE() isa TDVP1_LSE
    @test !isdefined(GraftEvolution, :GSE_TDVP)
    @test !isdefined(GraftEvolution, :LSE_TDVP)
end

@testset "implicit-log one-site trapezoid contract" begin
    @test logarithmic_time_grid(0.01, 0.07) ==
          [0.0, 0.01, 0.02, 0.04, 0.07]
    @test logarithmic_time_grid(0.01, 0.08; nsteps_per_panel=2) ==
          [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08]
    @test_throws ArgumentError logarithmic_time_grid(0.0, 0.08)
    @test_throws ArgumentError LogGaussLegendre(0)

    topology = mps_topology(1)
    spin = spin_ops()
    physical_spaces = Dict(:site1 => spin.P)
    hamiltonian = OpSum() +
        Term(0.37, SiteOp(:site1, :Z, spin.Z)) +
        Term(-0.61, SiteOp(:site1, :X, spin.X))
    operator = ttno_from_opsum(
        hamiltonian, topology, physical_spaces; hermitian=true)
    state = random_ttns(
        Xoshiro(20260802), ComplexF64, topology, physical_spaces, ℂ^1)

    dense_operator = dense_hamiltonian(hamiltonian, state)
    initial = to_dense(state)
    step_size = 0.04
    identity_matrix = Matrix{ComplexF64}(I, length(initial), length(initial))
    reference = (identity_matrix + step_size * dense_operator / 2) \
                ((identity_matrix - step_size * dense_operator / 2) * initial)
    evolver = ImplicitLogTime(
        scheme=LogTrapezoid(), krylovdim=4, maxiter=3, tol=1e-11,
        fit_nsweeps=2, fit_tol=1e-12)

    step!(evolver, state, operator, -step_size)

    @test !supports_complex_step(typeof(evolver))
    @test evolver.last_info.converged == 1
    @test evolver.last_info.normres < 1e-10
    @test norm(to_dense(state) - reference) < 2e-7
end

@testset "subspace expansion selector contract" begin
    topology = mps_topology(1)
    spin = spin_ops()
    physical_spaces = Dict(:site1 => spin.P)
    hamiltonian = OpSum() + Term(0.25, SiteOp(:site1, :Z, spin.Z))
    operator = ttno_from_opsum(
        hamiltonian, topology, physical_spaces; hermitian=true)
    state = random_ttns(
        Xoshiro(2026080202), ComplexF64, topology, physical_spaces, ℂ^1)
    prepare! = GraftEvolution.Evolution._prepare_subspace_expansion!

    @test_throws ArgumentError prepare!(
        TDVP1_LSE(expand_scheme=:unknown), copy(state), operator, -0.01,
        "TDVP1_LSE",
    )
    @test_throws ArgumentError prepare!(
        TDVP1_LSE(expand_scheme=:rangefinder), copy(state), operator, -0.01,
        "TDVP1_LSE",
    )
    @test prepare!(
        TDVP1_LSE(expand_scheme=:directqr), copy(state), operator, -0.01,
        "TDVP1_LSE",
    ) !== nothing
end

include("cbe_selectors.jl")
include("lgvd_cbe.jl")
include("lgvd_shrewd_cbe.jl")
include("gse_bootstrap.jl")
