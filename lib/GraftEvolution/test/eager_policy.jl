using GraftEvolution: GlobalKrylov, TDVP1, TDVP2, TDVP1_CBE, TDVP1_GSE,
    TDVP1_LSE, step!
using GraftFoundation: TruncationScheme, mps_topology, ℂ
using GraftSymbolic: OpSum, SiteOp, Term, spin_ops
using GraftTestUtils: dense_hamiltonian, random_ttns, to_dense
using GraftTTNOBuild: ttno_from_opsum
using LinearAlgebra: norm
using Random: Xoshiro
using Test

const _EAGER_BACKEND_CALLS = NamedTuple[]

function GraftEvolution.Evolution.evolution_exponentiate_backend(
        map::GraftEvolution.Contractions.Planning.WorkspaceMap,
        time, x; kwargs...)
    push!(_EAGER_BACKEND_CALLS,
          (; kind=:local, time, eager=get(kwargs, :eager, missing)))
    return GraftEvolution.Evolution.KrylovKit.exponentiate(
        map, time, x; kwargs...)
end

function GraftEvolution.Evolution.evolution_exponentiate_backend(
        map::GraftEvolution.Evolution._GKOperator,
        time, x; kwargs...)
    push!(_EAGER_BACKEND_CALLS,
          (; kind=:global, time, eager=get(kwargs, :eager, missing)))
    return GraftEvolution.Evolution.KrylovKit.exponentiate(
        map, time, x; kwargs...)
end

@testset "Krylov eager policy defaults" begin
    @test TDVP1().eager
    @test TDVP2().eager
    @test TDVP1_CBE().eager
    @test TDVP1_GSE(ancillary_shift=0.1).eager
    @test TDVP1_LSE().eager
    @test GlobalKrylov().eager
    @test TDVP2().fuse_turning

    @test !TDVP1(eager=false).eager
    @test !TDVP2(eager=false).eager
    @test !TDVP1_CBE(eager=false).eager
    @test !TDVP1_GSE(ancillary_shift=0.1, eager=false).eager
    @test !TDVP1_LSE(eager=false).eager
    @test !GlobalKrylov(eager=false).eager
    @test !TDVP2(fuse_turning=false).fuse_turning
end

function _turning_fixture(nsites::Int, seed::Int)
    topology = mps_topology(nsites)
    spin = spin_ops()
    spaces = Dict(Symbol(:site, site) => spin.P for site in 1:nsites)
    hamiltonian = OpSum()
    for site in 1:nsites
        name = Symbol(:Z, site)
        hamiltonian += Term(0.13 * site,
                            SiteOp(Symbol(:site, site), name, spin.Z))
    end
    for site in 1:(nsites - 1)
        left = Symbol(:site, site)
        right = Symbol(:site, site + 1)
        hamiltonian += Term(
            -0.21,
            SiteOp(left, Symbol(:X, site, :l), spin.X),
            SiteOp(right, Symbol(:X, site, :r), spin.X),
        )
    end
    operator = ttno_from_opsum(
        hamiltonian, topology, spaces; hermitian=true)
    state = random_ttns(
        Xoshiro(seed), ComplexF64, topology, spaces, ℂ^2)
    return state, operator
end

@testset "TDVP1 order-2 fuses the turning site and forwards eager" begin
    state, operator, _ = _gse_fixture()
    empty!(_EAGER_BACKEND_CALLS)
    step!(TDVP1(order=2, eager=false, krylovdim=8, tol=1e-11,
                verbose=false), state, operator, -0.02im)

    local_calls = filter(call -> call.kind == :local, _EAGER_BACKEND_CALLS)
    @test length(local_calls) == 9
    @test all(call -> call.eager === false, local_calls)
    @test count(call -> call.time == -0.02im, local_calls) == 1
    @test count(call -> call.time == -0.01im, local_calls) == 4
    @test count(call -> call.time == 0.01im, local_calls) == 4

    single_state, single_operator = _turning_fixture(1, 2026080301)
    empty!(_EAGER_BACKEND_CALLS)
    step!(TDVP1(order=2, verbose=false),
          single_state, single_operator, -0.02im)
    @test _EAGER_BACKEND_CALLS ==
          [(; kind=:local, time=-0.02im, eager=true)]
end

@testset "TDVP2 order-2 fuses the turning bond" begin
    state, operator = _turning_fixture(3, 2026080302)
    empty!(_EAGER_BACKEND_CALLS)
    step!(TDVP2(
        order=2,
        trunc=TruncationScheme(maxdim=16),
        eager=false,
        krylovdim=8,
        tol=1e-11,
        verbose=false,
    ), state, operator, -0.02im)
    @test length(_EAGER_BACKEND_CALLS) == 5
    @test all(call -> call.kind == :local && call.eager === false,
              _EAGER_BACKEND_CALLS)
    @test count(call -> call.time == -0.02im,
                _EAGER_BACKEND_CALLS) == 1
    @test count(call -> call.time == -0.01im,
                _EAGER_BACKEND_CALLS) == 2
    @test count(call -> call.time == 0.01im,
                _EAGER_BACKEND_CALLS) == 2

    bond_state, bond_operator = _turning_fixture(2, 2026080303)
    empty!(_EAGER_BACKEND_CALLS)
    step!(TDVP2(
        order=2,
        trunc=TruncationScheme(maxdim=16),
        verbose=false,
    ), bond_state, bond_operator, -0.02im)
    @test _EAGER_BACKEND_CALLS ==
          [(; kind=:local, time=-0.02im, eager=true)]

    optout_state, optout_operator = _turning_fixture(3, 2026080306)
    empty!(_EAGER_BACKEND_CALLS)
    step!(TDVP2(
        order=2,
        trunc=TruncationScheme(),
        fuse_turning=false,
        krylovdim=8,
        tol=1e-11,
        verbose=false,
    ), optout_state, optout_operator, -0.02im)
    @test length(_EAGER_BACKEND_CALLS) == 6
    @test count(call -> call.time == -0.01im,
                _EAGER_BACKEND_CALLS) == 4
    @test count(call -> call.time == 0.01im,
                _EAGER_BACKEND_CALLS) == 2
    @test all(call -> call.time != -0.02im, _EAGER_BACKEND_CALLS)
end

@testset "turning fusion matches manual unfused full-rank sweeps" begin
    step_size = -0.02im
    initial1, operator1 = _turning_fixture(3, 2026080304)
    fused1 = copy(initial1)
    unfused1 = copy(initial1)
    fused_ev1 = TDVP1(order=2, krylovdim=12, tol=1e-12, verbose=false)
    unfused_ev1 = TDVP1(order=2, krylovdim=12, tol=1e-12, verbose=false)
    unfused_ev1.cache = GraftEvolution.Evolution.EnvCache(unfused1.topo)
    empty!(_EAGER_BACKEND_CALLS)
    step!(fused_ev1, fused1, operator1, step_size)
    GraftEvolution.Evolution._tdvp1_sweep!(
        unfused_ev1, unfused1, operator1, step_size / 2; rev=false)
    GraftEvolution.Evolution._tdvp1_sweep!(
        unfused_ev1, unfused1, operator1, step_size / 2; rev=true)
    @test norm(to_dense(fused1) - to_dense(unfused1)) < 1e-10

    initial2, operator2 = _turning_fixture(3, 2026080305)
    fused2 = copy(initial2)
    public_unfused2 = copy(initial2)
    unfused2 = copy(initial2)
    trunc = TruncationScheme()
    fused_ev2 = TDVP2(
        order=2, trunc=trunc, krylovdim=12, tol=1e-12,
        contraction_optimize=true, verbose=false)
    unfused_ev2 = TDVP2(
        order=2, trunc=trunc, krylovdim=12, tol=1e-12,
        contraction_optimize=true, verbose=false)
    public_unfused_ev2 = TDVP2(
        order=2, trunc=trunc, krylovdim=12, tol=1e-12,
        fuse_turning=false, contraction_optimize=true, verbose=false)
    unfused_ev2.cache = GraftEvolution.Evolution.EnvCache(unfused2.topo)
    empty!(_EAGER_BACKEND_CALLS)
    step!(fused_ev2, fused2, operator2, step_size)
    step!(public_unfused_ev2, public_unfused2, operator2, step_size)
    GraftEvolution.Evolution._tdvp2_sweep!(
        unfused_ev2, unfused2, operator2, step_size / 2; rev=false)
    GraftEvolution.Evolution._tdvp2_sweep!(
        unfused_ev2, unfused2, operator2, step_size / 2; rev=true)
    @test norm(to_dense(fused2) - to_dense(unfused2)) < 1e-9
    @test norm(to_dense(public_unfused2) - to_dense(unfused2)) < 1e-12
end

@testset "TDVP2 forced truncation stays within an exact-oracle budget" begin
    initial, operator, hamiltonian = _gse_fixture()
    step_size = -0.02im
    exact = exp(step_size * dense_hamiltonian(hamiltonian, initial)) *
        to_dense(initial)
    fused = copy(initial)
    unfused = copy(initial)
    untruncated = copy(initial)
    forced = TruncationScheme(maxdim=1)
    fused_ev = TDVP2(
        order=2, trunc=forced, krylovdim=12, tol=1e-12,
        contraction_optimize=true, verbose=false)
    unfused_ev = TDVP2(
        order=2, trunc=forced, krylovdim=12, tol=1e-12,
        fuse_turning=false, contraction_optimize=true, verbose=false)

    empty!(_EAGER_BACKEND_CALLS)
    step!(fused_ev, fused, operator, step_size)
    step!(unfused_ev, unfused, operator, step_size)
    step!(TDVP2(
        order=2, trunc=TruncationScheme(), krylovdim=12, tol=1e-12,
        contraction_optimize=true, verbose=false,
    ), untruncated, operator, step_size)

    fused_dense = to_dense(fused)
    unfused_dense = to_dense(unfused)
    diagnostics = (;
        fused_error=norm(fused_dense - exact),
        unfused_error=norm(unfused_dense - exact),
        fused_prior_delta=norm(fused_dense - unfused_dense),
        fused_norm=norm(fused_dense),
        unfused_norm=norm(unfused_dense),
    )
    @test all(isfinite, diagnostics)
    @test GraftEvolution.Evolution._tdvp_max_bond_dim(untruncated) > 1
    @test GraftEvolution.Evolution._tdvp_max_bond_dim(fused) == 1
    @test GraftEvolution.Evolution._tdvp_max_bond_dim(unfused) == 1
    @test 0.999 <= diagnostics.fused_norm <= 1.001
    @test 0.999 <= diagnostics.unfused_norm <= 1.001

    # Fixture-specific limits with headroom over the measured deterministic
    # errors; they do not assert a universal accuracy ordering between paths.
    @test diagnostics.fused_error <= 0.017
    @test diagnostics.unfused_error <= 0.017
    @test diagnostics.fused_prior_delta <= 2.0e-5
end

@testset "GSE and LSE wrappers forward eager" begin
    gse_state, operator, _ = _gse_fixture()
    empty!(_EAGER_BACKEND_CALLS)
    step!(_paper_gse(order=1, eager=false), gse_state, operator, -0.01im)
    @test length(_EAGER_BACKEND_CALLS) == 5
    @test all(call -> call.kind == :local && call.eager === false,
              _EAGER_BACKEND_CALLS)

    lse_state, _, _ = _gse_fixture()
    empty!(_EAGER_BACKEND_CALLS)
    step!(TDVP1_LSE(
        order=1,
        trunc=TruncationScheme(maxdim=8),
        max_add=2,
        eager=false,
        krylovdim=8,
        tol=1e-11,
        verbose=false,
    ), lse_state, operator, -0.01im)
    @test length(_EAGER_BACKEND_CALLS) == 5
    @test all(call -> call.kind == :local && call.eager === false,
              _EAGER_BACKEND_CALLS)

    lse_order2, _, _ = _gse_fixture()
    empty!(_EAGER_BACKEND_CALLS)
    step!(TDVP1_LSE(
        order=2,
        trunc=TruncationScheme(maxdim=8),
        max_add=2,
        eager=false,
        krylovdim=8,
        tol=1e-11,
        verbose=false,
    ), lse_order2, operator, -0.02im)
    @test length(_EAGER_BACKEND_CALLS) == 10
    @test all(call -> call.kind == :local && call.eager === false,
              _EAGER_BACKEND_CALLS)
    @test count(call -> call.time == -0.01im,
                _EAGER_BACKEND_CALLS) == 6
    @test count(call -> call.time == 0.01im,
                _EAGER_BACKEND_CALLS) == 4
    @test all(call -> call.time != -0.02im, _EAGER_BACKEND_CALLS)
end

@testset "GlobalKrylov forwards eager without changing the result" begin
    initial, operator, _ = _gse_fixture()
    eager_state = copy(initial)
    full_state = copy(initial)
    settings = (;
        krylovdim=8,
        maxiter=4,
        fit_nsweeps=4,
        fit_tol=1e-11,
        tol=1e-10,
        fit_verbose=false,
    )

    empty!(_EAGER_BACKEND_CALLS)
    step!(GlobalKrylov(; settings...), eager_state, operator, -0.01im)
    @test _EAGER_BACKEND_CALLS ==
          [(; kind=:global, time=-0.01im, eager=true)]

    empty!(_EAGER_BACKEND_CALLS)
    step!(GlobalKrylov(; settings..., eager=false),
          full_state, operator, -0.01im)
    @test _EAGER_BACKEND_CALLS ==
          [(; kind=:global, time=-0.01im, eager=false)]
    @test norm(to_dense(eager_state) - to_dense(full_state)) < 1e-10
end
