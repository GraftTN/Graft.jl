using Test
using GRAFT
using GRAFT.TestUtils
using GRAFT.Backend: ℂ
using GRAFT.Trees: edges
using LinearAlgebra: I, Hermitian, dot, eigvals, norm
using Random: Xoshiro

function _ilt_tfi(topo; J=1.0, g=0.41)
    S = spin_ops()
    H = OpSum()
    for (child, parent) in edges(topo)
        H += Term(-J, SiteOp(nodeid(topo, child), :Z, S.Z),
                  SiteOp(nodeid(topo, parent), :Z, S.Z))
    end
    for n in 1:nnodes(topo)
        H += Term(-g, SiteOp(nodeid(topo, n), :X, S.X))
    end
    return H
end

_ilt_normalized(v) = v / norm(v)
_ilt_state_error(v, ref) = norm(_ilt_normalized(v) - _ilt_normalized(ref))

function _ilt_run(ev, ψ0, O, grid)
    ψ = copy(ψ0)
    for (a, b) in zip(grid[1:end-1], grid[2:end])
        step!(ev, ψ, O, -(b - a))
    end
    return ψ
end

@testset "implicit logarithmic grids and paper schemes" begin
    @test logarithmic_time_grid(0.01, 0.08; nsteps_per_panel=2) ==
          [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08]
    @test_throws ArgumentError logarithmic_time_grid(0.0, 1.0)
    @test_throws ArgumentError logarithmic_time_grid(0.01, 0.07)
    @test_throws ArgumentError logarithmic_time_grid(0.01, 0.08;
                                                     nsteps_per_panel=0)
    @test_throws ArgumentError LogGaussLegendre(0)

    # One-step formula checks: default is the paper's trapezoid scheme;
    # one-stage Gauss collocation is algebraically the same method; backward
    # Euler remains available only when selected explicitly.
    topo2 = mps_topology(1)
    phys2 = Dict(:site1 => spin_ops().P)
    S2 = spin_ops()
    H2 = OpSum() + Term(0.37, SiteOp(:site1, :Z, S2.Z)) +
         Term(-0.61, SiteOp(:site1, :X, S2.X))
    O2 = ttno_from_opsum(H2, topo2, phys2; hermitian=true)
    ψ2 = random_ttns(Xoshiro(260602930), ComplexF64, topo2, phys2, ℂ^1)
    Hd2 = dense_hamiltonian(H2, ψ2)
    v2 = to_dense(ψ2)
    I2 = Matrix{ComplexF64}(I, length(v2), length(v2))
    h = 0.04
    trap_ref = (I2 + h * Hd2 / 2) \ ((I2 - h * Hd2 / 2) * v2)
    be_ref = (I2 + h * Hd2) \ v2

    common = (; krylovdim=4, maxiter=3, tol=1e-11,
              fit_nsweeps=2, fit_tol=1e-12)
    ψtrap = copy(ψ2)
    trap = ImplicitLogTime(; common...)
    step!(trap, ψtrap, O2, -h)
    @test trap.scheme isa LogTrapezoid
    @test trap.last_info.converged == 1
    @test norm(to_dense(ψtrap) - trap_ref) < 2e-7

    ψbe = copy(ψ2)
    step!(ImplicitLogTime(; scheme=LogBackwardEuler(), common...), ψbe, O2, -h)
    @test norm(to_dense(ψbe) - be_ref) < 2e-7

    ψg1 = copy(ψ2)
    g1 = ImplicitLogTime(; scheme=LogGaussLegendre(1), common...)
    step!(g1, ψg1, O2, -h)
    @test length(g1.last_stage_infos) == 1
    @test norm(to_dense(ψg1) - trap_ref) < 2e-7

    ψg2 = copy(ψ2)
    g2 = ImplicitLogTime(; scheme=LogGaussLegendre(2), common...)
    step!(g2, ψg2, O2, -h)
    exact2 = exp(-h * Hd2) * v2
    @test length(g2.last_stage_infos) == 2
    @test g2.last_info.converged == 1
    @test norm(to_dense(ψg2) - exact2) < 2e-7

    ψreal = random_ttns(Xoshiro(260602931), Float64, topo2, phys2, ℂ^1)
    @test_throws ArgumentError step!(ImplicitLogTime(scheme=LogGaussLegendre(2)),
                                     ψreal, O2, -h)

    # On the one-site exact manifold, verify the paper's n^-2 panel convergence
    # and spectral convergence as the number of Gauss nodes increases.
    τfirst, τmax = 0.02, 0.08
    exact = exp(-τmax * Hd2) * v2
    trap_errors = Float64[]
    for n in (1, 2, 4)
        grid = logarithmic_time_grid(τfirst, τmax; nsteps_per_panel=n)
        ev = ImplicitLogTime(; common...)
        push!(trap_errors, _ilt_state_error(to_dense(_ilt_run(ev, ψ2, O2, grid)),
                                            exact))
    end
    @test trap_errors[2] < trap_errors[1] / 3
    @test trap_errors[3] < trap_errors[2] / 3

    panel_grid = logarithmic_time_grid(τfirst, τmax)
    spectral_errors = Float64[]
    for stages in (1, 2, 3)
        ev = ImplicitLogTime(; scheme=LogGaussLegendre(stages), common...)
        result = _ilt_run(ev, ψ2, O2, panel_grid)
        @test ev.last_info.converged == 1
        push!(spectral_errors, _ilt_state_error(to_dense(result), exact))
    end
    @test spectral_errors[2] < spectral_errors[1] / 5
    @test spectral_errors[3] < spectral_errors[2] / 5

    # A genuine branching tree: spin root plus two leaves. One step of each
    # paper scheme must match its independent dense formula. This is the tree
    # topology gate; convergence-order scans above stay on the cheap exact
    # manifold so the default CI does not spend minutes repeating ALS solves.
    topo = star_topology(2, 1; center=:imp, prefix=:arm)
    phys = Dict(nodeid(topo, n) => spin_ops().P for n in 1:nnodes(topo))
    H0 = _ilt_tfi(topo; g=0.33)
    probe = random_ttns(Xoshiro(260602932), ComplexF64, topo, phys, ℂ^2)
    H0d = dense_hamiltonian(H0, probe)
    E0 = minimum(eigvals(Hermitian(H0d)))
    S = spin_ops()
    K = H0 + Term(-E0, SiteOp(:imp, :I, S.I))
    O = ttno_from_opsum(K, topo, phys; hermitian=true)
    Kd = dense_hamiltonian(K, probe)
    v0 = to_dense(probe)
    htree = 0.02
    Itree = Matrix{ComplexF64}(I, length(v0), length(v0))
    tree_trap_ref = (Itree + htree * Kd / 2) \
                    ((Itree - htree * Kd / 2) * v0)
    tree_common = (; krylovdim=4, maxiter=2, tol=1e-8,
                   fit_nsweeps=2, fit_tol=1e-9)
    tree_trap = copy(probe)
    step!(ImplicitLogTime(; tree_common...), tree_trap, O, -htree)
    @test _ilt_state_error(to_dense(tree_trap), tree_trap_ref) < 2e-5

    tree_spectral = copy(probe)
    step!(ImplicitLogTime(; scheme=LogGaussLegendre(2), tree_common...),
          tree_spectral, O, -htree)
    @test _ilt_state_error(to_dense(tree_spectral), exp(-htree * Kd) * v0) < 2e-5

    # A-stability smoke: a panel far larger than the inverse spectral radius
    # remains finite for both paper schemes.
    for scheme in (LogTrapezoid(), LogGaussLegendre(3))
        ψlarge = copy(ψ2)
        step!(ImplicitLogTime(; scheme, common...), ψlarge, O2, -8.0)
        @test all(isfinite, to_dense(ψlarge))
    end
end
