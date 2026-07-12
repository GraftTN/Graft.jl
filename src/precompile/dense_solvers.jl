# Ground-state and time-evolution families on a two-site dense fixture. Bond-1
# inputs force the expansion algorithms through their growth paths.
PrecompileTools.@compile_workload begin
    let
        topo = mps_topology(2)
        spins = spin_ops()
        phys = Dict(:site1 => spins.P, :site2 => spins.P)
        hamiltonian = OpSum() +
            Term(-1.0, SiteOp(:site1, :Z, spins.Z),
                       SiteOp(:site2, :Z, spins.Z)) +
            Term(-0.3, SiteOp(:site1, :X, spins.X)) +
            Term(-0.3, SiteOp(:site2, :X, spins.X))
        operator = ttno_from_opsum(hamiltonian, topo, phys; hermitian=true)
        state = TestUtils.random_ttns(
            Random.Xoshiro(0xbb67ae85), ComplexF64, topo, phys, Backend.ℂ^2)
        growth_state = TestUtils.random_ttns(
            Random.Xoshiro(0x3c6ef372), ComplexF64, topo, phys, Backend.ℂ^1)
        trunc = TruncationScheme(maxdim=2)

        dmrg1!(copy(state), operator;
               nsweeps=1, krylovdim=4, verbose=false)
        dmrg2!(copy(state), operator;
               trunc, nsweeps=1, krylovdim=4, verbose=false)
        dmrg1_3s!(copy(growth_state), operator;
                  trunc, nsweeps=1, max_add=1, krylovdim=4, verbose=false)

        evolve!(TDVP1(order=1, krylovdim=4, tol=1e-8, verbose=false),
                copy(state), operator, -0.01im, 1)
        step!(TDVP2(order=1, trunc=trunc, krylovdim=4,
                    tol=1e-8, verbose=false),
              copy(state), operator, -0.01im)
        step!(TDVP1_CBE(order=1, trunc=trunc, d_tilde_max=1,
                        krylovdim=4, tol=1e-8, verbose=false),
              copy(growth_state), operator, -0.01im)
        step!(GSE_TDVP(order=1, trunc=trunc, max_add=1,
                       krylovdim=4, tol=1e-8, verbose=false),
              copy(growth_state), operator, -0.01im)
        step!(LSE_TDVP(order=1, trunc=trunc, max_add=1,
                       krylovdim=4, tol=1e-8, verbose=false),
              copy(growth_state), operator, -0.01im)
        step!(GlobalKrylov(krylovdim=4, maxiter=2, tol=1e-8,
                           fit_nsweeps=1, fit_tol=1e-8),
              copy(state), operator, -0.01im)

        linsolve!(copy(state), operator, state;
                  a0=1.0, a1=0.01, krylovdim=4, maxiter=2,
                  tol=1e-8, fit_nsweeps=1, fit_tol=1e-8)
        for scheme in (LogTrapezoid(), LogBackwardEuler(),
                       LogGaussLegendre(1))
            step!(ImplicitLogTime(scheme=scheme, krylovdim=4, maxiter=2,
                                  tol=1e-8, fit_nsweeps=1, fit_tol=1e-8),
                  copy(state), operator, -0.01)
        end
    end
end
