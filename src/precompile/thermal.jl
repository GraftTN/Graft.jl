# Purification construction, finite-temperature propagation/observables, and
# both thermal and zero-temperature correlator drivers.
PrecompileTools.@compile_workload begin
    let
        spins = spin_ops()
        topo = mps_topology(1)
        phys = Dict(:site1 => spins.P)
        hamiltonian = OpSum() +
            Term(0.5, SiteOp(:site1, :Z, spins.Z))
        problem = purification_problem(
            hamiltonian, topo, phys; hermitian=true)
        observable = physical_ttno(
            problem,
            OpSum() + Term(1.0, SiteOp(:site1, :Z, spins.Z));
            hermitian=true)
        state0 = infinite_temperature_state(problem)
        thermal_expect(state0, observable)

        beta = 0.02
        evolver = TDVP2(order=1, trunc=TruncationScheme(maxdim=2),
                        krylovdim=4, tol=1e-8, verbose=false)
        trajectory = thermalize(
            Purified(), problem, beta; evolver, nsteps=1,
            save_betas=[beta / 2])
        state_at(trajectory, beta / 2)
        thermal_expect(trajectory, observable)
        thermal_correlator(
            Purified(), problem,
            :site1 => spins.Z, :site1 => spins.Z,
            beta, [0.0, beta];
            evolver, trajectory, prop_nsteps=1)

        correlator_series(
            state0.psi, 0.0,
            :site1 => spins.X, :site1 => spins.X,
            (0.0, 0.01);
            H=problem.K, evolver=evolver,
            metadata=(; workload=:precompile))
    end
end
