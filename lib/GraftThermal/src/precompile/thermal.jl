# Purification construction, finite-temperature propagation/observables, and
# both thermal and zero-temperature correlator drivers belong to GraftThermal.
PrecompileTools.@compile_workload begin
    let
        spins = Symbolic.spin_ops()
        topo = Trees.mps_topology(1)
        phys = Dict(:site1 => spins.P)
        hamiltonian = Symbolic.OpSum() +
            Symbolic.Term(0.5, Symbolic.SiteOp(:site1, :Z, spins.Z))
        problem = Thermal.purification_problem(
            hamiltonian, topo, phys; hermitian=true)
        observable = Thermal.physical_ttno(
            problem,
            Symbolic.OpSum() +
                Symbolic.Term(
                    1.0, Symbolic.SiteOp(:site1, :Z, spins.Z));
            hermitian=true,
        )
        state0 = Thermal.infinite_temperature_state(problem)
        Thermal.thermal_expect(state0, observable)

        beta = 0.02
        evolver = Evolution.TDVP2(
            order=1, trunc=Backend.TruncationScheme(maxdim=2),
            krylovdim=4, tol=1e-8, verbose=false,
        )
        trajectory = Thermal.thermalize(
            Thermal.Purified(), problem, beta;
            evolver, nsteps=1, save_betas=[beta / 2],
        )
        Thermal.state_at(trajectory, beta / 2)
        Thermal.thermal_expect(trajectory, observable)
        Thermal.thermal_correlator(
            Thermal.Purified(), problem,
            :site1 => spins.Z, :site1 => spins.Z,
            beta, [0.0, beta];
            evolver, trajectory, prop_nsteps=1,
        )

        Evolution.correlator_series(
            state0.psi, 0.0,
            :site1 => spins.X, :site1 => spins.X,
            (0.0, 0.01);
            H=problem.K, evolver=evolver,
            metadata=(; workload=:precompile),
        )
    end
end
