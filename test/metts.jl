using Graft.Thermal

struct METTSLocalZEvolver <: Evolver
    coefficient::Float64
end

struct METTSSiteZEvolver <: Evolver
    site::Symbol
    coefficient::Float64
end

function Graft.step!(ev::METTSLocalZEvolver, psi::TTNS, ::TTNO, dz::Number)
    site = :site1
    P = physspace(psi, nodeindex(topology(psi), site))
    gate = TensorMap(
        ComplexF64[exp(dz * ev.coefficient) 0;
                   0 exp(-dz * ev.coefficient)],
        P ← P,
    )
    result = apply_local(psi, gate, site)
    psi.tensors .= result.tensors
    psi.center = center(result)
    return psi
end

function Graft.step!(ev::METTSSiteZEvolver, psi::TTNS, ::TTNO, dz::Number)
    P = physspace(psi, nodeindex(topology(psi), ev.site))
    gate = TensorMap(
        ComplexF64[exp(dz * ev.coefficient) 0;
                   0 exp(-dz * ev.coefficient)],
        P ← P,
    )
    result = apply_local(psi, gate, ev.site)
    psi.tensors .= result.tensors
    psi.center = center(result)
    return psi
end

@graft_testset "M2 purified real-time auxiliary backward evolution" begin
    spin = spin_ops()
    topo = mps_topology(1)
    phys = Dict(:site1 => spin.P)
    H = OpSum() + Term(0.5, SiteOp(:site1, :Z, spin.Z))
    problem = purification_problem(H, topo, phys; hermitian=true)
    times = collect(0.0:0.2:1.0)

    trajectory = thermalize(
        Purified(), problem, 0.0;
        evolver=METTSSiteZEvolver(:site1, 0.5), nsteps=1)
    plain = thermal_realtime_correlator(
        Purified(), problem, :site1 => spin.X, :site1 => spin.X, times;
        evolver=METTSSiteZEvolver(:site1, 0.5), trajectory)
    backward = thermal_realtime_correlator(
        Purified(aux_evolution=:backward), problem,
        :site1 => spin.X, :site1 => spin.X, times;
        evolver=METTSSiteZEvolver(:site1, 0.5),
        aux_evolver=METTSSiteZEvolver(:site1_thermal, 0.5),
        trajectory)
    @test real.(plain.values) ≈ cos.(times) atol=1e-12
    @test maximum(abs, imag.(plain.values)) < 1e-12
    @test backward.values ≈ plain.values atol=1e-12
    @test backward.metadata.aux_evolution == :backward

    custom_aux = Graft.Thermal._automatic_aux_hamiltonian(problem)
    custom = thermal_realtime_correlator(
        Purified(aux_evolution=custom_aux), problem,
        :site1 => spin.X, :site1 => spin.X, times;
        evolver=METTSSiteZEvolver(:site1, 0.5),
        aux_evolver=METTSSiteZEvolver(:site1_thermal, 0.5),
        trajectory)
    @test custom.values ≈ plain.values atol=1e-12
    @test_throws ArgumentError thermal_realtime_correlator(
        Purified(), problem, :site1 => spin.X, :site1 => spin.X, times;
        evolver=METTSSiteZEvolver(:site1, 0.5), trajectory,
        aux_hamiltonian=custom_aux)
end

@graft_testset "M2 HybridMETTS partial purification" begin
    spin = spin_ops()
    topo = mps_topology(2)
    phys = Dict(:site1 => spin.P, :site2 => spin.P)
    H = OpSum() + Term(0.5, SiteOp(:site1, :Z, spin.Z))
    problem = purification_problem(H, topo, phys; hermitian=true)
    observable = physical_ttno(problem, H; hermitian=true)

    chain = thermalize(
        HybridMETTS(; rng=Xoshiro(20260728), sampled_sites=[:site1],
                    collapse_basis=:alternating, burnin=10,
                    nsamples=200, thin=1),
        problem, 1.0;
        evolver=METTSLocalZEvolver(0.5), nsteps=1,
    )
    stats = metts_statistics(chain, observable)
    exact_energy = -0.5 * tanh(0.5)
    @test length(chain.samples) == 200
    @test abs(real(stats.mean) - exact_energy) < 6stats.stderr
    @test chain.metadata.representation == :hybrid_metts
    @test chain.metadata.sampled_sites == [:site1]
    @test chain.metadata.purified_sites == [:site2]
    @test chain.metadata.resets_purified_groups
    @test topology(chain.final_product) == problem.topo_doubled
    @test thermal_expect(chain, observable) ≈ stats.mean

    @test_throws ArgumentError HybridMETTS(
        ; rng=Xoshiro(1), sampled_sites=Symbol[])
    @test_throws ArgumentError thermalize(
        HybridMETTS(; rng=Xoshiro(1), sampled_sites=[:site1, :site2],
                    nsamples=2),
        problem, 1.0; evolver=METTSLocalZEvolver(0.5), nsteps=1)
end

@graft_testset "M2 METTS sampling and statistics" begin
    spin = spin_ops()
    topo = mps_topology(1)
    phys = Dict(:site1 => spin.P)
    H = OpSum() + Term(0.5, SiteOp(:site1, :Z, spin.Z))
    problem = purification_problem(H, topo, phys; hermitian=true)
    observable = physical_ttno(problem, H; hermitian=true, doubled=false)

    chain = thermalize(
        METTS(; rng=Xoshiro(20260727), collapse_basis=:alternating,
              burnin=20, nsamples=400, thin=1),
        problem, 1.0;
        evolver=METTSLocalZEvolver(0.5), nsteps=1,
    )
    stats = metts_statistics(chain, observable)
    exact_energy = -0.5 * tanh(0.5)
    @test length(chain.samples) == 400
    @test abs(real(stats.mean) - exact_energy) < 5stats.stderr
    @test stats.effective_samples <= stats.nsamples
    @test stats.tau_int >= 0.5
    @test topology(chain.final_product) == topo
    @test chain.metadata.representation == :metts
    @test thermal_expect(chain, observable) ≈ stats.mean

    repeat_chain = thermalize(
        METTS(; rng=Xoshiro(20260727), collapse_basis=:alternating,
              burnin=20, nsamples=400, thin=1),
        problem, 1.0;
        evolver=METTSLocalZEvolver(0.5), nsteps=1,
    )
    @test [sample.outcomes for sample in chain.samples] ==
          [sample.outcomes for sample in repeat_chain.samples]
    @test thermal_expect(chain, observable) ==
          thermal_expect(repeat_chain, observable)

    continued = thermalize(
        METTS(; rng=Xoshiro(999), collapse_basis=:alternating,
              burnin=999, nsamples=10, thin=1),
        problem, 1.0;
        evolver=METTSLocalZEvolver(0.5), nsteps=1,
        resume_from=chain,
    )
    one_shot = thermalize(
        METTS(; rng=Xoshiro(20260727), collapse_basis=:alternating,
              burnin=20, nsamples=410, thin=1),
        problem, 1.0;
        evolver=METTSLocalZEvolver(0.5), nsteps=1,
    )
    @test length(continued.samples) == 410
    @test [sample.outcomes for sample in continued.samples] ==
          [sample.outcomes for sample in one_shot.samples]
    @test continued.total_steps == one_shot.total_steps
    @test continued.burnin == 20

    seeded = thermalize(
        METTS(; rng=Xoshiro(77), collapse_basis=:computational,
              burnin=0, nsamples=2),
        problem, 1.0;
        evolver=METTSLocalZEvolver(0.5), nsteps=1,
        initial_state=chain.final_product, collapse_initial=true)
    @test haskey(first(seeded.samples).outcomes, :site1)
    mktempdir() do directory
        path = joinpath(directory, "metts.jld2")
        checkpoint!(seeded, path; metadata=(; representation=:metts))
        restored = resume(path)
        @test restored.metadata.representation == :metts
        disk_continued = thermalize(
            METTS(; rng=Xoshiro(999), collapse_basis=:computational,
                  burnin=99, nsamples=2),
            problem, 1.0;
            evolver=METTSLocalZEvolver(0.5), nsteps=1,
            resume_from=restored.state)
        direct = thermalize(
            METTS(; rng=Xoshiro(77), collapse_basis=:computational,
                  burnin=0, nsamples=4),
            problem, 1.0;
            evolver=METTSLocalZEvolver(0.5), nsteps=1,
            initial_state=chain.final_product, collapse_initial=true)
        @test [sample.outcomes for sample in disk_continued.samples] ==
              [sample.outcomes for sample in direct.samples]
        @test thermal_expect(disk_continued, observable) ==
              thermal_expect(direct, observable)
    end

    @test_throws ArgumentError METTS(; rng=Xoshiro(1), nsamples=0)
    @test_throws ArgumentError thermalize(
        METTS(; rng=Xoshiro(1), nsamples=2),
        problem, 1.0; evolver=METTSLocalZEvolver(0.5), nsteps=1,
        collapse_initial=true)
    @test_throws ArgumentError thermalize(
        METTS(; rng=Xoshiro(1), nsamples=2),
        problem, 1.0; evolver=METTSLocalZEvolver(0.5), nsteps=1,
        state_transform=state -> begin
            state.tensors[state.center] *= 0
            state
        end)
    @test_throws ArgumentError thermalize(
        METTS(; rng=Xoshiro(1), nsamples=2),
        problem, -1.0; evolver=METTSLocalZEvolver(0.5), nsteps=1)
    @test_throws ArgumentError thermal_expect(
        chain, physical_ttno(problem, H; hermitian=true))
end

@graft_testset "M2 graded METTS conditional collapse" begin
    fermion = fermion_ops_z2()
    topo = mps_topology(2)
    phys = Dict(:site1 => fermion.P, :site2 => fermion.P)
    H = OpSum() +
        Term(0.2, SiteOp(:site1, :N, fermion.N)) +
        Term(-0.3, SiteOp(:site2, :N, fermion.N))
    problem = purification_problem(H, topo, phys; hermitian=true)
    even = FermionParity(0)
    odd = FermionParity(1)
    bond = Graft.Backend.Vect[FermionParity](even => 1, odd => 1)
    initial = random_ttns(Xoshiro(26060727), ComplexF64, topo, phys, bond)
    chain = thermalize(
        METTS(; rng=Xoshiro(26060728), collapse_basis=:computational,
              burnin=0, nsamples=2),
        problem, 0.0;
        evolver=METTSLocalZEvolver(0.0), nsteps=1,
        initial_state=initial, collapse_initial=true)
    @test length(chain.samples) == 2
    @test all(length(sample.outcomes) == 2 for sample in chain.samples)
    @test all(
        haskey(sample.outcomes, site)
        for sample in chain.samples for site in (:site1, :site2))

    descriptors = Dict(
        :site1 => (even, ComplexF64[1]),
        :site2 => (even, ComplexF64[1]))
    even_product = Graft.Thermal._metts_product_state(
        ComplexF64, topo, phys, descriptors)
    @test norm(even_product) ≈ 1
end
