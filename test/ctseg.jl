@graft_testset "M2 finite-mode CTSEG action" begin
    action = FiniteModeAction(
        beta=5.0,
        mu=0.3,
        static_interaction=2.0,
        n0=1.0,
        bath_energies=[-0.7, 0.4],
        bath_couplings=ComplexF64[0.25 + 0.1im, -0.2im],
        boson_frequencies=[0.6, 1.1],
        boson_couplings=[0.3, -0.2],
        orbital_convention=:spinful,
        density_convention=:shifted)

    delta = hybridization_iw(action, -2:2)
    @test delta.frequencies ==
        [(2n + 1) * pi / action.beta for n in -2:2]
    @test delta.values ≈ [
        sum(abs2(V) / (im * omega - epsilon)
            for (epsilon, V) in zip(
                action.bath_energies, action.bath_couplings))
        for omega in delta.frequencies
    ]

    interaction = retarded_interaction_iv(action, 0:3)
    @test interaction.frequencies ==
        [2n * pi / action.beta for n in 0:3]
    @test interaction.values ≈ [
        -sum(2g^2 * omega0 / (nu^2 + omega0^2)
             for (omega0, g) in zip(
                 action.boson_frequencies, action.boson_couplings))
        for nu in interaction.frequencies
    ]
    @test finite_mode_hash(action) == finite_mode_hash(action)
    changed = FiniteModeAction(
        beta=action.beta,
        mu=action.mu,
        static_interaction=action.static_interaction,
        n0=action.n0,
        bath_energies=action.bath_energies,
        bath_couplings=action.bath_couplings,
        boson_frequencies=action.boson_frequencies,
        boson_couplings=[0.3, -0.21],
        orbital_convention=:spinful,
        density_convention=:shifted)
    @test finite_mode_hash(action) != finite_mode_hash(changed)

    @test_throws ArgumentError FiniteModeAction(
        beta=0, bath_energies=[], bath_couplings=[])
    @test_throws ArgumentError FiniteModeAction(
        beta=1, bath_energies=[0.0], bath_couplings=[])
    @test_throws ArgumentError FiniteModeAction(
        beta=1, boson_frequencies=[0.0], boson_couplings=[1.0])

    mktempdir() do directory
        input_path = joinpath(directory, "ctseg_input.csv")
        @test write_ctseg_input_csv(
            input_path, action;
            fermionic_indices=0:2, bosonic_indices=0:1) == input_path
        input_lines = readlines(input_path)
        @test length(input_lines) == 1 + 3 + 2
        @test all(contains(finite_mode_hash(action)), input_lines[2:end])
        @test contains("-0.7;0.4")(input_lines[2])
        @test contains("0.25:0.1")(input_lines[2])
    end
end

@graft_testset "M2 CTSEG artifact and acceptance gate" begin
    action = FiniteModeAction(
        beta=4.0,
        mu=0.1,
        static_interaction=1.5,
        n0=0.5,
        bath_energies=[0.2],
        bath_couplings=[0.35],
        boson_frequencies=[0.8],
        boson_couplings=[0.25])
    metadata = CTSEGMetadata(
        action;
        equilibrated=true,
        error_stable=true,
        nsamples=20_000,
        jackknife_bins=40,
        warmup_cycles=100,
        length_cycle=8,
        seed=17)
    artifact = CTSEGArtifact(metadata, [
        CTSEGDatum(:density, :scalar, 0.0, 0.52, 0.01),
        CTSEGDatum(:Gtau, :tau, 1.0, -0.31, 0.02),
        CTSEGDatum(:Giw, :fermionic_iw, pi / 4, -0.2 - 0.4im, 0.01),
    ])

    mktempdir() do directory
        path = joinpath(directory, "ctseg_results.csv")
        @test write_ctseg_results_csv(path, artifact) == path
        restored = read_ctseg_results_csv(path)
        @test restored.metadata == artifact.metadata
        @test restored.data == artifact.data
        @test restored.metadata.cycles_per_replica == 500
        @test restored.metadata.warmup_cycles == 100
        @test restored.metadata.length_cycle == 8
        @test restored.metadata.seed == 17
    end

    reference = [
        ThermalBenchmarkDatum(
            :density, :scalar, 0.0, 0.50;
            deterministic_error=0.001, cutoff_error=0.002),
        ThermalBenchmarkDatum(
            :Gtau, :tau, 1.0, -0.28;
            stderr=0.005, deterministic_error=0.001),
        ThermalBenchmarkDatum(
            :Giw, :fermionic_iw, pi / 4, -0.19 - 0.38im;
            cutoff_error=0.002),
    ]
    comparison = compare_ctseg(reference, artifact; action)
    @test comparison.passed
    @test length(comparison.rows) == 3
    @test comparison.sigma == 3
    @test comparison.rows[1].threshold ≈
        3 * 0.01 + 0.001 + 0.002

    failed_reference = copy(reference)
    failed_reference[1] =
        ThermalBenchmarkDatum(:density, :scalar, 0.0, 0.4)
    failed = compare_ctseg(failed_reference, artifact; action)
    @test !failed.passed
    @test !failed.rows[1].passed

    unstable = CTSEGArtifact(
        CTSEGMetadata(
            action;
            equilibrated=true,
            error_stable=false,
            nsamples=100,
            jackknife_bins=10),
        artifact.data)
    @test_throws ArgumentError compare_ctseg(reference, unstable; action)
    @test_throws ArgumentError compare_ctseg(
        reference[1:2], artifact; action)
    @test_throws ArgumentError CTSEGArtifact(
        metadata, [artifact.data[1], artifact.data[1]])
end

@graft_testset "M2 CTSEG readiness certification" begin
    action = FiniteModeAction(
        beta=2.0,
        mu=0.1,
        bath_energies=[0.2],
        bath_couplings=[0.3],
        boson_frequencies=[0.8],
        boson_couplings=[0.2])
    short_metadata = CTSEGMetadata(
        action;
        equilibrated=false,
        error_stable=false,
        nsamples=4_000,
        jackknife_bins=8,
        cycles_per_replica=500,
        warmup_cycles=100,
        length_cycle=16,
        seed=11)
    long_metadata = CTSEGMetadata(
        action;
        equilibrated=false,
        error_stable=false,
        nsamples=8_000,
        jackknife_bins=8,
        cycles_per_replica=1_000,
        warmup_cycles=200,
        length_cycle=16,
        seed=29)
    short = CTSEGArtifact(short_metadata, [
        CTSEGDatum(:density, :scalar, 0.0, 0.50, 0.02),
        CTSEGDatum(:Gtau, :tau, 0.0, -0.50, 0.02),
        CTSEGDatum(:Gtau, :tau, 1.0, -0.40, 0.03),
    ])
    long = CTSEGArtifact(long_metadata, [
        CTSEGDatum(:density, :scalar, 0.0, 0.505, 0.014),
        CTSEGDatum(:Gtau, :tau, 0.0, -0.495, 0.014),
        CTSEGDatum(:Gtau, :tau, 1.0, -0.41, 0.021),
    ])

    report = assess_ctseg_readiness(short, long)
    @test report.passed
    @test report.sample_ratio == 2
    @test report.observed_outliers == 0
    @test all(<=(report.error_ratio_limit),
              values(report.observable_error_ratios))

    certified = certify_ctseg(short, long)
    @test certified.metadata.equilibrated
    @test certified.metadata.error_stable
    @test certified.data == long.data

    same_seed = CTSEGArtifact(
        CTSEGMetadata(
            action;
            equilibrated=false,
            error_stable=false,
            nsamples=8_000,
            jackknife_bins=8,
            cycles_per_replica=1_000,
            warmup_cycles=200,
            length_cycle=16,
            seed=11),
        long.data)
    @test_throws ArgumentError assess_ctseg_readiness(short, same_seed)

    noisy = CTSEGArtifact(long_metadata, [
        CTSEGDatum(:density, :scalar, 0.0, 0.505, 0.05),
        long.data[2],
        long.data[3],
    ])
    @test !assess_ctseg_readiness(short, noisy).passed
    @test_throws ArgumentError certify_ctseg(short, noisy)
end
