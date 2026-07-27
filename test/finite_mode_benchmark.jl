function _benchmark_data(offset=0.0; lbo=0.0)
    return [
        ThermalBenchmarkDatum(:logZ, :scalar, 0.0, 1.2 + offset;
                              lbo_error=lbo),
        ThermalBenchmarkDatum(:density, :scalar, 0.0, 0.5 + offset;
                              lbo_error=lbo),
        ThermalBenchmarkDatum(:boson_occupation, :scalar, 0.0, 0.2 + offset;
                              lbo_error=lbo),
        ThermalBenchmarkDatum(:Gtau, :tau, 0.0, -0.5 + offset;
                              lbo_error=lbo),
        ThermalBenchmarkDatum(:Gtau, :tau, 1.0, -0.3 + offset;
                              lbo_error=lbo),
        ThermalBenchmarkDatum(:chi_nn, :tau, 0.0, 0.25 + offset;
                              lbo_error=lbo),
        ThermalBenchmarkDatum(:Giw, :fermionic_iw, pi, -0.2im + offset;
                              lbo_error=lbo),
        ThermalBenchmarkDatum(:chi_nn_iv, :bosonic_iv, 0.0, 0.1 + offset;
                              lbo_error=lbo),
    ]
end

@graft_testset "M2 finite-mode benchmark records and gates" begin
    action = FiniteModeAction(
        beta=2.0,
        n0=0.5,
        bath_energies=[0.1],
        bath_couplings=[0.2],
        boson_frequencies=[0.7],
        boson_couplings=[0.3])
    cells = [
        FiniteModeBenchmarkCell(
            :spinless_one_mode, action, :ed, :plain, nmax,
            _benchmark_data(offset);
            truncation=(; scheme=:none),
            propagation_grid=[0.0, 1.0, 2.0],
            wall_time=0.1)
        for (nmax, offset) in [(1, 0.03), (2, 0.005), (3, 0.001)]
    ]
    report = assess_boson_cutoff(cells; target_uncertainty=0.01)
    @test report.converged
    @test report.final_pair == (2, 3)
    @test length(report.rows) == 2 * length(_benchmark_data())
    @test any(!row.passed for row in report.rows
              if row.low_nmax == 1)

    tight = assess_boson_cutoff(cells; target_uncertainty=1e-4)
    @test !tight.converged

    plain = cells[end]
    pp = FiniteModeBenchmarkCell(
        :spinless_one_mode, action, :ed, :pp_untruncated, 3,
        _benchmark_data(0.001 + 1e-11);
        truncation=(; pp_bond=:untruncated),
        propagation_grid=[0.0, 1.0, 2.0])
    exact_pp = compare_representations(plain, pp; tolerance=1e-10)
    @test exact_pp.passed

    pp_lbo = FiniteModeBenchmarkCell(
        :spinless_one_mode, action, :ed, :pp_lbo, 3,
        _benchmark_data(0.001 + 2e-3; lbo=3e-3);
        truncation=(; pp_maxdim=2),
        propagation_grid=[0.0, 1.0, 2.0])
    lbo = compare_representations(plain, pp_lbo; tolerance=1e-5)
    @test lbo.passed
    @test all(row.lbo_allowance == 3e-3 for row in lbo.rows)

    @test_throws ArgumentError FiniteModeBenchmarkCell(
        :incomplete, action, :ed, :plain, 2,
        _benchmark_data()[1:3])
    @test_throws ArgumentError assess_boson_cutoff(
        cells[1:1]; target_uncertainty=0.1)

    mktempdir() do directory
        path = joinpath(directory, "p4.csv")
        @test write_finite_mode_benchmark_csv(path, cells) == path
        lines = readlines(path)
        @test length(lines) == 1 + length(cells) * length(_benchmark_data())
        @test contains(finite_mode_hash(action))(lines[2])
    end
end
