using Test
using Graft

@testset "Hamiltonian bath pole fitting support matrix" begin
    P = Partition([[:imp]])
    J(ω) = 0.2 * ω
    νs = [0.0, 0.4, 1.0, 2.0, 4.0, 8.0]
    exact_poles = [0.75, 1.25, 1.75]
    exact_residues = [0.04, 0.02, 0.01]
    Uν = [sum(2 * ω * r / (ν^2 + ω^2)
              for (ω, r) in zip(exact_poles, exact_residues)) for ν in νs]

    bath_real_0 = fit_bath(J, P; nmodes=3, ωmin=0.5, ωmax=2.0)
    @test bath_real_0 isa RealPoles
    @test bath_real_0.diagnostics.domain == :real_axis
    @test matsubara_reconstruct(bath_real_0, νs) ≈
          [sum(2 * ω * r / (ν^2 + ω^2)
               for (ω, r) in zip(bath_real_0.poles, bath_real_0.residues)) for ν in νs]

    bath_matsu_0 = fit_bath((; frequencies=im .* νs, values=Uν), P;
                            domain=:matsubara, nmodes=3, ωmin=0.5, ωmax=2.0)
    @test bath_matsu_0 isa RealPoles
    @test bath_matsu_0.diagnostics.domain == :matsubara
    @test bath_matsu_0.poles ≈ exact_poles
    @test bath_matsu_0.residues ≈ exact_residues atol = 1e-10
    @test matsubara_reconstruct(bath_matsu_0, im .* νs) ≈ Uν atol = 1e-10
    @test bath_matsu_0.diagnostics.block_diagnostics[1].relative_residual < 1e-10

    T = 0.5
    beta = inv(T)
    bose = [inv(expm1(beta * ω)) for ω in bath_real_0.poles]
    bath_real_T = fit_bath(J, P; T, nmodes=3, ωmin=0.5, ωmax=2.0)
    @test bath_real_T isa ThermofieldRealPoles
    @test bath_real_T.statistics == :boson
    @test bath_real_T.diagnostics.representation == :thermofield_star
    @test bath_real_T.diagnostics.domain == :real_axis
    @test bath_real_T.source.poles ≈ bath_real_0.poles
    @test bath_real_T.source.residues ≈ bath_real_0.residues
    @test bath_real_T.emission.residues ≈ (bose .+ 1) .* bath_real_0.residues
    @test bath_real_T.absorption.residues ≈ bose .* bath_real_0.residues
    @test bath_real_T.emission.residues .- bath_real_T.absorption.residues ≈
          bath_real_T.source.residues
    @test bath_real_T.diagnostics.source_reconstruction.relative_residual < 1e-14
    @test bath_real_T.diagnostics.beta_delta_energy ≈ beta * 0.5
    @test matsubara_reconstruct(bath_real_T, νs; channel=:source) ≈
          matsubara_reconstruct(bath_real_0, νs)
    @test matsubara_reconstruct(bath_real_T, νs; channel=:emission) -
          matsubara_reconstruct(bath_real_T, νs; channel=:absorption) ≈
          matsubara_reconstruct(bath_real_T, νs; channel=:source)

    bath_matsu_T = fit_bath((; frequencies=νs, values=Uν), P;
                            T, domain=:matsubara, nmodes=3,
                            ωmin=0.5, ωmax=2.0)
    @test bath_matsu_T isa ThermofieldRealPoles
    @test bath_matsu_T.diagnostics.domain == :matsubara
    @test bath_matsu_T.source.poles ≈ exact_poles
    @test bath_matsu_T.source.residues ≈ exact_residues atol = 1e-10
    nB = [inv(expm1(beta * ω)) for ω in exact_poles]
    @test bath_matsu_T.emission.residues ≈ (nB .+ 1) .* exact_residues atol = 1e-10
    @test bath_matsu_T.absorption.residues ≈ nB .* exact_residues atol = 1e-10
    @test matsubara_reconstruct(bath_matsu_T, im .* νs; channel=:source) ≈ Uν atol = 1e-10
    @test bath_matsu_T.source.diagnostics.block_diagnostics[1].relative_residual < 1e-10
    @test bath_matsu_T.diagnostics.source_reconstruction.relative_residual < 1e-14

    topo = TreeTopology(:imp, Pair{Symbol,Symbol}[])
    mounted = mount_bath(topo, bath_real_T, P; prefix=:tf)
    @test length(mounted.emission.sites) == 3
    @test length(mounted.absorption.sites) == 3
    @test length(mounted.sites) == 6
    @test nnodes(mounted.topology) == 7
    @test_throws MethodError mount_bath(topo, ComplexPoles(), P)
    @test_throws ArgumentError fit_bath((; times=[0.0], values=[1.0]), P;
                                        pole_family=:complex, domain=:time_bcf,
                                        nmodes=1, ωmin=0.1, ωmax=1.0)

    @test_throws ArgumentError fit_bath(J, P; kind=:fermion, T,
                                        nmodes=2, ωmin=0.1, ωmax=1.0)
    @test_throws ArgumentError BosonBath(J; partition=P, topology=topo,
                                         matter_ops=spin_ops(), boson_ops=boson_ops(1),
                                         T, nmodes=2, ωmin=0.5, ωmax=1.5)
end
