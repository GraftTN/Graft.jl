const LegacyLowering =
    GraftTTNOBuild.LegacyTTNOBuild.LegacyLoweringInterface

@testset "legacy TTNO builder" begin
    spin = Symbolic.spin_ops()
    topo = Trees.mps_topology(2)
    phys = Dict(:site1 => spin.P, :site2 => spin.P)
    hamiltonian = Symbolic.OpSum() + Symbolic.Term(
        -0.7,
        Symbolic.SiteOp(:site1, :X, spin.X),
        Symbolic.SiteOp(:site2, :Z, spin.Z),
    )

    operator = GraftTTNOBuild.ttno_from_opsum(
        hamiltonian, topo, phys; hermitian=true)
    @test operator isa Networks.TTNO
    @test Networks.topology(operator) == topo
    @test Networks.check_arrows(operator)
    @test Networks.ishermitian(operator)
end

@testset "braided lowering certificate" begin
    fermion = Symbolic.fermion_ops_z2()
    topo = Trees.mps_topology(2)
    phys = Dict(:site1 => fermion.P, :site2 => fermion.P)
    factors = [
        Symbolic.SiteOp(:site1, :Cd, fermion.Cd),
        Symbolic.SiteOp(:site2, :C, fermion.C),
    ]
    ops = Dict(
        Trees.nodeindex(topo, factor.site) => factor for factor in factors)
    opnodes = sort!(collect(keys(ops)))
    unit_sector = one(Symbolic.charge(first(factors)))

    plan = LegacyLowering._build_braided_term_plan(
        topo,
        LegacyLowering._Euler(topo),
        phys,
        ops,
        opnodes,
        unit_sector,
        1.0,
    )
    @test plan.uses_certificate
    @test sort(plan.canonical_word) == opnodes
    @test sort(plan.native_word) == opnodes
    @test plan.restriction_charge[topo.root] == unit_sector
    @test prod(local_plan.word_scale for local_plan in plan.local_plans) ==
        plan.certificate_scale

    hopping = Symbolic.OpSum() + Symbolic.Term(1.0, factors...)
    operator = GraftTTNOBuild.ttno_from_opsum(hopping, topo, phys)
    @test operator isa Networks.TTNO
    @test Networks.check_arrows(operator)
end
