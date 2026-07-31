# SD2 — direct-sum realization tests (state-diagram-compiler plan v2, SD2
# deliverables; ADR-0003 decision 5 dual-producer view gate).
#
# Action equality is established against the legacy compiler on every
# supported input and additionally against the independent
# `dense_hamiltonian` kron oracle wherever that oracle is convention-valid
# (it carries no fermionic exchange signs, so graded fermionic inputs pin
# exact legacy equality instead; the legacy fermionic action is itself
# pinned by the permanent braided-sign regressions).

using GraftTestUtils: to_dense, dense_hamiltonian
using Graft.Backend: ⊠, ⊗

"Realize `H` through the typed direct-sum pipeline and return (O, report, plan, input, exps)."
function _sd2_realize(H, topo, phys; hermitian=false)
    hermiticity = hermitian ? TB.AssertedHermitian() : TB.NoHermiticityAssertion()
    input = TB.TTNOBuildInput(H, topo, phys; hermiticity)
    exps = TB.lower_terms(input, TB.AbelianScalarLowering())
    plan = TB.merge_channels(input, exps, TB.DirectSumMerge())
    O, report = TB.realize_ttno(input, plan)
    return O, report, plan, input, exps
end

"Assert direct-sum action equality vs legacy (exact) and the kron oracle."
function _sd2_check_action(H, topo, phys; kron_oracle::Bool, atol=1e-12)
    O, report, plan, input, exps = _sd2_realize(H, topo, phys)
    L = ttno_from_opsum(H, topo, phys)
    @test norm(to_dense(O) - to_dense(L)) < atol
    if kron_oracle
        @test norm(to_dense(O) - dense_hamiltonian(H, topo, phys)) < atol
    end
    @test report.plan_kind == :direct_sum
    @test report.nterms == length(H.terms)
    @test length(report.edges) == nnodes(topo) - 1
    @test all(e.dimension == length(H.terms) for e in report.edges)
    return O, report, plan, input, exps
end

@graft_testset "SD2 direct-sum action equality (dense, U1, boson, PP)" begin
    S = spin_ops()
    topo = mps_topology(3)
    phys = Dict(:site1 => S.P, :site2 => S.P, :site3 => S.P)
    H = OpSum()
    H += Term(-1.0, SiteOp(:site1, :Z, S.Z), SiteOp(:site2, :Z, S.Z))
    H += Term(-1.0, SiteOp(:site2, :Z, S.Z), SiteOp(:site3, :Z, S.Z))
    H += Term(-0.7, SiteOp(:site1, :X, S.X))
    H += Term(-0.7, SiteOp(:site2, :X, S.X))
    H += Term(-0.7, SiteOp(:site3, :X, S.X))
    _sd2_check_action(H, topo, phys; kron_oracle=true)

    # Physless branching node (star center without a physical space).
    star = star_topology(2, 1)
    sphys = Dict(:b1_1 => S.P, :b2_1 => S.P)
    Hs = OpSum()
    Hs += Term(0.8, SiteOp(:b1_1, :Z, S.Z), SiteOp(:b2_1, :Z, S.Z))
    Hs += Term(-0.3, SiteOp(:b1_1, :X, S.X))
    _sd2_check_action(Hs, star, sphys; kron_oracle=true)

    # Abelian U(1) spin with charged factors.
    U = spin_ops_u1()
    physu = Dict(:site1 => U.P, :site2 => U.P, :site3 => U.P)
    Hu = OpSum()
    Hu += Term(0.5, SiteOp(:site1, :Sp, U.Sp), SiteOp(:site2, :Sm, U.Sm))
    Hu += Term(0.5, SiteOp(:site2, :Sp, U.Sp), SiteOp(:site3, :Sm, U.Sm))
    Hu += Term(0.25, SiteOp(:site1, :Z, U.Z), SiteOp(:site3, :Z, U.Z))
    Hu += Term(0.1, SiteOp(:site2, :N, U.N))
    _sd2_check_action(Hu, topo, physu; kron_oracle=true)

    # Number-conserving bosonic U(1).
    B = boson_ops_u1(2)
    physb = Dict(:site1 => B.P, :site2 => B.P, :site3 => B.P)
    Hb = OpSum()
    Hb += Term(0.4, SiteOp(:site1, :Bd, B.Bd), SiteOp(:site2, :B, B.B))
    Hb += Term(0.4, SiteOp(:site2, :Bd, B.Bd), SiteOp(:site1, :B, B.B))
    Hb += Term(0.9, SiteOp(:site2, :N, B.N))
    Hb += Term(0.9, SiteOp(:site3, :N, B.N))
    _sd2_check_action(Hb, topo, physb; kron_oracle=true)

    # Projected-purification U(1) through the ppdress rewrite.
    Bo = boson_ops(2)
    ptopo = mps_topology(2)
    pphys = Dict(:site1 => Bo.P, :site2 => Bo.P)
    Hp = OpSum()
    Hp += Term(0.6, SiteOp(:site1, :Bd, Bo.Bd), SiteOp(:site2, :B, Bo.B))
    Hp += Term(0.6, SiteOp(:site1, :B, Bo.B), SiteOp(:site2, :Bd, Bo.Bd))
    Hp += Term(0.2, SiteOp(:site1, :N, Bo.N))
    Hpp, ptopo2, pphys2 = ppdress(Hp, ptopo, pphys; nmax=2)
    _sd2_check_action(Hpp, ptopo2, pphys2; kron_oracle=true)
end

@graft_testset "SD2 direct-sum action equality (graded fermionic, multimode)" begin
    F = fermion_ops_z2()
    topo = mps_topology(3)
    physf = Dict(:site1 => F.P, :site2 => F.P, :site3 => F.P)
    Hf = OpSum()
    Hf += Term(1.0, SiteOp(:site1, :Cd, F.Cd), SiteOp(:site3, :C, F.C))
    Hf += Term(1.0, SiteOp(:site3, :Cd, F.Cd), SiteOp(:site1, :C, F.C))
    Hf += Term(0.5, SiteOp(:site2, :N, F.N))
    _sd2_check_action(Hf, topo, physf; kron_oracle=false)

    # Fermionic fork with a longer-range crossing.
    fork = fork_topology(2, 1)
    sites = [nodeid(fork, i) for i in 1:nnodes(fork)]
    physk = Dict(s => F.P for s in sites)
    Hk = OpSum()
    Hk += Term(0.7, SiteOp(sites[1], :Cd, F.Cd), SiteOp(sites[end], :C, F.C))
    Hk += Term(0.7, SiteOp(sites[end], :Cd, F.Cd), SiteOp(sites[1], :C, F.C))
    Hk += Term(0.2, SiteOp(sites[2], :N, F.N))
    _sd2_check_action(Hk, fork, physk; kron_oracle=false)

    # Multimode fZ2 ⊠ U(1) carrier with sector degeneracy (CG-009 contract).
    M = _sd2_multimode_carrier(2)
    mtopo = mps_topology(2)
    mphys = Dict(:site1 => M.P, :site2 => M.P)
    Hm = OpSum()
    Hm += Term(0.8, SiteOp(:site1, :Cd1, M.Cd[1]), SiteOp(:site2, :C2, M.C[2]))
    Hm += Term(0.8, SiteOp(:site2, :Cd2, M.Cd[2]), SiteOp(:site1, :C1, M.C[1]))
    Hm += Term(0.3, SiteOp(:site1, :N2, M.N[2]))
    _sd2_check_action(Hm, mtopo, mphys; kron_oracle=false)
end

@graft_testset "SD2 dual-producer realization view gate" begin
    for build in (_sd1_dense_input, _sd1_fz2_input)
        input, topo = build()
        exps = TB.lower_terms(input, TB.AbelianScalarLowering())
        plan = TB.merge_channels(input, exps, TB.DirectSumMerge())
        zsd = TB.StateDiagram(input.topology, input.provenance, exps,
                              TB.MergeProofStep[], TB.OptimizerLog())
        v1 = TB.realization_view(plan)
        v2 = TB.realization_view(zsd)
        @test v1 == v2
        @test hash(v1) == hash(v2)
        O1, r1 = TB.realize_ttno(input, plan)
        O2, r2 = TB.realize_ttno(input, zsd)
        @test all(O1[i] == O2[i] for i in 1:nnodes(topo))
        @test r1.plan_kind == :direct_sum
        @test r2.plan_kind == :state_diagram
        @test r1.edges == r2.edges
        @test r1.nentries == r2.nentries
    end
end

@graft_testset "SD2 determinism and report stability" begin
    input, _ = _sd1_dense_input()
    exps = TB.lower_terms(input, TB.AbelianScalarLowering())
    plan = TB.merge_channels(input, exps, TB.DirectSumMerge())
    O1, r1 = TB.realize_ttno(input, plan)
    O2, r2 = TB.realize_ttno(input, plan)
    @test all(O1[i] == O2[i] for i in 1:nnodes(input.topology))
    @test r1 == r2
    @test hash(r1) == hash(r2)

    # Term order is not an action signal.
    S = spin_ops()
    topo = mps_topology(3)
    phys = Dict(:site1 => S.P, :site2 => S.P, :site3 => S.P)
    terms = [
        Term(-1.0, SiteOp(:site1, :Z, S.Z), SiteOp(:site2, :Z, S.Z)),
        Term(-0.7, SiteOp(:site2, :X, S.X)),
        Term(0.4, SiteOp(:site1, :X, S.X), SiteOp(:site3, :X, S.X)),
    ]
    Ha = OpSum() + terms[1] + terms[2] + terms[3]
    Hb = OpSum() + terms[3] + terms[1] + terms[2]
    Oa, _, _, _, _ = _sd2_realize(Ha, topo, phys)
    Ob, _, _, _, _ = _sd2_realize(Hb, topo, phys)
    @test norm(to_dense(Oa) - to_dense(Ob)) < 1e-12

    # Hermiticity contract propagates into the TTNO trait and the report.
    Oh, rh, _, _, _ = _sd2_realize(Ha, topo, phys; hermitian=true)
    @test Graft.Networks.ishermitian(Oh)
    @test rh.hermitian_asserted
    On, rn, _, _, _ = _sd2_realize(Ha, topo, phys; hermitian=false)
    @test !Graft.Networks.ishermitian(On)
    @test !rn.hermitian_asserted
end

@graft_testset "SD2 fail-closed realization gates" begin
    input1, _ = _sd1_dense_input()
    exps1 = TB.lower_terms(input1, TB.AbelianScalarLowering())
    plan1 = TB.merge_channels(input1, exps1, TB.DirectSumMerge())

    # Provenance mismatch fails closed.
    input2, _ = _sd1_fz2_input()
    @test_throws ArgumentError TB.realize_ttno(input2, plan1)

    # Nonzero-merge StateDiagram plans are not realizable before SD3.
    span = TB.ChannelSpan(TB.PLAIN_IDLE, ())
    fakeproof = TB.MergeProofStep(
        :identical_hyperedge, 2, span, TB.IdentitySpanRelation(span),
        TB.StructuralIdentityWitness(:identical_hyperedge),
        [TB.DegeneracyLabel(span, 2)], [TB.DegeneracyLabel(span, 1)])
    sd = TB.StateDiagram(input1.topology, input1.provenance, exps1,
                         [fakeproof], TB.OptimizerLog())
    @test_throws ArgumentError TB.realize_ttno(input1, sd)

    # Unsupported categories fail closed at the merge kernel.
    su2 = Graft.Backend.Vect[Graft.Backend.SU2Irrep](
        Graft.Backend.SU2Irrep(0) => 1, Graft.Backend.SU2Irrep(1//2) => 1)
    topo = mps_topology(2)
    Hsu2 = OpSum() + Term(1.0, SiteOp(:site1, :I, Graft.Backend.id(su2)))
    inputsu2 = TB.TTNOBuildInput(Hsu2, topo, Dict(:site1 => su2, :site2 => su2))
    @test_throws TB.MissingCategoryCapability TB.merge_channels(
        inputsu2, TB.TermTTNOExpansion{Graft.Backend.SU2Irrep,Float64}[],
        TB.DirectSumMerge())
end
