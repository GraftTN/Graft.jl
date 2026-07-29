# SD3 — exact structural sharing tests (state-diagram-compiler plan v2, SD3
# deliverables): identical-hyperedge/identical-subtree merging with typed
# proofs, idempotence, insertion-order invariance, route/frame overmerge
# guards, and replayable/reversible proofs.

using Graft.Backend: dim

function _sd3_build(H, topo, phys)
    input = TB.TTNOBuildInput(H, topo, phys)
    exps = TB.lower_terms(input, TB.AbelianScalarLowering())
    ds = TB.merge_channels(input, exps, TB.DirectSumMerge())
    sd = TB.merge_channels(input, exps,
                           TB.StateDiagramMerge(TB.StructuralOptimizer()))
    return input, exps, ds, sd
end

function _sd3_check(H, topo, phys; atol=1e-12)
    input, exps, ds, sd = _sd3_build(H, topo, phys)
    Ods, _ = TB.realize_ttno(input, ds)
    Osd, rsd = TB.realize_ttno(input, sd)
    L = ttno_from_opsum(H, topo, phys)
    @test norm(to_dense(Osd) - to_dense(Ods)) < atol
    @test norm(to_dense(Osd) - to_dense(L)) < atol
    # Structural sharing reproduces the legacy state-diagram bond layout.
    for edge in rsd.edges
        c = nodeindex(topo, edge.child)
        @test edge.dimension == dim(virtualspace(L, c))
    end
    @test rsd.plan_kind == :state_diagram
    @test rsd.proofs_applied == length(sd.proofs)
    @test !isempty(sd.log.entries)
    # Reversal restores the exact lowering output.
    @test TB.unmerge_expansions(sd) == exps
    # Idempotence: a second structural pass finds nothing to merge.
    sd2 = TB.merge_channels(input, sd.expansions,
                            TB.StateDiagramMerge(TB.StructuralOptimizer()))
    @test isempty(sd2.proofs)
    @test sd2.expansions == sd.expansions
    return input, exps, ds, sd, rsd
end

@graft_testset "SD3 structural merge reproduces legacy bond layout" begin
    S = spin_ops()
    topo = mps_topology(3)
    phys = Dict(:site1 => S.P, :site2 => S.P, :site3 => S.P)
    H = OpSum()
    H += Term(-1.0, SiteOp(:site1, :Z, S.Z), SiteOp(:site2, :Z, S.Z))
    H += Term(-1.0, SiteOp(:site2, :Z, S.Z), SiteOp(:site3, :Z, S.Z))
    for site in (:site1, :site2, :site3)
        H += Term(-0.7, SiteOp(site, :X, S.X))
    end
    _sd3_check(H, topo, phys)

    # Fork with a physless-independent branch structure.
    fork = fork_topology(2, 1)
    sites = [nodeid(fork, i) for i in 1:nnodes(fork)]
    physk = Dict(s => S.P for s in sites)
    Hk = OpSum()
    Hk += Term(0.9, SiteOp(sites[1], :Z, S.Z), SiteOp(sites[2], :Z, S.Z))
    Hk += Term(0.5, SiteOp(sites[3], :X, S.X), SiteOp(sites[4], :X, S.X))
    Hk += Term(-0.2, SiteOp(sites[2], :X, S.X))
    _sd3_check(Hk, fork, physk)

    U = spin_ops_u1()
    physu = Dict(:site1 => U.P, :site2 => U.P, :site3 => U.P)
    Hu = OpSum()
    Hu += Term(0.5, SiteOp(:site1, :Sp, U.Sp), SiteOp(:site2, :Sm, U.Sm))
    Hu += Term(0.5, SiteOp(:site2, :Sp, U.Sp), SiteOp(:site3, :Sm, U.Sm))
    Hu += Term(0.25, SiteOp(:site1, :Z, U.Z), SiteOp(:site3, :Z, U.Z))
    Hu += Term(0.1, SiteOp(:site2, :N, U.N))
    _sd3_check(Hu, topo, physu)

    F = fermion_ops_z2()
    physf = Dict(:site1 => F.P, :site2 => F.P, :site3 => F.P)
    Hf = OpSum()
    Hf += Term(1.0, SiteOp(:site1, :Cd, F.Cd), SiteOp(:site3, :C, F.C))
    Hf += Term(1.0, SiteOp(:site3, :Cd, F.Cd), SiteOp(:site1, :C, F.C))
    Hf += Term(0.5, SiteOp(:site2, :N, F.N))
    _sd3_check(Hf, topo, physf)

    physfk = Dict(s => F.P for s in sites)
    Hfk = OpSum()
    Hfk += Term(0.7, SiteOp(sites[1], :Cd, F.Cd), SiteOp(sites[end], :C, F.C))
    Hfk += Term(0.7, SiteOp(sites[end], :Cd, F.Cd), SiteOp(sites[1], :C, F.C))
    Hfk += Term(0.2, SiteOp(sites[2], :N, F.N))
    _sd3_check(Hfk, fork, physfk)
end

@graft_testset "SD3 order invariance and identity classes" begin
    S = spin_ops()
    topo = mps_topology(3)
    phys = Dict(:site1 => S.P, :site2 => S.P, :site3 => S.P)
    terms = [
        Term(-1.0, SiteOp(:site1, :Z, S.Z), SiteOp(:site2, :Z, S.Z)),
        Term(-0.7, SiteOp(:site2, :X, S.X)),
        Term(0.4, SiteOp(:site1, :X, S.X), SiteOp(:site3, :X, S.X)),
        Term(0.2, SiteOp(:site3, :X, S.X)),
    ]
    Ha = OpSum() + terms[1] + terms[2] + terms[3] + terms[4]
    Hb = OpSum() + terms[4] + terms[2] + terms[1] + terms[3]
    inputa, _, _, sda = _sd3_build(Ha, topo, phys)
    inputb, _, _, sdb = _sd3_build(Hb, topo, phys)
    Oa, ra = TB.realize_ttno(inputa, sda)
    Ob, rb = TB.realize_ttno(inputb, sdb)
    @test norm(to_dense(Oa) - to_dense(Ob)) < 1e-12
    @test [e.dimension for e in ra.edges] == [e.dimension for e in rb.edges]

    # Explicit :I is an ACTIVE restriction and never merges into plain idle
    # transport without a proved canonicalization step.
    Hi = OpSum() +
        Term(0.5, SiteOp(:site1, :I, S.I), SiteOp(:site2, :X, S.X)) +
        Term(0.3, SiteOp(:site2, :X, S.X))
    inputi, _, _, sdi = _sd3_build(Hi, topo, phys)
    i1 = nodeindex(topo, :site1)
    classes = Set(ch.degeneracy.span.class
                  for exp in sdi.expansions
                  for ch in (exp.edge_channels[i1],)
                  if ch isa TB.ChannelIdentity)
    @test TB.ACTIVE in classes && TB.PLAIN_IDLE in classes
    Oi, ri = TB.realize_ttno(inputi, sdi)
    Li = ttno_from_opsum(Hi, topo, phys)
    @test norm(to_dense(Oi) - to_dense(Li)) < 1e-12
end

@graft_testset "SD3 overmerge guards (route and frame axes)" begin
    S = spin_ops()
    topo = mps_topology(2)
    phys = Dict(:site1 => S.P, :site2 => S.P)
    # Two identical terms produce mergeable equal-signature channels: the
    # control case merges them into one channel copy.
    H = OpSum() +
        Term(0.5, SiteOp(:site1, :X, S.X), SiteOp(:site2, :Z, S.Z)) +
        Term(0.25, SiteOp(:site1, :X, S.X), SiteOp(:site2, :Z, S.Z))
    input = TB.TTNOBuildInput(H, topo, phys)
    exps = TB.lower_terms(input, TB.AbelianScalarLowering())
    kernel = TB.StateDiagramMerge(TB.StructuralOptimizer())
    sd = TB.merge_channels(input, exps, kernel)
    c = nodeindex(topo, :site1)
    merged_channels = unique([exp.edge_channels[c] for exp in sd.expansions])
    @test length(merged_channels) == 1
    @test length(sd.proofs) == 1
    @test only(sd.proofs).witness.kind === :identical_subtree
    O, _ = TB.realize_ttno(input, sd)
    L = ttno_from_opsum(H, topo, phys)
    @test norm(to_dense(O) - to_dense(L)) < 1e-12

    # Same channels, but one frame axis modified: signatures split and the
    # merge refuses (same charge, different frame).
    Q = TB.input_sectortype(input)
    framed = let ch = exps[2].edge_channels[c]
        TB.ChannelIdentity{Q}(ch.sector, ch.route, ch.multiplicity,
                              ch.degeneracy, ch.orientation,
                              TB.AbelianFrameCertificate((2 => ch.sector,)))
    end
    exps_frame = [exps[1], TB._rewrite_channel(exps[2], c, framed)]
    sd_frame = TB.merge_channels(input, exps_frame, kernel)
    @test isempty([p for p in sd_frame.proofs if p.edge == c])
    @test length(unique([exp.edge_channels[c]
                         for exp in sd_frame.expansions])) == 2

    # Same outer sector, different canonical route: no merge. The U(1) route
    # [+2, -1] fuses to the same +1 sector as [+1].
    U = spin_ops_u1()
    physu = Dict(:site1 => U.P, :site2 => U.P)
    Hu = OpSum() +
        Term(0.5, SiteOp(:site1, :Sp, U.Sp), SiteOp(:site2, :Sm, U.Sm)) +
        Term(0.25, SiteOp(:site1, :Sp, U.Sp), SiteOp(:site2, :Sm, U.Sm))
    inputu = TB.TTNOBuildInput(Hu, topo, physu)
    expsu = TB.lower_terms(inputu, TB.AbelianScalarLowering())
    QU = TB.input_sectortype(inputu)
    rerouted = let ch = expsu[2].edge_channels[c]
        TB.ChannelIdentity{QU}(ch.sector,
                               TB.FusionRoute([U1Irrep(2), U1Irrep(-1)],
                                              [U1Irrep(2), U1Irrep(1)]),
                               ch.multiplicity, ch.degeneracy,
                               ch.orientation, ch.frame)
    end
    expsu_route = [expsu[1], TB._rewrite_channel(expsu[2], c, rerouted)]
    sdu = TB.merge_channels(inputu, expsu_route, kernel)
    @test isempty([p for p in sdu.proofs if p.edge == c])
    @test length(unique([exp.edge_channels[c]
                         for exp in sdu.expansions])) == 2

    # Corrupted proofs fail closed at realization.
    span = only(sd.proofs).span
    bogus = TB.MergeProofStep(
        :restriction_channel_merge, c, span, TB.IdentitySpanRelation(span),
        TB.StructuralIdentityWitness(:identical_subtree),
        [TB.DegeneracyLabel(span, 1)], [TB.DegeneracyLabel(span, 1)])
    corrupt = TB.StateDiagram(sd.topology, sd.provenance, sd.expansions,
                              vcat(sd.proofs, [bogus]), sd.log)
    @test_throws ArgumentError TB.realize_ttno(input, corrupt)
end
