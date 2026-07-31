# SD1/CAT0 — typed input and categorical IR seam tests
# (state-diagram-compiler plan v2, SD1 deliverables; ADR-0003 decisions 2-4;
# representability audit findings F1-F3).

const TB = Graft.StateDiagram

include("statediagram_fixtures.jl")

@graft_testset "SD1 capability matrix and fail-closed services" begin
    @test TB.is_supported_profile(TB.category_semantics(Graft.Backend.ComplexSpace))
    @test TB.is_supported_profile(TB.category_semantics(typeof(U1Space(0 => 1))))
    fz2 = Graft.Backend.Vect[FermionParity](FermionParity(0) => 1)
    @test TB.is_supported_profile(TB.category_semantics(typeof(fz2)))

    su2 = Graft.Backend.Vect[Graft.Backend.SU2Irrep](
        Graft.Backend.SU2Irrep(0) => 1, Graft.Backend.SU2Irrep(1//2) => 1)
    csu2 = TB.category_semantics(typeof(su2))
    @test !TB.is_supported_profile(csu2)
    report = TB.capability_report(csu2)
    @test !TB.is_supported(report)
    @test any(m -> m.service === :lowering && m.operation === :lower_terms &&
                  m.requirement === :UniqueFusion, report.missing)
    @test any(m -> m.service === :equivalence, report.missing)
    @test any(m -> m.service === :materialization, report.missing)
    # The supported profile has an empty missing matrix.
    @test TB.is_supported(TB.capability_report(
        TB.category_semantics(Graft.Backend.ComplexSpace)))

    # Fail closed before densification: input construction is data-only,
    # lowering rejects with the typed missing capability.
    topo = mps_topology(2)
    Isu2 = Graft.Backend.id(su2)
    H = OpSum() + Term(1.0, SiteOp(:site1, :I, Isu2))
    input = TB.TTNOBuildInput(H, topo, Dict(:site1 => su2, :site2 => su2))
    err = try
        TB.lower_terms(input, TB.AbelianScalarLowering())
        nothing
    catch e
        e
    end
    @test err isa TB.MissingCategoryCapability
    @test err.service === :lowering
    @test err.operation === :lower_terms
    @test err.requirement === :UniqueFusion
end

@graft_testset "SD1 channel identity axes are load-bearing" begin
    Q = U1Irrep
    key = TB.LocalOpKey(:Sp, "P", Q(1))
    route = TB.FusionRoute([Q(1)], [Q(1)])
    span = TB.ChannelSpan(TB.ACTIVE, ((1, key),))
    base = TB.ChannelIdentity{Q}(
        Q(1), route, TB.MultiplicityLabels(),
        TB.DegeneracyLabel(span, 1), TB.ChannelOrientation(),
        TB.AbelianFrameCertificate(),
    )
    remake(; sector=Q(1), r=route, mu=TB.MultiplicityLabels(),
           d=TB.DegeneracyLabel(span, 1), o=TB.ChannelOrientation(),
           f=TB.AbelianFrameCertificate()) =
        TB.ChannelIdentity{Q}(sector, r, mu, d, o, f)

    @test base == remake()
    @test hash(base) == hash(remake())

    variants = [
        remake(sector=Q(-1)),
        remake(r=TB.FusionRoute([Q(1), Q(0)], [Q(1), Q(1)])),
        remake(mu=TB.MultiplicityLabels([1])),
        remake(d=TB.DegeneracyLabel(span, 2)),
        remake(d=TB.DegeneracyLabel(
            TB.ChannelSpan(TB.ACTIVE, ((1, TB.LocalOpKey(:Sm, "P", Q(-1))),)), 1)),
        remake(o=TB.ChannelOrientation(true, true)),
        remake(f=TB.AbelianFrameCertificate((2 => Q(1),))),
    ]
    for variant in variants
        @test variant != base
        @test hash(variant) != hash(base)
    end
    # Explicit vs omitted identity are distinct transition classes.
    ikey = TB.LocalOpKey(:I, "P", Q(0))
    @test TB.ExplicitLocalTransition(ikey) != TB.OmittedIdentityTransition()

    svc = TB.AbelianEquivalenceService()
    # Copy index is outside the boundary signature and mixing key.
    copy2 = remake(d=TB.DegeneracyLabel(span, 2))
    @test TB.boundary_signature(svc, base) == TB.boundary_signature(svc, copy2)
    @test TB.mixing_block_key(svc, base) == TB.mixing_block_key(svc, copy2)
    # Span content is inside the signature but outside the mixing key.
    other_span = remake(d=TB.DegeneracyLabel(
        TB.ChannelSpan(TB.ACTIVE, ((1, TB.LocalOpKey(:Sm, "P", Q(-1))),)), 1))
    @test TB.boundary_signature(svc, base) != TB.boundary_signature(svc, other_span)
    @test TB.mixing_block_key(svc, base) == TB.mixing_block_key(svc, other_span)
    # Frame participates in both.
    framed = remake(f=TB.AbelianFrameCertificate((2 => Q(1),)))
    @test TB.boundary_signature(svc, base) != TB.boundary_signature(svc, framed)
    @test TB.mixing_block_key(svc, base) != TB.mixing_block_key(svc, framed)

    # Route canonicalization is owned by canonicalize_channel (finding F3).
    @test TB.canonicalize_channel(svc, base) == base
    bad_route = remake(r=TB.FusionRoute([Q(1)], [Q(0)]))
    @test_throws ArgumentError TB.canonicalize_channel(svc, bad_route)
    bad_mult = remake(mu=TB.MultiplicityLabels([1]))
    @test_throws ArgumentError TB.canonicalize_channel(svc, bad_mult)

    # Span basis relations: identity on equal spans, fail-closed otherwise.
    rel = TB.channel_basis_relation(svc, span, span)
    @test rel isa TB.IdentitySpanRelation
    @test TB.validate_basis_proof(svc, rel)
    other = TB.ChannelSpan(TB.ACTIVE, ((1, TB.LocalOpKey(:Sm, "P", Q(-1))),))
    norel = TB.channel_basis_relation(svc, span, other)
    @test norel isa TB.NoKnownSpanRelation
    @test !TB.validate_basis_proof(svc, norel)
end

@graft_testset "SD1 lowering: coefficient ownership and anchors" begin
    input, topo = _sd1_dense_input()
    exps = TB.lower_terms(input, TB.AbelianScalarLowering())
    @test length(exps) == 3
    for exp in exps
        @test TB.validate_expansion(input, exp)
        anchors = [n for n in 1:nnodes(topo)
                   if TB.is_anchor_slot(exp.hyperedges[n].coeff)]
        @test anchors == [exp.anchor_node]
    end
    # The chain is rooted at :site3, so {site1,site2} completes at :site2,
    # {site2,site3} at the root :site3, and the one-site term at its node.
    @test exps[1].anchor_node == nodeindex(topo, :site2)
    @test exps[2].anchor_node == nodeindex(topo, :site3)
    @test exps[3].anchor_node == nodeindex(topo, :site2)

    # Constant and all-identity terms anchor at the root hyperedge.
    S = spin_ops()
    phys = Dict(:site1 => S.P, :site2 => S.P, :site3 => S.P)
    Hc = OpSum() + Term(2.5) +
        Term(1.5, SiteOp(:site1, :I, S.I)) +
        Term(-0.25, SiteOp(:site1, :X, S.X))
    cinput = TB.TTNOBuildInput(Hc, topo, phys)
    cexps = TB.lower_terms(cinput, TB.AbelianScalarLowering())
    for exp in cexps
        @test TB.validate_expansion(cinput, exp)
    end
    @test cexps[1].anchor_node == topo.root
    @test cexps[2].anchor_node == topo.root
    @test cexps[3].anchor_node == nodeindex(topo, :site1)
    @test all(ch isa TB.RootBoundary || ch.degeneracy.span.class == TB.PLAIN_IDLE
              for ch in cexps[1].edge_channels)
    # Explicit and omitted identity are distinct transition classes: the
    # explicit :I factor owns an ExplicitLocalTransition even though its
    # single-site term completes at its own node (DONE above).
    i1 = nodeindex(topo, :site1)
    @test cexps[2].edge_channels[i1].degeneracy.span.class == TB.DONE_TRANSPORT
    @test cexps[2].hyperedges[i1].transition isa TB.ExplicitLocalTransition
    @test cexps[1].hyperedges[i1].transition isa TB.OmittedIdentityTransition

    # In a multi-factor term an explicit :I is an ACTIVE partial restriction,
    # never plain idle transport.
    Hm = OpSum() + Term(0.5, SiteOp(:site1, :I, S.I), SiteOp(:site2, :X, S.X))
    minput = TB.TTNOBuildInput(Hm, topo, phys)
    mexps = TB.lower_terms(minput, TB.AbelianScalarLowering())
    @test TB.validate_expansion(minput, mexps[1])
    mch = mexps[1].edge_channels[i1]
    @test mch.degeneracy.span.class == TB.ACTIVE
    @test mexps[1].hyperedges[i1].transition isa TB.ExplicitLocalTransition
    @test mexps[1].anchor_node == nodeindex(topo, :site2)

    # Graded lowering records the SD0 certificate as provenance.
    finput, ftopo = _sd1_fz2_input()
    fexps = TB.lower_terms(finput, TB.AbelianScalarLowering())
    for exp in fexps
        @test TB.validate_expansion(finput, exp)
    end
    hop = fexps[1]
    @test hop.provenance.uses_certificate
    @test !isempty(hop.provenance.canonical_word)
    @test !isempty(hop.provenance.native_word)
    @test abs(hop.provenance.certificate_scale) == 1.0
    # The odd restriction charge threads through ACTIVE channels.
    odd = FermionParity(1)
    active = [ch for ch in hop.edge_channels
              if !(ch isa TB.RootBoundary) &&
                 ch.degeneracy.span.class == TB.ACTIVE]
    @test !isempty(active)
    @test any(ch -> ch.sector == odd, active)
    for ch in active
        @test length(ch.route.leaves) == length(ch.route.intermediates)
    end
end

@graft_testset "SD1 operator-key collision fails closed" begin
    S = spin_ops()
    topo = mps_topology(2)
    phys = Dict(:site1 => S.P, :site2 => S.P)
    fake = Graft.Backend.TensorMap(ComplexF64[1 0; 0 1], S.P ← S.P)
    Hbad = OpSum() +
        Term(1.0, SiteOp(:site1, :Z, S.Z)) +
        Term(1.0, SiteOp(:site2, :Z, fake))
    @test_throws ArgumentError TB.TTNOBuildInput(Hbad, topo, phys)

    # Identical maps under one key deduplicate without error; distinct
    # charges under one name are distinct keys.
    U = spin_ops_u1()
    physu = Dict(:site1 => U.P, :site2 => U.P)
    Hok = OpSum() +
        Term(1.0, SiteOp(:site1, :Z, U.Z), SiteOp(:site2, :Z, U.Z)) +
        Term(0.5, SiteOp(:site1, :Sp, U.Sp), SiteOp(:site2, :Sm, U.Sm))
    input = TB.TTNOBuildInput(Hok, topo, physu)
    # :Z deduplicates across sites under one key; :Sp/:Sm are distinct keys.
    @test length(input.operator_table) == 3
    kZ = TB.local_op_key(SiteOp(:site1, :Z, U.Z))
    kSp = TB.local_op_key(SiteOp(:site1, :Sp, U.Sp))
    @test kZ != kSp
    @test kZ.charge == U1Irrep(0)
    @test kSp.charge == U1Irrep(1)
end

@graft_testset "SD1 structural equality and serialization determinism" begin
    input1, _ = _sd1_dense_input()
    input2, _ = _sd1_dense_input()
    @test input1 == input2
    @test hash(input1) == hash(input2)
    @test input1.provenance == input2.provenance
    @test input1.provenance.schema == TB.IR_SCHEMA_VERSION

    exps1 = TB.lower_terms(input1, TB.AbelianScalarLowering())
    exps2 = TB.lower_terms(input2, TB.AbelianScalarLowering())
    @test exps1 == exps2
    @test hash(exps1) == hash(exps2)
    @test TB.serialize_expansions(input1, exps1) ==
        TB.serialize_expansions(input2, exps2)

    # Hermiticity contract participates in identity and provenance.
    hinput, _ = _sd1_dense_input(hermiticity=TB.AssertedHermitian())
    @test hinput != input1
    @test hinput.provenance != input1.provenance

    finput1, _ = _sd1_fz2_input()
    finput2, _ = _sd1_fz2_input()
    fexps1 = TB.lower_terms(finput1, TB.AbelianScalarLowering())
    fexps2 = TB.lower_terms(finput2, TB.AbelianScalarLowering())
    @test fexps1 == fexps2
    @test TB.serialize_expansions(finput1, fexps1) ==
        TB.serialize_expansions(finput2, fexps2)
end

@graft_testset "SD1 versioned serialization golden fixtures" begin
    header = "(graft-ttno-ir (schema $(TB.IR_SCHEMA_VERSION))"
    input, _ = _sd1_dense_input()
    exps = TB.lower_terms(input, TB.AbelianScalarLowering())
    text = TB.serialize_expansions(input, exps)
    @test startswith(text, header)

    finput, _ = _sd1_fz2_input()
    fexps = TB.lower_terms(finput, TB.AbelianScalarLowering())
    ftext = TB.serialize_expansions(finput, fexps)
    @test startswith(ftext, header)

    fixtures = joinpath(@__DIR__, "fixtures", "statediagram")
    golden_dense = joinpath(fixtures, "sd1_ir_dense_tfi.sexp")
    golden_fz2 = joinpath(fixtures, "sd1_ir_fz2_hopping.sexp")
    @test isfile(golden_dense)
    @test isfile(golden_fz2)
    @test read(golden_dense, String) == text
    @test read(golden_fz2, String) == ftext
end
