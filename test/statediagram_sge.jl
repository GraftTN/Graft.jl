# SD5 — restricted exact SGE and proof-backed cover selection tests
# (state-diagram-compiler plan v2, SD5 deliverables): exact expression
# algebra, restricted elimination with typed operation logs, independent
# raw-versus-eliminated covers with the strict raw-wins-ties rule, the
# nontrivial 2x2 basis-transport fixture, and exact reversal.

using Graft.Backend: Trivial

const _SD5_C = Float64
_g(pairs...) = TB._gexpr(collect(Tuple{Int,_SD5_C}, pairs))

"Rebind one expansion's coefficient atom (shared-atom fixture surgery)."
function _sd5_share_atom(exp::TB.TermTTNOExpansion{Q,C},
                         atom::TB.CoeffAtom; scale=nothing) where {Q,C}
    hyper = copy(exp.hyperedges)
    a = exp.anchor_node
    old = hyper[a].coeff::TB.CoeffAtomSlot
    s = scale === nothing ? old.scale : C(scale)
    hyper[a] = TB.TermHyperedge{Q,C}(a, hyper[a].transition,
                                     hyper[a].certificate,
                                     TB.CoeffAtomSlot(atom, s))
    return TB.TermTTNOExpansion{Q,C}(exp.term_ordinal, atom, a,
                                     exp.edge_channels, hyper,
                                     exp.provenance)
end

@graft_testset "SD5 exact expression algebra and restricted elimination" begin
    a = _g((1, 1.0))
    b = _g((2, 1.0))
    @test TB._gexpr_proportional(a, a) == 1.0
    @test TB._gexpr_proportional(_g((1, -1.0)), a) == -1.0
    @test TB._gexpr_proportional(a, b) === nothing        # opaque atoms
    @test TB._gexpr_proportional(_g((1, 1.0), (2, 1.0)),
                                 _g((1, 1.0), (2, -1.0))) === nothing
    @test TB._gexpr_proportional(_g((1, -2.0), (2, -4.0)),
                                 _g((1, 2.0), (2, 4.0))) == -1.0
    @test TB._gexpr_iszero(TB._gexpr_axpy(a, -1.0, a))

    # Row elimination on an atom-sharing 2x2 Gamma.
    z = TB.GammaExpr{_SD5_C}()
    m = [ _g((1, 1.0)) _g((2, 1.0));
          _g((1, 1.0)) _g((2, 1.0)) ]
    elim, ops = TB._restricted_sge(m)
    @test length(ops) == 1
    @test only(ops) isa TB.SGERowElimination
    @test only(ops).factor == 1.0
    @test TB._gexpr_iszero(elim[2, 1]) && TB._gexpr_iszero(elim[2, 2])

    # Negated proportional rows.
    mneg = [ _g((1, 1.0)) _g((2, -1.0));
             _g((1, -1.0)) _g((2, 1.0)) ]
    _, opsneg = TB._restricted_sge(mneg)
    @test only(opsneg).factor == -1.0

    # Column elimination.
    mcol = [ _g((1, 1.0)) _g((1, -1.0));
             _g((2, 1.0)) _g((2, -1.0)) ]
    elimc, opsc = TB._restricted_sge(mcol)
    @test any(op -> op isa TB.SGEColElimination, opsc)
    @test TB._gexpr_iszero(elimc[1, 2]) && TB._gexpr_iszero(elimc[2, 2])

    # Distinct atoms: nothing is ever eliminated (floating equality is
    # never inferred, rational pivots are never invented).
    mfree = [ _g((1, 1.0)) _g((2, 1.0));
              _g((3, 1.0)) _g((4, 1.0)) ]
    _, opsfree = TB._restricted_sge(mfree)
    @test isempty(opsfree)
    @test z == TB.GammaExpr{_SD5_C}()
end

@graft_testset "SD5 eliminated cover wins strictly and reconstructs exactly" begin
    S = spin_ops()
    topo = mps_topology(2)
    phys = Dict(:site1 => S.P, :site2 => S.P)
    a, b = 0.8, -0.35
    H = OpSum() +
        Term(a, SiteOp(:site1, :X, S.X), SiteOp(:site2, :Z, S.Z)) +
        Term(b, SiteOp(:site1, :X, S.X), SiteOp(:site2, :X, S.X)) +
        Term(a, SiteOp(:site1, :Y, S.Y), SiteOp(:site2, :Z, S.Z)) +
        Term(b, SiteOp(:site1, :Y, S.Y), SiteOp(:site2, :X, S.X))
    input = TB.TTNOBuildInput(H, topo, phys)
    exps = TB.lower_terms(input, TB.AbelianScalarLowering())
    # Shared-atom surgery: terms 3 and 4 reuse the atoms of terms 1 and 2
    # (equal coefficient values keep the action equal to the OpSum).
    exps2 = [exps[1], exps[2],
             _sd5_share_atom(exps[3], TB.CoeffAtom(1)),
             _sd5_share_atom(exps[4], TB.CoeffAtom(2))]
    for exp in exps2
        @test TB.validate_expansion(input, exp)
    end

    ds = TB.merge_channels(input, exps2, TB.DirectSumMerge())
    Ods, _ = TB.realize_ttno(input, ds)
    sd = TB.merge_channels(input, exps2,
                           TB.StateDiagramMerge(TB.SGEOptimizer()))
    Osd, rsd = TB.realize_ttno(input, sd)

    # (X+Y) ⊗ (a Z + b X): one channel where the raw cover needs two.
    @test only(rsd.edges).dimension == 1
    @test norm(to_dense(Osd) - to_dense(Ods)) < 1e-12
    elim_proofs = [p for p in sd.proofs if p.kind === :gamma_eliminated_cover]
    @test length(elim_proofs) == 1
    witness = only(elim_proofs).witness::TB.EliminationCoverWitness
    @test witness.raw_cover_size == 2
    @test witness.eliminated_cover_size == 1
    @test witness.eliminated_cover_size < witness.raw_cover_size
    @test length(witness.operations) == 1
    @test only(witness.operations).factor == 1.0
    @test length(witness.slot_rewrites) == 2
    @test all(r -> r[3] == 1.0, witness.slot_rewrites)
    @test any(e -> e.operation === :gamma_eliminated_cover, sd.log.entries)

    # Exact reversal restores the shared-atom lowering output.
    @test TB.unmerge_expansions(sd) == exps2
    @test TB.validate_merge_plan(input, sd)

    # Determinism.
    sdb = TB.merge_channels(input, exps2,
                            TB.StateDiagramMerge(TB.SGEOptimizer()))
    @test sd.proofs == sdb.proofs && sd.expansions == sdb.expansions

    # Without shared atoms the same Hamiltonian finds no exact relation:
    # elimination is a proven no-op and the block stays at the raw result.
    sdplain = TB.merge_channels(input, exps,
                                TB.StateDiagramMerge(TB.SGEOptimizer()))
    @test isempty([p for p in sdplain.proofs
                   if p.kind === :gamma_eliminated_cover])
    Oplain, rplain = TB.realize_ttno(input, sdplain)
    L = ttno_from_opsum(H, topo, phys)
    @test only(rplain.edges).dimension == dim(virtualspace(L, nodeindex(topo, :site1)))
    @test norm(to_dense(Oplain) - to_dense(L)) < 1e-12
end

@graft_testset "SD5 raw wins ties and negated relations reconstruct" begin
    S = spin_ops()
    topo = mps_topology(2)
    phys = Dict(:site1 => S.P, :site2 => S.P)
    a = 0.6
    # Two shared-atom terms with one common above context: the raw column
    # cover already reaches size 1; elimination ties and raw is selected.
    H = OpSum() +
        Term(a, SiteOp(:site1, :X, S.X), SiteOp(:site2, :Z, S.Z)) +
        Term(a, SiteOp(:site1, :Y, S.Y), SiteOp(:site2, :Z, S.Z))
    input = TB.TTNOBuildInput(H, topo, phys)
    exps = TB.lower_terms(input, TB.AbelianScalarLowering())
    exps2 = [exps[1], _sd5_share_atom(exps[2], TB.CoeffAtom(1))]
    sd = TB.merge_channels(input, exps2,
                           TB.StateDiagramMerge(TB.SGEOptimizer()))
    @test isempty([p for p in sd.proofs if p.kind === :gamma_eliminated_cover])
    @test length([p for p in sd.proofs if p.kind === :gamma_raw_cover]) == 1
    @test any(e -> e.operation === :cover_selection &&
                  occursin("raw", e.detail), sd.log.entries)
    O, r = TB.realize_ttno(input, sd)
    ds = TB.merge_channels(input, exps2, TB.DirectSumMerge())
    Ods, _ = TB.realize_ttno(input, ds)
    @test only(r.edges).dimension == 1
    @test norm(to_dense(O) - to_dense(Ods)) < 1e-12

    # Negated shared atom: the -1 proportional row eliminates and the ±1
    # slot rewrite reconstructs (X - Y) ⊗ (a Z + b X) exactly.
    b = -0.15
    Hn = OpSum() +
        Term(a, SiteOp(:site1, :X, S.X), SiteOp(:site2, :Z, S.Z)) +
        Term(b, SiteOp(:site1, :X, S.X), SiteOp(:site2, :X, S.X)) +
        Term(-a, SiteOp(:site1, :Y, S.Y), SiteOp(:site2, :Z, S.Z)) +
        Term(-b, SiteOp(:site1, :Y, S.Y), SiteOp(:site2, :X, S.X))
    inputn = TB.TTNOBuildInput(Hn, topo, phys)
    expsn = TB.lower_terms(inputn, TB.AbelianScalarLowering())
    # Terms 3/4 reuse atoms 1/2 with the exact scalar -1, reproducing their
    # -a/-b coefficients as -1 times the shared atoms.
    expsn2 = [expsn[1], expsn[2],
              _sd5_share_atom(expsn[3], TB.CoeffAtom(1); scale=-1.0),
              _sd5_share_atom(expsn[4], TB.CoeffAtom(2); scale=-1.0)]
    dsn = TB.merge_channels(inputn, expsn2, TB.DirectSumMerge())
    Odsn, _ = TB.realize_ttno(inputn, dsn)
    sdn = TB.merge_channels(inputn, expsn2,
                            TB.StateDiagramMerge(TB.SGEOptimizer()))
    On, rn = TB.realize_ttno(inputn, sdn)
    @test norm(to_dense(On) - to_dense(Odsn)) < 1e-12
    elimn = [p for p in sdn.proofs if p.kind === :gamma_eliminated_cover]
    if !isempty(elimn)
        wn = only(elimn).witness::TB.EliminationCoverWitness
        @test only(wn.operations).factor == -1.0
        @test only(rn.edges).dimension == 1
    end
    @test TB.unmerge_expansions(sdn) == expsn2
end

@graft_testset "SD5 nontrivial basis transport does not merge" begin
    Q = U1Irrep
    key = TB.LocalOpKey(:Sp, "P", Q(1))
    span1 = TB.ChannelSpan(TB.ACTIVE, ((1, key),))
    span2 = TB.ChannelSpan(TB.ACTIVE, ((2, key),))
    svc = TB.AbelianEquivalenceService()

    # A nontrivial exact 2x2 transport validates as a basis relation...
    swap = ComplexF64[0 1; 1 0]
    transport = TB.SpanBasisTransport(span1, span2, swap, swap)
    @test TB.validate_basis_proof(svc, transport)
    rot = ComplexF64[0 -1; 1 0]
    rotinv = ComplexF64[0 1; -1 0]
    transport2 = TB.SpanBasisTransport(span1, span2, rot, rotinv)
    @test TB.validate_basis_proof(svc, transport2)
    bad = TB.SpanBasisTransport(span1, span2, rot, rot)
    @test !TB.validate_basis_proof(svc, bad)

    # ...but transport alone never merges: a proof citing a transport with
    # no admissible redundancy witness fails closed at realization.
    input, topo = _sd1_dense_input()
    exps = TB.lower_terms(input, TB.AbelianScalarLowering())
    sd = TB.merge_channels(input, exps,
                           TB.StateDiagramMerge(TB.SGEOptimizer()))
    c = first([i for i in 1:nnodes(topo) if i != topo.root])
    bogus = TB.MergeProofStep(
        :basis_transport_only, c, span1, transport,
        TB.StructuralIdentityWitness(:basis_transport),
        [TB.DegeneracyLabel(span1, 2)], [TB.DegeneracyLabel(span1, 1)])
    corrupt = TB.StateDiagram(sd.topology, sd.provenance, sd.expansions,
                              vcat(sd.proofs, [bogus]), sd.log)
    @test_throws ArgumentError TB.realize_ttno(input, corrupt)

    # The SGE optimizer over the plain input equals direct sum everywhere
    # (the Abelian Gamma/SGE gate closes on action equality).
    ds = TB.merge_channels(input, exps, TB.DirectSumMerge())
    Ods, _ = TB.realize_ttno(input, ds)
    Osd, _ = TB.realize_ttno(input, sd)
    @test norm(to_dense(Osd) - to_dense(Ods)) < 1e-12
end
