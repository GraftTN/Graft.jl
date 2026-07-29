# SD4 — raw Gamma and minimum vertex cover tests (state-diagram-compiler
# plan v2, SD4 deliverables): Gamma block partition, deterministic
# Hopcroft-Karp matching, Konig cover, raw-cover reconstruction with exact
# logs, action equality against direct sum, and randomized small-tree
# properties.

using Graft.Backend: dim
using Random: Xoshiro, randperm

const _SD4_RNG = Xoshiro(0x5d4c0ffee)

function _sd4_build(H, topo, phys)
    input = TB.TTNOBuildInput(H, topo, phys)
    exps = TB.lower_terms(input, TB.AbelianScalarLowering())
    ds = TB.merge_channels(input, exps, TB.DirectSumMerge())
    sd = TB.merge_channels(input, exps,
                           TB.StateDiagramMerge(TB.GammaCoverOptimizer()))
    return input, exps, ds, sd
end

function _sd4_check(H, topo, phys; atol=1e-12)
    input, exps, ds, sd = _sd4_build(H, topo, phys)
    Ods, _ = TB.realize_ttno(input, ds)
    Osd, rsd = TB.realize_ttno(input, sd)
    @test norm(to_dense(Osd) - to_dense(Ods)) < atol
    @test TB.unmerge_expansions(sd) == exps
    return input, exps, ds, sd, rsd, Osd, Ods
end

@graft_testset "SD4 Hopcroft-Karp and Konig are deterministic and minimal" begin
    # Path graph rows {1,2}, cols {1,2}: (1,1), (1,2), (2,2).
    adj = [[1, 2], [2]]
    mr, mc = TB._hopcroft_karp(2, 2, adj)
    @test count(!=(0), mr) == 2
    cr, cc = TB._konig_cover(2, 2, adj, mr, mc)
    @test length(cr) + length(cc) == 2
    # Star: three rows all pointing at one column: cover is that column.
    adj2 = [[1], [1], [1]]
    mr2, mc2 = TB._hopcroft_karp(3, 1, adj2)
    @test count(!=(0), mr2) == 1
    cr2, cc2 = TB._konig_cover(3, 1, adj2, mr2, mc2)
    @test isempty(cr2) && cc2 == [1]
    # One row fanning out to three columns: cover is the row.
    adj3 = [[1, 2, 3]]
    mr3, mc3 = TB._hopcroft_karp(1, 3, adj3)
    cr3, cc3 = TB._konig_cover(1, 3, adj3, mr3, mc3)
    @test cr3 == [1] && isempty(cc3)
    # Determinism across repeated runs.
    @test TB._hopcroft_karp(2, 2, adj) == (mr, mc)
    @test TB._konig_cover(2, 2, adj, mr, mc) == (cr, cc)
end

@graft_testset "SD4 raw cover beats the legacy channel structure" begin
    # a·XZ + b·YZ share their above context: the column cover realizes one
    # combined channel where the legacy compiler needs two.
    S = spin_ops()
    topo = mps_topology(2)
    phys = Dict(:site1 => S.P, :site2 => S.P)
    H = OpSum() +
        Term(0.8, SiteOp(:site1, :X, S.X), SiteOp(:site2, :Z, S.Z)) +
        Term(-0.3, SiteOp(:site1, :Y, S.Y), SiteOp(:site2, :Z, S.Z))
    input, exps, ds, sd, rsd, Osd, _ = _sd4_check(H, topo, phys)
    L = ttno_from_opsum(H, topo, phys)
    c = nodeindex(topo, :site1)
    legacy_dim = dim(virtualspace(L, c))
    @test legacy_dim == 2
    @test only(rsd.edges).dimension == 1
    @test norm(to_dense(Osd) - to_dense(L)) < 1e-12
    @test any(p -> p.kind === :gamma_raw_cover, sd.proofs)
    witness = only([p.witness for p in sd.proofs if p.kind === :gamma_raw_cover])
    @test witness isa TB.GammaCoverWitness
    @test length(witness.rows) == 2 && length(witness.cover_cols) == 1
    @test length(witness.moved_atoms) == 2
    # The optimizer log records the block, matching, and cover sizes.
    @test any(e -> e.operation === :gamma_raw_cover, sd.log.entries)

    # Larger shared-tail family: sum_i c_i A_i x B collapses to one channel.
    H3 = OpSum() +
        Term(0.5, SiteOp(:site1, :X, S.X), SiteOp(:site2, :X, S.X)) +
        Term(0.25, SiteOp(:site1, :Y, S.Y), SiteOp(:site2, :X, S.X)) +
        Term(-0.75, SiteOp(:site1, :Z, S.Z), SiteOp(:site2, :X, S.X))
    _, _, _, sd3, rsd3, Osd3, _ = _sd4_check(H3, topo, phys)
    @test only(rsd3.edges).dimension == 1
    L3 = ttno_from_opsum(H3, topo, phys)
    @test dim(virtualspace(L3, c)) == 3
    @test norm(to_dense(Osd3) - to_dense(L3)) < 1e-12

    # Mixed structure: a shared tail plus an independent term keeps the
    # cover honest (no overmerge of distinct above contexts).
    H4 = OpSum() +
        Term(0.5, SiteOp(:site1, :X, S.X), SiteOp(:site2, :Z, S.Z)) +
        Term(0.25, SiteOp(:site1, :Y, S.Y), SiteOp(:site2, :Z, S.Z)) +
        Term(-0.4, SiteOp(:site1, :Z, S.Z), SiteOp(:site2, :X, S.X)) +
        Term(0.1, SiteOp(:site2, :X, S.X))
    _, _, _, sd4, rsd4, Osd4, Ods4 = _sd4_check(H4, topo, phys)
    @test norm(to_dense(Osd4) - to_dense(Ods4)) < 1e-12
    L4 = ttno_from_opsum(H4, topo, phys)
    @test only(rsd4.edges).dimension < dim(virtualspace(L4, c))

    # U(1) charged shared tails merge within their sector block.
    U = spin_ops_u1()
    physu = Dict(:site1 => U.P, :site2 => U.P)
    Hu = OpSum() +
        Term(0.5, SiteOp(:site1, :Sp, U.Sp), SiteOp(:site2, :Sm, U.Sm)) +
        Term(0.5, SiteOp(:site1, :Sm, U.Sm), SiteOp(:site2, :Sp, U.Sp)) +
        Term(0.25, SiteOp(:site1, :Z, U.Z), SiteOp(:site2, :Z, U.Z)) +
        Term(0.1, SiteOp(:site1, :N, U.N), SiteOp(:site2, :Z, U.Z))
    inputu, _, dsu, sdu = _sd4_build(Hu, topo, physu)
    Ou, ru = TB.realize_ttno(inputu, sdu)
    Odsu, _ = TB.realize_ttno(inputu, dsu)
    Lu = ttno_from_opsum(Hu, topo, physu)
    @test norm(to_dense(Ou) - to_dense(Odsu)) < 1e-12
    @test norm(to_dense(Ou) - to_dense(Lu)) < 1e-12
    # Z-block channels (Z and N tails both consumed by Z above) merge; the
    # charged Sp/Sm blocks stay separate.
    @test only(ru.edges).dimension < dim(virtualspace(Lu, c))

    # Fermionic shared tails: graded blocks merge without sign corruption.
    F = fermion_ops_z2()
    physf = Dict(:site1 => F.P, :site2 => F.P)
    Hf = OpSum() +
        Term(1.0, SiteOp(:site1, :Cd, F.Cd), SiteOp(:site2, :C, F.C)) +
        Term(1.0, SiteOp(:site1, :C, F.C), SiteOp(:site2, :Cd, F.Cd)) +
        Term(0.5, SiteOp(:site1, :N, F.N), SiteOp(:site2, :N, F.N)) +
        Term(0.25, SiteOp(:site1, :I, F.I), SiteOp(:site2, :N, F.N))
    inputf, _, dsf, sdf = _sd4_build(Hf, topo, physf)
    Of, rf = TB.realize_ttno(inputf, sdf)
    Odsf, _ = TB.realize_ttno(inputf, dsf)
    @test norm(to_dense(Of) - to_dense(Odsf)) < 1e-12
    Lf = ttno_from_opsum(Hf, topo, physf)
    @test norm(to_dense(Of) - to_dense(Lf)) < 1e-12
end

@graft_testset "SD4 reconstruction proofs, idempotence, and determinism" begin
    S = spin_ops()
    topo = mps_topology(3)
    phys = Dict(:site1 => S.P, :site2 => S.P, :site3 => S.P)
    H = OpSum() +
        Term(0.8, SiteOp(:site1, :X, S.X), SiteOp(:site2, :Z, S.Z)) +
        Term(-0.3, SiteOp(:site1, :Y, S.Y), SiteOp(:site2, :Z, S.Z)) +
        Term(0.4, SiteOp(:site2, :X, S.X), SiteOp(:site3, :Z, S.Z)) +
        Term(0.2, SiteOp(:site1, :Z, S.Z), SiteOp(:site3, :X, S.X)) +
        Term(-0.7, SiteOp(:site2, :X, S.X))
    input, exps, ds, sd, rsd, Osd, Ods = _sd4_check(H, topo, phys)

    # Realization validates gamma merge plans (replay) and the anchors moved
    # by column covers still occur exactly once per term.
    @test TB.validate_merge_plan(input, sd)
    for exp in sd.expansions
        @test TB.validate_expansion(input, exp)
    end

    # Idempotence: rerunning the full optimizer on the merged expansions
    # produces no further gamma reductions.
    sd2 = TB.merge_channels(input, sd.expansions,
                            TB.StateDiagramMerge(TB.GammaCoverOptimizer()))
    @test isempty([p for p in sd2.proofs if p.kind === :gamma_raw_cover])

    # Determinism: repeated builds serialize identically.
    _, _, _, sdb = _sd4_build(H, topo, phys)
    @test sd.proofs == sdb.proofs
    @test sd.expansions == sdb.expansions
    @test TB.serialize_expansions(input, sd.expansions) ==
        TB.serialize_expansions(input, sdb.expansions)

    # Term-order invariance of the realized action and bond dimensions.
    Hp = OpSum() + H.terms[4] + H.terms[1] + H.terms[5] + H.terms[2] + H.terms[3]
    inputp, _, _, sdp = _sd4_build(Hp, topo, phys)
    Op, rp = TB.realize_ttno(inputp, sdp)
    @test norm(to_dense(Op) - to_dense(Osd)) < 1e-12
    @test sort([e.dimension for e in rp.edges]) ==
        sort([e.dimension for e in rsd.edges])
end

@graft_testset "SD4 randomized small-tree action properties" begin
    S = spin_ops()
    opset = [(:X, S.X), (:Y, S.Y), (:Z, S.Z), (:N, S.N)]
    topologies = [
        mps_topology(2), mps_topology(3), mps_topology(4),
        fork_topology(2, 1), star_topology(2, 1), binary_topology(2),
    ]
    for (case, topo) in enumerate(topologies)
        sites = [nodeid(topo, i) for i in 1:nnodes(topo)]
        phys = Dict(s => S.P for s in sites)
        for trial in 1:3
            nterms_ = rand(_SD4_RNG, 2:6)
            H = OpSum()
            for _ in 1:nterms_
                nfac = rand(_SD4_RNG, 1:min(3, length(sites)))
                chosen = sites[randperm(_SD4_RNG, length(sites))[1:nfac]]
                ops = SiteOp[]
                for s in chosen
                    name, op = opset[rand(_SD4_RNG, 1:length(opset))]
                    push!(ops, SiteOp(s, name, op))
                end
                H += Term(randn(_SD4_RNG), ops...)
            end
            input, exps, ds, sd = _sd4_build(H, topo, phys)
            Ods, _ = TB.realize_ttno(input, ds)
            Osd, _ = TB.realize_ttno(input, sd)
            @test norm(to_dense(Osd) - to_dense(Ods)) < 1e-10
            @test TB.unmerge_expansions(sd) == exps
        end
    end
end
