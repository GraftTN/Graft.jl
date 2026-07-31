# SD7 — extended promotion gate (state-diagram-compiler plan v2, SD7).
#
# The extended gate runs the typed compiler over the full TTNO build matrix
# while the legacy compiler remains the package default. Passing this gate
# establishes promotion readiness only: the actual default switch and the
# legacy quarantine are explicitly deferred and require separate approval
# (user instruction 2026-07-29 — the default compiler is not switched in
# this lane).

using Graft.Backend: dim
using Random: Xoshiro, randperm

const _SD7_RNG = Xoshiro(0x50d7a7e5)

function _sd7_gate(H, topo, phys; kernels, atol=1e-11)
    L = ttno_from_opsum(H, topo, phys)
    dense_L = to_dense(L)
    for merge in kernels
        O, report, provenance = compile_ttno(H, topo, phys; merge)
        @test norm(to_dense(O) - dense_L) < atol
        if merge isa DirectSumMerge
            @test provenance isa TTNOExactProvenance
        else
            @test isnothing(provenance)
        end
        if merge isa StateDiagramMerge
            for edge in report.edges
                @test edge.dimension <=
                    dim(virtualspace(L, nodeindex(topo, edge.child)))
            end
        end
        O2, report2, provenance2 = compile_ttno(H, topo, phys; merge)
        @test report == report2
        @test typeof(provenance2) === typeof(provenance)
        @test all(O[i] == O2[i] for i in 1:nnodes(topo))
    end
end

@graft_extended_testset "SD7 extended gate: full build matrix, legacy default" begin
    all_kernels = [DirectSumMerge(),
                   StateDiagramMerge(StructuralOptimizer()),
                   StateDiagramMerge(GammaCoverOptimizer()),
                   StateDiagramMerge(SGEOptimizer())]
    sge_only = [StateDiagramMerge(SGEOptimizer())]

    # Spin across the complete topology family, all four kernels.
    for (name, topo, physical) in _sd6_topologies()
        H, phys = _sd6_spin_h(topo, physical)
        _sd7_gate(H, topo, phys; kernels=all_kernels)
    end

    # Charged and graded physics across chain, fork, and star.
    U = spin_ops_u1()
    F = fermion_ops_z2()
    B = boson_ops_u1(2)
    for topo in (mps_topology(3), fork_topology(2, 1), star_topology(2, 1))
        sites = [nodeid(topo, i) for i in 1:nnodes(topo)]
        physu = Dict(s => U.P for s in sites)
        Hu = OpSum()
        for (a, b) in zip(sites[1:end-1], sites[2:end])
            Hu += Term(0.5, SiteOp(a, :Sp, U.Sp), SiteOp(b, :Sm, U.Sm))
            Hu += Term(0.5, SiteOp(a, :Sm, U.Sm), SiteOp(b, :Sp, U.Sp))
        end
        Hu += Term(0.25, SiteOp(sites[1], :Z, U.Z), SiteOp(sites[end], :Z, U.Z))
        _sd7_gate(Hu, topo, physu; kernels=sge_only)

        physf = Dict(s => F.P for s in sites)
        Hf = OpSum()
        Hf += Term(1.0, SiteOp(sites[1], :Cd, F.Cd), SiteOp(sites[end], :C, F.C))
        Hf += Term(1.0, SiteOp(sites[end], :Cd, F.Cd), SiteOp(sites[1], :C, F.C))
        Hf += Term(0.5, SiteOp(sites[2], :N, F.N))
        _sd7_gate(Hf, topo, physf; kernels=sge_only)

        physb = Dict(s => B.P for s in sites)
        Hb = OpSum()
        for (a, b) in zip(sites[1:end-1], sites[2:end])
            Hb += Term(0.4, SiteOp(a, :Bd, B.Bd), SiteOp(b, :B, B.B))
            Hb += Term(0.4, SiteOp(b, :Bd, B.Bd), SiteOp(a, :B, B.B))
        end
        _sd7_gate(Hb, topo, physb; kernels=sge_only)
    end

    # Projected purification and the multimode carrier.
    Bo = boson_ops(2)
    ptopo = mps_topology(2)
    pphys = Dict(:site1 => Bo.P, :site2 => Bo.P)
    Hp = OpSum()
    Hp += Term(0.6, SiteOp(:site1, :Bd, Bo.Bd), SiteOp(:site2, :B, Bo.B))
    Hp += Term(0.6, SiteOp(:site1, :B, Bo.B), SiteOp(:site2, :Bd, Bo.Bd))
    Hp += Term(0.2, SiteOp(:site1, :N, Bo.N))
    Hpp, ptopo2, pphys2 = ppdress(Hp, ptopo, pphys; nmax=2)
    _sd7_gate(Hpp, ptopo2, pphys2; kernels=sge_only)

    M = _sd2_multimode_carrier(2)
    mtopo = mps_topology(2)
    mphys = Dict(:site1 => M.P, :site2 => M.P)
    Hm = OpSum()
    Hm += Term(0.8, SiteOp(:site1, :Cd1, M.Cd[1]), SiteOp(:site2, :C2, M.C[2]))
    Hm += Term(0.8, SiteOp(:site2, :Cd2, M.Cd[2]), SiteOp(:site1, :C1, M.C[1]))
    Hm += Term(0.3, SiteOp(:site1, :N2, M.N[2]))
    _sd7_gate(Hm, mtopo, mphys; kernels=sge_only)

    # Randomized dense sweep over the topology family.
    S = spin_ops()
    opset = [(:X, S.X), (:Y, S.Y), (:Z, S.Z), (:N, S.N)]
    for topo in (mps_topology(4), fork_topology(2, 1), binary_topology(2))
        sites = [nodeid(topo, i) for i in 1:nnodes(topo)]
        phys = Dict(s => S.P for s in sites)
        for _ in 1:4
            H = OpSum()
            for _ in 1:rand(_SD7_RNG, 3:7)
                nfac = rand(_SD7_RNG, 1:min(3, length(sites)))
                chosen = sites[randperm(_SD7_RNG, length(sites))[1:nfac]]
                ops = SiteOp[]
                for s in chosen
                    name, op = opset[rand(_SD7_RNG, 1:length(opset))]
                    push!(ops, SiteOp(s, name, op))
                end
                H += Term(randn(_SD7_RNG), ops...)
            end
            _sd7_gate(H, topo, phys; kernels=sge_only, atol=1e-10)
        end
    end

    # Promotion readiness only: the default compiler is unchanged. The
    # public default entry point remains the legacy ttno_from_opsum, and
    # the typed compiler stays opt-in behind compile_ttno.
    @test Graft.ttno_from_opsum === Graft.TTNOBuild.ttno_from_opsum
    @test Graft.compile_ttno === Graft.TTNOBuild.compile_ttno
end
