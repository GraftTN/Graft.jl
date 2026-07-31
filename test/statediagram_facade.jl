# SD6 — typed opt-in compiler facade, reports, and benchmarks
# (state-diagram-compiler plan v2, SD6 deliverables). The legacy compiler
# stays the default and the explicit oracle; every fixture asserts exact
# action equality and bond dimensions no larger than legacy.

using Graft.Backend: dim

"Representative topology matrix: chain, star, fork, binary, small T3NS, Cayley-like."
function _sd6_topologies()
    t3ns = TreeTopology(:root, [
        :root => :junction,
        :junction => :left,
        :junction => :right,
    ])
    cayley = TreeTopology(:c, [
        :c => :a, :c => :b,
        :a => :a1, :a => :a2,
        :b => :b1, :b => :b2,
    ])
    return [
        (:chain, mps_topology(4), nothing),
        (:star, star_topology(2, 1), nothing),
        (:fork, fork_topology(2, 1), nothing),
        (:binary, binary_topology(2), nothing),
        (:t3ns, t3ns, Set([:left, :right])),      # physless root/junction
        (:cayley, cayley, nothing),
    ]
end

"Spin Hamiltonian over the physical sites of a topology."
function _sd6_spin_h(topo, physical_sites)
    S = spin_ops()
    sites = [nodeid(topo, i) for i in 1:nnodes(topo)
             if physical_sites === nothing || nodeid(topo, i) in physical_sites]
    H = OpSum()
    for (a, b) in zip(sites[1:end-1], sites[2:end])
        H += Term(-1.0, SiteOp(a, :Z, S.Z), SiteOp(b, :Z, S.Z))
    end
    for s in sites
        H += Term(-0.7, SiteOp(s, :X, S.X))
    end
    H += Term(0.4, SiteOp(sites[1], :Z, S.Z), SiteOp(sites[end], :Z, S.Z))
    phys = Dict(s => S.P for s in sites)
    return H, phys
end

function _sd6_compare(H, topo, phys; kernels)
    L = ttno_from_opsum(H, topo, phys)
    dense_L = to_dense(L)
    for merge in kernels
        O, report, provenance = compile_ttno(H, topo, phys; merge)
        @test norm(to_dense(O) - dense_L) < 1e-11
        @test report isa TTNOBuildReport
        @test TB.is_supported(report.capability)
        if merge isa DirectSumMerge
            @test provenance isa TTNOExactProvenance
        else
            @test isnothing(provenance)
        end
        if merge isa StateDiagramMerge
            # Optimizing kernels never exceed the legacy bond layout; the
            # direct-sum oracle is intentionally uncompressed.
            for edge in report.edges
                c = nodeindex(topo, edge.child)
                @test edge.dimension <= dim(virtualspace(L, c))
            end
        end
        # Deterministic reports and bond layout across repeated runs.
        O2, report2, provenance2 = compile_ttno(H, topo, phys; merge)
        @test report == report2
        @test typeof(provenance2) === typeof(provenance)
        @test all(O[i] == O2[i] for i in 1:nnodes(topo))
    end
end

@graft_testset "SD6 public compiler facade types" begin
    @test AbstractOperatorLoweringKernel ===
        Graft.TTNOBuild.AbstractOperatorLoweringKernel
    @test AbstractTTNOMergeKernel === Graft.TTNOBuild.AbstractTTNOMergeKernel
    @test MissingCategoryCapability === Graft.TTNOBuild.MissingCategoryCapability
    @test AbelianScalarLowering() isa AbstractOperatorLoweringKernel
    @test DirectSumMerge() isa AbstractTTNOMergeKernel
end

@graft_testset "SD6 opt-in facade over the topology matrix (spin)" begin
    kernels = [DirectSumMerge(),
               StateDiagramMerge(StructuralOptimizer()),
               StateDiagramMerge(SGEOptimizer())]
    for (name, topo, physical) in _sd6_topologies()
        H, phys = _sd6_spin_h(topo, physical)
        _sd6_compare(H, topo, phys; kernels)
    end
end

@graft_testset "SD6 opt-in facade over the physics matrix (chain)" begin
    kernels = [DirectSumMerge(), StateDiagramMerge(SGEOptimizer())]
    topo = mps_topology(3)

    U = spin_ops_u1()
    physu = Dict(:site1 => U.P, :site2 => U.P, :site3 => U.P)
    Hu = OpSum()
    Hu += Term(0.5, SiteOp(:site1, :Sp, U.Sp), SiteOp(:site2, :Sm, U.Sm))
    Hu += Term(0.5, SiteOp(:site2, :Sp, U.Sp), SiteOp(:site3, :Sm, U.Sm))
    Hu += Term(0.25, SiteOp(:site1, :Z, U.Z), SiteOp(:site3, :Z, U.Z))
    _sd6_compare(Hu, topo, physu; kernels)

    B = boson_ops_u1(2)
    physb = Dict(:site1 => B.P, :site2 => B.P, :site3 => B.P)
    Hb = OpSum()
    Hb += Term(0.4, SiteOp(:site1, :Bd, B.Bd), SiteOp(:site2, :B, B.B))
    Hb += Term(0.4, SiteOp(:site2, :Bd, B.Bd), SiteOp(:site1, :B, B.B))
    Hb += Term(0.9, SiteOp(:site2, :N, B.N))
    _sd6_compare(Hb, topo, physb; kernels)

    Bo = boson_ops(2)
    ptopo = mps_topology(2)
    pphys = Dict(:site1 => Bo.P, :site2 => Bo.P)
    Hp = OpSum()
    Hp += Term(0.6, SiteOp(:site1, :Bd, Bo.Bd), SiteOp(:site2, :B, Bo.B))
    Hp += Term(0.6, SiteOp(:site1, :B, Bo.B), SiteOp(:site2, :Bd, Bo.Bd))
    Hp += Term(0.2, SiteOp(:site1, :N, Bo.N))
    Hpp, ptopo2, pphys2 = ppdress(Hp, ptopo, pphys; nmax=2)
    _sd6_compare(Hpp, ptopo2, pphys2; kernels)

    F = fermion_ops_z2()
    physf = Dict(:site1 => F.P, :site2 => F.P, :site3 => F.P)
    Hf = OpSum()
    Hf += Term(1.0, SiteOp(:site1, :Cd, F.Cd), SiteOp(:site3, :C, F.C))
    Hf += Term(1.0, SiteOp(:site3, :Cd, F.Cd), SiteOp(:site1, :C, F.C))
    Hf += Term(0.5, SiteOp(:site2, :N, F.N))
    _sd6_compare(Hf, topo, physf; kernels)

    M = _sd2_multimode_carrier(2)
    mtopo = mps_topology(2)
    mphys = Dict(:site1 => M.P, :site2 => M.P)
    Hm = OpSum()
    Hm += Term(0.8, SiteOp(:site1, :Cd1, M.Cd[1]), SiteOp(:site2, :C2, M.C[2]))
    Hm += Term(0.8, SiteOp(:site2, :Cd2, M.Cd[2]), SiteOp(:site1, :C1, M.C[1]))
    Hm += Term(0.3, SiteOp(:site1, :N2, M.N[2]))
    _sd6_compare(Hm, mtopo, mphys; kernels)

    # Fallback behavior: unsupported categories fail closed through the
    # facade with the typed missing capability.
    su2 = Graft.Backend.Vect[Graft.Backend.SU2Irrep](
        Graft.Backend.SU2Irrep(0) => 1, Graft.Backend.SU2Irrep(1//2) => 1)
    stopo = mps_topology(2)
    Hsu2 = OpSum() + Term(1.0, SiteOp(:site1, :I, Graft.Backend.id(su2)))
    @test_throws Graft.TTNOBuild.MissingCategoryCapability compile_ttno(
        Hsu2, stopo, Dict(:site1 => su2, :site2 => su2))
end

@graft_testset "SD6 build-time and allocation characterization" begin
    S = spin_ops()
    topo = mps_topology(8)
    sites = [Symbol(:site, i) for i in 1:8]
    phys = Dict(s => S.P for s in sites)
    H = OpSum()
    for (a, b) in zip(sites[1:end-1], sites[2:end])
        H += Term(-1.0, SiteOp(a, :Z, S.Z), SiteOp(b, :Z, S.Z))
        H += Term(0.5, SiteOp(a, :Sp, S.Sp), SiteOp(b, :Sm, S.Sm))
        H += Term(0.5, SiteOp(a, :Sm, S.Sm), SiteOp(b, :Sp, S.Sp))
    end
    for s in sites
        H += Term(-0.7, SiteOp(s, :X, S.X))
    end

    # Warm up both compilers, then characterize one clean run each.
    ttno_from_opsum(H, topo, phys)
    compile_ttno(H, topo, phys)
    legacy = @timed ttno_from_opsum(H, topo, phys)
    typed = @timed compile_ttno(H, topo, phys)
    O, report, provenance = typed.value
    L = legacy.value
    @test isnothing(provenance)
    println("[sd6-bench] legacy: time=$(round(legacy.time; digits=4))s ",
            "alloc=$(legacy.bytes) bytes")
    println("[sd6-bench] typed(SGE): time=$(round(typed.time; digits=4))s ",
            "alloc=$(typed.bytes) bytes")
    println("[sd6-bench] bond dims legacy=",
            [dim(virtualspace(L, c)) for c in 1:nnodes(topo) if c != topo.root],
            " typed=", [e.dimension for e in report.edges])
    @test norm(to_dense(O) - to_dense(L)) < 1e-10
    @test all(e.dimension <= dim(virtualspace(L, nodeindex(topo, e.child)))
              for e in report.edges)
    @test isfinite(typed.time) && typed.time < 120
    # Report stability across clean repeated runs.
    _, report2, provenance2 = compile_ttno(H, topo, phys)
    @test report == report2
    @test isnothing(provenance2)
end
