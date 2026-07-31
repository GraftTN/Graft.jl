# CP1d — compression integration and performance acceptance
# (ttno-compression plan v2, CP1d deliverables): raw compiler output,
# structurally merged compiler output, and external TTNO inputs through the
# three-stage pipeline; compiler-certified provenance; DMRG and TDVP
# observables with exact-rank and bounded approximate compression; and
# runtime/allocation/bond-reduction characterization.

using Graft.TestUtils: to_dense, dense_hamiltonian, exact_groundstate,
    exact_evolve, random_ttns, product_ttns
using Graft.Backend: dim, ℂ
using Random: Xoshiro

function _cp1d_tfi(n)
    S = spin_ops()
    sites = [Symbol(:site, i) for i in 1:n]
    H = OpSum()
    for (a, b) in zip(sites[1:end-1], sites[2:end])
        H += Term(-1.0, SiteOp(a, :Z, S.Z), SiteOp(b, :Z, S.Z))
    end
    for s in sites
        H += Term(-0.7, SiteOp(s, :X, S.X))
    end
    return H, mps_topology(n), Dict(s => S.P for s in sites)
end

@graft_testset "CP1d compiler outputs through the three-stage pipeline" begin
    H, topo, phys = _cp1d_tfi(4)
    reference = dense_hamiltonian(H, topo, phys)

    # Raw (direct-sum) compiler output: per-term transport channels are
    # bitwise duplicates, so exact Stage 1 removes them with witnesses.
    Oraw, _, raw_provenance =
        compile_ttno(H, topo, phys; merge=DirectSumMerge())
    @test raw_provenance isa TTNOExactProvenance
    raw_before = sum(dim(virtualspace(Oraw, c))
                     for c in 1:nnodes(topo) if c != topo.root)
    report_raw = compress!(Oraw; compression_atol=1e-12)
    @test norm(to_dense(Oraw) - reference) < 1e-10
    @test report_raw.total_after_dimension < raw_before
    @test sum(e.exact_witness_count for e in report_raw.edges) > 0
    @test any(e.exact_deparallelized_rank < e.input_rank
              for e in report_raw.edges)
    @test all(e.exact_deparallelized_rank <= e.input_rank
              for e in report_raw.edges)

    # Structurally merged compiler output.
    Osm, _, sm_provenance = compile_ttno(
        H, topo, phys; merge=StateDiagramMerge(StructuralOptimizer()))
    @test isnothing(sm_provenance)
    report_sm = compress!(Osm; compression_atol=1e-12)
    @test norm(to_dense(Osm) - reference) < 1e-10

    # Fully optimized compiler output and the external legacy TTNO.
    Osge, _, sge_provenance = compile_ttno(H, topo, phys)
    @test isnothing(sge_provenance)
    report_sge = compress!(Osge; compression_atol=1e-12)
    @test norm(to_dense(Osge) - reference) < 1e-10
    Oext = ttno_from_opsum(H, topo, phys)
    report_ext = compress!(Oext; compression_atol=1e-12)
    @test norm(to_dense(Oext) - reference) < 1e-10
    # All three converge to the same compressed bond profile.
    @test [e.retained_svd_rank for e in report_sge.edges] ==
        [e.retained_svd_rank for e in report_ext.edges]
end

@graft_testset "CP1d compiler-certified provenance feeds exact Stage 1" begin
    # Two completed terms with identical structure and different
    # coefficients realize proportional done-channel columns: bitwise
    # Stage 1 cannot certify them, compiler provenance can.
    S = spin_ops()
    topo = mps_topology(3)
    phys = Dict(:site1 => S.P, :site2 => S.P, :site3 => S.P)
    H = OpSum() +
        Term(0.7, SiteOp(:site1, :X, S.X), SiteOp(:site2, :Z, S.Z)) +
        Term(-0.31, SiteOp(:site1, :X, S.X), SiteOp(:site2, :Z, S.Z)) +
        Term(0.4, SiteOp(:site2, :X, S.X), SiteOp(:site3, :X, S.X))
    O1, _, provenance1 =
        compile_ttno(H, topo, phys; merge=DirectSumMerge())
    O2, _, provenance =
        compile_ttno(H, topo, phys; merge=DirectSumMerge())
    @test provenance1 isa TTNOExactProvenance
    @test provenance isa TTNOExactProvenance
    @test !isempty(provenance.relations)
    @test any(r -> r.factor ≈ -0.31 / 0.7, provenance.relations)

    reference = to_dense(O1)
    plain = compress!(O1; compression_atol=1e-12)
    certified = compress!(O2; compression_atol=1e-12, provenance)
    @test norm(to_dense(O2) - reference) < 1e-10
    witnesses = reduce(vcat, [e.witnesses for e in certified.edges])
    @test any(w -> w.source === :provenance, witnesses)
    # The certified run removes at least as much in exact Stage 1.
    @test sum(e.exact_deparallelized_rank for e in certified.edges) <=
        sum(e.exact_deparallelized_rank for e in plain.edges)
    @test all(isempty(e.fallback_reasons) for e in certified.edges)
end

@graft_testset "CP1d DMRG and TDVP observables under compression" begin
    H, topo, phys = _cp1d_tfi(4)
    Hd = dense_hamiltonian(H, topo, phys)
    E0, _ = exact_groundstate(Hd)

    for (label, prep) in [
        ("exact-rank", O -> compress!(O; compression_atol=1e-12)),
        ("approximate", O -> compress!(O; compression_atol=0.0,
                                       mode=:approximate,
                                       scheme=TruncationScheme(atol=1e-8))),
    ]
        O, _, _ = compile_ttno(H, topo, phys; hermitian=true)
        prep(O)
        rng = Xoshiro(0xcafe1d)
        ψ = random_ttns(rng, ComplexF64, topo, phys, ℂ^4)
        _, Es = dmrg2!(ψ, O; nsweeps=6, verbose=false)
        @test isapprox(real(Es[end]), E0; atol=1e-6)
    end

    # TDVP trajectory against exact dense propagation.
    Ht, topot, physt = _cp1d_tfi(3)
    Ot, _, _ = compile_ttno(Ht, topot, physt; hermitian=true)
    compress!(Ot; compression_atol=1e-12)
    Hdt = dense_hamiltonian(Ht, topot, physt)
    rng = Xoshiro(0x7d0f)
    ψ0 = random_ttns(rng, ComplexF64, topot, physt, ℂ^4)
    normalize!(ψ0)
    v = to_dense(ψ0)
    ψ = deepcopy(ψ0)
    dt = 0.05
    ev = TDVP1(order=2, verbose=false)
    for _ in 1:5
        step!(ev, ψ, Ot, -im * dt)
        v = exact_evolve(Hdt, v, -im * dt)
    end
    overlap = abs(dot(to_dense(ψ), v))
    @test overlap > 1 - 1e-8
end

@graft_testset "CP1d performance characterization and report determinism" begin
    S = spin_ops()
    sites = [Symbol(:site, i) for i in 1:8]
    H = OpSum()
    for (a, b) in zip(sites[1:end-1], sites[2:end])
        H += Term(-1.0, SiteOp(a, :Z, S.Z), SiteOp(b, :Z, S.Z))
        H += Term(0.5, SiteOp(a, :Sp, S.Sp), SiteOp(b, :Sm, S.Sm))
        H += Term(0.5, SiteOp(a, :Sm, S.Sm), SiteOp(b, :Sp, S.Sp))
    end
    for s in sites
        H += Term(-0.7, SiteOp(s, :X, S.X))
    end
    topo = mps_topology(8)
    phys = Dict(s => S.P for s in sites)

    O_warm, _, _ = compile_ttno(H, topo, phys; merge=DirectSumMerge())
    compress!(O_warm; compression_atol=1e-12)
    O1, _, _ = compile_ttno(H, topo, phys; merge=DirectSumMerge())
    before = sum(dim(virtualspace(O1, c)) for c in 1:nnodes(topo)
                 if c != topo.root)
    stats = @timed compress!(O1; compression_atol=1e-12)
    report1 = stats.value
    println("[cp1d-bench] compress!(direct-sum chain-8): ",
            "time=$(round(stats.time; digits=4))s alloc=$(stats.bytes) bytes ",
            "bond $(before) -> $(report1.total_after_dimension) ",
            "(ratio $(round(report1.compression_ratio; digits=4)))")
    @test report1.total_after_dimension < before
    @test isfinite(stats.time) && stats.time < 120

    O2, _, _ = compile_ttno(H, topo, phys; merge=DirectSumMerge())
    report2 = compress!(O2; compression_atol=1e-12)
    @test report1.stage_trace == report2.stage_trace
    @test [(e.child, e.input_rank, e.exact_deparallelized_rank,
            e.post_qr_rank, e.retained_svd_rank)
           for e in report1.edges] ==
          [(e.child, e.input_rank, e.exact_deparallelized_rank,
            e.post_qr_rank, e.retained_svd_rank)
           for e in report2.edges]
    @test all(O1.tensors[i] == O2.tensors[i] for i in eachindex(O1.tensors))
end
