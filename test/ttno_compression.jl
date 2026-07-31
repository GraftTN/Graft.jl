using Test
using Graft
using GraftTestUtils
using Graft.Backend: U1Irrep, FermionParity, Trivial, SU2Irrep, Vect, ⊗, ←,
    oneunit, codomain, dim, domain, flip, isdual, numind, ℂ, TensorMap
using LinearAlgebra: norm

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

function _redundant_dense_ttno()
    topo = mps_topology(2)
    ops = spin_ops()
    phys = Dict(:site1 => ops.P, :site2 => ops.P)
    H = OpSum()
    H += Term(0.7, SiteOp(:site1, :x_left, ops.X), SiteOp(:site2, :z_left, ops.Z))
    H += Term(-0.2, SiteOp(:site1, :x_right, ops.X), SiteOp(:site2, :z_right, ops.Z))
    return ttno_from_opsum(H, topo, phys; hermitian=true)
end

function _redundant_u1_ttno()
    topo = mps_topology(2)
    ops = boson_ops_u1(1)
    phys = Dict(:site1 => ops.P, :site2 => ops.P)
    H = OpSum()
    H += Term(0.6, SiteOp(:site1, :bd_left, ops.Bd), SiteOp(:site2, :b_left, ops.B))
    H += Term(-0.15, SiteOp(:site1, :bd_right, ops.Bd), SiteOp(:site2, :b_right, ops.B))
    H += Term(0.6, SiteOp(:site1, :b_left, ops.B), SiteOp(:site2, :bd_left, ops.Bd))
    H += Term(-0.15, SiteOp(:site1, :b_right, ops.B), SiteOp(:site2, :bd_right, ops.Bd))
    return ttno_from_opsum(H, topo, phys; hermitian=true)
end

function _redundant_star_ttno()
    topo = star_topology(2, 1)
    ops = spin_ops()
    phys = Dict(nodeid(topo, i) => ops.P for i in 1:nnodes(topo))
    H = OpSum()
    H += Term(0.5, SiteOp(:b1_1, :x_left, ops.X), SiteOp(:center, :z_left, ops.Z))
    H += Term(-0.1, SiteOp(:b1_1, :x_right, ops.X), SiteOp(:center, :z_right, ops.Z))
    H += Term(0.3, SiteOp(:b2_1, :y_left, ops.Y), SiteOp(:center, :x_center_left, ops.X))
    H += Term(-0.2, SiteOp(:b2_1, :y_right, ops.Y), SiteOp(:center, :x_center_right, ops.X))
    return ttno_from_opsum(H, topo, phys; hermitian=true)
end

function _redundant_physless_ttno()
    topo = TreeTopology(:root, [
        :root => :junction,
        :junction => :left,
        :junction => :right,
    ])
    ops = spin_ops()
    phys = Dict(:left => ops.P, :right => ops.P)
    H = OpSum()
    H += Term(0.4, SiteOp(:left, :x_left, ops.X), SiteOp(:right, :z_left, ops.Z))
    H += Term(-0.1, SiteOp(:left, :x_right, ops.X), SiteOp(:right, :z_right, ops.Z))
    return ttno_from_opsum(H, topo, phys; hermitian=true)
end

function _redundant_fz2_ttno()
    topo = mps_topology(2)
    ops = fermion_ops_z2()
    phys = Dict(:site1 => ops.P, :site2 => ops.P)
    H = OpSum()
    H += Term(-1.0, SiteOp(:site1, :cd_left, ops.Cd), SiteOp(:site2, :c_left, ops.C))
    H += Term(0.25, SiteOp(:site1, :cd_right, ops.Cd), SiteOp(:site2, :c_right, ops.C))
    H += Term(-1.0, SiteOp(:site1, :c_left, ops.C), SiteOp(:site2, :cd_left, ops.Cd))
    H += Term(0.25, SiteOp(:site1, :c_right, ops.C), SiteOp(:site2, :cd_right, ops.Cd))
    return ttno_from_opsum(H, topo, phys; hermitian=true)
end

function _dual_virtual_fz2_ttno()
    O = _redundant_fz2_ttno()
    t = topology(O)
    child = only(t.children[t.root])
    parent = t.parent[child]
    slot = Graft.Trees.childslot(t, parent, child)
    O.tensors[parent] = flip(O.tensors[parent], slot)
    O.tensors[child] = flip(O.tensors[child], numind(O.tensors[child]))
    @assert check_arrows(O)
    @assert isdual(virtualspace(O, child))
    return O
end

"Hand-built 2-site dense TTNO with three channels: [X, 0.5X, Z]."
function _proportional_channel_ttno()
    topo = mps_topology(2)          # root :site2, child :site1
    ops = spin_ops()
    X = ComplexF64[0 1; 1 0]
    Z = ComplexF64[1 0; 0 -1]
    Y2 = ComplexF64[0.3 0; 0 1.1]
    W1 = zeros(ComplexF64, 2, 2, 3)
    W1[:, :, 1] = X
    W1[:, :, 2] = 0.5 .* X
    W1[:, :, 3] = Z
    W2 = zeros(ComplexF64, 3, 2, 2, 1)
    W2[1, :, :, 1] = Z
    W2[2, :, :, 1] = Y2
    W2[3, :, :, 1] = X
    child = TensorMap(W1, ops.P ← ops.P ⊗ ℂ^3)
    root = TensorMap(W2, ℂ^3 ⊗ ops.P ← ops.P ⊗ ℂ^1)
    return TTNO(topo, [root, child])
end

function _copy_ttno(O::TTNO)
    return TTNO(topology(O), copy.(O.tensors); ishermitian=O.ishermitian)
end

function _fz2_product_basis(topo, phys)
    labels = [(FermionParity(left), FermionParity(right))
              for left in 0:1 for right in 0:1]
    return [(label, product_ttns(
                ComplexF64, topo, phys,
                Dict(:site1 => label[1], :site2 => label[2]),
            )) for label in labels]
end

function _fz2_action_matrix(O::TTNO, basis)
    D = length(basis)
    values = zeros(ComplexF64, D, D)
    root = topology(O).root
    plan_cache = EnvCache(topology(O))
    for (column, (_, input_state)) in enumerate(basis)
        output = apply(O, input_state; optimize=false)
        output_root = domain(output.tensors[root])[1]
        for (row, (_, output_state)) in enumerate(basis)
            domain(output_state.tensors[root])[1] == output_root || continue
            values[row, column] = inner(
                output_state, output; plan_cache, optimize=false,
            )
        end
    end
    return values
end

"Assert the CP1 global phase ordering from the report's stage trace."
function _assert_stage_ordering(report)
    stages = [first(step) for step in report.stage_trace]
    last_dep = findlast(==(:exact_deparallelize), stages)
    first_qr = findfirst(==(:qr), stages)
    last_qr = findlast(==(:qr), stages)
    first_svd = findfirst(==(:svd), stages)
    @test last_dep !== nothing && first_qr !== nothing && first_svd !== nothing
    @test last_dep < first_qr
    @test last_qr < first_svd
    nedges = length(report.edges)
    @test count(==(:exact_deparallelize), stages) == nedges
    @test count(==(:qr), stages) == nedges
    @test count(==(:svd), stages) == nedges
end

@graft_testset "TTNO three-stage compression (exact Stage 1, QR, SVD)" begin
    O = _redundant_dense_ttno()
    child = only(topology(O).children[topology(O).root])
    before = dim(virtualspace(O, child))
    reference = to_dense(O)
    identity_before = O
    report = compress!(O; compression_atol=1e-12)

    @test O === identity_before
    @test report isa TTNOCompressionReport
    @test report.mode === :exact_rank
    @test report.sweep_root == :site2
    _assert_stage_ordering(report)
    @test report.total_before_dimension == before
    @test report.total_after_dimension < before
    @test report.compression_ratio == report.total_after_dimension / report.total_before_dimension
    @test report.aggregate_local_discarded_norm ≤ 1e-10
    @test length(report.edges) == 1
    edge = only(report.edges)
    @test edge.child == :site1 && edge.parent == :site2
    # The identical X-channel pair is a bitwise Stage-1 duplicate: exact
    # deparallelization removes it before any factorization runs.
    @test edge.exact_deparallelized_rank < edge.input_rank
    @test edge.exact_witness_count >= 1
    @test any(w -> w.source === :bitwise_duplicate, edge.witnesses)
    @test isempty(edge.fallback_reasons)
    @test edge.retained_svd_rank < edge.input_rank
    @test all(s -> s.exact_deparallelized_rank <= s.input_rank, edge.sectors)
    @test all(s -> s.retained_svd_rank <= s.post_qr_rank, edge.sectors)
    @test edge.physical_discarded_norm == 0.0
    @test check_arrows(O)
    @test O.ishermitian
    @test norm(to_dense(O) - reference) < 1e-10

    # Exact-rank mode rejects every physical truncation request; invalid
    # options never mutate the operand (transactional boundary).
    Ofrozen = _redundant_dense_ttno()
    frozen_tensors = copy.(Ofrozen.tensors)
    @test_throws ArgumentError compress!(
        Ofrozen;
        compression_atol=1e-12,
        scheme=TruncationScheme(maxdim=1),
    )
    @test_throws ArgumentError compress!(Ofrozen; compression_atol=1e-12, mode=:truncate)
    @test_throws ArgumentError compress!(
        Ofrozen; compression_atol=0.0, mode=:approximate,
    )
    @test_throws ArgumentError compress!(
        Ofrozen; compression_atol=1e-12, mode=:approximate,
        scheme=TruncationScheme(maxdim=1),
    )
    @test all(Ofrozen.tensors[i] == frozen_tensors[i]
              for i in eachindex(frozen_tensors))

    Ophysless = _redundant_physless_ttno()
    reference_physless = to_dense(Ophysless)
    report_physless = compress!(Ophysless; compression_atol=1e-12)
    @test length(report_physless.edges) == 3
    @test Set(edge.child for edge in report_physless.edges) == Set([:left, :right, :junction])
    _assert_stage_ordering(report_physless)
    @test check_arrows(Ophysless)
    @test norm(to_dense(Ophysless) - reference_physless) < 1e-10

    Ou1 = _redundant_u1_ttno()
    child_u1 = only(topology(Ou1).children[topology(Ou1).root])
    reference_u1 = to_dense(Ou1)
    report_u1 = compress!(Ou1; compression_atol=1e-12)
    @test check_arrows(Ou1)
    @test norm(to_dense(Ou1) - reference_u1) < 1e-10
    @test all(s -> s.sector isa U1Irrep, only(report_u1.edges).sectors)
    @test only(report_u1.edges).exact_witness_count >= 1
    @test all(s -> s.retained_svd_rank <= s.input_rank,
              only(report_u1.edges).sectors)
    @test dim(virtualspace(Ou1, child_u1)) <= report_u1.total_before_dimension

    if GRAFT_EXTENDED_TESTS
        Ostar = _redundant_star_ttno()
        reference_star = to_dense(Ostar)
        report_star = compress!(Ostar; compression_atol=1e-12)
        @test length(report_star.edges) == 2
        @test Set(edge.child for edge in report_star.edges) == Set([:b1_1, :b2_1])
        @test all(!isempty(edge.sectors) for edge in report_star.edges)
        @test check_arrows(Ostar)
        @test norm(to_dense(Ostar) - reference_star) < 1e-10

        Of = _redundant_fz2_ttno()
        reference_f = to_dense(Of)
        report_f = compress!(Of; compression_atol=1e-12)
        @test check_arrows(Of)
        @test norm(to_dense(Of) - reference_f) < 1e-10
        @test all(s -> s.sector isa FermionParity, only(report_f.edges).sectors)
        @test all(s -> s.retained_svd_rank <= s.input_rank,
                  only(report_f.edges).sectors)
    end

    topo_su2 = mps_topology(1)
    Psu2 = Vect[SU2Irrep](SU2Irrep(0) => 1, SU2Irrep(1 // 2) => 1)
    Wsu2 = zeros(ComplexF64, Psu2 ← Psu2 ⊗ oneunit(Psu2))
    Osu2 = TTNO(topo_su2, [Wsu2])
    @test_throws ArgumentError compress!(Osu2; compression_atol=1e-12)
end

@graft_testset "CP1a exact provenance witnesses and undercompression" begin
    # Without a certificate the 0.5-proportional pair is retained by Stage 1
    # (undercompression is valid); Stage 3's numerical-zero cutoff still
    # removes the rank deficiency, but the exact-deparallelized rank proves
    # Stage 1 never used numerical rank estimation.
    O_plain = _proportional_channel_ttno()
    reference = to_dense(O_plain)
    report_plain = compress!(O_plain; compression_atol=1e-12)
    edge_plain = only(report_plain.edges)
    @test edge_plain.exact_deparallelized_rank == 3
    @test edge_plain.exact_witness_count == 0
    @test edge_plain.retained_svd_rank == 2
    @test norm(to_dense(O_plain) - reference) < 1e-10

    # With the exact certificate, Stage 1 removes the proportional channel
    # and records a replayable provenance witness.
    O_prov = _proportional_channel_ttno()
    relation = TTNOExactChannelRelation{Trivial}(:site1, Trivial(), 1, 2, 0.5)
    report_prov = compress!(
        O_prov; compression_atol=1e-12,
        provenance=TTNOExactProvenance{Trivial}([relation]))
    edge_prov = only(report_prov.edges)
    @test edge_prov.exact_deparallelized_rank == 2
    @test edge_prov.exact_witness_count == 1
    witness = only(edge_prov.witnesses)
    @test witness.source === :provenance
    @test witness.kept == 1 && witness.removed == 2
    @test witness.factor == 0.5
    @test norm(to_dense(O_prov) - reference) < 1e-10

    # A certificate that disagrees with the data is rejected with a recorded
    # fallback reason: the channels are retained, never overcompressed.
    O_bad = _proportional_channel_ttno()
    bad = TTNOExactChannelRelation{Trivial}(:site1, Trivial(), 1, 3, 0.5)
    report_bad = compress!(
        O_bad; compression_atol=1e-12,
        provenance=TTNOExactProvenance{Trivial}([bad]))
    edge_bad = only(report_bad.edges)
    @test edge_bad.exact_deparallelized_rank == 3
    @test edge_bad.exact_witness_count == 0
    @test !isempty(edge_bad.fallback_reasons)
    @test norm(to_dense(O_bad) - reference) < 1e-10

    # Provenance naming an unknown node fails closed without mutation.
    O_unknown = _proportional_channel_ttno()
    unknown_tensors = copy.(O_unknown.tensors)
    ghost = TTNOExactChannelRelation{Trivial}(:ghost, Trivial(), 1, 2, 0.5)
    @test_throws ArgumentError compress!(
        O_unknown; compression_atol=1e-12,
        provenance=TTNOExactProvenance{Trivial}([ghost]))
    @test all(O_unknown.tensors[i] == unknown_tensors[i]
              for i in eachindex(unknown_tensors))
end

@graft_testset "CP1c approximate mode is explicit and reported" begin
    O = _proportional_channel_ttno()
    reference = to_dense(O)
    report = compress!(O; compression_atol=0.0, mode=:approximate,
                       scheme=TruncationScheme(maxdim=1))
    edge = only(report.edges)
    @test report.mode === :approximate
    @test edge.retained_svd_rank == 1
    @test edge.physical_discarded_norm > 0.0
    @test edge.numerical_zero_discarded_norm == 0.0
    @test report.aggregate_local_discarded_norm >= edge.physical_discarded_norm
    # The truncated action differs from the reference by an amount governed
    # by the reported edge-local discarded norms (declared local metric).
    err = norm(to_dense(O) - reference)
    @test err > 0.0
    @test err <= 10 * report.aggregate_local_discarded_norm + 1e-10

    # Tightening the budget (larger maxdim) does not increase the discarded
    # weight.
    O_loose = _proportional_channel_ttno()
    report_loose = compress!(O_loose; compression_atol=0.0, mode=:approximate,
                             scheme=TruncationScheme(maxdim=2))
    @test report_loose.aggregate_local_discarded_norm <=
        report.aggregate_local_discarded_norm + 1e-12
    @test norm(to_dense(O_loose) - reference) < 1e-10
end

@graft_testset "CP1b root-choice coverage and dual-edge action" begin
    # The same operator on chains rooted at either end compresses to the
    # same action; the sweep root is recorded in the report.
    ops = spin_ops()
    H(sites) = begin
        acc = OpSum()
        acc += Term(0.7, SiteOp(sites[1], :x_left, ops.X), SiteOp(sites[2], :z_left, ops.Z))
        acc += Term(-0.2, SiteOp(sites[1], :x_right, ops.X), SiteOp(sites[2], :z_right, ops.Z))
        acc += Term(0.4, SiteOp(sites[2], :y, ops.Z), SiteOp(sites[3], :y, ops.Z))
        acc
    end
    topoA = mps_topology(3)         # rooted at :site3
    topoB = TreeTopology(:site1, [:site1 => :site2, :site2 => :site3])
    phys = Dict(:site1 => ops.P, :site2 => ops.P, :site3 => ops.P)
    OA = ttno_from_opsum(H([:site1, :site2, :site3]), topoA, phys)
    OB = ttno_from_opsum(H([:site1, :site2, :site3]), topoB, phys)
    refA = to_dense(OA)
    reportA = compress!(OA; compression_atol=1e-12)
    reportB = compress!(OB; compression_atol=1e-12)
    @test reportA.sweep_root == :site3
    @test reportB.sweep_root == :site1
    @test norm(to_dense(OA) - refA) < 1e-10
    # Both rootings realize the same operator (site kron order matches
    # to_dense's shared physical ordering only per topology, so compare each
    # against its own uncompressed reference).
    OB_ref = ttno_from_opsum(H([:site1, :site2, :site3]), topoB, phys)
    @test norm(to_dense(OB) - to_dense(OB_ref)) < 1e-10

    # Dual virtual edge: every stage preserves the operator action.
    Odual = _dual_virtual_fz2_ttno()
    dual_topology = topology(Odual)
    dual_phys = Dict(:site1 => fermion_ops_z2().P, :site2 => fermion_ops_z2().P)
    dual_basis = _fz2_product_basis(dual_topology, dual_phys)
    reference_dual_action = _fz2_action_matrix(Odual, dual_basis)

    if GRAFT_EXTENDED_TESTS
        dual_child = only(dual_topology.children[dual_topology.root])
        norel = TTNOExactChannelRelation{FermionParity}[]

        Odual_stage1 = _copy_ttno(Odual)
        Graft.Networks._stage1_exact_deparallelize_edge!(Odual_stage1, dual_child, norel)
        @test _fz2_action_matrix(Odual_stage1, dual_basis) ≈ reference_dual_action atol=1e-12

        Odual_qr = _copy_ttno(Odual)
        Graft.Networks._stage1_exact_deparallelize_edge!(Odual_qr, dual_child, norel)
        Graft.Networks._qr_canonicalize_edge!(Odual_qr, dual_child)
        @test _fz2_action_matrix(Odual_qr, dual_basis) ≈ reference_dual_action atol=1e-12

        Odual_svd = _copy_ttno(Odual)
        Graft.Networks._stage1_exact_deparallelize_edge!(Odual_svd, dual_child, norel)
        Graft.Networks._qr_canonicalize_edge!(Odual_svd, dual_child)
        Graft.Networks._stage3_split_edge!(
            Odual_svd, dual_topology.root, dual_child,
            Graft.Networks._numerical_zero_scheme(1e-12))
        @test _fz2_action_matrix(Odual_svd, dual_basis) ≈ reference_dual_action atol=1e-12
    end

    report_dual = compress!(Odual; compression_atol=1e-12)
    @test check_arrows(Odual)
    @test _fz2_action_matrix(Odual, dual_basis) ≈ reference_dual_action atol=1e-12
    @test all(s -> s.sector isa FermionParity, only(report_dual.edges).sectors)

    # Repeated runs are deterministic stage for stage.
    Odet1 = _redundant_dense_ttno()
    Odet2 = _redundant_dense_ttno()
    rep1 = compress!(Odet1; compression_atol=1e-12)
    rep2 = compress!(Odet2; compression_atol=1e-12)
    @test rep1.stage_trace == rep2.stage_trace
    @test [(e.input_rank, e.exact_deparallelized_rank, e.post_qr_rank,
            e.retained_svd_rank) for e in rep1.edges] ==
          [(e.input_rank, e.exact_deparallelized_rank, e.post_qr_rank,
            e.retained_svd_rank) for e in rep2.edges]
    @test all(Odet1.tensors[i] == Odet2.tensors[i]
              for i in eachindex(Odet1.tensors))
end
