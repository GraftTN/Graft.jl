using Test
using Graft
using Graft.TestUtils
using Graft.Backend: U1Irrep, FermionParity, SU2Irrep, Vect, ⊗, ←, oneunit,
    codomain, dim, domain, flip, isdual, numind
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

function _requires_pivotal_link(O)
    t = topology(O)
    child = only(t.children[t.root])
    unfolded = Graft.Networks._compression_unfold(O.tensors[child])
    _, S, Vᴴ, _ = Graft.Backend.split_svd_with_error(
        unfolded,
        TruncationScheme(atol=1e-12),
    )
    link = S * Vᴴ
    return isdual(codomain(link)[1]) != isdual(domain(link)[1])
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
    for (column, (_, input_state)) in enumerate(basis)
        output = apply(O, input_state)
        output_root = domain(output.tensors[root])[1]
        for (row, (_, output_state)) in enumerate(basis)
            domain(output_state.tensors[root])[1] == output_root || continue
            values[row, column] = inner(output_state, output)
        end
    end
    return values
end


@graft_testset "TTNO sector-aware compression" begin
    factor_ops = spin_ops()
    U, S, Vᴴ = Graft.Backend.split_svd(factor_ops.X, TruncationScheme())
    @test U * S * Vᴴ ≈ factor_ops.X

    O = _redundant_dense_ttno()
    child = only(topology(O).children[topology(O).root])
    before = dim(virtualspace(O, child))
    reference = to_dense(O)
    identity_before = O
    report = compress!(O; compression_atol=1e-12)

    @test O === identity_before
    @test report isa TTNOCompressionReport
    @test report.mode === :exact_rank
    @test report.total_before_dimension == before
    @test report.total_after_dimension < before
    @test report.compression_ratio == report.total_after_dimension / report.total_before_dimension
    @test report.aggregate_local_discarded_norm ≤ 1e-10
    @test length(report.edges) == 1
    edge = only(report.edges)
    @test edge.child == :site1 && edge.parent == :site2
    @test edge.after_svd_dimension < edge.before_dimension
    @test all(s -> s.after_deparallelization_dimension <= s.before_dimension,
              edge.sectors)
    @test all(s -> s.after_qr_dimension == s.after_deparallelization_dimension,
              edge.sectors)
    @test all(s -> s.after_svd_dimension == s.after_qr_dimension, edge.sectors)
    @test check_arrows(O)
    @test O.ishermitian
    @test norm(to_dense(O) - reference) < 1e-10

    @test_throws ArgumentError compress!(
        _redundant_dense_ttno();
        compression_atol=1e-12,
        scheme=TruncationScheme(maxdim=1),
    )
    @test_throws ArgumentError compress!(
        _redundant_dense_ttno();
        compression_atol=1e-12,
        mode=:truncate,
    )

    Ophysless = _redundant_physless_ttno()
    reference_physless = to_dense(Ophysless)
    report_physless = compress!(Ophysless; compression_atol=1e-12)
    @test length(report_physless.edges) == 3
    @test Set(edge.child for edge in report_physless.edges) == Set([:left, :right, :junction])
    @test check_arrows(Ophysless)
    @test norm(to_dense(Ophysless) - reference_physless) < 1e-10

    Ou1 = _redundant_u1_ttno()
    child_u1 = only(topology(Ou1).children[topology(Ou1).root])
    reference_u1 = to_dense(Ou1)
    report_u1 = compress!(Ou1; compression_atol=1e-12)
    @test check_arrows(Ou1)
    @test norm(to_dense(Ou1) - reference_u1) < 1e-10
    @test all(s -> s.sector isa U1Irrep, only(report_u1.edges).sectors)
    @test all(s -> s.after_svd_dimension <= s.before_dimension,
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
        @test all(s -> s.after_svd_dimension <= s.before_dimension,
                  only(report_f.edges).sectors)
    end

    Odual = _dual_virtual_fz2_ttno()
    @test _requires_pivotal_link(Odual)
    dual_topology = topology(Odual)
    dual_phys = Dict(:site1 => fermion_ops_z2().P, :site2 => fermion_ops_z2().P)
    dual_basis = _fz2_product_basis(dual_topology, dual_phys)
    reference_dual_action = _fz2_action_matrix(Odual, dual_basis)

    if GRAFT_EXTENDED_TESTS
        dual_child = only(dual_topology.children[dual_topology.root])

        # Each exact-rank stage must preserve the operator action, not merely a
        # raw TTNO contraction. The dual edge forces the factor codomain/domain
        # orientations to differ and therefore catches an extra pivotal twist.
        Odual_deparallel = _copy_ttno(Odual)
        Graft.Networks._deparallelize_edge!(Odual_deparallel, dual_child, 1e-12)
        @test _fz2_action_matrix(Odual_deparallel, dual_basis) ≈ reference_dual_action atol=1e-12

        Odual_qr = _copy_ttno(Odual)
        Graft.Networks._deparallelize_edge!(Odual_qr, dual_child, 1e-12)
        Graft.Networks._qr_canonicalize_edge!(Odual_qr, dual_child)
        @test _fz2_action_matrix(Odual_qr, dual_basis) ≈ reference_dual_action atol=1e-12

        Odual_svd = _copy_ttno(Odual)
        Graft.Networks._deparallelize_edge!(Odual_svd, dual_child, 1e-12)
        Graft.Networks._qr_canonicalize_edge!(Odual_svd, dual_child)
        Graft.Networks._svd_compress_edge!(Odual_svd, dual_child, TruncationScheme())
        @test _fz2_action_matrix(Odual_svd, dual_basis) ≈ reference_dual_action atol=1e-12
    end

    report_dual = compress!(Odual; compression_atol=1e-12)
    @test check_arrows(Odual)
    @test _fz2_action_matrix(Odual, dual_basis) ≈ reference_dual_action atol=1e-12
    @test all(s -> s.sector isa FermionParity, only(report_dual.edges).sectors)

    topo_su2 = mps_topology(1)
    Psu2 = Vect[SU2Irrep](SU2Irrep(0) => 1, SU2Irrep(1 // 2) => 1)
    Wsu2 = zeros(ComplexF64, Psu2 ← Psu2 ⊗ oneunit(Psu2))
    Osu2 = TTNO(topo_su2, [Wsu2])
    @test_throws ArgumentError compress!(Osu2; compression_atol=1e-12)
end
