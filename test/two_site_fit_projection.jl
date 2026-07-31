using Test
using LinearAlgebra: dot, norm
using Random: Xoshiro, randn
using Graft
using Graft.Backend
using Graft.TestUtils

const _FIT = Graft.Contractions
const _CON = Graft.Contractions

function _projection_probe(target, source, n, m;
                           target_center::Symbol, source_center::Int,
                           seed::Integer, dense::Bool=false)
    gauged_target = move_center!(
        copy(target), target_center === :n ? n : m)
    gauged_source = move_center!(copy(source), source_center)
    projected = _FIT._fit_two_site_tensor(
        _FIT._FitCache(topology(target)),
        gauged_target, gauged_source, n, m,
    )
    reference = _FIT._fit_two_site_tensor_ncon_reference(
        _FIT._FitCache(topology(target)),
        gauged_target, gauged_source, n, m,
    )

    @test space(projected) == _CON.two_site_space(gauged_target, n, m)
    @test norm(projected - reference) ≤
          2e-12 * max(norm(reference), 1.0)

    probe_tensor = randn(Xoshiro(seed), ComplexF64, space(projected))
    probe_state = copy(gauged_target)
    _CON.split_two_site!(
        probe_state, probe_tensor, n, m; center_on=target_center)
    @test norm(_CON.two_site_tensor(probe_state, n, m) - probe_tensor) ≤
          2e-12 * max(norm(probe_tensor), 1.0)

    overlap = dense ?
        dot(categorical_coordinates(probe_state),
            categorical_coordinates(gauged_source)) :
        inner(probe_state, gauged_source)
    @test dot(probe_tensor, projected) ≈ overlap atol=2e-12 rtol=2e-12

    self_projection = _FIT._fit_two_site_tensor(
        _FIT._FitCache(topology(target)),
        gauged_target, gauged_target, n, m,
    )
    @test norm(
        self_projection - _CON.two_site_tensor(gauged_target, n, m),
    ) ≤ 2e-12
    return projected
end

@testset "two-site fit projection" begin
    @testset "ComplexSpace chain, physical root, and gauge covariance" begin
        topo = mps_topology(3)
        phys = Dict(nodeid(topo, i) => spin_ops().P for i in 1:nnodes(topo))
        target = random_ttns(
            Xoshiro(0x2001), ComplexF64, topo, phys, ℂ^2)
        source = random_ttns(
            Xoshiro(0x2002), ComplexF64, topo, phys, ℂ^3)
        m = topo.root
        n = only(topo.children[m])
        leaf = only(topo.children[n])

        from_n = _projection_probe(
            target, source, n, m;
            target_center=:n, source_center=leaf, seed=0x2003,
            dense=true,
        )
        from_m = _projection_probe(
            target, source, n, m;
            target_center=:m, source_center=m, seed=0x2004,
            dense=true,
        )
        @test norm(from_n - from_m) ≤
              3e-12 * max(norm(from_n), 1.0)

        off_edge = move_center!(copy(target), leaf)
        @test_throws ArgumentError _FIT._fit_two_site_tensor(
            _FIT._FitCache(topo), off_edge, source, n, m)
        @test_throws ArgumentError _FIT._fit_two_site_tensor(
            _FIT._FitCache(topo), move_center!(copy(target), n),
            source, m, n)
    end

    @testset "ComplexSpace branching with a physless root" begin
        topo = binary_topology(2; prefix=:fitroot)
        phys = Dict(nodeid(topo, i) => spin_ops().P for i in leaves(topo))
        target = random_ttns(
            Xoshiro(0x2101), ComplexF64, topo, phys, ℂ^2)
        source = random_ttns(
            Xoshiro(0x2102), ComplexF64, topo, phys, ℂ^3)
        m = topo.root
        n = first(topo.children[m])
        opposite_leaf = last(leaves(topo))

        projected = _projection_probe(
            target, source, n, m;
            target_center=:m, source_center=opposite_leaf, seed=0x2103,
            dense=true,
        )
        @test numout(projected) == numind(projected)
        @test numin(projected) == 0
    end

    @testset "FermionParity branching and legitimate virtual gauges" begin
        fermion = fermion_ops_z2()
        even, odd = FermionParity(0), FermionParity(1)
        small = Vect[FermionParity](even => 1, odd => 1)
        wide = Vect[FermionParity](even => 2, odd => 2)
        topo = TreeTopology(:root, [
            :root => :active, :root => :sibling, :active => :leaf,
        ])
        phys = Dict(site => fermion.P
                    for site in (:root, :active, :sibling, :leaf))
        target = random_ttns(
            Xoshiro(0x2201), ComplexF64, topo, phys, small)
        source = random_ttns(
            Xoshiro(0x2202), ComplexF64, topo, phys, wide)
        m = topo.root
        n = nodeindex(topo, :active)
        sibling = nodeindex(topo, :sibling)
        leaf = nodeindex(topo, :leaf)

        from_sibling = _projection_probe(
            target, source, n, m;
            target_center=:n, source_center=sibling, seed=0x2203,
        )
        from_leaf = _projection_probe(
            target, source, n, m;
            target_center=:m, source_center=leaf, seed=0x2204,
        )
        @test norm(from_sibling - from_leaf) ≤
              3e-12 * max(norm(from_sibling), 1.0)
    end

    @testset "U1 graded chain" begin
        physical = U1Space(0 => 1, 1 => 1)
        small = U1Space(-1 => 1, 0 => 2, 1 => 1)
        wide = U1Space(-2 => 1, -1 => 2, 0 => 3, 1 => 2, 2 => 1)
        topo = mps_topology(3)
        phys = Dict(nodeid(topo, i) => physical for i in 1:nnodes(topo))
        target = random_ttns(
            Xoshiro(0x2301), ComplexF64, topo, phys, small)
        source = random_ttns(
            Xoshiro(0x2302), ComplexF64, topo, phys, wide)
        m = topo.root
        n = only(topo.children[m])
        leaf = only(topo.children[n])

        projected = _projection_probe(
            target, source, n, m;
            target_center=:m, source_center=leaf, seed=0x2303,
        )
        @test check_arrows(move_center!(copy(target), m))
        @test norm(projected) > 0
    end
end
