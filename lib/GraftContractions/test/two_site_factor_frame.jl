using GraftContractions: EnvCache, contract_oriented_two_site, eff_h2,
    oriented_two_site_factor_frame, two_site_tensor
using GraftFoundation: FermionParity, mps_topology, norm, numout, permute, ℂ
using GraftNetworks: move_center!
using GraftSymbolic: OpSum, SiteOp, Term, fermion_ops_z2, spin_ops
using GraftTestUtils: product_ttns, random_ttns
using GraftTTNOBuild: ttno_from_opsum
using Random: Xoshiro
using Test

function _factor_dense_fixture(; seed=2026080211)
    topo = mps_topology(2)
    spin = spin_ops()
    physical = Dict(:site1 => spin.P, :site2 => spin.P)
    terms = OpSum() +
        Term(0.37, SiteOp(:site1, :Z, spin.Z)) +
        Term(-0.19, SiteOp(:site2, :X, spin.X)) +
        Term(0.23, SiteOp(:site1, :X, spin.X),
                   SiteOp(:site2, :Z, spin.Z))
    H = ttno_from_opsum(terms, topo, physical; hermitian=true)
    psi = random_ttns(
        Xoshiro(seed), ComplexF64, topo, physical, ℂ^1; center=topo.root)
    return psi, H
end

function _factor_graded_fixture()
    topo = mps_topology(2)
    fermion = fermion_ops_z2()
    physical = Dict(:site1 => fermion.P, :site2 => fermion.P)
    basis = Dict(:site1 => FermionParity(0), :site2 => FermionParity(1))
    terms = OpSum() +
        Term(-0.5, SiteOp(:site1, :N, fermion.N)) +
        Term(-1.0, SiteOp(:site1, :Cd, fermion.Cd),
                   SiteOp(:site2, :C, fermion.C)) +
        Term(-1.0, SiteOp(:site1, :C, fermion.C),
                   SiteOp(:site2, :Cd, fermion.Cd))
    H = ttno_from_opsum(terms, topo, physical; hermitian=true)
    return product_ttns(ComplexF64, topo, physical, basis), H
end

function _factor_c1_error(psi, H, source, target)
    cache = EnvCache(psi.topo)
    frame = oriented_two_site_factor_frame(cache, psi, H, source, target)
    factorized = contract_oriented_two_site(cache, frame)
    child, parent = psi.topo.parent[source] == target ?
        (source, target) : (target, source)
    theta = two_site_tensor(psi, child, parent)
    reference = eff_h2(cache, psi, H, child, parent)(theta)
    if source == parent
        ns = numout(frame.source_action)
        nt = numout(frame.target_action)
        factorized = permute(
            factorized,
            ((ntuple(i -> ns + i, nt)..., ntuple(identity, ns)...), ()))
    end
    return norm(factorized - reference), norm(reference)
end

@testset "oriented two-site factor frame dense C1 both directions" begin
    psi, H = _factor_dense_fixture()
    for (source, target) in ((1, 2), (2, 1))
        move_center!(psi, source)
        error, scale = _factor_c1_error(psi, H, source, target)
        @test error <= 1e-12 * max(scale, 1.0)
    end
end

@testset "oriented two-site factor frame graded safety" begin
    psi, H = _factor_graded_fixture()
    for (source, target) in ((1, 2), (2, 1))
        move_center!(psi, source)
        error, scale = _factor_c1_error(psi, H, source, target)
        @test error <= 1e-11 * max(scale, 1.0)
    end
end
