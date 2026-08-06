import GraftContractions
using GraftContractions: DistributedChannelAdmissionError, EnvCache,
    contract_biprojected_two_site,
    contract_oriented_two_site, contract_projected_two_site, eff_h2,
    oriented_two_site_factor_frame, two_site_tensor
using GraftFoundation: FermionParity, left_null, left_orth, mps_topology,
    norm, numout, permute, ℂ
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

function _factor_single_channel_fixture(; seed=2026080717)
    topo = mps_topology(2)
    spin = spin_ops()
    physical = Dict(:site1 => spin.P, :site2 => spin.P)
    terms = OpSum() + Term(
        0.23, SiteOp(:site1, :X, spin.X), SiteOp(:site2, :Z, spin.Z))
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

function _factor_biprojected_error(psi, H, source, target)
    cache = EnvCache(psi.topo)
    source_tensor, _ = GraftContractions.Contractions._oriented_site_tensor(
        psi, source, target)
    target_tensor, _ = GraftContractions.Contractions._oriented_site_tensor(
        psi, target, source)
    source_basis, link = left_orth(source_tensor)
    frame = oriented_two_site_factor_frame(
        cache, psi, H, source, target; source_tensor=source_basis)
    target_basis = left_null(target_tensor)
    source_projection = left_null(source_basis)
    reference = source_projection' * contract_projected_two_site(
        cache, frame, link, target_basis)
    factorized = contract_biprojected_two_site(
        cache, frame, link, source_projection, target_basis)
    channelized = contract_biprojected_two_site(
        cache, frame, link, source_projection, target_basis;
        threaded_channels=true, channel_slices=2, channel_minbatch=1,
        channel_min_flops=0,
        channel_memory_cap_bytes=1_000_000_000)
    return norm(factorized - reference), norm(channelized - reference),
           norm(reference)
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


@testset "oriented two-site biprojection dense and graded" begin
    for (psi, H) in (_factor_dense_fixture(), _factor_graded_fixture())
        for (source, target) in ((1, 2), (2, 1))
            move_center!(psi, source)
            error, channel_error, scale = _factor_biprojected_error(
                psi, H, source, target)
            @test error <= 1e-11 * max(scale, 1.0)
            @test channel_error <= 1e-11 * max(scale, 1.0)
        end
    end
end

struct _FactorDistributedContext <: GraftContractions.Parallel.AbstractDistributedContext
    rank::Int
    size::Int
end
GraftContractions.Parallel.distributed_rank(context::_FactorDistributedContext) =
    context.rank
GraftContractions.Parallel.distributed_size(context::_FactorDistributedContext) =
    context.size

mutable struct _ReplayFactorContext <: GraftContractions.Parallel.AbstractDistributedContext
    remote::Any
end
GraftContractions.Parallel.distributed_rank(::_ReplayFactorContext) = 0
GraftContractions.Parallel.distributed_size(::_ReplayFactorContext) = 2
function GraftContractions.Parallel.distributed_allreduce_sum!(
        context::_ReplayFactorContext, value)
    value isa AbstractArray ||
        GraftContractions.Contractions.axpy!(1, context.remote, value)
    return value
end
GraftContractions.Parallel.distributed_broadcast!(
    ::_ReplayFactorContext, value) = value

function _factor_remote_rank_slices(cache, frame, link,
                                    source_basis, target_basis)
    contractions = GraftContractions.Contractions
    source_projected = source_basis' * frame.source_action
    target_projected = target_basis' * frame.target_action
    spec = contractions._biprojected_two_site_spec(
        source_projected, target_projected)
    operands = (source_projected, link, target_projected)
    operator_space = contractions.space(
        source_projected, contractions.numind(source_projected))
    groups = contractions._biprojected_channel_groups(
        operator_space, max(2, 2 * Threads.nthreads()))
    indices = collect(2:2:length(groups))
    sliced_operands = Vector{Any}(undef, length(groups))
    plans = Vector{Any}(undef, length(groups))
    for slice in indices
        sliced_operands[slice] = contractions._biprojected_slice_operands(
            operands, groups[slice])
        plans[slice] = contractions._cache_get_or_plan!(
            cache, Symbol("test_remote_biprojected_channel_", slice),
            spec, sliced_operands[slice],
            contractions.scalartype(source_projected);
            optimize=true, sector_aware=true)
    end
    return contractions._sum_biprojected_slices(
        plans, sliced_operands, indices; threaded=false, minbatch=1)
end

@testset "factorized distributed channel admission has no materialized fallback" begin
    psi, H = _factor_dense_fixture(seed=2026080714)
    move_center!(psi, 1)
    cache = EnvCache(psi.topo)
    source_tensor, _ = GraftContractions.Contractions._oriented_site_tensor(
        psi, 1, 2)
    target_tensor, _ = GraftContractions.Contractions._oriented_site_tensor(
        psi, 2, 1)
    source_basis, link = left_orth(source_tensor)
    target_basis, _ = left_orth(target_tensor)
    frame = oriented_two_site_factor_frame(
        cache, psi, H, 1, 2; source_tensor=source_basis)
    contractions = GraftContractions.Contractions
    source_projected = source_basis' * frame.source_action
    operator_space = contractions.space(
        source_projected, contractions.numind(source_projected))
    impossible_ranks = contractions.dim(operator_space) + 1
    @test_throws DistributedChannelAdmissionError contract_biprojected_two_site(
        cache, frame, link, source_basis, target_basis;
        channel_slices=2,
        channel_memory_cap_bytes=1_000_000_000,
        distributed=_FactorDistributedContext(0, impossible_ranks))
end

@testset "factorized distributed channel reduction matches unsliced" begin
    psi, H = _factor_dense_fixture(seed=2026080716)
    move_center!(psi, 1)
    cache = EnvCache(psi.topo)
    source_tensor, _ = GraftContractions.Contractions._oriented_site_tensor(
        psi, 1, 2)
    target_tensor, _ = GraftContractions.Contractions._oriented_site_tensor(
        psi, 2, 1)
    source_basis, link = left_orth(source_tensor)
    target_basis, _ = left_orth(target_tensor)
    frame = oriented_two_site_factor_frame(
        cache, psi, H, 1, 2; source_tensor=source_basis)
    reference = contract_biprojected_two_site(
        cache, frame, link, source_basis, target_basis)
    remote = _factor_remote_rank_slices(
        cache, frame, link, source_basis, target_basis)
    distributed = contract_biprojected_two_site(
        cache, frame, link, source_basis, target_basis;
        channel_slices=2,
        channel_memory_cap_bytes=1_000_000_000,
        distributed=_ReplayFactorContext(remote))

    @test norm(distributed - reference) <=
        1e-12 * max(norm(reference), 1.0)
end

@testset "single-channel distributed request stays factorized and unsliced" begin
    psi, H = _factor_single_channel_fixture()
    move_center!(psi, 1)
    cache = EnvCache(psi.topo)
    source_tensor, _ = GraftContractions.Contractions._oriented_site_tensor(
        psi, 1, 2)
    target_tensor, _ = GraftContractions.Contractions._oriented_site_tensor(
        psi, 2, 1)
    source_basis, link = left_orth(source_tensor)
    target_basis, _ = left_orth(target_tensor)
    frame = oriented_two_site_factor_frame(
        cache, psi, H, 1, 2; source_tensor=source_basis)
    contractions = GraftContractions.Contractions
    projected = source_basis' * frame.source_action
    operator_space = contractions.space(
        projected, contractions.numind(projected))
    @test contractions.dim(operator_space) == 1

    reference = contract_biprojected_two_site(
        cache, frame, link, source_basis, target_basis)
    distributed = contract_biprojected_two_site(
        cache, frame, link, source_basis, target_basis;
        distributed=_FactorDistributedContext(0, 2))
    @test norm(distributed - reference) <=
        1e-12 * max(norm(reference), 1.0)
end
