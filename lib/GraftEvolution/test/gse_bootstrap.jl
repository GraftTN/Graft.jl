using Test
using GraftEvolution: GSEInfo, GlobalSubspaceExpansionInfo,
    GlobalSubspaceEdgeInfo, TDVP1_GSE, gse_enrich!, step!
using GraftFoundation: TruncationScheme, mps_topology, ℂ
using GraftNetworks: center
using GraftSymbolic: OpSum, SiteOp, Term, spin_ops
using GraftTestUtils: dense_hamiltonian, product_ttns, to_dense
using GraftTTNOBuild: ttno_from_opsum
using LinearAlgebra: norm

function _gse_fixture(::Type{T}=ComplexF64) where {T<:Number}
    topology = mps_topology(3)
    spin = spin_ops()
    spaces = Dict(Symbol(:site, site) => spin.P for site in 1:3)
    hamiltonian = OpSum() +
        Term(0.7,
             SiteOp(:site1, :X12, spin.X),
             SiteOp(:site2, :X21, spin.X)) +
        Term(-0.4,
             SiteOp(:site2, :X23, spin.X),
             SiteOp(:site3, :X32, spin.X)) +
        Term(0.2, SiteOp(:site2, :Z2, spin.Z))
    operator = ttno_from_opsum(
        hamiltonian, topology, spaces; hermitian=true)
    state = product_ttns(
        T,
        topology,
        Dict(Symbol(:site, site) => T[1, 0] for site in 1:3),
    )
    return state, operator, hamiltonian
end

function _paper_gse(; kwargs...)
    defaults = (;
        ancillary_order=2,
        ancillary_shift=0.17,
        ancillary_trunc=TruncationScheme(maxdim=16),
        trunc=TruncationScheme(maxdim=8),
        max_add=4,
        enrichment_atol=0.0,
        enrichment_rtol=1e-12,
        max_exact_bond=128,
        max_exact_payload=100_000,
        order=1,
        krylovdim=12,
        tol=1e-12,
        verbose=false,
    )
    return TDVP1_GSE(; merge(defaults, (; kwargs...))...)
end

@testset "Yang--White ancillary enrichment" begin
    state, operator, _ = _gse_fixture()
    initial = to_dense(state)
    initial_center = center(state)
    evolver = _paper_gse()

    returned, info = gse_enrich!(evolver, state, operator)

    @test returned === state
    @test info === evolver.last_info
    @test info isa GSEInfo
    @test info.ancillary_shift == 0.17
    @test info.action_count == 2
    @test length(info.ancillary_truncation_errors) == 2
    @test info.ancillary_generation_seconds >= 0
    @test info.enrichment_seconds >= 0
    @test info.propagation_seconds == 0
    @test info.expansion isa GlobalSubspaceExpansionInfo
    @test all(edge isa GlobalSubspaceEdgeInfo for edge in info.expansion.edges)
    @test length(info.expansion.edges) == 2
    @test any(edge.rank_added > 0 for edge in info.expansion.edges)
    @test info.expansion.state_embedding_error < 1e-12
    @test norm(to_dense(state) - initial) < 1e-12
    @test info.enriched_bond_dimensions == info.final_bond_dimensions
    @test center(state) != initial_center || initial_center == state.topo.root
end

@testset "caller-owned ancillary shift is independent of dz" begin
    first_state, operator, _ = _gse_fixture()
    second_state = copy(first_state)
    first = _paper_gse(ancillary_shift=0.031)
    second = _paper_gse(ancillary_shift=0.031)

    _, first_info = gse_enrich!(first, first_state, operator)
    _, second_info = gse_enrich!(second, second_state, operator)

    @test first_info.ancillary_shift == 0.031
    @test second_info.ancillary_shift == 0.031
    @test first_info.enriched_bond_dimensions ==
        second_info.enriched_bond_dimensions
    @test [edge.rank_added for edge in first_info.expansion.edges] ==
        [edge.rank_added for edge in second_info.expansion.edges]
end

@testset "GSE plus TDVP1 and typed diagnostics" begin
    state, operator, hamiltonian = _gse_fixture()
    initial = to_dense(state)
    dense_operator = dense_hamiltonian(hamiltonian, state)
    step_size = -0.01im
    reference = exp(step_size * dense_operator) * initial
    evolver = _paper_gse()

    step!(evolver, state, operator, step_size)

    @test evolver.last_info isa GSEInfo
    @test evolver.last_info.propagation_seconds >= 0
    @test norm(to_dense(state) - reference) < 1e-9
    @test evolver.last_info.final_bond_dimensions ==
        [2, 2]
end

struct _GSEBadStep <: Number end

@testset "TDVP1_GSE step is transactional" begin
    state, operator, _ = _gse_fixture()
    before = to_dense(state)
    before_center = center(state)
    evolver = _paper_gse()

    @test_throws Exception step!(evolver, state, operator, _GSEBadStep())
    @test norm(to_dense(state) - before) == 0
    @test center(state) == before_center
    @test evolver.last_info === nothing
    @test evolver.cache === nothing
end

@testset "ancillary resource guards fail before state enrichment" begin
    state, operator, _ = _gse_fixture()
    before = to_dense(state)
    evolver = _paper_gse(max_exact_bond=1)

    @test_throws ArgumentError gse_enrich!(evolver, state, operator)
    @test norm(to_dense(state) - before) == 0
    @test evolver.last_info === nothing
end
