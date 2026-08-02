using Test
using GraftFoundation: TensorMap, TruncationScheme, dim, mps_topology,
    nnodes, nchildren, postorder, star_topology, ℂ, ⊗, ←
using GraftNetworks: TTNS, center, check_arrows, exact_linear_combination,
    global_subspace_expand!, topology, virtualspace

function _gse_product_state(topo, occupations)
    unit = ℂ^1
    tensors = map(1:nnodes(topo)) do node
        children = nchildren(topo, node)
        child_space = children == 0 ? one(unit) :
            reduce(⊗, ntuple(_ -> unit, children))
        local_state = ComplexF64[
            occupations[node] == 1,
            occupations[node] == 2,
        ]
        array = reshape(
            local_state,
            ntuple(_ -> 1, children)...,
            2,
            1,
        )
        TensorMap(array, child_space ⊗ ℂ^2 ← unit)
    end
    return TTNS(topo, tensors, topo.root)
end

function _gse_correlated_ancillary(topo)
    first_state = _gse_product_state(topo, fill(1, nnodes(topo)))
    second_state = _gse_product_state(topo, fill(2, nnodes(topo)))
    return exact_linear_combination(
        [first_state, second_state], ComplexF64[1, 1])
end

@testset "synchronous global subspace expansion" begin
    topology = mps_topology(3)
    input = _gse_product_state(topology, fill(1, nnodes(topology)))

    for reverse in (false, true)
        state = copy(input)
        ancillary = _gse_correlated_ancillary(topology)
        ancillary_before = copy(ancillary)
        _, report = global_subspace_expand!(
            state,
            [ancillary];
            trunc=TruncationScheme(maxdim=4),
            max_add=2,
            reverse,
        )

        @test length(report.edges) == nnodes(topology) - 1
        @test all(edge.rank_before == 1 for edge in report.edges)
        @test all(edge.rank_after == 2 for edge in report.edges)
        @test all(edge.rank_added == 1 for edge in report.edges)
        @test report.state_embedding_error < 1e-13
        @test report.ancillary_projection_error < 1e-13
        @test all(dim(virtualspace(state, child)) == 2
                  for child in 1:nnodes(topology)
                  if topology.parent[child] != 0)
        @test all(norm(ancillary[node] - ancillary_before[node]) == 0
                  for node in 1:nnodes(topology))
        @test center(state) == (reverse ? first(postorder(topology)) : topology.root)
        @test check_arrows(state)
        @test check_arrows(ancillary)
    end
end

@testset "branching common-basis installation" begin
    topology = star_topology(3, 1)
    state = _gse_product_state(topology, fill(1, nnodes(topology)))
    ancillary = _gse_correlated_ancillary(topology)
    ancillary_before = copy(ancillary)
    _, report = global_subspace_expand!(
        state,
        [ancillary];
        trunc=TruncationScheme(maxdim=3),
        max_add=1,
    )

    @test length(report.edges) == 3
    @test all(edge.rank_added == 1 for edge in report.edges)
    @test all(dim(virtualspace(state, child)) == 2 for child in 2:4)
    @test all(norm(ancillary[node] - ancillary_before[node]) == 0
              for node in 1:nnodes(topology))
    @test report.state_embedding_error < 1e-13
    @test check_arrows(state)
    @test check_arrows(ancillary)
end

@testset "global subspace expansion guards" begin
    topology = mps_topology(2)
    state = _gse_product_state(topology, fill(1, nnodes(topology)))
    ancillary = _gse_correlated_ancillary(topology)
    @test_throws ArgumentError global_subspace_expand!(
        copy(state), TTNS[])
    @test_throws ArgumentError global_subspace_expand!(
        copy(state), [copy(ancillary)]; max_add=-1)
    @test_throws ArgumentError global_subspace_expand!(
        copy(state), [copy(ancillary)];
        trunc=TruncationScheme(maxdim=0))
end
