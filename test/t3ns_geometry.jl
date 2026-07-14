using Test
using Graft

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

@graft_testset "T3NS geometry and FTPS flavor semantics" begin
    fork = fork_topology(4, 2)
    @test !is_t3ns(fork; physical=[Symbol(:spine, i) for i in 1:4])

    t3ns = TreeTopology(:branch, [
        :branch => :left,
        :branch => :right,
        :left => :left_bath,
        :right => :right_bath,
    ])
    @test is_t3ns(t3ns; physical=[:left, :right, :left_bath, :right_bath])

    three_way = star_topology(3, 1)
    @test is_t3ns(three_way; physical=[:b1_1, :b2_1, :b3_1])
    @test !is_t3ns(three_way; physical=[:center, :b1_1, :b2_1, :b3_1])

    if GRAFT_EXTENDED_TESTS
        @test nodeid(fork, fork.root) == :spine1
        @test all(nodeid(fork, i) in fork.ids for i in 1:nnodes(fork))
        @test :tooth1_1 in fork.ids && :tooth4_2 in fork.ids
        @test_throws ArgumentError fork_topology(0, 1)
        @test_throws ArgumentError fork_topology(1, -1)
        @test !is_t3ns(t3ns; physical=[:left, :left])
        @test !is_t3ns(t3ns; physical=[:missing])
    end
end
