using GraftFoundation
using Test

@testset "GraftFoundation TensorKit transformer adapter" begin
    transformer_threads =
        GraftFoundation.Backend.tensor_transformer_threads
    set_transformer_threads! =
        GraftFoundation.Backend.set_tensor_transformer_threads!
    original = transformer_threads()
    requested = Base.Threads.nthreads() > 1 && original == 1 ? 2 : 1
    try
        @test set_transformer_threads!(requested) == requested
        @test transformer_threads() == requested
        @test_throws ArgumentError set_transformer_threads!(0)
        @test_throws ArgumentError set_transformer_threads!(
            Base.Threads.nthreads() + 1)
        @test transformer_threads() == requested
    finally
        set_transformer_threads!(original)
    end
end

@testset "GraftFoundation topology and T3NS geometry" begin
    star = star_topology(3, 2)

    @test nnodes(star) == 7
    @test nodeid(star, star.root) == :center
    @test isleaf(star, nodeindex(star, :b1_2))
    @test postorder(star)[end] == star.root
    @test preorder(star)[1] == star.root
    @test allunique(postorder(star))
    @test Set(edges(star)) == Set([
        (nodeindex(star, :b1_1), star.root),
        (nodeindex(star, :b1_2), nodeindex(star, :b1_1)),
        (nodeindex(star, :b2_1), star.root),
        (nodeindex(star, :b2_2), nodeindex(star, :b2_1)),
        (nodeindex(star, :b3_1), star.root),
        (nodeindex(star, :b3_2), nodeindex(star, :b3_1)),
    ])

    path = path_between(
        star,
        nodeindex(star, :b1_2),
        nodeindex(star, :b3_2),
    )
    @test nodeid.(Ref(star), path) == [:b1_2, :b1_1, :center, :b3_1, :b3_2]
    @test Set(subtree_nodes(
        star,
        nodeindex(star, :b1_1),
        star.root,
    )) == Set([nodeindex(star, :b1_1), nodeindex(star, :b1_2)])

    copy_star = star_topology(3, 2)
    @test star == copy_star
    @test hash(star) == hash(copy_star)
    @test binary_topology(2) != star

    @test is_t3ns(star)
    @test is_t3ns(star; physical=[:b1_1, :b1_2, :b2_1, :b2_2, :b3_1, :b3_2])
    @test !is_t3ns(star; physical=[:center])
    @test !is_t3ns(star; physical=[:b1_1, :b1_1])
    @test !is_t3ns(star; physical=[:missing])

    chain = mps_topology(5)
    @test is_t3ns(chain; physical=[Symbol(:site, i) for i in 1:5])

    fork = fork_topology(4, 2)
    @test !is_t3ns(fork; physical=[Symbol(:spine, i) for i in 1:4])
    @test_throws ArgumentError fork_topology(0, 1)
    @test_throws ArgumentError fork_topology(1, -1)

    t3ns = TreeTopology(:branch, [
        :branch => :left,
        :branch => :right,
        :left => :left_bath,
        :right => :right_bath,
    ])
    @test is_t3ns(t3ns; physical=[:left, :right, :left_bath, :right_bath])

    mounted = mount_chain(t3ns, :left_bath, 2; prefix=:boson)
    @test nnodes(mounted) == nnodes(t3ns) + 2
    @test nnodes(t3ns) == 5
    @test nodeid(mounted, path_between(
        mounted,
        nodeindex(mounted, :boson2),
        nodeindex(mounted, :branch),
    )[1]) == :boson2

    @test_throws ArgumentError TreeTopology(:root, [:missing => :child])
    @test_throws ArgumentError TreeTopology(:root, [:root => :child, :root => :child])
end

@testset "GraftFoundation owner-local load graph" begin
    loaded = Set(nameof(module_) for module_ in values(Base.loaded_modules))
    @test :GraftFoundation in loaded
    @test :Graft ∉ loaded
    @test :GraftTestUtils ∉ loaded
end
