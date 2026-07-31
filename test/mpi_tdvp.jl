using Test
using Graft
using MPI
using Graft.Backend: ℂ, AbstractTensorMap
using GraftTestUtils: random_ttns, to_dense
using LinearAlgebra: BLAS, norm
using Random: MersenneTwister

struct RankFailingMap{C}
    context::C
end

Graft.Contractions.Planning.workspace_map(map::RankFailingMap) = map

function (map::RankFailingMap)(x::AbstractTensorMap)
    distributed_rank(map.context) == 1 &&
        error("intentional rank-local matvec failure")
    return copy(x)
end

function tdvp_kwargs(context, size)
    return (;
        order=2,
        krylovdim=8,
        tol=1e-12,
        verbose=false,
        distributed=context,
        channel_slices=size,
        channel_minbatch=1,
        channel_min_flops=0,
        channel_memory_cap_bytes=64_000_000)
end

function compare_serial_and_distributed!(
        context, size, seed, operator, serial_evolver, distributed_evolver)
    serial_state = copy(seed)
    distributed_state = copy(seed)
    dz = -0.01im

    step!(serial_evolver, serial_state, operator, dz)
    step!(distributed_evolver, distributed_state, operator, dz)

    serial_dense = to_dense(serial_state)
    distributed_dense = to_dense(distributed_state)
    scale = max(norm(serial_dense), 1.0)
    local_error = norm(distributed_dense - serial_dense) / scale
    rank_errors = distributed_allgather(context, local_error)
    rank_states = distributed_allgather(context, distributed_dense)
    rank_state_error = maximum(
        norm(state - first(rank_states)) / scale for state in rank_states)

    @test maximum(rank_errors) <= 5e-10
    @test rank_state_error <= 5e-12
    @test length(rank_states) == size
    return nothing
end

function main()
    context = mpi_context()
    rank = distributed_rank(context)
    size = distributed_size(context)

    @test size == parse(Int, get(ENV, "GRAFT_MPI_EXPECTED_SIZE", "6"))
    @test Threads.nthreads() ==
          parse(Int, get(ENV, "GRAFT_MPI_EXPECTED_THREADS", "4"))
    @test BLAS.get_num_threads() == 1

    spin = spin_ops()
    topo = star_topology(3, 1; center=:spin, prefix=:bath)
    phys = Dict(nodeid(topo, node) => spin.P for node in 1:nnodes(topo))
    terms = OpSum()
    for (child, parent) in Graft.Trees.edges(topo)
        terms += Term(
            0.19 + 0.01child,
            SiteOp(nodeid(topo, child), :Z, spin.Z),
            SiteOp(nodeid(topo, parent), :Z, spin.Z))
        terms += Term(
            -0.11,
            SiteOp(nodeid(topo, child), :X, spin.X),
            SiteOp(nodeid(topo, parent), :X, spin.X))
    end
    for node in 1:nnodes(topo)
        terms += Term(
            0.23 + 0.02node,
            SiteOp(nodeid(topo, node), :X, spin.X))
        terms += Term(
            -0.07,
            SiteOp(nodeid(topo, node), :Z, spin.Z))
    end
    operator = ttno_from_opsum(terms, topo, phys; hermitian=true)
    seed = random_ttns(
        MersenneTwister(260728), ComplexF64, topo, phys, ℂ^3)

    @testset "multi-site MPI TDVP1 matches serial" begin
        compare_serial_and_distributed!(
            context, size, seed, operator,
            TDVP1(; order=2, krylovdim=8, tol=1e-12, verbose=false),
            TDVP1(; tdvp_kwargs(context, size)...))
    end

    @testset "multi-site MPI TDVP2 matches serial" begin
        trunc = TruncationScheme(; maxdim=8, atol=1e-12)
        compare_serial_and_distributed!(
            context, size, seed, operator,
            TDVP2(;
                order=2,
                trunc,
                krylovdim=8,
                tol=1e-12,
                verbose=false),
            TDVP2(;
                trunc,
                tdvp_kwargs(context, size)...))
    end

    @testset "multi-site MPI TDVP1-CBE matches serial" begin
        trunc = TruncationScheme(; maxdim=8, atol=1e-12)
        compare_serial_and_distributed!(
            context, size, seed, operator,
            TDVP1_CBE(;
                order=2,
                trunc,
                d_tilde_max=2,
                enr_rtol=1e-12,
                enr_atol=1e-12,
                krylovdim=8,
                tol=1e-12,
                verbose=false),
            TDVP1_CBE(;
                trunc,
                d_tilde_max=2,
                enr_rtol=1e-12,
                enr_atol=1e-12,
                tdvp_kwargs(context, size)...))
    end

    @testset "rank-local Krylov failure terminates collectively" begin
        message = try
            distributed_exponentiate(
                context,
                RankFailingMap(context),
                seed.tensors[topo.root],
                -0.01im;
                ishermitian=true,
                krylovdim=4)
            ""
        catch err
            sprint(showerror, err)
        end
        rank_messages = distributed_allgather(context, message)
        @test all(
            occursin("intentional rank-local matvec failure", item)
            for item in rank_messages)
    end

    if rank == distributed_root(context)
        println(
            "Graft multi-site MPI TDVP validation passed: ranks=$size, ",
            "threads=$(Threads.nthreads())")
    end
    return nothing
end

main()
