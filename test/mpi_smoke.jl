using Test
using Graft
using MPI
using LinearAlgebra: BLAS
using Graft.Backend: ℂ, ←, blocks
using Graft.TestUtils: random_ttns
using LinearAlgebra: norm
using Random: MersenneTwister, Xoshiro, rand

context = mpi_context()
rank = distributed_rank(context)
size = distributed_size(context)

struct MPIIdentityEvolver <: Evolver end

@testset "MPI runtime and collectives" begin
    @test size == parse(Int, get(ENV, "GRAFT_MPI_EXPECTED_SIZE", "6"))
    @test Threads.nthreads() ==
          parse(Int, get(ENV, "GRAFT_MPI_EXPECTED_THREADS", "4"))
    @test 0 <= rank < size
    @test context.threadlevel >= MPI.THREAD_FUNNELED
    @test BLAS.get_num_threads() == 1

    values = ComplexF64[rank + 1, (rank + 1)im]
    @test distributed_allreduce_sum!(context, values) === values
    expected_sum = size * (size + 1) / 2
    @test values ≈ ComplexF64[expected_sum, expected_sum * im]

    root_values = rank == distributed_root(context) ?
                  [pi, exp(1.0)] : zeros(2)
    @test distributed_broadcast!(context, root_values) === root_values
    @test root_values ≈ [pi, exp(1.0)]

    tensor = zeros(ComplexF64, ℂ^2 ← ℂ^2)
    for (_, data) in blocks(tensor)
        fill!(data, rank + 1)
    end
    @test distributed_allreduce_sum!(context, tensor) === tensor
    for (_, data) in blocks(tensor)
        @test all(≈(expected_sum), data)
    end

    distributed_barrier(context)
end

@testset "sample-parallel METTS" begin
    spin = spin_ops()
    topo = mps_topology(1)
    phys = Dict(:site1 => spin.P)
    terms = OpSum() + Term(
        0.5, SiteOp(:site1, :Z, spin.Z))
    problem = purification_problem(terms, topo, phys; hermitian=true)
    observable = physical_ttno(
        problem, terms; hermitian=true, doubled=false)
    trajectory = thermalize(
        METTS(;
            rng=Xoshiro(260728),
            collapse_basis=:computational,
            burnin=0,
            nsamples=2size,
            thin=1),
        problem, 0.0;
        evolver=MPIIdentityEvolver(),
        nsteps=1,
        distributed=context)

    @test trajectory isa DistributedMETTSTrajectory
    @test trajectory.global_nsamples == 2size
    @test trajectory.samples_per_rank == fill(2, size)
    @test length(trajectory.local_chain.samples) == 2
    stats = metts_statistics(trajectory, observable; maxlag=0)
    @test stats.nsamples == 2size
    @test thermal_expect(trajectory, observable) ≈ stats.mean

    rng_probe = rand(copy(trajectory.local_chain.rng), UInt64)
    rank_rng_probes = distributed_allgather(context, rng_probe)
    @test length(unique(rank_rng_probes)) == size

    checkpoint_path = MPI.bcast(
        rank == distributed_root(context) ? tempname() : "",
        distributed_root(context), MPI.COMM_WORLD)
    checkpoint!(
        trajectory, checkpoint_path;
        metadata=(; purpose=:mpi_smoke))
    restored = resume_mpi(checkpoint_path, context)
    @test restored.metadata.purpose == :mpi_smoke
    @test restored.state.global_nsamples == trajectory.global_nsamples
    @test restored.state.samples_per_rank == trajectory.samples_per_rank
    @test [
        sample.outcomes for sample in restored.state.local_chain.samples
    ] == [
        sample.outcomes for sample in trajectory.local_chain.samples
    ]
end

@testset "distributed effective operator channels" begin
    spin = spin_ops()
    topo = star_topology(3, 1)
    phys = Dict(nodeid(topo, i) => spin.P for i in 1:nnodes(topo))
    terms = OpSum()
    for (child, parent) in Graft.Trees.edges(topo)
        terms += Term(
            0.17, SiteOp(nodeid(topo, child), :Z, spin.Z),
            SiteOp(nodeid(topo, parent), :Z, spin.Z))
    end
    for node in 1:nnodes(topo)
        terms += Term(
            0.31 + 0.01node,
            SiteOp(nodeid(topo, node), :X, spin.X))
    end
    operator = ttno_from_opsum(terms, topo, phys; hermitian=true)
    psi = random_ttns(
        MersenneTwister(260728), ComplexF64, topo, phys, ℂ^3)
    root = topo.root

    serial_map = eff_h1(EnvCache(topo), psi, operator, root)
    distributed_map = eff_h1(
        EnvCache(topo), psi, operator, root;
        distributed=context,
        channel_slices=size,
        channel_minbatch=1,
        channel_min_flops=0,
        channel_memory_cap_bytes=1_000_000_000)
    @test distributed_map isa DistributedChannelEffectiveMap
    @test all(i -> 1 <= i <= length(distributed_map.effective.maps),
              distributed_map.local_indices)
    reference = serial_map(psi.tensors[root])
    actual = distributed_map(psi.tensors[root])
    @test norm(actual - reference) <=
          5e-12 * max(norm(reference), 1.0)
end

@testset "distributed snapshot Gram assembly" begin
    spin = spin_ops()
    topo = mps_topology(1)
    phys = Dict(:site1 => spin.P)
    operator = ttno_from_opsum(
        OpSum() + Term(1.0, SiteOp(:site1, :Z, spin.Z)),
        topo, phys; hermitian=true)
    psi = random_ttns(Xoshiro(26060729), ComplexF64, topo, phys, ℂ^1)
    snapshots = [
        psi,
        apply_local(psi, spin.X, :site1),
        apply_local(psi, spin.Z, :site1),
    ]
    serial = complex_time_krylov(snapshots, operator)
    parallel = complex_time_krylov(
        snapshots, operator; distributed=context)
    @test parallel.overlap ≈ serial.overlap atol=1e-12
    @test parallel.hamiltonian ≈ serial.hamiltonian atol=1e-12
    @test parallel.values ≈ serial.values atol=1e-12
end

if rank == distributed_root(context)
    println("Graft MPI smoke passed: ranks=$size, threads=$(Threads.nthreads())")
end
