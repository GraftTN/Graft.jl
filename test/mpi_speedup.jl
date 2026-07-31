using Test
using Graft
using MPI
using Graft.Backend: ℂ
using GraftTestUtils: random_ttns
using LinearAlgebra: BLAS
using Random: MersenneTwister
using Statistics: median

context = mpi_context()
rank = distributed_rank(context)
size = distributed_size(context)
root = distributed_root(context)

size == parse(Int, get(ENV, "GRAFT_MPI_EXPECTED_SIZE", "6")) ||
    error("MPI speedup test must run with the requested rank count")
Threads.nthreads() ==
    parse(Int, get(ENV, "GRAFT_MPI_EXPECTED_THREADS", "4")) ||
    error("MPI speedup test must run with the requested thread count")
BLAS.get_num_threads() == 1 ||
    error("MPI speedup test requires one BLAS thread per rank")

function benchmark_fixture()
    topo = star_topology(3, 2)
    spin = spin_ops()
    phys = Dict(nodeid(topo, i) => spin.P for i in 1:nnodes(topo))
    terms = OpSum()
    for (child, parent) in Graft.Trees.edges(topo)
        terms += Term(
            0.19, SiteOp(nodeid(topo, child), :Z, spin.Z),
            SiteOp(nodeid(topo, parent), :Z, spin.Z))
    end
    for node in 1:nnodes(topo)
        terms += Term(
            0.37 + 0.01node,
            SiteOp(nodeid(topo, node), :X, spin.X))
    end
    operator = ttno_from_opsum(terms, topo, phys; hermitian=true)
    snapshot_count = parse(Int, get(ENV, "GRAFT_MPI_BENCH_SNAPSHOTS", "12"))
    bond_dimension = parse(Int, get(ENV, "GRAFT_MPI_BENCH_BOND_DIM", "6"))
    snapshots = [
        random_ttns(
            MersenneTwister(260728 + i), ComplexF64, topo, phys,
            ℂ^bond_dimension)
        for i in 1:snapshot_count
    ]
    foreach(normalize!, snapshots)
    return snapshots, operator
end

function root_serial_time(f, samples)
    times = Float64[]
    for _ in 1:samples
        distributed_barrier(context)
        if rank == root
            GC.gc()
            start = time_ns()
            f()
            push!(times, (time_ns() - start) / 1e9)
        end
        distributed_barrier(context)
    end
    value = rank == root ? median(times) : 0.0
    buffer = [value]
    distributed_broadcast!(context, buffer)
    return only(buffer)
end

function distributed_time(f, samples)
    times = Float64[]
    for _ in 1:samples
        GC.gc()
        distributed_barrier(context)
        start = time_ns()
        f()
        elapsed = [(time_ns() - start) / 1e9]
        MPI.Allreduce!(elapsed, MPI.MAX, MPI.COMM_WORLD)
        rank == root && push!(times, only(elapsed))
    end
    value = rank == root ? median(times) : 0.0
    buffer = [value]
    distributed_broadcast!(context, buffer)
    return only(buffer)
end

snapshots, operator = benchmark_fixture()
serial_run() = complex_time_krylov(snapshots, operator)
mpi_run() = complex_time_krylov(snapshots, operator; distributed=context)

# Compile both paths before recording wall time.
if rank == root
    serial_run()
end
distributed_barrier(context)
mpi_run()
distributed_barrier(context)

samples = parse(Int, get(ENV, "GRAFT_MPI_BENCH_SAMPLES", "3"))
serial_seconds = root_serial_time(serial_run, samples)
mpi_seconds = distributed_time(mpi_run, samples)
speedup = serial_seconds / mpi_seconds
minimum_speedup = parse(
    Float64, get(ENV, "GRAFT_MPI_MIN_SPEEDUP", "1.05"))

@testset "MPI Krylov speedup" begin
    @test isfinite(serial_seconds) && serial_seconds > 0
    @test isfinite(mpi_seconds) && mpi_seconds > 0
    @test speedup >= minimum_speedup
end

if rank == root
    println(
        "Graft MPI speedup: serial=$(round(serial_seconds; digits=4)) s, ",
        "mpi=$(round(mpi_seconds; digits=4)) s, ",
        "speedup=$(round(speedup; digits=3))x ",
        "(ranks=$size, threads/rank=$(Threads.nthreads()))")
end
