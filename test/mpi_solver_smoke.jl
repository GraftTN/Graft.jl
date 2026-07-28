using Test
using Graft
using MPI
using Graft.Backend: ℂ, norm
using Graft.TestUtils: random_ttns
using LinearAlgebra: BLAS
using Random: MersenneTwister

const eigsolve = Graft.GroundState.eigsolve
const exponentiate = Graft.Evolution.exponentiate

function main()
context = mpi_context()
rank = distributed_rank(context)
size = distributed_size(context)

@test size == parse(Int, get(ENV, "GRAFT_MPI_EXPECTED_SIZE", "6"))
@test Threads.nthreads() ==
      parse(Int, get(ENV, "GRAFT_MPI_EXPECTED_THREADS", "4"))
@test BLAS.get_num_threads() == 1

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

reference_value = if rank == distributed_root(context)
    reference_values, _, _ = eigsolve(
        serial_map, psi.tensors[root], 1, :SR;
        ishermitian=true, krylovdim=6)
    first(reference_values)
else
    0.0
end
reference_buffer = [reference_value]
distributed_broadcast!(context, reference_buffer)
reference_value = only(reference_buffer)

values, _, info = distributed_eigsolve(
    context, distributed_map, psi.tensors[root], 1, :SR;
    ishermitian=true, krylovdim=6)

rank_values = distributed_allgather(context, first(values))
@testset "distributed adaptive Krylov solver" begin
    @test info.converged >= 1
    @test maximum(abs.(rank_values .- first(rank_values))) == 0
    @test first(values) ≈ reference_value atol=1e-10
end

serial_evolved = if rank == distributed_root(context)
    first(exponentiate(
        serial_map, -0.01im, psi.tensors[root];
        ishermitian=true, krylovdim=6))
else
    nothing
end
evolved, _ = distributed_exponentiate(
    context, distributed_map, psi.tensors[root], -0.01im;
    ishermitian=true, krylovdim=6)
evolution_error = rank == distributed_root(context) ?
                  norm(evolved - serial_evolved) : 0.0
error_buffer = [evolution_error]
distributed_broadcast!(context, error_buffer)
rank_norms = distributed_allgather(context, norm(evolved))
@testset "distributed adaptive exponential action" begin
    @test only(error_buffer) <= 1e-10 * max(norm(evolved), 1.0)
    @test maximum(rank_norms) - minimum(rank_norms) == 0
end

single_topo = mps_topology(1)
single_phys = Dict(:site1 => spin.P)
single_terms = OpSum() +
    Term(0.6, SiteOp(:site1, :Z, spin.Z)) +
    Term(0.2, SiteOp(:site1, :X, spin.X))
single_operator = ttno_from_opsum(
    single_terms, single_topo, single_phys; hermitian=true)
single_state = random_ttns(
    MersenneTwister(260730), ComplexF64,
    single_topo, single_phys, ℂ^1)
single_state, energies = dmrg1!(
    single_state, single_operator;
    nsweeps=1,
    krylovdim=4,
    verbose=false,
    distributed=context,
    channel_memory_cap_bytes=1_000_000)
rank_energies = distributed_allgather(context, last(energies))
@testset "distributed public solver integration" begin
    @test maximum(rank_energies) - minimum(rank_energies) == 0
end

tdvp = TDVP1(;
    order=1,
    krylovdim=4,
    verbose=false,
    distributed=context,
    channel_memory_cap_bytes=1_000_000)
step!(tdvp, single_state, single_operator, -0.01im)
rank_expectations = distributed_allgather(
    context, expect(single_state, single_operator))
@testset "distributed public evolver integration" begin
    @test maximum(abs.(rank_expectations .- first(rank_expectations))) == 0
end

if rank == distributed_root(context)
    println(
        "Graft MPI solver smoke passed: ranks=$size, ",
        "threads=$(Threads.nthreads())")
end
end

main()
