using GraftParallel
using Test

@testset "GraftParallel process-startup runtime policy" begin
    configured = configure_parallel_runtime!()
    @test configured.blas_threads == 1
    @test configured.strided_threads == 1
    @test configured.transformer_threads == 1
    @test configured.configured
    @test configured.active_regions == 0
    @test parallel_runtime_config() == configured

    repeated = configure_parallel_runtime!(
        ; blas_threads=1, strided_threads=1, transformer_threads=1)
    @test repeated.generation == configured.generation

    before = parallel_runtime_config()
    @test_throws ArgumentError configure_parallel_runtime!(
        ; transformer_threads=Base.Threads.nthreads() + 1)
    @test parallel_runtime_config() == before
    @test GraftParallel.Parallel._validated_runtime_thread_count(
        :blas_threads, Base.Threads.nthreads() + 1) ==
          Base.Threads.nthreads() + 1
    @test GraftParallel.Parallel._validated_runtime_thread_count(
        :strided_threads, Base.Threads.nthreads() + 1) ==
          Base.Threads.nthreads() + 1
    @test_throws ArgumentError configure_parallel_runtime!(; blas_threads=0)
    @test_throws ArgumentError configure_parallel_runtime!(; strided_threads=0)
    @test_throws ArgumentError configure_parallel_runtime!(
        ; transformer_threads=0)

    state = Dict(:blas => 1, :strided => 1, :transformer => 1)
    calls = Pair{Symbol,Int}[]
    setters = (;
        blas=value -> begin
            push!(calls, :blas => value)
            state[:blas] = value
        end,
        strided=value -> begin
            push!(calls, :strided => value)
            state[:strided] = value
        end,
        transformer=value -> begin
            push!(calls, :transformer => value)
            value == 2 && error("injected transformer failure")
            state[:transformer] = value
        end,
    )
    current = (;
        blas_threads=1, strided_threads=1, transformer_threads=1)
    requested = (;
        blas_threads=2, strided_threads=2, transformer_threads=2)
    @test_throws ErrorException GraftParallel.Parallel._apply_runtime_thread_counts!(
        current, requested, setters)
    @test state == Dict(:blas => 1, :strided => 1, :transformer => 1)
    @test calls == [
        :blas => 2,
        :strided => 2,
        :transformer => 2,
        :transformer => 1,
        :strided => 1,
        :blas => 1,
    ]

    if Base.Threads.nthreads() > 1
        GraftParallel.Parallel._with_parallel_runtime_region() do
            active = parallel_runtime_config()
            @test active.active_regions == 1
            failure = try
                configure_parallel_runtime!(
                    ; blas_threads=active.blas_threads,
                    strided_threads=active.strided_threads,
                    transformer_threads=2)
                nothing
            catch error
                error
            end
            @test failure isa ParallelRuntimeConfigurationError
            if failure isa ParallelRuntimeConfigurationError
                @test failure.current.transformer_threads == 1
                @test failure.requested.transformer_threads == 2
            end
            @test parallel_runtime_config().transformer_threads == 1
        end
        @test parallel_runtime_config().active_regions == 0
    end
end

@testset "GraftParallel threaded fan-out" begin
    serial = zeros(Int, 8)
    threaded_foreach(1:8; threaded=false) do item
        serial[item] = item^2
    end
    @test serial == [item^2 for item in 1:8]

    threaded = zeros(Int, 8)
    threaded_foreach((item for item in 1:8); threaded=true, minbatch=1) do item
        threaded[item] = 2item
    end
    @test threaded == [2item for item in 1:8]
    @test_throws ArgumentError threaded_foreach(identity, [1]; minbatch=0)
end

@testset "GraftParallel bounded admission" begin
    completed = zeros(Int, 8)
    diagnostics = bounded_threaded_foreach(
        1:8;
        threaded=true,
        minbatch=1,
        retained_memory_bytes=10,
        item_memory_bytes=20,
        memory_cap_bytes=50,
    ) do item
        completed[item] = item
    end

    @test completed == collect(1:8)
    @test diagnostics.completed_items == 8
    @test diagnostics.cancelled_items == 0
    @test diagnostics.peak_admitted_bytes <= diagnostics.memory_cap_bytes
    @test diagnostics.worker_limit <= Base.Threads.nthreads()
    @test diagnostics.mode ==
          (Base.Threads.nthreads() > 1 ? :threaded : :serial)

    admitted = Int[]
    rejection = try
        bounded_threaded_foreach(
            1:3;
            threaded=true,
            minbatch=1,
            retained_memory_bytes=10,
            item_memory_bytes=20,
            memory_cap_bytes=29,
        ) do item
            push!(admitted, item)
        end
        nothing
    catch error
        error
    end

    @test rejection isa BoundedFanoutAdmissionError
    @test isempty(admitted)
    @test rejection.item_index == 1
    @test rejection.item_id == 1
    @test occursin("admission rejected item 1", sprint(showerror, rejection))
end

@testset "GraftParallel failure cancellation" begin
    worker_count = Base.Threads.nthreads()
    item_count = max(2worker_count, 4)
    attempted = zeros(UInt8, item_count)

    failure = try
        bounded_threaded_foreach(
            1:item_count;
            threaded=true,
            minbatch=1,
            item_memory_bytes=1,
            memory_cap_bytes=worker_count,
            item_id=(index, item) -> Symbol(:item_, item),
        ) do item
            attempted[item] = 0x01
            item == 2 && error("injected item 2")
        end
        nothing
    catch error
        error
    end

    @test failure isa BoundedFanoutItemError
    @test failure.item_index == 2
    @test failure.item_id == :item_2
    @test occursin("injected item 2", sprint(showerror, failure))

    expected_attempted = worker_count == 1 ? 2 : worker_count
    @test failure.attempted_items == expected_attempted
    @test failure.cancelled_items == item_count - expected_attempted
    @test count(!iszero, attempted) == expected_attempted
    @test all(iszero, @view attempted[(expected_attempted + 1):end])
end

@testset "GraftParallel owner-local load graph" begin
    loaded = Set(nameof(module_) for module_ in values(Base.loaded_modules))
    @test :GraftParallel in loaded
    @test :MPI ∉ loaded
    @test :Graft ∉ loaded
    @test :GraftTestUtils ∉ loaded
end
