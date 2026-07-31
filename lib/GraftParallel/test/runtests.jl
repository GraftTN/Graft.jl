using GraftParallel
using Test

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
