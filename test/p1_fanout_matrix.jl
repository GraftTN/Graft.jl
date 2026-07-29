using Test
using Graft
using Graft.Backend: ←, blocks
using Graft.Contractions: _rsvd_random_probe
using Random: Xoshiro, rand

struct P1NoOpEvolver <: Evolver end
Graft.step!(::P1NoOpEvolver, psi::TTNS, ::TTNO, ::Number) = psi

struct P1FailingEvolver <: Evolver end
Graft.step!(::P1FailingEvolver, ::TTNS, ::TTNO, ::Number) =
    error("injected thermal propagation failure")

@testset "P1 bounded fan-out ($(Base.Threads.nthreads()) Julia threads)" begin
    @testset "generic admission, cancellation, and item indexing" begin
        visited = zeros(Int, 8)
        diagnostics = bounded_threaded_foreach(
            1:8;
            threaded=true,
            minbatch=1,
            retained_memory_bytes=10,
            item_memory_bytes=20,
            memory_cap_bytes=50,
        ) do item
            visited[item] = Base.Threads.threadid()
        end
        @test all(!iszero, visited)
        @test diagnostics.peak_admitted_bytes <= 50
        @test diagnostics.mode ==
              (Base.Threads.nthreads() > 1 ? :threaded : :serial)

        started = Int[]
        admission_error = try
            bounded_threaded_foreach(
                1:4;
                threaded=true,
                minbatch=1,
                retained_memory_bytes=10,
                item_memory_bytes=20,
                memory_cap_bytes=29,
            ) do item
                push!(started, item)
            end
            nothing
        catch err
            err
        end
        @test admission_error isa BoundedFanoutAdmissionError
        @test isempty(started)

        item_error = try
            bounded_threaded_foreach(
                1:16;
                threaded=true,
                minbatch=1,
                item_memory_bytes=1,
                memory_cap_bytes=max(Base.Threads.nthreads(), 1),
                item_id=(index, item) -> (; index, item),
            ) do item
                item in (2, 3) && error("injected item $item")
            end
            nothing
        catch err
            err
        end
        @test item_error isa BoundedFanoutItemError
        @test item_error.item_index == 2
        @test item_error.item_id == (; index=2, item=2)
        @test item_error.cancelled_items > 0

        # Exercise repeated batch publication. This catches accidental capture
        # and reuse of loop-local batch state across threaded regions.
        publication_stable = true
        if Base.Threads.nthreads() > 1
            for _ in 1:10_000
                published = zeros(Int, 9)
                bounded_threaded_foreach(
                    1:9;
                    threaded=true,
                    minbatch=1,
                    item_memory_bytes=1,
                    memory_cap_bytes=Base.Threads.nthreads(),
                ) do item
                    published[item] = item
                end
                if published != collect(1:9)
                    publication_stable = false
                    break
                end
            end
        end
        @test publication_stable
    end

    @testset "thermal correlator contract" begin
        spin = spin_ops()
        topology = mps_topology(1)
        physical = Dict(:site1 => spin.P)
        generator = OpSum() + Term(0.2, SiteOp(:site1, :Z, spin.Z))
        problem = purification_problem(
            generator, topology, physical; hermitian=true)
        beta = 1.0
        taus = [0.0, 0.25, 0.5, 0.75, 1.0]
        trajectory = thermalize(
            Purified(),
            problem,
            beta;
            evolver=P1NoOpEvolver(),
            nsteps=4,
            save_betas=sort(unique(beta .- taus)),
        )

        serial = thermal_correlator(
            Purified(),
            problem,
            :site1 => spin.Z,
            :site1 => spin.Z,
            beta,
            taus;
            evolver=P1NoOpEvolver(),
            trajectory,
            prop_nsteps=1,
            threaded=false,
        )
        parallel = thermal_correlator(
            Purified(),
            problem,
            :site1 => spin.Z,
            :site1 => spin.Z,
            beta,
            taus;
            evolver=P1NoOpEvolver(),
            trajectory,
            prop_nsteps=1,
            threaded=true,
            minbatch=1,
            task_memory_cap_bytes=typemax(Int),
            task_workspace_memory_bytes=1024^2,
        )
        @test parallel.values == serial.values
        @test parallel.metadata.fanout.mode ==
              (Base.Threads.nthreads() > 1 ? :threaded : :serial)
        missing_workspace = thermal_correlator(
            Purified(),
            problem,
            :site1 => spin.Z,
            :site1 => spin.Z,
            beta,
            taus;
            evolver=P1NoOpEvolver(),
            trajectory,
            prop_nsteps=1,
            threaded=true,
            minbatch=1,
            task_memory_cap_bytes=typemax(Int),
        )
        @test missing_workspace.metadata.fanout.mode == :serial
        @test missing_workspace.metadata.fanout.fallback ==
              :missing_task_workspace_memory

        too_small = serial.metadata.fanout.peak_admitted_bytes - 1
        admission_error = try
            thermal_correlator(
                Purified(),
                problem,
                :site1 => spin.Z,
                :site1 => spin.Z,
                beta,
                taus;
                evolver=P1NoOpEvolver(),
                trajectory,
                prop_nsteps=1,
                threaded=true,
                minbatch=1,
                task_memory_cap_bytes=too_small,
                task_workspace_memory_bytes=1024^2,
            )
            nothing
        catch err
            err
        end
        @test admission_error isa BoundedFanoutAdmissionError
        @test admission_error.item_index == 1
        @test admission_error.item_id.tau_index == 1

        item_error = try
            thermal_correlator(
                Purified(),
                problem,
                :site1 => spin.Z,
                :site1 => spin.Z,
                beta,
                taus;
                evolver=P1FailingEvolver(),
                trajectory,
                prop_nsteps=1,
                threaded=true,
                minbatch=1,
                task_memory_cap_bytes=typemax(Int),
                task_workspace_memory_bytes=1024^2,
            )
            nothing
        catch err
            err
        end
        @test item_error isa BoundedFanoutItemError
        @test item_error.item_index == 2
        @test item_error.item_id.tau_index == 2
        @test item_error.item_id.tau == 0.25
        @test occursin(
            "injected thermal propagation failure",
            sprint(showerror, item_error),
        )
    end

    @testset "RSVD probe-block contract" begin
        spin = spin_ops_u1()
        target = spin.P ← spin.P
        serial_rng = Xoshiro(2026072901)
        threaded_rng = Xoshiro(2026072901)
        serial_diagnostics = Ref{Any}(nothing)
        threaded_diagnostics = Ref{Any}(nothing)
        serial = _rsvd_random_probe(
            serial_rng,
            ComplexF64,
            target;
            threaded=false,
            minbatch=1,
            memory_cap_bytes=typemax(Int),
            task_workspace_memory_bytes=1024^2,
            fanout_diagnostics=serial_diagnostics,
        )
        parallel = _rsvd_random_probe(
            threaded_rng,
            ComplexF64,
            target;
            threaded=true,
            minbatch=1,
            memory_cap_bytes=typemax(Int),
            task_workspace_memory_bytes=1024^2,
            fanout_diagnostics=threaded_diagnostics,
        )
        @test iszero(Graft.Backend.norm(parallel - serial))
        @test rand(threaded_rng, UInt64) == rand(serial_rng, UInt64)
        @test threaded_diagnostics[].mode ==
              (Base.Threads.nthreads() > 1 ? :threaded : :serial)
        @test threaded_diagnostics[].peak_admitted_bytes <= typemax(Int)
        missing_diagnostics = Ref{Any}(nothing)
        _rsvd_random_probe(
            Xoshiro(2026072902),
            ComplexF64,
            target;
            threaded=true,
            minbatch=1,
            memory_cap_bytes=typemax(Int),
            fanout_diagnostics=missing_diagnostics,
        )
        @test missing_diagnostics[].mode == :serial
        @test missing_diagnostics[].fallback ==
              :missing_task_workspace_memory

        cap = serial_diagnostics[].peak_admitted_bytes - 1
        rejected_rng = Xoshiro(1)
        rejection_control_rng = Xoshiro(1)
        admission_error = try
            _rsvd_random_probe(
                rejected_rng,
                ComplexF64,
                target;
                threaded=true,
                minbatch=1,
                memory_cap_bytes=cap,
                task_workspace_memory_bytes=1024^2,
            )
            nothing
        catch err
            err
        end
        @test admission_error isa BoundedFanoutAdmissionError
        @test admission_error.item_id.probe_block_index ==
              admission_error.item_index
        @test rand(rejected_rng, UInt64) ==
              rand(rejection_control_rng, UInt64)

        item_error = try
            _rsvd_random_probe(
                Xoshiro(2),
                ComplexF64,
                target;
                threaded=true,
                minbatch=1,
                memory_cap_bytes=typemax(Int),
                task_workspace_memory_bytes=1024^2,
                block_fill=(rng, block_, index) ->
                    index == 1 ? error("injected RSVD block failure") :
                    fill!(block_, zero(eltype(block_))),
            )
            nothing
        catch err
            err
        end
        @test item_error isa BoundedFanoutItemError
        @test item_error.item_index == 1
        @test item_error.item_id.probe_block_index == 1
        @test occursin(
            "injected RSVD block failure",
            sprint(showerror, item_error),
        )
    end
end
