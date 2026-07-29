using Test
using TOML

include(joinpath(@__DIR__, "..", "benchmark", "sysimage_builder.jl"))

@testset "optional sysimage builder identity checks" begin
    @test main(["--check"]) == 0
    mktempdir() do directory
        metadata = joinpath(directory, "graft.so.toml")
        open(metadata, "w") do io
            TOML.print(io, identities(); sorted=true)
        end
        @test isnothing(verify_metadata(metadata))

        mismatched = TOML.parsefile(metadata)
        mismatched["julia_version"] = "0.0.0"
        open(metadata, "w") do io
            TOML.print(io, mismatched; sorted=true)
        end
        error = try
            verify_metadata(metadata)
            nothing
        catch err
            err
        end
        @test error isa ErrorException
        @test occursin(
            "compatibility check failed before solver execution",
            sprint(showerror, error),
        )
    end
end
