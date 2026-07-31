using Test

import GraftFoundation
import GraftNetworks
import GraftStateDiagram
import GraftSymbolic
import GraftTTNOBuild

const Backend = GraftFoundation.Backend
const Trees = GraftFoundation.Trees
const Networks = GraftNetworks.Networks
const Symbolic = GraftSymbolic.Symbolic
const StateDiagram = GraftStateDiagram.StateDiagramCompiler

@testset "GraftStateDiagram" begin
    include("ir_pipeline.jl")
end

@testset "owner-local load graph" begin
    loaded = Set(pkgid.name for pkgid in keys(Base.loaded_modules))
    forbidden = Set([
        "Graft",
        "GraftTestUtils",
        "GraftContractions",
        "GraftGroundState",
        "GraftEvolution",
        "GraftSpectral",
        "GraftThermal",
    ])
    @test isempty(intersect(loaded, forbidden))
end
