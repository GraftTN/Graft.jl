using Test

import GraftFoundation
import GraftNetworks
import GraftSymbolic
import GraftTTNOBuild

const Backend = GraftFoundation.Backend
const Trees = GraftFoundation.Trees
const Networks = GraftNetworks.Networks
const Symbolic = GraftSymbolic.Symbolic

@testset "GraftTTNOBuild" begin
    include("thc.jl")
    include("legacy_builder.jl")
end

@testset "owner-local load graph" begin
    loaded = Set(pkgid.name for pkgid in keys(Base.loaded_modules))
    forbidden = Set([
        "Graft",
        "GraftStateDiagram",
        "GraftGroundState",
        "GraftEvolution",
        "GraftSpectral",
        "GraftThermal",
    ])
    @test isempty(intersect(loaded, forbidden))
end
