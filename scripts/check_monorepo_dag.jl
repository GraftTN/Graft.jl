using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const EXPECTED = Dict(
    "GraftFoundation" => Set{String}(),
    "GraftParallel" => Set(["GraftFoundation"]),
    "GraftPlanning" => Set(["GraftFoundation"]),
    "GraftNetworks" => Set(["GraftFoundation", "GraftPlanning"]),
    "GraftContractions" => Set([
        "GraftFoundation", "GraftParallel", "GraftPlanning", "GraftNetworks",
    ]),
    "GraftSymbolic" => Set(["GraftFoundation"]),
    "GraftTTNOBuild" => Set([
        "GraftFoundation", "GraftNetworks", "GraftSymbolic",
    ]),
    "GraftStateDiagram" => Set([
        "GraftFoundation", "GraftNetworks", "GraftSymbolic", "GraftTTNOBuild",
    ]),
    "GraftGroundState" => Set([
        "GraftFoundation", "GraftNetworks", "GraftContractions", "GraftParallel",
    ]),
    "GraftEvolution" => Set([
        "GraftFoundation", "GraftNetworks", "GraftContractions", "GraftParallel",
    ]),
    "GraftSpectral" => Set([
        "GraftNetworks", "GraftContractions", "GraftEvolution", "GraftParallel",
    ]),
    "GraftThermal" => Set([
        "GraftFoundation", "GraftNetworks", "GraftContractions", "GraftSymbolic",
        "GraftTTNOBuild", "GraftEvolution", "GraftParallel",
    ]),
)

function project(path::AbstractString)
    isfile(path) || error("missing project: $path")
    return TOML.parsefile(path)
end

function visit!(name::String, edges, temporary::Set{String}, permanent::Set{String})
    name in permanent && return
    name in temporary && error("runtime package cycle reaches $name")
    push!(temporary, name)
    for dependency in edges[name]
        visit!(dependency, edges, temporary, permanent)
    end
    delete!(temporary, name)
    push!(permanent, name)
    return
end

function main()
    names = Set(keys(EXPECTED))
    discovered = Set(
        entry for entry in readdir(joinpath(ROOT, "lib"))
        if isfile(joinpath(ROOT, "lib", entry, "Project.toml")) &&
            entry != "GraftTestUtils"
    )
    discovered == names || error(
        "runtime package set mismatch: expected $(sort!(collect(names))), " *
        "found $(sort!(collect(discovered)))")

    edges = Dict{String,Set{String}}()
    for name in names
        data = project(joinpath(ROOT, "lib", name, "Project.toml"))
        data["name"] == name || error("$name project declares $(data["name"])")
        dependencies = Set(intersect(keys(get(data, "deps", Dict())), names))
        dependencies == EXPECTED[name] || error(
            "$name internal dependencies are $(sort!(collect(dependencies))); " *
            "expected $(sort!(collect(EXPECTED[name])))")
        "GraftTestUtils" in keys(get(data, "deps", Dict())) &&
            error("runtime package $name depends on GraftTestUtils")
        edges[name] = dependencies
    end

    permanent = Set{String}()
    for name in names
        visit!(name, edges, Set{String}(), permanent)
    end

    root = project(joinpath(ROOT, "Project.toml"))
    rootdeps = Set(intersect(keys(root["deps"]), names))
    rootsources = Set(intersect(keys(root["sources"]), names))
    rootdeps == names || error("umbrella does not depend on exactly twelve packages")
    rootsources == names || error("umbrella does not source exactly twelve packages")
    haskey(root["deps"], "GraftTestUtils") &&
        error("umbrella runtime graph contains GraftTestUtils")

    println("monorepo runtime DAG ok: 12 subpackages, 1 umbrella, no test-support edge")
end

main()
