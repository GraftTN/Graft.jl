"""
Check the split umbrella's module/type aliases.

Optional external artifacts are read-only inputs; this script never creates a
pre-split baseline:

    julia --project=. scripts/check_umbrella_compat.jl \
        [--exports-snapshot PATH] [--jld2-fixture PATH]

The export snapshot is a newline-delimited list of root public names. Blank
lines and lines beginning with `#` are ignored; either `name` or `Graft.name`
is accepted. The JLD2 fixture is loaded in full so every stored object path is
reconstructed through the compatibility namespaces.
"""

import Graft
import GraftContractions
import GraftEvolution
import GraftFoundation
import GraftGroundState
import GraftNetworks
import GraftParallel
import GraftPlanning
import GraftSpectral
import GraftStateDiagram
import GraftSymbolic
import GraftTTNOBuild
import GraftThermal
import JLD2

function require_identity(label::AbstractString, actual, expected)
    actual === expected || error(
        "$label is $(repr(actual)); expected identity with $(repr(expected))")
    return nothing
end

function owner_concrete_types(owner::Module)
    result = Symbol[]
    for name in names(owner; all=true, imported=false)
        Base.isidentifier(String(name)) || continue
        isdefined(owner, name) || continue
        binding = getfield(owner, name)
        binding isa Union{DataType,UnionAll} || continue
        datatype = Base.unwrap_unionall(binding)
        parentmodule(datatype) === owner || continue
        isabstracttype(datatype) && continue
        push!(result, name)
    end
    return sort!(unique!(result); by=string)
end

function check_module_aliases()
    require_identity("Graft.Backend", Graft.Backend, GraftFoundation.Backend)
    require_identity("Graft.Trees", Graft.Trees, GraftFoundation.Trees)
    require_identity("Graft.Networks", Graft.Networks, GraftNetworks.Networks)
    require_identity(
        "Graft.Contractions", Graft.Contractions,
        GraftContractions.Contractions)
    require_identity("Graft.Parallel", Graft.Parallel, GraftParallel.Parallel)
    require_identity("Graft.Planning", Graft.Planning, GraftPlanning.Planning)
    require_identity("Graft.Symbolic", Graft.Symbolic, GraftSymbolic.Symbolic)
    require_identity(
        "Graft.GroundState", Graft.GroundState,
        GraftGroundState.GroundState)
    require_identity("Graft.Evolution", Graft.Evolution, GraftEvolution.Evolution)
    require_identity("Graft.Spectral", Graft.Spectral, GraftSpectral.Spectral)
    require_identity("Graft.Thermal", Graft.Thermal, GraftThermal.Thermal)
    require_identity(
        "Graft.Contractions.Planning", Graft.Contractions.Planning,
        GraftPlanning.Planning)
    require_identity(
        "Graft.StateDiagram", Graft.StateDiagram,
        GraftStateDiagram.StateDiagramCompiler)
    isdefined(Graft, :TestUtils) &&
        error("Graft.TestUtils must not be reachable from the runtime umbrella")
    return nothing
end

function check_ttno_aliases()
    compatibility = Graft.TTNOBuild
    legacy = GraftTTNOBuild.LegacyTTNOBuild
    typed = GraftStateDiagram.StateDiagramCompiler

    require_identity(
        "Graft.TTNOBuild.LegacyTTNOBuild",
        compatibility.LegacyTTNOBuild, legacy)
    require_identity(
        "Graft.TTNOBuild.StateDiagramAPI",
        compatibility.StateDiagramAPI, typed)

    for name in (
        :ttno_from_opsum, :THCFactorization, :THCReport, :isdf_thc, :fit_thc,
        :reconstruct_thc, :_net_u1_charge, :_build_braided_term_plan, :_Euler,
        :_input_twist_parity,
    )
        require_identity(
            "Graft.TTNOBuild.$name", getfield(compatibility, name),
            getfield(legacy, name))
    end

    for name in (
        :compile_ttno, :AbstractOperatorLoweringKernel,
        :AbelianScalarLowering, :AbstractTTNOMergeKernel, :DirectSumMerge,
        :StateDiagramMerge, :StructuralOptimizer, :GammaCoverOptimizer,
        :SGEOptimizer, :MissingCategoryCapability, :TTNOBuildReport,
        :TTNOBuildEdgeReport, :compiler_exact_provenance,
    )
        require_identity(
            "Graft.TTNOBuild.$name", getfield(compatibility, name),
            getfield(typed, name))
    end

    for name in owner_concrete_types(typed)
        isdefined(compatibility, name) || error(
            "Graft.TTNOBuild is missing concrete typed alias $name")
        require_identity(
            "Graft.TTNOBuild.$name", getfield(compatibility, name),
            getfield(typed, name))
    end
    return nothing
end

function snapshot_names(path::AbstractString)
    isfile(path) || error("export snapshot does not exist: $path")
    result = Set{String}()
    for raw in eachline(path)
        name = strip(raw)
        (isempty(name) || startswith(name, '#')) && continue
        startswith(name, "Graft.") && (name = name[7:end])
        startswith(name, ':') && (name = name[2:end])
        isempty(name) || push!(result, name)
    end
    isempty(result) && error("export snapshot is empty: $path")
    return result
end

function check_export_snapshot(path::AbstractString)
    expected = snapshot_names(path)
    actual = Set(string.(names(Graft; all=false, imported=false)))
    missing = sort!(collect(setdiff(expected, actual)))
    added = sort!(collect(setdiff(actual, expected)))
    isempty(missing) && isempty(added) || error(
        "root export snapshot mismatch; missing=$(missing), added=$(added)")
    return nothing
end

function load_jld2_fixture(path::AbstractString)
    isfile(path) || error("JLD2 fixture does not exist: $path")
    values = JLD2.load(path)
    println("loaded pre-split JLD2 fixture keys: ",
            join(sort!(string.(collect(keys(values)))), ", "))
    return nothing
end

function parse_options(args::Vector{String})
    exports_snapshot = nothing
    jld2_fixture = nothing
    index = 1
    while index <= length(args)
        option = args[index]
        option in ("--exports-snapshot", "--jld2-fixture") ||
            error("unknown option: $option")
        index == length(args) && error("missing path after $option")
        path = args[index + 1]
        option == "--exports-snapshot" ?
            (exports_snapshot = path) : (jld2_fixture = path)
        index += 2
    end
    return (; exports_snapshot, jld2_fixture)
end

function main(args::Vector{String})
    options = parse_options(args)
    check_module_aliases()
    check_ttno_aliases()
    options.exports_snapshot === nothing ||
        check_export_snapshot(options.exports_snapshot)
    options.jld2_fixture === nothing || load_jld2_fixture(options.jld2_fixture)
    println("umbrella compatibility aliases ok")
    return nothing
end

main(ARGS)
