#!/usr/bin/env julia

"""
Measure Julia package-image invalidation before or after the Graft monorepo split.

The live checkout is read-only. The script copies it (excluding `.git`) into a
temporary workspace, uses a completely isolated `JULIA_DEPOT_PATH`, builds one
umbrella baseline, and restores that baseline before every comment-only edit.
Use `--help` for the command-line contract.
"""

using Dates
using Printf
using SHA
using UUIDs

const SCRIPT_ROOT = normpath(joinpath(@__DIR__, ".."))
const CACHE_VERSION = "v$(VERSION.major).$(VERSION.minor)"
const IMAGE_SUFFIXES = Set((".ji", ".so", ".dylib", ".dll"))

struct Target
    name::String
    source::String
    load_mpi::Bool
end

const SPLIT_TARGETS = Target[
    Target("parallel", "lib/GraftParallel/src/GraftParallel.jl", false),
    Target("planning", "lib/GraftPlanning/src/GraftPlanning.jl", false),
    Target("contractions", "lib/GraftContractions/src/GraftContractions.jl", false),
    Target("symbolic", "lib/GraftSymbolic/src/GraftSymbolic.jl", false),
    Target("ttnobuild", "lib/GraftTTNOBuild/src/GraftTTNOBuild.jl", false),
    Target("statediagram", "lib/GraftStateDiagram/src/GraftStateDiagram.jl", false),
    Target("evolution", "lib/GraftEvolution/src/GraftEvolution.jl", false),
    Target("thermal", "lib/GraftThermal/src/GraftThermal.jl", false),
    Target("mpi-parallel", "lib/GraftParallel/ext/GraftParallelMPIExt.jl", true),
    Target("mpi-groundstate", "lib/GraftGroundState/ext/GraftGroundStateMPIExt.jl", true),
    Target("mpi-evolution", "lib/GraftEvolution/ext/GraftEvolutionMPIExt.jl", true),
    Target("mpi-umbrella", "ext/GraftMPIExt.jl", true),
]
const PRESPLIT_TARGETS = Target[
    Target("parallel", "src/Parallel/Parallel.jl", false),
    Target("planning", "src/Contractions/planning/Planning.jl", false),
    Target("contractions", "src/Contractions/Contractions.jl", false),
    Target("symbolic", "src/Symbolic/Symbolic.jl", false),
    Target("ttnobuild", "src/TTNOBuild/statediagram.jl", false),
    Target("statediagram", "src/TTNOBuild/ir.jl", false),
    Target("evolution", "src/Evolution/Evolution.jl", false),
    Target("thermal", "src/Thermal/Thermal.jl", false),
]
const RUNTIME_PACKAGES = [
    "Graft", "GraftFoundation", "GraftParallel", "GraftPlanning",
    "GraftNetworks", "GraftContractions", "GraftSymbolic",
    "GraftTTNOBuild", "GraftStateDiagram", "GraftGroundState",
    "GraftEvolution", "GraftSpectral", "GraftThermal",
]
const MPI_EXTENSION_IMAGES = [
    "GraftMPIExt", "GraftParallelMPIExt", "GraftGroundStateMPIExt",
    "GraftEvolutionMPIExt",
]

struct Options
    layout::String
    source_root::String
    targets::Vector{Target}
    tsv::Union{Nothing,String}
    json::Union{Nothing,String}
    text::Union{Nothing,String}
    keep_temp::Union{Nothing,String}
    overwrite::Bool
    dry_run::Bool
    help::Bool
end

function usage(io::IO=stdout)
    println(io, """
Usage:
  julia --project=. scripts/measure_monorepo_invalidation.jl \\
      --output-prefix /absolute/path/graft-invalidation \\
      [--targets parallel,planning,...] [--keep-temp /absolute/path] [--overwrite]

Layout selection:
  --layout split         Measure the split monorepo (default). The source root
                         defaults to the checkout containing this script.
  --layout presplit      Measure the monolithic before-layout. Requires an
                         absolute --source-root and rejects all MPI targets.
  --source-root ABS_PATH Read-only source checkout to copy into isolation.

Output selection (required except with --dry-run):
  --output-prefix PATH   Write PATH.tsv, PATH.json, and PATH.txt.
  --tsv PATH             Explicit TSV path; requires --json and --text.
  --json PATH            Explicit JSON path; requires --tsv and --text.
  --text PATH            Explicit text-report path; requires --tsv and --json.

Probe selection:
  --targets LIST         Comma-separated targets. Default: all.
                         Runtime: parallel, planning, contractions, symbolic,
                         ttnobuild, statediagram, evolution, thermal.
                         MPI: mpi-parallel, mpi-groundstate, mpi-evolution,
                         mpi-umbrella. Alias `mpi` expands all four MPI targets.
                         Alias `all` expands every target available in the layout.

Safety and diagnostics:
  --keep-temp PATH       Keep the isolated source, depot, cache backups, and logs
                         in a new PATH. By default a temporary directory is removed.
  --overwrite            Permit replacement of the three report files. The default
                         is to refuse every existing output.
  --dry-run              Validate paths and print the planned isolated operations;
                         do not copy, edit, resolve, compile, load, or write reports.
  -h, --help             Show this help.

The script runs only on Julia 1.12. Output and --keep-temp paths inside either the
script checkout or selected source checkout are rejected. Every measured edit is
an appended comment in the isolated source copy; the source file and Graft*
compiled-cache baseline are restored before the next target. Package-image
snapshots include the actual .ji/.so/.dylib/.dll paths, byte counts, mtimes in
nanoseconds, and SHA-256 digests. Presplit runs assert only the monolithic Graft
package image; split runs assert the umbrella and twelve runtime package images.
""")
end

function take_value(args::Vector{String}, index::Int, option::String)
    index == length(args) && error("missing value after $option")
    return args[index + 1], index + 2
end

available_targets(layout::AbstractString) =
    layout == "split" ? SPLIT_TARGETS : PRESPLIT_TARGETS

function expand_targets(specification::String, layout::String)
    targets = available_targets(layout)
    target_by_name = Dict(target.name => target for target in targets)
    requested = split(lowercase(specification), ','; keepempty=false)
    isempty(requested) && error("--targets must not be empty")
    names = String[]
    for raw in requested
        name = strip(raw)
        isempty(name) && continue
        expanded = if name == "all"
            [target.name for target in targets]
        elseif name == "mpi"
            layout == "presplit" && error(
                "MPI targets are unavailable with --layout presplit")
            [target.name for target in targets if target.load_mpi]
        else
            haskey(target_by_name, name) || error(
                "target '$name' is unavailable for layout '$layout'; " *
                "run with --help for the target list")
            [name]
        end
        for item in expanded
            item in names || push!(names, item)
        end
    end
    isempty(names) && error("--targets must select at least one target")
    return [target_by_name[name] for name in names]
end

function parse_options(args::Vector{String})
    layout = "split"
    source_root = nothing
    target_specification = "all"
    output_prefix = nothing
    tsv = nothing
    json = nothing
    text = nothing
    keep_temp = nothing
    overwrite = false
    dry_run = false
    help = false

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument in ("-h", "--help")
            help = true
            index += 1
        elseif argument == "--overwrite"
            overwrite = true
            index += 1
        elseif argument == "--dry-run"
            dry_run = true
            index += 1
        elseif startswith(argument, "--layout=")
            layout = lowercase(split(argument, '='; limit=2)[2])
            index += 1
        elseif argument == "--layout"
            layout, index = take_value(args, index, argument)
            layout = lowercase(layout)
        elseif startswith(argument, "--source-root=")
            source_root = split(argument, '='; limit=2)[2]
            index += 1
        elseif argument == "--source-root"
            source_root, index = take_value(args, index, argument)
        elseif startswith(argument, "--targets=")
            target_specification = split(argument, '='; limit=2)[2]
            index += 1
        elseif argument == "--targets"
            target_specification, index = take_value(args, index, argument)
        elseif startswith(argument, "--output-prefix=")
            output_prefix = split(argument, '='; limit=2)[2]
            index += 1
        elseif argument == "--output-prefix"
            output_prefix, index = take_value(args, index, argument)
        elseif startswith(argument, "--tsv=")
            tsv = split(argument, '='; limit=2)[2]
            index += 1
        elseif argument == "--tsv"
            tsv, index = take_value(args, index, argument)
        elseif startswith(argument, "--json=")
            json = split(argument, '='; limit=2)[2]
            index += 1
        elseif argument == "--json"
            json, index = take_value(args, index, argument)
        elseif startswith(argument, "--text=")
            text = split(argument, '='; limit=2)[2]
            index += 1
        elseif argument == "--text"
            text, index = take_value(args, index, argument)
        elseif startswith(argument, "--keep-temp=")
            keep_temp = split(argument, '='; limit=2)[2]
            index += 1
        elseif argument == "--keep-temp"
            keep_temp, index = take_value(args, index, argument)
        else
            error("unknown option: $argument")
        end
    end

    if output_prefix !== nothing
        any(value !== nothing for value in (tsv, json, text)) && error(
            "--output-prefix cannot be combined with --tsv, --json, or --text")
        tsv = string(output_prefix, ".tsv")
        json = string(output_prefix, ".json")
        text = string(output_prefix, ".txt")
    end
    explicit_count = count(value !== nothing for value in (tsv, json, text))
    explicit_count in (0, 3) || error(
        "specify all of --tsv, --json, and --text, or use --output-prefix")
    !help && !dry_run && explicit_count == 0 && error(
        "report paths are required; use --output-prefix or all explicit paths")

    layout in ("split", "presplit") || error(
        "--layout must be 'split' or 'presplit'; got '$layout'")
    if source_root === nothing
        layout == "presplit" && error(
            "--layout presplit requires an absolute --source-root")
        source_root = SCRIPT_ROOT
    else
        isabspath(source_root) || error("--source-root must be an absolute path")
        source_root = normpath(source_root)
    end
    absolute(value) = value === nothing ? nothing : abspath(value)
    return Options(
        layout, source_root, expand_targets(target_specification, layout),
        absolute(tsv), absolute(json), absolute(text), absolute(keep_temp),
        overwrite, dry_run, help)
end

function is_within(path::AbstractString, root::AbstractString)
    relative = relpath(abspath(path), abspath(root))
    return relative == "." ||
        !(relative == ".." || startswith(relative, string("..", Base.Filesystem.path_separator)))
end

function validate_paths(options::Options)
    isdir(options.source_root) || error(
        "source root does not exist or is not a directory: $(options.source_root)")
    isfile(joinpath(options.source_root, "Project.toml")) || error(
        "source root is missing Project.toml: $(options.source_root)")
    outputs = String[path for path in (options.tsv, options.json, options.text)
                     if path !== nothing]
    length(unique(normpath.(outputs))) == length(outputs) ||
        error("TSV, JSON, and text outputs must be distinct paths")
    for path in outputs
        (is_within(path, SCRIPT_ROOT) || is_within(path, options.source_root)) &&
            error("report path must be outside the script and source checkouts: $path")
        isdir(path) && error("report path is a directory: $path")
        ispath(path) && !options.overwrite && error(
            "refusing to overwrite existing report (pass --overwrite): $path")
    end
    if options.keep_temp !== nothing
        (is_within(options.keep_temp, SCRIPT_ROOT) ||
         is_within(options.keep_temp, options.source_root)) && error(
            "--keep-temp must be outside the script and source checkouts: " *
            options.keep_temp)
        ispath(options.keep_temp) && error(
            "--keep-temp path must not already exist: $(options.keep_temp)")
    end
    for target in options.targets
        path = joinpath(options.source_root, target.source)
        isfile(path) || error("target source does not exist: $path")
    end
    return nothing
end

function copy_checkout(source_root::AbstractString, destination::AbstractString)
    mkdir(destination)
    for entry in readdir(source_root)
        entry == ".git" && continue
        cp(joinpath(source_root, entry), joinpath(destination, entry);
           follow_symlinks=false)
    end
    return nothing
end

function command_string(command::Cmd)
    return sprint(show, command)
end

function tail_text(path::AbstractString, maximum_bytes::Int=8000)
    bytes = read(path)
    first_index = max(1, length(bytes) - maximum_bytes + 1)
    return String(bytes[first_index:end])
end

function run_julia_phase(source_root::AbstractString, depot::AbstractString,
                         logs::AbstractString, name::AbstractString,
                         code::AbstractString)
    mkpath(logs)
    log_path = joinpath(logs, string(name, ".log"))
    command = `$(Base.julia_cmd()) --startup-file=no --history-file=no --project=$source_root -e $code`
    environment = copy(ENV)
    environment["JULIA_DEPOT_PATH"] = depot
    environment["JULIA_LOAD_PATH"] = "@:@stdlib"
    started = time_ns()
    process = open(log_path, "w") do io
        process = run(pipeline(setenv(command, environment), stdout=io, stderr=io);
                      wait=false)
        wait(process)
        return process
    end
    elapsed = (time_ns() - started) / 1.0e9
    success(process) || error(
        "Julia phase '$name' failed ($(command_string(command))). Log tail:\n" *
        tail_text(log_path))
    return Dict{String,Any}(
        "name" => String(name),
        "seconds" => elapsed,
        "command" => command_string(command),
        "log_sha256" => bytes2hex(SHA.sha256(read(log_path))),
        "log_bytes" => filesize(log_path),
    )
end

cache_root(depot::AbstractString) = joinpath(depot, "compiled", CACHE_VERSION)

function is_image_file(path::AbstractString)
    return splitext(path)[2] in IMAGE_SUFFIXES
end

function snapshot_images(depot::AbstractString)
    root = cache_root(depot)
    snapshot = Dict{String,Any}()
    isdir(root) || return snapshot
    for package in sort!(filter(name -> startswith(name, "Graft"), readdir(root)))
        package_root = joinpath(root, package)
        isdir(package_root) || continue
        for (directory, _, files) in walkdir(package_root)
            for file in sort!(files)
                path = joinpath(directory, file)
                isfile(path) && is_image_file(path) || continue
                relative = relpath(path, root)
                status = stat(path)
                snapshot[relative] = Dict{String,Any}(
                    "package" => package,
                    "path" => relative,
                    "bytes" => status.size,
                    "mtime_ns" => round(Int, status.mtime * 1.0e9),
                    "sha256" => bytes2hex(SHA.sha256(read(path))),
                )
            end
        end
    end
    return snapshot
end

function assert_packages(snapshot::Dict{String,Any}, required::Vector{String},
                         label::AbstractString)
    present = Set(String(record["package"]) for record in values(snapshot))
    missing = sort!(collect(setdiff(Set(required), present)))
    isempty(missing) || error(
        "$label package-image snapshot is missing: $(join(missing, ", "))")
    return nothing
end

function image_equal(before::Dict{String,Any}, after::Dict{String,Any})
    return before["bytes"] == after["bytes"] &&
        before["mtime_ns"] == after["mtime_ns"] &&
        before["sha256"] == after["sha256"]
end

function compare_images(before::Dict{String,Any}, after::Dict{String,Any})
    paths = sort!(collect(union(Set(keys(before)), Set(keys(after)))))
    files = Dict{String,Any}[]
    package_changed = Dict{String,Bool}()
    packages = Set{String}()
    for path in paths
        before_record = get(before, path, nothing)
        after_record = get(after, path, nothing)
        package = String((before_record === nothing ? after_record : before_record)["package"])
        push!(packages, package)
        classification = if before_record === nothing
            "created"
        elseif after_record === nothing
            "deleted"
        elseif image_equal(before_record, after_record)
            "reused"
        else
            "modified"
        end
        package_changed[package] = get(package_changed, package, false) ||
            classification != "reused"
        push!(files, Dict{String,Any}(
            "package" => package,
            "path" => path,
            "classification" => classification,
            "before" => before_record,
            "after" => after_record,
        ))
    end
    changed_packages = sort!([package for package in packages
                              if get(package_changed, package, false)])
    reused_packages = sort!([package for package in packages
                             if !get(package_changed, package, false)])
    return Dict{String,Any}(
        "changed_packages" => changed_packages,
        "reused_packages" => reused_packages,
        "created_images" => [record["path"] for record in files
                             if record["classification"] == "created"],
        "modified_images" => [record["path"] for record in files
                              if record["classification"] == "modified"],
        "deleted_images" => [record["path"] for record in files
                             if record["classification"] == "deleted"],
        "reused_images" => [record["path"] for record in files
                            if record["classification"] == "reused"],
        "files" => files,
    )
end

function cache_entries(root::AbstractString)
    isdir(root) || return String[]
    return sort!(filter(name -> startswith(name, "Graft") &&
                                isdir(joinpath(root, name)), readdir(root)))
end

function preserved_copy(source::AbstractString, destination::AbstractString)
    run(`/bin/cp -pR $source $destination`)
    return nothing
end

function save_cache_baseline(depot::AbstractString, destination::AbstractString)
    ispath(destination) && rm(destination; recursive=true, force=true)
    mkdir(destination)
    root = cache_root(depot)
    for entry in cache_entries(root)
        preserved_copy(joinpath(root, entry), joinpath(destination, entry))
    end
    isempty(readdir(destination)) && error("no Graft* package-image cache to save")
    return nothing
end

function restore_cache_baseline(depot::AbstractString, source::AbstractString)
    root = cache_root(depot)
    mkpath(root)
    for entry in cache_entries(root)
        rm(joinpath(root, entry); recursive=true, force=true)
    end
    for entry in sort!(readdir(source))
        preserved_copy(joinpath(source, entry), joinpath(root, entry))
    end
    return nothing
end

function same_snapshot_content(left::Dict{String,Any}, right::Dict{String,Any})
    keys(left) == keys(right) || return false
    return all(left[path]["bytes"] == right[path]["bytes"] &&
               left[path]["sha256"] == right[path]["sha256"] for path in keys(left))
end

function append_probe_comment(path::AbstractString, target::Target)
    marker = "# Graft monorepo invalidation probe: $(target.name) $(uuid4())"
    open(path, "a") do io
        write(io, '\n', marker, '\n')
    end
    return marker
end

function measure_target(target::Target, source_root::AbstractString,
                        depot::AbstractString, logs::AbstractString,
                        baseline_cache::AbstractString,
                        reference_snapshot::Dict{String,Any})
    restore_cache_baseline(depot, baseline_cache)
    before = snapshot_images(depot)
    same_snapshot_content(before, reference_snapshot) || error(
        "restored cache content differs from the recorded baseline for $(target.name)")

    source_path = joinpath(source_root, target.source)
    backup_path = joinpath(dirname(baseline_cache), "source-backups", target.name)
    mkpath(dirname(backup_path))
    preserved_copy(source_path, backup_path)
    original_sha256 = bytes2hex(SHA.sha256(read(source_path)))
    marker = append_probe_comment(source_path, target)
    phase = nothing
    after = Dict{String,Any}()
    try
        expression = target.load_mpi ? "using Graft; using MPI" : "using Graft"
        phase = run_julia_phase(
            source_root, depot, logs, string("edit-load-", target.name), expression)
        after = snapshot_images(depot)
    finally
        preserved_copy(backup_path, source_path)
        restored_sha256 = bytes2hex(SHA.sha256(read(source_path)))
        restored_sha256 == original_sha256 || error(
            "failed to restore isolated source after target $(target.name)")
        restore_cache_baseline(depot, baseline_cache)
    end

    comparison = compare_images(before, after)
    return Dict{String,Any}(
        "target" => target.name,
        "edited_file" => target.source,
        "comment_marker" => marker,
        "load_expression" => target.load_mpi ? "using Graft; using MPI" : "using Graft",
        "edit_load" => phase,
        "baseline_snapshot" => [before[path] for path in sort!(collect(keys(before)))],
        "edited_snapshot" => [after[path] for path in sort!(collect(keys(after)))],
        "comparison" => comparison,
    )
end

function git_read(source_root::AbstractString, arguments...)
    command = Cmd(String["git", "-C", source_root, string.(arguments)...])
    try
        return chomp(read(command, String))
    catch
        return "unavailable"
    end
end

function tsv_field(value)
    return replace(string(value), '\t' => "\\t", '\n' => "\\n", '\r' => "\\r")
end

function write_tsv(io::IO, results::Vector{Dict{String,Any}},
                   layout::AbstractString, source_root::AbstractString)
    headers = [
        "layout", "source_root", "target", "edited_file", "edit_load_seconds",
        "package", "package_status", "image_status", "image_path",
        "before_bytes", "after_bytes", "before_mtime_ns", "after_mtime_ns",
        "before_sha256", "after_sha256",
    ]
    println(io, join(headers, '\t'))
    for result in results
        comparison = result["comparison"]
        changed = Set(comparison["changed_packages"])
        for record in comparison["files"]
            before = record["before"]
            after = record["after"]
            package = record["package"]
            row = Any[
                layout, source_root, result["target"], result["edited_file"],
                result["edit_load"]["seconds"], package,
                package in changed ? "changed" : "reused",
                record["classification"], record["path"],
                before === nothing ? "" : before["bytes"],
                after === nothing ? "" : after["bytes"],
                before === nothing ? "" : before["mtime_ns"],
                after === nothing ? "" : after["mtime_ns"],
                before === nothing ? "" : before["sha256"],
                after === nothing ? "" : after["sha256"],
            ]
            println(io, join(tsv_field.(row), '\t'))
        end
    end
end

function json_string(io::IO, value::AbstractString)
    write(io, '"')
    for character in value
        if character == '"'
            write(io, "\\\"")
        elseif character == '\\'
            write(io, "\\\\")
        elseif character == '\b'
            write(io, "\\b")
        elseif character == '\f'
            write(io, "\\f")
        elseif character == '\n'
            write(io, "\\n")
        elseif character == '\r'
            write(io, "\\r")
        elseif character == '\t'
            write(io, "\\t")
        elseif Int(character) < 0x20
            @printf(io, "\\u%04x", Int(character))
        else
            write(io, character)
        end
    end
    write(io, '"')
end

function write_json(io::IO, value, indent::Int=0)
    if value === nothing
        write(io, "null")
    elseif value isa Bool
        write(io, value ? "true" : "false")
    elseif value isa Integer || value isa AbstractFloat
        write(io, string(value))
    elseif value isa AbstractString
        json_string(io, value)
    elseif value isa AbstractVector
        isempty(value) && return write(io, "[]")
        write(io, "[\n")
        for (index, item) in enumerate(value)
            write(io, " "^(indent + 2))
            write_json(io, item, indent + 2)
            index == length(value) || write(io, ',')
            write(io, '\n')
        end
        write(io, " "^indent, ']')
    elseif value isa AbstractDict
        keys_sorted = sort!(string.(collect(keys(value))))
        isempty(keys_sorted) && return write(io, "{}")
        write(io, "{\n")
        for (index, key) in enumerate(keys_sorted)
            write(io, " "^(indent + 2))
            json_string(io, key)
            write(io, ": ")
            write_json(io, value[key], indent + 2)
            index == length(keys_sorted) || write(io, ',')
            write(io, '\n')
        end
        write(io, " "^indent, '}')
    else
        error("cannot encode JSON value of type $(typeof(value))")
    end
    return nothing
end

function write_text(io::IO, report::Dict{String,Any})
    println(io, "Graft monorepo package-image invalidation measurement")
    println(io, "Julia: ", report["julia_version"])
    println(io, "Layout: ", report["layout"])
    println(io, "Live source (read-only input): ", report["live_source"])
    println(io, "Git HEAD: ", report["git_head"])
    println(io, "Git status at copy time:\n", report["git_status"])
    println(io, "\nBaseline phases:")
    for phase in report["baseline_phases"]
        @printf(io, "  %-16s %10.6f s  log sha256 %s\n",
                phase["name"], phase["seconds"], phase["log_sha256"])
        println(io, "    ", phase["command"])
    end
    for result in report["targets"]
        comparison = result["comparison"]
        println(io, "\nTarget: ", result["target"])
        println(io, "  comment-only edit: ", result["edited_file"])
        println(io, "  load expression: ", result["load_expression"])
        @printf(io, "  fresh load: %.6f s\n", result["edit_load"]["seconds"])
        println(io, "  changed package images: ",
                join(comparison["changed_packages"], ", "))
        println(io, "  reused package images: ",
                join(comparison["reused_packages"], ", "))
        for classification in ("created", "modified", "deleted", "reused")
            paths = comparison[string(classification, "_images")]
            println(io, "  ", classification, " image files (", length(paths), "):")
            for path in paths
                println(io, "    ", path)
            end
        end
    end
end

function atomic_write(path::AbstractString, overwrite::Bool, writer::Function)
    parent = dirname(path)
    mkpath(parent)
    temporary, io = mktemp(parent; cleanup=false)
    try
        writer(io)
        close(io)
        mv(temporary, path; force=overwrite)
    catch
        isopen(io) && close(io)
        ispath(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return nothing
end

function write_reports(options::Options, report::Dict{String,Any})
    results = report["targets"]
    atomic_write(options.tsv, options.overwrite) do io
        write_tsv(io, results, report["layout"], report["live_source"])
    end
    atomic_write(options.json, options.overwrite) do io
        write_json(io, report)
        write(io, '\n')
    end
    atomic_write(options.text, options.overwrite) do io
        write_text(io, report)
    end
    return nothing
end

function print_dry_run(options::Options)
    println("dry run: no files will be copied, edited, compiled, or written")
    println("layout: ", options.layout)
    println("live source (read-only): ", options.source_root)
    println("Julia: ", VERSION)
    println("temporary policy: ", options.keep_temp === nothing ?
            "automatic cleanup" : "keep at $(options.keep_temp)")
    if options.tsv !== nothing
        println("reports: ", options.tsv, ", ", options.json, ", ", options.text)
    else
        println("reports: none requested for dry run")
    end
    println("baseline: Pkg.resolve -> Pkg.instantiate -> Pkg.precompile -> using Graft")
    any(target.load_mpi for target in options.targets) &&
        println("MPI baseline: using Graft; using MPI")
    println("targets:")
    for target in options.targets
        println("  ", target.name, ": append comment to ", target.source,
                "; fresh ", target.load_mpi ? "using Graft; using MPI" : "using Graft")
    end
    return nothing
end

function execute(options::Options, work_root::AbstractString)
    source_root = joinpath(work_root, "source")
    depot = joinpath(work_root, "depot")
    logs = joinpath(work_root, "logs")
    backups = joinpath(work_root, "cache-baselines")
    copy_checkout(options.source_root, source_root)
    mkdir(depot)
    mkpath(backups)

    baseline_phases = Dict{String,Any}[]
    for (name, code) in (
        ("resolve", "import Pkg; Pkg.resolve()"),
        ("instantiate", "import Pkg; Pkg.instantiate()"),
        ("precompile", "import Pkg; Pkg.precompile()"),
        ("load-graft", "using Graft"),
    )
        push!(baseline_phases,
              run_julia_phase(source_root, depot, logs, name, code))
    end

    umbrella_snapshot = snapshot_images(depot)
    required_packages = options.layout == "split" ? RUNTIME_PACKAGES : ["Graft"]
    assert_packages(umbrella_snapshot, required_packages, "umbrella baseline")
    umbrella_cache = joinpath(backups, "umbrella")
    save_cache_baseline(depot, umbrella_cache)

    mpi_snapshot = Dict{String,Any}()
    mpi_cache = joinpath(backups, "mpi")
    if any(target.load_mpi for target in options.targets)
        restore_cache_baseline(depot, umbrella_cache)
        push!(baseline_phases, run_julia_phase(
            source_root, depot, logs, "load-graft-mpi", "using Graft; using MPI"))
        mpi_snapshot = snapshot_images(depot)
        assert_packages(mpi_snapshot, vcat(RUNTIME_PACKAGES, MPI_EXTENSION_IMAGES),
                        "MPI baseline")
        save_cache_baseline(depot, mpi_cache)
    end

    results = Dict{String,Any}[]
    for target in options.targets
        baseline_cache = target.load_mpi ? mpi_cache : umbrella_cache
        reference_snapshot = target.load_mpi ? mpi_snapshot : umbrella_snapshot
        push!(results, measure_target(
            target, source_root, depot, logs, baseline_cache, reference_snapshot))
    end

    report = Dict{String,Any}(
        "schema_version" => 1,
        "generated_at_utc" => string(now(UTC)),
        "julia_version" => string(VERSION),
        "julia_executable" => joinpath(Sys.BINDIR, Base.julia_exename()),
        "layout" => options.layout,
        "live_source" => options.source_root,
        "live_source_was_modified" => false,
        "git_head" => git_read(options.source_root, "rev-parse", "HEAD"),
        "git_status" => git_read(options.source_root, "status", "--short", "--branch"),
        "cache_version" => CACHE_VERSION,
        "image_suffixes" => sort!(collect(IMAGE_SUFFIXES)),
        "baseline_phases" => baseline_phases,
        "targets" => results,
        "temporary_workspace" => options.keep_temp === nothing ? nothing : work_root,
    )
    write_reports(options, report)
    return report
end

function main(args::Vector{String})
    VERSION.major == 1 && VERSION.minor == 12 || error(
        "measure_monorepo_invalidation.jl requires Julia 1.12; got $(VERSION)")
    options = parse_options(args)
    if options.help
        usage()
        return nothing
    end
    validate_paths(options)
    if options.dry_run
        print_dry_run(options)
        return nothing
    end

    if options.keep_temp === nothing
        work_root = mktempdir(; prefix="graft-monorepo-invalidation-")
        try
            execute(options, work_root)
        finally
            ispath(work_root) && rm(work_root; recursive=true, force=true)
        end
    else
        mkpath(dirname(options.keep_temp))
        mkdir(options.keep_temp)
        execute(options, options.keep_temp)
    end
    println("wrote reports: $(options.tsv), $(options.json), $(options.text)")
    return nothing
end

main(ARGS)
