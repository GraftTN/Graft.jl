#!/usr/bin/env julia

using SHA
using TOML
using Libdl

const REPOSITORY = normpath(joinpath(@__DIR__, ".."))
const PROJECT_FILE = joinpath(REPOSITORY, "Project.toml")
const MANIFEST_FILE = joinpath(REPOSITORY, "Manifest.toml")

function project_fingerprint()
    digest = SHA.sha256(vcat(read(PROJECT_FILE), read(MANIFEST_FILE)))
    return bytes2hex(digest)
end

function source_fingerprint()
    files = String[PROJECT_FILE, MANIFEST_FILE]
    for (directory, _, names) in walkdir(joinpath(REPOSITORY, "src"))
        for name in names
            endswith(name, ".jl") && push!(files, joinpath(directory, name))
        end
    end
    sort!(files)
    payload = UInt8[]
    for path in files
        append!(payload, codeunits(relpath(path, REPOSITORY)))
        push!(payload, 0x00)
        append!(payload, read(path))
        push!(payload, 0x00)
    end
    return bytes2hex(SHA.sha256(payload))
end

function source_commit()
    try
        return readchomp(`git -C $REPOSITORY rev-parse HEAD`)
    catch
        return "unknown"
    end
end

function identities()
    project = TOML.parsefile(PROJECT_FILE)
    return Dict(
        "format_version" => 1,
        "package" => "Graft",
        "package_version" => get(project, "version", "unknown"),
        "julia_version" => string(VERSION),
        "machine" => string(Sys.MACHINE),
        "cpu_target" => "generic",
        "project_fingerprint_sha256" => project_fingerprint(),
        "source_fingerprint_sha256" => source_fingerprint(),
        "source_commit" => source_commit(),
        "full_precompile_workload" => true,
        "contains_checkpoint_data" => false,
        "contains_mpi_context" => false,
        "contains_application_data" => false,
    )
end

function print_check()
    available = Base.find_package("PackageCompiler") !== nothing
    println(
        "GRAFT_SYSIMAGE_CHECK ",
        "packagecompiler_available=$available ",
        "julia_version=$(VERSION) ",
        "machine=$(Sys.MACHINE) ",
        "project_fingerprint_sha256=$(project_fingerprint()) ",
        "source_fingerprint_sha256=$(source_fingerprint()) ",
        "source_commit=$(source_commit())",
    )
    return available
end

function verify_metadata(path)
    isfile(path) || error("sysimage metadata does not exist: $path")
    recorded = TOML.parsefile(path)
    expected = identities()
    mismatches = String[]
    for key in (
        "format_version",
        "package",
        "package_version",
        "julia_version",
        "machine",
        "cpu_target",
        "project_fingerprint_sha256",
        "source_fingerprint_sha256",
        "source_commit",
    )
        get(recorded, key, nothing) == expected[key] ||
            push!(mismatches,
                  "$key=$(repr(get(recorded, key, nothing))) expected $(repr(expected[key]))")
    end
    isempty(mismatches) ||
        error("sysimage compatibility check failed before solver execution: " *
              join(mismatches, "; "))
    println(
        "GRAFT_SYSIMAGE_VERIFY compatible=true metadata=$(abspath(path)) ",
        "julia_version=$(VERSION) machine=$(Sys.MACHINE)",
    )
    return nothing
end

function build_sysimage(path)
    Base.find_package("PackageCompiler") !== nothing ||
        error("PackageCompiler is unavailable; install it in a dedicated " *
              "builder environment, not as a Graft runtime dependency")
    output = abspath(path)
    endswith(output, "." * Libdl.dlext) ||
        error("sysimage output must end in .$(Libdl.dlext)")
    ispath(output) &&
        error("refusing to overwrite existing sysimage: $output")
    metadata_path = output * ".toml"
    ispath(metadata_path) &&
        error("refusing to overwrite existing metadata: $metadata_path")
    mkpath(dirname(output))

    # The package-owned full workload contains deterministic in-memory
    # fixtures only. Checkpoints, communicators, machine paths, and application
    # data are deliberately absent.
    ENV["GRAFT_FULL_PRECOMPILE"] = "true"
    @eval import PackageCompiler
    PackageCompiler.create_sysimage(
        [:Graft];
        project=REPOSITORY,
        sysimage_path=output,
        incremental=false,
        cpu_target="generic",
    )
    open(metadata_path, "w") do io
        TOML.print(io, identities(); sorted=true)
    end
    println(
        "GRAFT_SYSIMAGE_BUILD sysimage=$output metadata=$metadata_path ",
        "project_fingerprint_sha256=$(project_fingerprint())",
        " source_fingerprint_sha256=$(source_fingerprint())",
    )
    return nothing
end

function usage()
    println(stderr, """
Usage:
  julia --project=. benchmark/sysimage_builder.jl --check
  julia --project=. benchmark/sysimage_builder.jl --verify METADATA.toml
  julia --project=. benchmark/sysimage_builder.jl --build OUTPUT.$(Libdl.dlext)
""")
end

function main(args)
    if args == ["--check"]
        print_check()
    elseif length(args) == 2 && args[1] == "--verify"
        verify_metadata(args[2])
    elseif length(args) == 2 && args[1] == "--build"
        build_sysimage(args[2])
    else
        usage()
        return 2
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
