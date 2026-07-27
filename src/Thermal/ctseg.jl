"""
    FiniteModeAction(; beta, mu=0, static_interaction=0, n0=0,
                     bath_energies, bath_couplings,
                     boson_frequencies, boson_couplings,
                     orbital_convention=:spinless,
                     density_convention=:shifted)

Finite Anderson-Holstein action shared by Graft, dense ED, and CTSEG. The
fermionic and bosonic baths are kept as explicit real-frequency modes so the
Matsubara kernels can be generated without fitting.
"""
struct FiniteModeAction
    beta::Float64
    mu::Float64
    static_interaction::Float64
    n0::Float64
    bath_energies::Vector{Float64}
    bath_couplings::Vector{ComplexF64}
    boson_frequencies::Vector{Float64}
    boson_couplings::Vector{Float64}
    orbital_convention::Symbol
    density_convention::Symbol
end

function FiniteModeAction(; beta::Real,
                          mu::Real=0,
                          static_interaction::Real=0,
                          n0::Real=0,
                          bath_energies=Float64[],
                          bath_couplings=ComplexF64[],
                          boson_frequencies=Float64[],
                          boson_couplings=Float64[],
                          orbital_convention::Symbol=:spinless,
                          density_convention::Symbol=:shifted)
    β = Float64(beta)
    μ = Float64(mu)
    U = Float64(static_interaction)
    shift = Float64(n0)
    eps = Float64.(collect(bath_energies))
    hybridizations = ComplexF64.(collect(bath_couplings))
    omegas = Float64.(collect(boson_frequencies))
    couplings = Float64.(collect(boson_couplings))

    isfinite(β) && β > 0 ||
        throw(ArgumentError("beta must be finite and positive"))
    all(isfinite, (μ, U, shift)) ||
        throw(ArgumentError("mu, static_interaction, and n0 must be finite"))
    length(eps) == length(hybridizations) ||
        throw(ArgumentError("bath_energies and bath_couplings must have equal length"))
    length(omegas) == length(couplings) ||
        throw(ArgumentError("boson_frequencies and boson_couplings must have equal length"))
    all(isfinite, eps) ||
        throw(ArgumentError("bath energies must be finite"))
    all(z -> isfinite(real(z)) && isfinite(imag(z)), hybridizations) ||
        throw(ArgumentError("bath couplings must be finite"))
    all(isfinite, omegas) && all(>(0), omegas) ||
        throw(ArgumentError("boson frequencies must be finite and positive"))
    all(isfinite, couplings) ||
        throw(ArgumentError("boson couplings must be finite"))
    orbital_convention in (:spinless, :spinful) ||
        throw(ArgumentError("orbital_convention must be :spinless or :spinful"))
    density_convention in (:shifted, :unshifted) ||
        throw(ArgumentError("density_convention must be :shifted or :unshifted"))

    return FiniteModeAction(
        β, μ, U, shift, eps, hybridizations, omegas, couplings,
        orbital_convention, density_convention)
end

"""
    finite_mode_hash(action)

Stable FNV-1a fingerprint of every action parameter and convention. The hash
is an exchange-integrity check, not a cryptographic signature.
"""
function finite_mode_hash(action::FiniteModeAction)
    fields = String[
        bitstring(action.beta),
        bitstring(action.mu),
        bitstring(action.static_interaction),
        bitstring(action.n0),
        String(action.orbital_convention),
        String(action.density_convention),
    ]
    append!(fields, bitstring.(action.bath_energies))
    for coupling in action.bath_couplings
        push!(fields, bitstring(real(coupling)), bitstring(imag(coupling)))
    end
    append!(fields, bitstring.(action.boson_frequencies))
    append!(fields, bitstring.(action.boson_couplings))

    hash = UInt64(0xcbf29ce484222325)
    for byte in codeunits(join(fields, "|"))
        hash = (hash ⊻ UInt64(byte)) * UInt64(0x100000001b3)
    end
    return string(hash; base=16, pad=16)
end

fermionic_frequency(action::FiniteModeAction, n::Integer) =
    (2Int(n) + 1) * pi / action.beta

bosonic_frequency(action::FiniteModeAction, n::Integer) =
    2Int(n) * pi / action.beta

"""
    hybridization_iw(action, indices=0:15)

Exact finite-mode hybridization
`Delta(i*omega_n) = sum_p |V_p|^2 / (i*omega_n - epsilon_p)`.
"""
function hybridization_iw(action::FiniteModeAction, indices=0:15)
    ns = Int.(collect(indices))
    frequencies = fermionic_frequency.(Ref(action), ns)
    values = ComplexF64[
        sum(abs2(V) / (im * frequency - epsilon)
            for (epsilon, V) in zip(
                action.bath_energies, action.bath_couplings);
            init=0.0 + 0.0im)
        for frequency in frequencies
    ]
    return MatsubaraSeries(
        ns, frequencies, values,
        (; beta=action.beta, statistics=:fermionic,
           kernel=:hybridization, action_hash=finite_mode_hash(action)))
end

"""
    retarded_interaction_iv(action, indices=0:15)

Exact finite-mode retarded interaction
`U_ret(i*nu_n) = -sum_l 2*g_l^2*omega_l/(nu_n^2+omega_l^2)`.
"""
function retarded_interaction_iv(action::FiniteModeAction, indices=0:15)
    ns = Int.(collect(indices))
    frequencies = bosonic_frequency.(Ref(action), ns)
    values = ComplexF64[
        -sum(2g^2 * omega / (frequency^2 + omega^2)
             for (omega, g) in zip(
                 action.boson_frequencies, action.boson_couplings);
             init=0.0)
        for frequency in frequencies
    ]
    return MatsubaraSeries(
        ns, frequencies, values,
        (; beta=action.beta, statistics=:bosonic,
           kernel=:retarded_interaction, action_hash=finite_mode_hash(action)))
end

"""
    write_ctseg_input_csv(path, action; fermionic_indices=0:63,
                         bosonic_indices=0:63)

Write the exact finite-mode CTSEG kernels and all scalar conventions. Every
row repeats the action hash so independently sliced files remain auditable.
"""
function write_ctseg_input_csv(path::AbstractString,
                               action::FiniteModeAction;
                               fermionic_indices=0:63,
                               bosonic_indices=0:63)
    delta = hybridization_iw(action, fermionic_indices)
    interaction = retarded_interaction_iv(action, bosonic_indices)
    header = [
        "action_hash", "beta", "mu", "static_interaction", "n0",
        "orbital_convention", "density_convention", "bath_energies",
        "bath_couplings", "boson_frequencies", "boson_couplings",
        "kernel", "index", "frequency", "value_re", "value_im",
    ]
    open(path, "w") do io
        println(io, join(header, ','))
        _write_kernel_rows(io, action, :hybridization, delta)
        _write_kernel_rows(io, action, :retarded_interaction, interaction)
    end
    return path
end

function _write_kernel_rows(io, action, kernel, series)
    prefix = Any[
        finite_mode_hash(action), action.beta, action.mu,
        action.static_interaction, action.n0, action.orbital_convention,
        action.density_convention,
        join(action.bath_energies, ';'),
        join((_ctseg_complex_text(value)
              for value in action.bath_couplings), ';'),
        join(action.boson_frequencies, ';'),
        join(action.boson_couplings, ';'),
        kernel,
    ]
    for i in eachindex(series.indices)
        row = [
            prefix;
            series.indices[i];
            series.frequencies[i];
            real(series.values[i]);
            imag(series.values[i]);
        ]
        println(io, join(_csv_encode.(row), ','))
    end
    return nothing
end

_ctseg_complex_text(value) = string(real(value), ':', imag(value))

"""
    CTSEGMetadata

Metadata required before a stochastic CTSEG result may enter an acceptance
comparison. Sampling controls are retained so that readiness can be certified
from independent ensembles instead of asserted manually.
"""
struct CTSEGMetadata
    action_hash::String
    beta::Float64
    mu::Float64
    static_interaction::Float64
    n0::Float64
    orbital_convention::Symbol
    density_convention::Symbol
    endpoint_convention::Symbol
    density_connected::Bool
    fourier_convention::Symbol
    equilibrated::Bool
    error_stable::Bool
    nsamples::Int
    jackknife_bins::Int
    cycles_per_replica::Int
    warmup_cycles::Int
    length_cycle::Int
    seed::Int
end

function CTSEGMetadata(action::FiniteModeAction;
                       endpoint_convention::Symbol=:zero_plus,
                       density_connected::Bool=true,
                       fourier_convention::Symbol=:positive_exponential,
                       equilibrated::Bool,
                       error_stable::Bool,
                       nsamples::Integer,
                       jackknife_bins::Integer,
                       cycles_per_replica::Integer=cld(nsamples, jackknife_bins),
                       warmup_cycles::Integer=0,
                       length_cycle::Integer=1,
                       seed::Integer=0)
    endpoint_convention in (:zero_plus, :zero_minus) ||
        throw(ArgumentError("endpoint_convention must be :zero_plus or :zero_minus"))
    fourier_convention === :positive_exponential ||
        throw(ArgumentError("fourier_convention must be :positive_exponential"))
    nsamples > 0 || throw(ArgumentError("nsamples must be positive"))
    jackknife_bins > 1 ||
        throw(ArgumentError("jackknife_bins must exceed one"))
    cycles_per_replica > 0 ||
        throw(ArgumentError("cycles_per_replica must be positive"))
    warmup_cycles >= 0 ||
        throw(ArgumentError("warmup_cycles must be nonnegative"))
    length_cycle > 0 ||
        throw(ArgumentError("length_cycle must be positive"))
    nsamples == cycles_per_replica * jackknife_bins ||
        throw(ArgumentError(
            "nsamples must equal cycles_per_replica * jackknife_bins"))
    return CTSEGMetadata(
        finite_mode_hash(action), action.beta, action.mu,
        action.static_interaction, action.n0, action.orbital_convention,
        action.density_convention, endpoint_convention, density_connected,
        fourier_convention, equilibrated, error_stable,
        Int(nsamples), Int(jackknife_bins), Int(cycles_per_replica),
        Int(warmup_cycles), Int(length_cycle), Int(seed))
end

"""One scalar, imaginary-time, or Matsubara datum from CTSEG."""
struct CTSEGDatum
    observable::Symbol
    axis::Symbol
    coordinate::Float64
    mean::ComplexF64
    stderr::Float64
    function CTSEGDatum(observable::Symbol, axis::Symbol, coordinate::Real,
                        mean::Number, stderr::Real)
        axis in (:scalar, :tau, :fermionic_iw, :bosonic_iv) ||
            throw(ArgumentError("unsupported CTSEG axis $axis"))
        x = Float64(coordinate)
        error = Float64(stderr)
        isfinite(x) || throw(ArgumentError("datum coordinate must be finite"))
        isfinite(real(mean)) && isfinite(imag(mean)) ||
            throw(ArgumentError("datum mean must be finite"))
        isfinite(error) && error >= 0 ||
            throw(ArgumentError("datum stderr must be finite and nonnegative"))
        return new(observable, axis, x, ComplexF64(mean), error)
    end
end

struct CTSEGArtifact
    metadata::CTSEGMetadata
    data::Vector{CTSEGDatum}
    function CTSEGArtifact(metadata::CTSEGMetadata, data)
        values = CTSEGDatum[datum for datum in data]
        isempty(values) && throw(ArgumentError("CTSEG artifact has no data"))
        _check_unique_data(values)
        return new(metadata, values)
    end
end

"""Diagnostics used to certify an independent, longer CTSEG ensemble."""
struct CTSEGReadinessReport
    passed::Bool
    rows::Vector{NamedTuple}
    sigma::Float64
    max_sigma::Float64
    observed_outliers::Int
    allowed_outliers::Int
    sample_ratio::Float64
    observable_error_ratios::Dict{Symbol,Float64}
    error_ratio_limit::Float64
end

function _check_unique_data(data)
    seen = Set{Tuple{Symbol,Symbol,UInt64}}()
    for datum in data
        key = (datum.observable, datum.axis,
               reinterpret(UInt64, datum.coordinate))
        key in seen &&
            throw(ArgumentError("duplicate CTSEG datum $(key[1:2]) at $(datum.coordinate)"))
        push!(seen, key)
    end
    return nothing
end

"""
    ThermalBenchmarkDatum

Deterministic or sampled Graft/ED datum with a decomposed error budget.
`cutoff_error` is reserved for the residual boson-cutoff error and
`lbo_error` for a controlled PP bond truncation.
"""
struct ThermalBenchmarkDatum
    observable::Symbol
    axis::Symbol
    coordinate::Float64
    value::ComplexF64
    stderr::Float64
    deterministic_error::Float64
    cutoff_error::Float64
    lbo_error::Float64
    function ThermalBenchmarkDatum(observable::Symbol, axis::Symbol,
                                   coordinate::Real, value::Number;
                                   stderr::Real=0,
                                   deterministic_error::Real=0,
                                   cutoff_error::Real=0,
                                   lbo_error::Real=0)
        errors = Float64[
            stderr, deterministic_error, cutoff_error, lbo_error]
        all(x -> isfinite(x) && x >= 0, errors) ||
            throw(ArgumentError("benchmark errors must be finite and nonnegative"))
        return new(observable, axis, Float64(coordinate), ComplexF64(value),
                   errors...)
    end
end

struct CTSEGComparison
    passed::Bool
    rows::Vector{NamedTuple}
    sigma::Float64
    action_hash::String
end

"""
    assess_ctseg_readiness(short, long; kwargs...)

Compare independent CTSEG ensembles with identical actions and replica counts.
The longer ensemble must use a different seed, at least as much warmup, and
more samples. Mean stability is a global gate: a bounded fraction may exceed
`sigma` because a pointwise all-pass rule is invalid for a large grid, while
no point may exceed `max_sigma`. Error stability is checked per observable
against the expected inverse-square-root sample scaling.
"""
function assess_ctseg_readiness(
        short::CTSEGArtifact,
        long::CTSEGArtifact;
        sigma::Real=3,
        max_sigma::Real=4,
        max_outlier_fraction::Real=0.02,
        error_ratio_limit::Real=1.75,
        minimum_sample_ratio::Real=1.5)
    for (name, value) in (
            ("sigma", sigma),
            ("max_sigma", max_sigma),
            ("error_ratio_limit", error_ratio_limit),
            ("minimum_sample_ratio", minimum_sample_ratio))
        isfinite(value) && value > 0 ||
            throw(ArgumentError("$name must be finite and positive"))
    end
    max_sigma >= sigma ||
        throw(ArgumentError("max_sigma must be at least sigma"))
    isfinite(max_outlier_fraction) &&
        0 <= max_outlier_fraction < 1 ||
        throw(ArgumentError(
            "max_outlier_fraction must lie in [0, 1)"))

    _check_ctseg_metadata_match(short.metadata, long.metadata)
    short.metadata.jackknife_bins == long.metadata.jackknife_bins ||
        throw(ArgumentError(
            "CTSEG ensembles must use the same replica count"))
    short.metadata.length_cycle == long.metadata.length_cycle ||
        throw(ArgumentError(
            "CTSEG ensembles must use the same cycle length"))
    short.metadata.seed != long.metadata.seed ||
        throw(ArgumentError(
            "CTSEG ensembles must use independent seeds"))
    long.metadata.warmup_cycles >= short.metadata.warmup_cycles ||
        throw(ArgumentError(
            "long CTSEG ensemble must not use less warmup"))

    sample_ratio = long.metadata.nsamples / short.metadata.nsamples
    sample_ratio >= minimum_sample_ratio ||
        throw(ArgumentError(
            "long CTSEG ensemble does not meet minimum sample ratio"))
    long.metadata.cycles_per_replica >
        short.metadata.cycles_per_replica ||
        throw(ArgumentError(
            "long CTSEG ensemble must sample more cycles per replica"))

    short_data = _ctseg_data_map(short.data)
    long_data = _ctseg_data_map(long.data)
    keys(short_data) == keys(long_data) ||
        throw(ArgumentError(
            "CTSEG ensembles do not contain identical observable grids"))

    rows = NamedTuple[]
    ratios = Dict{Symbol,Vector{Float64}}()
    sigma_value = Float64(sigma)
    max_sigma_value = Float64(max_sigma)
    for key in sort!(collect(keys(short_data)); by=string)
        baseline = short_data[key]
        candidate = long_data[key]
        combined_error = hypot(baseline.stderr, candidate.stderr)
        delta = abs(baseline.mean - candidate.mean)
        zscore = combined_error == 0 ?
            (delta == 0 ? 0.0 : Inf) : delta / combined_error
        ratio = baseline.stderr == 0 ?
            (candidate.stderr == 0 ? 1.0 : Inf) :
            candidate.stderr / baseline.stderr
        push!(get!(ratios, candidate.observable, Float64[]), ratio)
        push!(rows, (;
            observable=candidate.observable,
            axis=candidate.axis,
            coordinate=candidate.coordinate,
            short_mean=baseline.mean,
            long_mean=candidate.mean,
            absolute_error=delta,
            combined_stderr=combined_error,
            zscore,
            error_ratio=ratio,
            outlier=zscore > sigma_value,
        ))
    end

    observed_outliers = count(row -> row.outlier, rows)
    allowed_outliers = ceil(Int, max_outlier_fraction * length(rows))
    mean_stable = observed_outliers <= allowed_outliers &&
        maximum(row -> row.zscore, rows) <= max_sigma_value
    expected_error_scale = inv(sqrt(sample_ratio))
    observable_error_ratios = Dict(
        observable => _ctseg_median(values) / expected_error_scale
        for (observable, values) in ratios)
    error_stable = all(
        ratio -> isfinite(ratio) && ratio <= error_ratio_limit,
        values(observable_error_ratios))

    return CTSEGReadinessReport(
        mean_stable && error_stable, rows, sigma_value, max_sigma_value,
        observed_outliers, allowed_outliers, sample_ratio,
        observable_error_ratios, Float64(error_ratio_limit))
end

"""
    certify_ctseg(short, long; kwargs...)

Return the longer artifact with readiness flags set only when
[`assess_ctseg_readiness`](@ref) passes.
"""
function certify_ctseg(short::CTSEGArtifact,
                       long::CTSEGArtifact;
                       kwargs...)
    report = assess_ctseg_readiness(short, long; kwargs...)
    report.passed || throw(ArgumentError(
        "CTSEG readiness failed: $(report.observed_outliers) > " *
        "$(report.allowed_outliers) mean outliers or unstable per-observable errors"))
    metadata = long.metadata
    certified = CTSEGMetadata(
        metadata.action_hash, metadata.beta, metadata.mu,
        metadata.static_interaction, metadata.n0,
        metadata.orbital_convention, metadata.density_convention,
        metadata.endpoint_convention, metadata.density_connected,
        metadata.fourier_convention, true, true, metadata.nsamples,
        metadata.jackknife_bins, metadata.cycles_per_replica,
        metadata.warmup_cycles, metadata.length_cycle, metadata.seed)
    return CTSEGArtifact(certified, long.data)
end

function _check_ctseg_metadata_match(left, right)
    fields = (
        :action_hash, :beta, :mu, :static_interaction, :n0,
        :orbital_convention, :density_convention, :endpoint_convention,
        :density_connected, :fourier_convention)
    all(field -> getfield(left, field) == getfield(right, field), fields) ||
        throw(ArgumentError(
            "CTSEG ensembles use different actions or conventions"))
    return nothing
end

function _ctseg_data_map(data)
    return Dict(
        (datum.observable, datum.axis,
         reinterpret(UInt64, datum.coordinate)) => datum
        for datum in data)
end

function _ctseg_median(values)
    sorted = sort(values)
    midpoint = length(sorted) ÷ 2
    return isodd(length(sorted)) ? sorted[midpoint + 1] :
        (sorted[midpoint] + sorted[midpoint + 1]) / 2
end

"""
    compare_ctseg(reference, artifact; sigma=3, coordinate_atol=0)

Compare common observables only after action/convention and Monte Carlo
readiness checks. The pointwise gate is

`abs(reference-ctseg) <= sigma*hypot(reference.stderr, ctseg.stderr)
                         + deterministic_error + cutoff_error + lbo_error`.
"""
function compare_ctseg(reference,
                       artifact::CTSEGArtifact;
                       action::Union{Nothing,FiniteModeAction}=nothing,
                       sigma::Real=3,
                       coordinate_atol::Real=0)
    metadata = artifact.metadata
    metadata.equilibrated ||
        throw(ArgumentError("CTSEG artifact is not marked equilibrated"))
    metadata.error_stable ||
        throw(ArgumentError("CTSEG artifact error bars are not marked stable"))
    sigma_value = Float64(sigma)
    isfinite(sigma_value) && sigma_value > 0 ||
        throw(ArgumentError("sigma must be finite and positive"))
    atol = Float64(coordinate_atol)
    isfinite(atol) && atol >= 0 ||
        throw(ArgumentError("coordinate_atol must be finite and nonnegative"))
    if action !== nothing
        _check_metadata(action, metadata)
    end

    refs = ThermalBenchmarkDatum[datum for datum in reference]
    isempty(refs) && throw(ArgumentError("reference has no data"))
    rows = NamedTuple[]
    for datum in artifact.data
        matches = filter(refs) do ref
            ref.observable == datum.observable &&
                ref.axis == datum.axis &&
                abs(ref.coordinate - datum.coordinate) <= atol
        end
        isempty(matches) &&
            throw(ArgumentError(
                "missing reference datum $(datum.observable)/$(datum.axis) at $(datum.coordinate)"))
        length(matches) == 1 ||
            throw(ArgumentError(
                "ambiguous reference datum $(datum.observable)/$(datum.axis) at $(datum.coordinate)"))
        ref = only(matches)
        delta = abs(ref.value - datum.mean)
        statistical = sigma_value * hypot(ref.stderr, datum.stderr)
        threshold = statistical + ref.deterministic_error +
            ref.cutoff_error + ref.lbo_error
        push!(rows, (;
            observable=datum.observable,
            axis=datum.axis,
            coordinate=datum.coordinate,
            reference=ref.value,
            ctseg=datum.mean,
            absolute_error=delta,
            statistical_allowance=statistical,
            deterministic_error=ref.deterministic_error,
            cutoff_error=ref.cutoff_error,
            lbo_error=ref.lbo_error,
            threshold,
            passed=delta <= threshold,
        ))
    end
    return CTSEGComparison(
        all(row -> row.passed, rows), rows, sigma_value, metadata.action_hash)
end

function _check_metadata(action, metadata)
    finite_mode_hash(action) == metadata.action_hash ||
        throw(ArgumentError("CTSEG action hash does not match the reference action"))
    action.beta == metadata.beta ||
        throw(ArgumentError("CTSEG beta does not match the reference action"))
    action.mu == metadata.mu ||
        throw(ArgumentError("CTSEG mu does not match the reference action"))
    action.static_interaction == metadata.static_interaction ||
        throw(ArgumentError("CTSEG static interaction does not match the reference action"))
    action.n0 == metadata.n0 ||
        throw(ArgumentError("CTSEG density shift does not match the reference action"))
    action.orbital_convention == metadata.orbital_convention ||
        throw(ArgumentError("CTSEG orbital convention does not match the reference action"))
    action.density_convention == metadata.density_convention ||
        throw(ArgumentError("CTSEG density convention does not match the reference action"))
    return nothing
end

"""
    write_ctseg_results_csv(path, artifact)
    read_ctseg_results_csv(path)

Strict CSV exchange for CTSEG means, standard errors, conventions, and
jackknife readiness metadata.
"""
function write_ctseg_results_csv(path::AbstractString,
                                 artifact::CTSEGArtifact)
    header = [
        "action_hash", "beta", "mu", "static_interaction", "n0",
        "orbital_convention", "density_convention", "endpoint_convention",
        "density_connected", "fourier_convention", "equilibrated",
        "error_stable", "nsamples", "jackknife_bins",
        "cycles_per_replica", "warmup_cycles", "length_cycle", "seed",
        "observable", "axis", "coordinate", "mean_re", "mean_im", "stderr",
    ]
    metadata = artifact.metadata
    prefix = Any[
        metadata.action_hash, metadata.beta, metadata.mu,
        metadata.static_interaction, metadata.n0,
        metadata.orbital_convention, metadata.density_convention,
        metadata.endpoint_convention, metadata.density_connected,
        metadata.fourier_convention, metadata.equilibrated,
        metadata.error_stable, metadata.nsamples, metadata.jackknife_bins,
        metadata.cycles_per_replica, metadata.warmup_cycles,
        metadata.length_cycle, metadata.seed,
    ]
    open(path, "w") do io
        println(io, join(header, ','))
        for datum in artifact.data
            row = [
                prefix;
                datum.observable;
                datum.axis;
                datum.coordinate;
                real(datum.mean);
                imag(datum.mean);
                datum.stderr;
            ]
            println(io, join(_csv_encode.(row), ','))
        end
    end
    return path
end

function read_ctseg_results_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("empty CTSEG CSV artifact"))
    header = _csv_decode(first(lines))
    expected = [
        "action_hash", "beta", "mu", "static_interaction", "n0",
        "orbital_convention", "density_convention", "endpoint_convention",
        "density_connected", "fourier_convention", "equilibrated",
        "error_stable", "nsamples", "jackknife_bins",
        "cycles_per_replica", "warmup_cycles", "length_cycle", "seed",
        "observable", "axis", "coordinate", "mean_re", "mean_im", "stderr",
    ]
    header == expected ||
        throw(ArgumentError("unexpected CTSEG CSV header"))
    length(lines) > 1 || throw(ArgumentError("CTSEG CSV artifact has no data"))

    rows = [_csv_decode(line) for line in lines[2:end] if !isempty(strip(line))]
    all(row -> length(row) == length(expected), rows) ||
        throw(ArgumentError("malformed CTSEG CSV row"))
    isempty(rows) && throw(ArgumentError("CTSEG CSV artifact has no data"))
    firstrow = first(rows)
    metadata = CTSEGMetadata(
        firstrow[1],
        _parse_float(firstrow[2], "beta"),
        _parse_float(firstrow[3], "mu"),
        _parse_float(firstrow[4], "static_interaction"),
        _parse_float(firstrow[5], "n0"),
        Symbol(firstrow[6]),
        Symbol(firstrow[7]),
        Symbol(firstrow[8]),
        _parse_bool(firstrow[9], "density_connected"),
        Symbol(firstrow[10]),
        _parse_bool(firstrow[11], "equilibrated"),
        _parse_bool(firstrow[12], "error_stable"),
        _parse_int(firstrow[13], "nsamples"),
        _parse_int(firstrow[14], "jackknife_bins"),
        _parse_int(firstrow[15], "cycles_per_replica"),
        _parse_int(firstrow[16], "warmup_cycles"),
        _parse_int(firstrow[17], "length_cycle"),
        _parse_int(firstrow[18], "seed"),
    )
    metadata.nsamples > 0 ||
        throw(ArgumentError("CTSEG nsamples must be positive"))
    metadata.jackknife_bins > 1 ||
        throw(ArgumentError("CTSEG jackknife_bins must exceed one"))
    metadata.cycles_per_replica > 0 ||
        throw(ArgumentError("CTSEG cycles_per_replica must be positive"))
    metadata.warmup_cycles >= 0 ||
        throw(ArgumentError("CTSEG warmup_cycles must be nonnegative"))
    metadata.length_cycle > 0 ||
        throw(ArgumentError("CTSEG length_cycle must be positive"))
    metadata.nsamples ==
        metadata.cycles_per_replica * metadata.jackknife_bins ||
        throw(ArgumentError(
            "CTSEG nsamples does not match replica sampling metadata"))

    data = CTSEGDatum[]
    for row in rows
        _row_metadata(row) == _row_metadata(firstrow) ||
            throw(ArgumentError("inconsistent metadata across CTSEG CSV rows"))
        push!(data, CTSEGDatum(
            Symbol(row[19]), Symbol(row[20]),
            _parse_float(row[21], "coordinate"),
            complex(_parse_float(row[22], "mean_re"),
                    _parse_float(row[23], "mean_im")),
            _parse_float(row[24], "stderr")))
    end
    return CTSEGArtifact(metadata, data)
end

_row_metadata(row) = row[1:18]

function _parse_float(value, name)
    parsed = tryparse(Float64, value)
    parsed === nothing &&
        throw(ArgumentError("invalid $name in CTSEG CSV"))
    return parsed
end

function _parse_int(value, name)
    parsed = tryparse(Int, value)
    parsed === nothing &&
        throw(ArgumentError("invalid $name in CTSEG CSV"))
    return parsed
end

function _parse_bool(value, name)
    value == "true" && return true
    value == "false" && return false
    throw(ArgumentError("invalid $name in CTSEG CSV"))
end

function _csv_encode(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) ||
            occursin('\n', text) || occursin('\r', text)
        return '"' * replace(text, "\"" => "\"\"") * '"'
    end
    return text
end

function _csv_decode(line::AbstractString)
    fields = String[]
    buffer = IOBuffer()
    quoted = false
    i = firstindex(line)
    while i <= lastindex(line)
        char = line[i]
        if quoted
            if char == '"'
                next = nextind(line, i)
                if next <= lastindex(line) && line[next] == '"'
                    write(buffer, '"')
                    i = next
                else
                    quoted = false
                end
            else
                write(buffer, char)
            end
        elseif char == ','
            push!(fields, String(take!(buffer)))
        elseif char == '"'
            position(buffer) == 0 ||
                throw(ArgumentError("malformed quote in CTSEG CSV"))
            quoted = true
        else
            write(buffer, char)
        end
        i = nextind(line, i)
    end
    quoted && throw(ArgumentError("unterminated quote in CTSEG CSV"))
    push!(fields, String(take!(buffer)))
    return fields
end
