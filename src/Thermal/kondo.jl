"""
    KondoScalingResult

Low-temperature scaling analysis of double-occupancy curves. `scales` are
defined from `1-D(T)/D(0) = (T/TK)^2` up to one common universal prefactor;
that common factor changes only `prefactor`, not the interaction exponent.
"""
struct KondoScalingResult
    interactions::Vector{Float64}
    scales::Vector{Float64}
    prefactor::Float64
    exponent::Float64
    r2::Float64
    quadratic_slopes::Vector{Float64}
    diagnostics::NamedTuple
end

"""Pointwise validation report for a discrete semicircular bath."""
struct SemicircularBathReport
    accepted::Bool
    max_error::Float64
    worst_frequency::Float64
    tolerance::Float64
    omega_min::Float64
    omega_max::Float64
    npoints::Int
    nsites::Int
end

"""
    gauss_semicircular_bath(nsites; half_bandwidth=1)

Gauss-Chebyshev discretization of the normalized semicircular density. This is
useful for finite-temperature smoke tests; paper-scale low-frequency Kondo
acceptance must use `validate_semicircular_bath` on an adaptive bath.
"""
function gauss_semicircular_bath(
        nsites::Integer; half_bandwidth::Real=1)
    nsites >= 1 || throw(ArgumentError("nsites must be positive"))
    D = Float64(half_bandwidth)
    isfinite(D) && D > 0 ||
        throw(ArgumentError("half_bandwidth must be finite and positive"))
    angles = [j * pi / (nsites + 1) for j in 1:nsites]
    energies = D .* cos.(angles)
    weights = 2 .* sin.(angles).^2 ./ (nsites + 1)
    return energies, sqrt.(weights)
end

"""
    semicircular_hybridization(z; half_bandwidth=1)

Analytic Stieltjes transform of
`rho(epsilon)=2*sqrt(D^2-epsilon^2)/(pi*D^2)` on `[-D,D]`.
The square-root branch is selected so the result behaves as `1/z`.
On the real axis inside the band the retarded (`+i0`) prescription is
returned, so `Im G < 0` for both signs of `Re z`.
"""
function semicircular_hybridization(z::Number; half_bandwidth::Real=1)
    D = Float64(half_bandwidth)
    isfinite(D) && D > 0 ||
        throw(ArgumentError("half_bandwidth must be finite and positive"))
    value = ComplexF64(z)
    root = sqrt(value^2 - D^2)
    if !iszero(imag(value))
        signbit(imag(root)) == signbit(imag(value)) || (root = -root)
    elseif abs(real(value)) < D
        signbit(imag(root)) && (root = -root)
    elseif !iszero(real(value))
        signbit(real(root)) == signbit(real(value)) || (root = -root)
    end
    return 2 * (value - root) / D^2
end

"""
    discrete_bath_hybridization(z, energies, couplings)

Hybridization of a real-pole bath,
`sum_j |V_j|^2/(z-epsilon_j)`.
"""
function discrete_bath_hybridization(z::Number, energies, couplings)
    epsilons = Float64.(collect(energies))
    values = ComplexF64.(collect(couplings))
    length(epsilons) == length(values) ||
        throw(ArgumentError("bath energies and couplings must have equal length"))
    isempty(epsilons) && throw(ArgumentError("bath must contain at least one site"))
    all(isfinite, epsilons) ||
        throw(ArgumentError("bath energies must be finite"))
    all(v -> isfinite(real(v)) && isfinite(imag(v)), values) ||
        throw(ArgumentError("bath couplings must be finite"))
    return sum(abs2(V) / (ComplexF64(z) - epsilon)
               for (epsilon, V) in zip(epsilons, values))
end

"""
    validate_semicircular_bath(energies, couplings;
                              omega_min=pi/1024, omega_max=100,
                              npoints=4096, tolerance=1e-6,
                              half_bandwidth=1)

Validate the pointwise absolute hybridization error on a logarithmically dense
positive imaginary-frequency grid. This is the paper-scale Kondo bath gate.
"""
function validate_semicircular_bath(
        energies, couplings;
        omega_min::Real=pi / 1024,
        omega_max::Real=100,
        npoints::Integer=4096,
        tolerance::Real=1e-6,
        half_bandwidth::Real=1)
    epsilons = Float64.(collect(energies))
    values = ComplexF64.(collect(couplings))
    length(epsilons) == length(values) ||
        throw(ArgumentError("bath energies and couplings must have equal length"))
    isempty(epsilons) && throw(ArgumentError("bath must contain at least one site"))
    lower = Float64(omega_min)
    upper = Float64(omega_max)
    tol = Float64(tolerance)
    isfinite(lower) && lower > 0 ||
        throw(ArgumentError("omega_min must be finite and positive"))
    isfinite(upper) && upper >= lower ||
        throw(ArgumentError("omega_max must be finite and at least omega_min"))
    npoints >= 2 || throw(ArgumentError("npoints must be at least two"))
    isfinite(tol) && tol > 0 ||
        throw(ArgumentError("tolerance must be finite and positive"))
    frequencies = exp.(range(log(lower), log(upper); length=Int(npoints)))
    errors = [
        abs(discrete_bath_hybridization(im * omega, epsilons, values) -
            semicircular_hybridization(
                im * omega; half_bandwidth))
        for omega in frequencies
    ]
    index = argmax(errors)
    maximum_error = Float64(errors[index])
    return SemicircularBathReport(
        maximum_error <= tol, maximum_error, frequencies[index], tol,
        lower, upper, Int(npoints), length(epsilons))
end

"""
    read_bath_csv(path)

Read a strict bath artifact with header
`energy,coupling_re,coupling_im`. Empty lines are ignored.
"""
function read_bath_csv(path::AbstractString)
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    isempty(lines) && throw(ArgumentError("empty bath CSV"))
    _csv_decode(first(lines)) ==
        ["energy", "coupling_re", "coupling_im"] ||
        throw(ArgumentError(
            "bath CSV header must be energy,coupling_re,coupling_im"))
    length(lines) > 1 || throw(ArgumentError("bath CSV has no poles"))
    energies = Float64[]
    couplings = ComplexF64[]
    for line in lines[2:end]
        row = _csv_decode(line)
        length(row) == 3 || throw(ArgumentError("malformed bath CSV row"))
        energy = tryparse(Float64, row[1])
        real_part = tryparse(Float64, row[2])
        imag_part = tryparse(Float64, row[3])
        any(isnothing, (energy, real_part, imag_part)) &&
            throw(ArgumentError("non-numeric bath CSV row"))
        push!(energies, energy)
        push!(couplings, complex(real_part, imag_part))
    end
    all(isfinite, energies) ||
        throw(ArgumentError("bath energies must be finite"))
    all(value -> isfinite(real(value)) && isfinite(imag(value)), couplings) ||
        throw(ArgumentError("bath couplings must be finite"))
    return energies, couplings
end

"""
    fit_kondo_scaling(interactions, temperatures, double_occupancies,
                      ground_double_occupancies; low_points=3)

For each interaction, fit the low-temperature Fermi-liquid form
`1-D(T)/D(0) = b(U) T^2` through the origin and define a relative Kondo scale
`TK=1/sqrt(b)`. Then fit `TK(U)=prefactor*exp(-exponent*U)`.

`double_occupancies` may be a matrix with one row per interaction or a vector
of vectors. Temperatures are shared by all rows and are sorted internally.
"""
function fit_kondo_scaling(interactions, temperatures, double_occupancies,
                           ground_double_occupancies;
                           low_points::Integer=3)
    us = Float64.(collect(interactions))
    ts = Float64.(collect(temperatures))
    d0 = Float64.(collect(ground_double_occupancies))
    length(us) >= 2 || throw(ArgumentError("Kondo scaling needs at least two interactions"))
    length(d0) == length(us) ||
        throw(ArgumentError("need one ground-state double occupancy per interaction"))
    all(t -> isfinite(t) && t > 0, ts) ||
        throw(ArgumentError("temperatures must be finite and positive"))
    low_points >= 2 || throw(ArgumentError("low_points must be at least two"))
    low_points <= length(ts) ||
        throw(ArgumentError("low_points exceeds the temperature grid"))
    curves = _kondo_curves(double_occupancies, length(us), length(ts))
    order = sortperm(ts)
    selected = order[1:Int(low_points)]
    x = abs2.(ts[selected])
    slopes = Vector{Float64}(undef, length(us))
    fit_errors = Vector{Float64}(undef, length(us))
    for i in eachindex(us)
        d0[i] > 0 || throw(ArgumentError("D(0) must be positive at U=$(us[i])"))
        y = 1 .- curves[i][selected] ./ d0[i]
        all(isfinite, y) || throw(ArgumentError("nonfinite double occupancy at U=$(us[i])"))
        slope = dot(x, y) / dot(x, x)
        slope > 0 || throw(ArgumentError(
            "low-temperature curve at U=$(us[i]) has nonpositive quadratic slope"))
        slopes[i] = slope
        fit_errors[i] = norm(y .- slope .* x) / max(norm(y), eps(Float64))
    end
    scales = inv.(sqrt.(slopes))
    design = hcat(ones(length(us)), us)
    coefficients = design \ log.(scales)
    fitted_logs = design * coefficients
    residual = log.(scales) .- fitted_logs
    total = log.(scales) .- sum(log.(scales)) / length(scales)
    r2 = 1 - sum(abs2, residual) / max(sum(abs2, total), eps(Float64))
    diagnostics = (;
        low_points=Int(low_points),
        selected_temperatures=ts[selected],
        curve_relative_errors=fit_errors,
        log_fit_residuals=residual,
    )
    return KondoScalingResult(
        us, scales, exp(coefficients[1]), -coefficients[2], r2,
        slopes, diagnostics)
end

function _kondo_curves(values::AbstractMatrix, ninteractions, ntemperatures)
    size(values) == (ninteractions, ntemperatures) ||
        throw(ArgumentError("double-occupancy matrix has the wrong shape"))
    return [Float64.(collect(values[i, :])) for i in 1:ninteractions]
end

function _kondo_curves(values, ninteractions, ntemperatures)
    length(values) == ninteractions ||
        throw(ArgumentError("need one double-occupancy curve per interaction"))
    curves = [Float64.(collect(curve)) for curve in values]
    all(length(curve) == ntemperatures for curve in curves) ||
        throw(ArgumentError("every double-occupancy curve must match temperatures"))
    return curves
end
