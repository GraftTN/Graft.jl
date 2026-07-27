"""
    MatsubaraSeries

Discrete imaginary-frequency values with integer Matsubara indices and
explicit statistics metadata.
"""
struct MatsubaraSeries{R<:Real,V,M<:NamedTuple}
    indices::Vector{Int}
    frequencies::Vector{R}
    values::Vector{V}
    metadata::M
end

Base.length(series::MatsubaraSeries) = length(series.indices)
Base.getindex(series::MatsubaraSeries, i::Int) =
    (series.frequencies[i], series.values[i])
Base.iterate(series::MatsubaraSeries, state...) =
    iterate(zip(series.frequencies, series.values), state...)

"""
    matsubara_transform(series; statistics, indices=0:15,
                        beta=nothing, weights=nothing)

Transform an imaginary-time `CorrelatorSeries` with
`integral_0^beta exp(im*omega_n*tau) C(tau) dtau`.

`statistics=:fermionic` uses `omega_n=(2n+1)pi/beta`;
`:bosonic` uses `omega_n=2n*pi/beta`. Without explicit quadrature `weights`,
the time grid must include both endpoints and nonuniform trapezoid weights are
used. Explicit weights support Gauss and other open quadratures.
"""
function matsubara_transform(series::CorrelatorSeries;
                             statistics::Symbol,
                             indices=0:15,
                             beta=nothing,
                             weights=nothing)
    statistics in (:fermionic, :bosonic) ||
        throw(ArgumentError("statistics must be :fermionic or :bosonic"))
    beta_value = beta === nothing ? _series_beta(series) : Float64(beta)
    isfinite(beta_value) && beta_value > 0 ||
        throw(ArgumentError("beta must be finite and positive"))
    times = Float64.(series.times)
    all(diff(times) .> 0) ||
        throw(ArgumentError("imaginary-time grid must be strictly increasing"))
    quadrature_weights = weights === nothing ?
        _trapezoid_weights(times, beta_value) : Float64.(collect(weights))
    length(quadrature_weights) == length(times) ||
        throw(ArgumentError("quadrature needs one weight per time point"))
    all(isfinite, quadrature_weights) ||
        throw(ArgumentError("quadrature weights must be finite"))

    ns = Int.(collect(indices))
    frequencies = statistics === :fermionic ?
        [(2n + 1) * pi / beta_value for n in ns] :
        [2n * pi / beta_value for n in ns]
    transformed = [
        _weighted_fourier(series.values, times, quadrature_weights, omega)
        for omega in frequencies
    ]
    metadata = (;
        beta=beta_value,
        statistics,
        quadrature=weights === nothing ? :trapezoid : :explicit,
        Nt=length(times),
    )
    return MatsubaraSeries(ns, frequencies, transformed, metadata)
end

function _series_beta(series::CorrelatorSeries)
    haskey(series.metadata, :beta) ||
        throw(ArgumentError("beta is absent from series metadata; pass beta explicitly"))
    return Float64(series.metadata.beta)
end

function _trapezoid_weights(times, beta)
    length(times) >= 2 ||
        throw(ArgumentError("trapezoid transform needs at least two time points"))
    isapprox(first(times), 0.0; atol=100eps(Float64), rtol=0) ||
        throw(ArgumentError("trapezoid time grid must start at 0"))
    isapprox(last(times), beta; atol=100eps(max(beta, 1.0)), rtol=0) ||
        throw(ArgumentError("trapezoid time grid must end at beta"))
    weights = Vector{Float64}(undef, length(times))
    weights[1] = (times[2] - times[1]) / 2
    weights[end] = (times[end] - times[end - 1]) / 2
    for i in 2:(length(times) - 1)
        weights[i] = (times[i + 1] - times[i - 1]) / 2
    end
    return weights
end

function _weighted_fourier(values, times, weights, omega)
    isempty(values) && throw(ArgumentError("cannot transform an empty series"))
    result = weights[1] * exp(im * omega * times[1]) * values[1]
    for i in 2:length(values)
        result += weights[i] * exp(im * omega * times[i]) * values[i]
    end
    return result
end
