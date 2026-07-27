"""
    LinearPredictionResult

Autoregressive continuation model. `coefficients[j]` multiplies the sample
`j` steps into the past, with the most recent sample first.
"""
struct LinearPredictionResult{T<:Number,S,D<:NamedTuple}
    coefficients::Vector{T}
    samples::Vector{S}
    dt::Float64
    t0::Float64
    diagnostics::D
end

function _lp_design(values::AbstractMatrix, order::Int)
    n, nchannels = size(values)
    n > order || throw(ArgumentError("linear prediction needs more samples than order"))
    rows = (n - order) * nchannels
    X = Matrix{ComplexF64}(undef, rows, order)
    y = Vector{ComplexF64}(undef, rows)
    row = 1
    for i in (order + 1):n, channel in 1:nchannels
        for j in 1:order
            X[row, j] = values[i - j, channel]
        end
        y[row] = values[i, channel]
        row += 1
    end
    return X, y
end

"""
    linear_prediction(times, samples; order, regularization=0)

Fit a shared autoregressive model to scalar or array-valued samples. A positive
`regularization` applies ridge stabilization to noisy or nearly dependent
histories.
"""
function linear_prediction(times::AbstractVector{<:Real}, samples;
                           order::Integer,
                           regularization::Real=0)
    length(times) == length(samples) ||
        throw(ArgumentError("times and samples must have equal length"))
    order >= 1 || throw(ArgumentError("order must be positive"))
    regularization >= 0 || throw(ArgumentError("regularization must be nonnegative"))
    dt = _uniform_times(times)
    values, _ = _sample_matrix(samples)
    X, y = _lp_design(values, Int(order))
    coefficients = if iszero(regularization)
        X \ y
    else
        (X' * X + regularization * I) \ (X' * y)
    end
    residual = X * coefficients - y
    diagnostics = (;
        method=:linear_prediction,
        order=Int(order),
        regularization=Float64(regularization),
        fit_rank=rank(X),
        condition_number=cond(X),
        l2err=norm(residual) / max(norm(y), eps(Float64)),
        dt,
        Nt=length(times),
    )
    return LinearPredictionResult(
        collect(coefficients), collect(samples), dt, Float64(first(times)),
        diagnostics)
end

linear_prediction(series::CorrelatorSeries; kwargs...) =
    linear_prediction(series.times, series.values; kwargs...)

"""
    predict(model, nfuture)

Return only the `nfuture` extrapolated samples, preserving the shape and element
type of the input samples.
"""
function predict(model::LinearPredictionResult, nfuture::Integer)
    nfuture >= 0 || throw(ArgumentError("nfuture must be nonnegative"))
    coefficient_type = eltype(model.coefficients)
    history = [one(coefficient_type) * sample for sample in model.samples]
    order = length(model.coefficients)
    length(history) >= order ||
        throw(ArgumentError("model history is shorter than its order"))
    output = Vector{eltype(history)}(undef, nfuture)
    for i in 1:nfuture
        value = model.coefficients[1] * history[end]
        for j in 2:order
            value += model.coefficients[j] * history[end - j + 1]
        end
        push!(history, value)
        output[i] = value
    end
    return output
end

"""
    exponential_sum(model; stabilize=false)

Convert a linear-prediction recurrence to the shared decay-pole convention.
With `stabilize=true`, roots outside the unit disk are projected radially onto
the unit circle before weights are refit; the modified root indices are stored
in diagnostics.
"""
function exponential_sum(model::LinearPredictionResult;
                         stabilize::Bool=false)
    order = length(model.coefficients)
    companion = zeros(ComplexF64, order, order)
    companion[1, :] .= model.coefficients
    for i in 2:order
        companion[i, i - 1] = 1
    end
    roots = eigvals(companion)
    unstable = findall(x -> abs(x) > 1 + sqrt(eps(Float64)), roots)
    if stabilize
        for i in unstable
            roots[i] /= abs(roots[i])
        end
    end
    poles = -log.(roots) ./ model.dt
    values, shape = _sample_matrix(model.samples)
    times = model.t0 .+
        model.dt .* collect(0:(length(model.samples) - 1))
    design = exp.(-times * transpose(poles))
    weight_matrix = design \ values
    weights = _restore_weights(weight_matrix, shape)
    fitted = design * weight_matrix
    diagnostics = merge(model.diagnostics, (;
        method=:linear_prediction_poles,
        roots,
        unstable_roots=unstable,
        stabilized=stabilize,
        pole_fit_error=norm(fitted - values) / max(norm(values), eps(Float64)),
    ))
    return ExponentialSum(poles, weights; diagnostics)
end
