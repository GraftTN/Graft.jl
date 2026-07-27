"""
    ExponentialSum(poles, weights; diagnostics=(;))

Representation of `sum_k weights[k] * exp(-poles[k] * t)`. A weight may be a
scalar, vector, or matrix. Poles use decay-rate convention, so stable
exponentials have nonnegative real parts.
"""
struct ExponentialSum{P<:Number,W,D<:NamedTuple}
    poles::Vector{P}
    weights::Vector{W}
    diagnostics::D
    function ExponentialSum(poles::AbstractVector{P},
                            weights::AbstractVector{W};
                            diagnostics::D=(;)) where {P<:Number,W,D<:NamedTuple}
        length(poles) == length(weights) ||
            throw(ArgumentError("ExponentialSum needs one weight per pole"))
        return new{P,W,D}(collect(poles), collect(weights), diagnostics)
    end
end

Base.length(s::ExponentialSum) = length(s.poles)
Base.isempty(s::ExponentialSum) = isempty(s.poles)

function evaluate(s::ExponentialSum, t::Number)
    isempty(s) && throw(ArgumentError("cannot evaluate an empty ExponentialSum"))
    value = s.weights[1] * exp(-s.poles[1] * t)
    for k in 2:length(s)
        value += s.weights[k] * exp(-s.poles[k] * t)
    end
    return value
end

evaluate(s::ExponentialSum, ts) = [evaluate(s, t) for t in ts]
(s::ExponentialSum)(t::Number) = evaluate(s, t)

"""
    rank_from_svals(s; M=nothing, err=nothing, err_type=:relative,
                    heuristic=:knee)

Choose a deterministic numerical rank. An explicit `M` wins. With `err`, keep
singular values above an absolute or relative threshold. Otherwise `:knee`
selects the largest adjacent logarithmic gap, while `:tolerance` applies the
standard floating-point rank threshold.
"""
function rank_from_svals(s::AbstractVector{<:Real};
                         M::Union{Nothing,Integer}=nothing,
                         err::Union{Nothing,Real}=nothing,
                         err_type::Symbol=:relative,
                         heuristic::Symbol=:knee)
    isempty(s) && throw(ArgumentError("singular-value list must be nonempty"))
    all(x -> x >= 0 && isfinite(x), s) ||
        throw(ArgumentError("singular values must be finite and nonnegative"))
    all(s[i] >= s[i + 1] for i in 1:(length(s) - 1)) ||
        throw(ArgumentError("singular values must be sorted nonincreasingly"))

    n = length(s)
    if M !== nothing
        1 <= M <= n || throw(ArgumentError("M must lie in 1:$n"))
        return Int(M)
    end

    if err !== nothing
        err >= 0 || throw(ArgumentError("err must be nonnegative"))
        threshold = if err_type === :relative
            Float64(err) * first(s)
        elseif err_type === :absolute
            Float64(err)
        else
            throw(ArgumentError("err_type must be :relative or :absolute"))
        end
        return clamp(count(x -> x > threshold, s), 1, n)
    end

    if heuristic === :tolerance
        threshold = max(n, 1) * eps(Float64) * first(s)
        return clamp(count(x -> x > threshold, s), 1, n)
    elseif heuristic === :knee
        n == 1 && return 1
        floor_value = max(first(s) * eps(Float64), floatmin(Float64))
        logs = log10.(max.(Float64.(s), floor_value))
        gaps = logs[1:end-1] .- logs[2:end]
        knee = argmax(gaps)
        # A flat spectrum has no meaningful knee. Use numerical rank instead.
        if gaps[knee] < 0.5
            threshold = n * eps(Float64) * first(s)
            return clamp(count(x -> x > threshold, s), 1, n)
        end
        return knee
    else
        throw(ArgumentError("heuristic must be :knee or :tolerance"))
    end
end

function _sample_matrix(samples)
    isempty(samples) && throw(ArgumentError("samples must be nonempty"))
    shape = size(first(samples))
    all(x -> size(x) == shape, samples) ||
        throw(ArgumentError("all samples must have the same shape"))
    rows = [vec(complex.(collect(x))) for x in samples]
    return Matrix(transpose(reduce(hcat, rows))), shape
end

_sample_matrix(samples::AbstractVector{<:Number}) =
    reshape(complex.(collect(samples)), :, 1), ()

function _uniform_times(times::AbstractVector{<:Real})
    length(times) >= 3 || throw(ArgumentError("at least three time samples are required"))
    dt = Float64(times[2] - times[1])
    dt > 0 || throw(ArgumentError("times must be strictly increasing"))
    all(isapprox(Float64(times[i] - times[i - 1]), dt;
                 rtol=1e-10, atol=10eps(dt)) for i in 3:length(times)) ||
        throw(ArgumentError("ESPRIT requires a uniform time grid"))
    return dt
end

function _restore_weights(weight_matrix::AbstractMatrix, shape)
    if isempty(shape)
        return vec(weight_matrix)
    end
    return [reshape(copy(weight_matrix[k, :]), shape) for k in axes(weight_matrix, 1)]
end

function _channel_relative_errors(reference::AbstractMatrix,
                                  fitted::AbstractMatrix)
    errors = Vector{Float64}(undef, size(reference, 2))
    for j in axes(reference, 2)
        denom = norm(view(reference, :, j))
        errors[j] = norm(view(reference, :, j) - view(fitted, :, j)) /
                    max(denom, eps(Float64))
    end
    return errors
end

"""
    esprit(times, samples; M=nothing, err=nothing, err_type=:relative,
           heuristic=:knee, matrix_mode=:stacked, hankel_rows=nothing)

Block-Hankel ESPRIT for scalar or array-valued samples on a uniform grid.
`:stacked` estimates shared poles from every channel. For matrix-valued data,
`:trace` and `:sum` estimate poles from a scalar reduction and then refit all
matrix weights.
"""
function esprit(times::AbstractVector{<:Real}, samples;
                M::Union{Nothing,Integer}=nothing,
                err::Union{Nothing,Real}=nothing,
                err_type::Symbol=:relative,
                heuristic::Symbol=:knee,
                matrix_mode::Symbol=:stacked,
                hankel_rows::Union{Nothing,Integer}=nothing)
    length(times) == length(samples) ||
        throw(ArgumentError("times and samples must have equal length"))
    dt = _uniform_times(times)
    values, shape = _sample_matrix(samples)
    n, nchannels = size(values)

    pole_values = if matrix_mode === :stacked
        values
    elseif matrix_mode === :trace
        length(shape) == 2 && shape[1] == shape[2] ||
            throw(ArgumentError("matrix_mode=:trace requires square matrix samples"))
        reshape([tr(reshape(values[i, :], shape)) for i in 1:n], n, 1)
    elseif matrix_mode === :sum
        reshape(vec(sum(values; dims=2)), n, 1)
    else
        throw(ArgumentError("matrix_mode must be :stacked, :trace, or :sum"))
    end

    L = hankel_rows === nothing ? n ÷ 2 : Int(hankel_rows)
    1 <= L < n || throw(ArgumentError("hankel_rows must lie in 1:$(n - 1)"))
    K = n - L
    H0 = Matrix{ComplexF64}(undef, size(pole_values, 2) * L, K)
    H1 = similar(H0)
    for i in 0:(L - 1), j in 0:(K - 1)
        row = (i * size(pole_values, 2) + 1):((i + 1) * size(pole_values, 2))
        H0[row, j + 1] .= pole_values[i + j + 1, :]
        H1[row, j + 1] .= pole_values[i + j + 2, :]
    end

    F = svd(H0)
    maxrank = min(size(H0)...)
    chosen = rank_from_svals(F.S; M, err, err_type, heuristic)
    chosen <= maxrank || throw(ArgumentError("chosen rank exceeds Hankel rank bound"))
    Ur = F.U[:, 1:chosen]
    Vr = F.V[:, 1:chosen]
    shift = Ur' * H1 * Vr * Diagonal(inv.(F.S[1:chosen]))
    lambda = eigvals(shift)
    poles = -log.(lambda) ./ dt
    order = sortperm(poles; by=z -> (real(z), imag(z)))
    poles = poles[order]

    t0 = Float64(first(times))
    relative_times = Float64.(times) .- t0
    design = exp.(-relative_times * transpose(poles))
    fitted_weights = design \ values
    absolute_weights = fitted_weights .*
        reshape(exp.(poles .* t0), :, 1)
    weights = _restore_weights(absolute_weights, shape)
    fitted = design * fitted_weights
    residual = fitted - values
    scale = max(norm(values), eps(Float64))
    aliasing = any(abs(imag(z)) > pi / dt * (1 + 100eps(Float64)) for z in poles)
    unstable = findall(z -> real(z) < -sqrt(eps(Float64)), poles)
    diagnostics = (;
        method=:esprit,
        matrix_mode,
        singular_values=collect(F.S),
        chosen_rank=chosen,
        rank_rule=M === nothing ? (err === nothing ? heuristic : err_type) : :explicit,
        dt,
        Tfit=Float64(last(times) - first(times)),
        Nt=n,
        t0,
        least_squares_rank=rank(design),
        condition_number=cond(design),
        maxerr=maximum(abs, residual),
        l2err=norm(residual) / scale,
        channel_relative_errors=_channel_relative_errors(values, fitted),
        aliasing_warning=aliasing,
        unstable_poles=unstable,
        nchannels,
    )
    return ExponentialSum(poles, weights; diagnostics)
end

esprit(series::CorrelatorSeries; kwargs...) =
    esprit(series.times, series.values; kwargs...)
