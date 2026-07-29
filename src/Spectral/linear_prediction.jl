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

function _lp_design(values::AbstractMatrix{<:Number}, order::Int)
    order > 0 || throw(ArgumentError("autoregressive order must be positive"))
    sample_count, channel_count = size(values)
    sample_count > order ||
        throw(ArgumentError("linear prediction needs more samples than order"))
    rows = (sample_count - order) * channel_count
    design = Matrix{ComplexF64}(undef, rows, order)
    targets = Vector{ComplexF64}(undef, rows)
    row = 1
    for sample in (order + 1):sample_count, channel in 1:channel_count
        for lag in 1:order
            design[row, lag] = values[sample - lag, channel]
        end
        targets[row] = values[sample, channel]
        row += 1
    end
    return design, targets
end

struct _ARSolve
    coefficients::Vector{ComplexF64}
    singular_values::Vector{Float64}
    design_rank::Int
    condition_number::Float64
    relative_residual::Float64
end

function _solve_ar(values::AbstractMatrix{<:Number}, order::Int,
                   regularization::Float64)
    isfinite(regularization) && regularization >= 0 ||
        throw(ArgumentError("regularization must be finite and nonnegative"))
    design, targets = _lp_design(values, order)
    coefficients = if iszero(regularization)
        design \ targets
    else
        (design' * design + regularization * I) \ (design' * targets)
    end
    residual = design * coefficients - targets
    return _ARSolve(
        ComplexF64.(coefficients),
        Float64.(svdvals(design)),
        rank(design),
        cond(design),
        norm(residual) / max(norm(targets), eps(Float64)),
    )
end

"""
    linear_prediction(times, samples; order, regularization=0)

Fit a shared autoregressive model to scalar or array-valued samples. A positive
`regularization` applies ridge stabilization to noisy or nearly dependent
histories. This result remains a sample-domain recurrence and does not apply
node mode policies.
"""
function linear_prediction(times::AbstractVector{<:Real}, samples;
                           order::Integer,
                           regularization::Real=0)
    order >= 1 || throw(ArgumentError("order must be positive"))
    penalty = Float64(regularization)
    sequence = UniformSequence(times, samples)
    solution = _solve_ar(sequence.values, Int(order), penalty)
    diagnostics = (;
        method=:linear_prediction,
        order=Int(order),
        regularization=penalty,
        fit_rank=solution.design_rank,
        condition_number=solution.condition_number,
        l2err=solution.relative_residual,
        dt=sequence.dt,
        Nt=length(sequence),
    )
    return LinearPredictionResult(
        solution.coefficients,
        collect(samples),
        sequence.dt,
        sequence.t0,
        diagnostics,
    )
end

linear_prediction(series::CorrelatorSeries; kwargs...) =
    linear_prediction(series.times, series.values; kwargs...)

"""
    predict(model, nfuture)

Return only the `nfuture` extrapolated samples, preserving the shape of the
input samples.
"""
function predict(model::LinearPredictionResult, nfuture::Integer)
    nfuture >= 0 || throw(ArgumentError("nfuture must be nonnegative"))
    coefficient_type = eltype(model.coefficients)
    history = [one(coefficient_type) * sample for sample in model.samples]
    order = length(model.coefficients)
    length(history) >= order ||
        throw(ArgumentError("model history is shorter than its order"))
    output = Vector{eltype(history)}(undef, nfuture)
    for index in 1:nfuture
        value = model.coefficients[1] * history[end]
        for lag in 2:order
            value += model.coefficients[lag] * history[end - lag + 1]
        end
        push!(history, value)
        output[index] = value
    end
    return output
end

"""
    ModePolicy

Typed policy for handling discrete shift nodes after a node estimator has
identified them. Direct `LinearPredictionResult` prediction never applies a
mode policy.
"""
abstract type ModePolicy end

"""
    KeepModes(; tolerance=sqrt(eps(Float64)))

Keep every identified node. Nodes with magnitude greater than
`1 + tolerance` are diagnosed but are not changed.
"""
struct KeepModes{T<:AbstractFloat} <: ModePolicy
    tolerance::T
    function KeepModes(tolerance::T) where {T<:AbstractFloat}
        isfinite(tolerance) && tolerance >= 0 ||
            throw(ArgumentError(
                "mode tolerance must be finite and nonnegative",
            ))
        return new{T}(tolerance)
    end
end

function KeepModes(; tolerance::Real=sqrt(eps(Float64)))
    value = Float64(tolerance)
    isfinite(value) && value >= 0 ||
        throw(ArgumentError("mode tolerance must be finite and nonnegative"))
    return KeepModes(value)
end

"""
    ProjectUnitCircle(; tolerance=sqrt(eps(Float64)))

Radially project nodes with magnitude greater than `1 + tolerance` onto the
unit circle. Each projection is recorded in `ARLeastSquaresDiagnostics`.
"""
struct ProjectUnitCircle{T<:AbstractFloat} <: ModePolicy
    tolerance::T
    function ProjectUnitCircle(tolerance::T) where {T<:AbstractFloat}
        isfinite(tolerance) && tolerance >= 0 ||
            throw(ArgumentError(
                "mode tolerance must be finite and nonnegative",
            ))
        return new{T}(tolerance)
    end
end

function ProjectUnitCircle(; tolerance::Real=sqrt(eps(Float64)))
    value = Float64(tolerance)
    isfinite(value) && value >= 0 ||
        throw(ArgumentError("mode tolerance must be finite and nonnegative"))
    return ProjectUnitCircle(value)
end

"""
    RejectOutsideUnitCircle(; tolerance=sqrt(eps(Float64)))

Reject the complete node estimate when any node has magnitude greater than
`1 + tolerance`. Every rejected node is recorded in the backend diagnostics.
"""
struct RejectOutsideUnitCircle{T<:AbstractFloat} <: ModePolicy
    tolerance::T
    function RejectOutsideUnitCircle(tolerance::T) where {T<:AbstractFloat}
        isfinite(tolerance) && tolerance >= 0 ||
            throw(ArgumentError(
                "mode tolerance must be finite and nonnegative",
            ))
        return new{T}(tolerance)
    end
end

function RejectOutsideUnitCircle(;
    tolerance::Real=sqrt(eps(Float64)),
)
    value = Float64(tolerance)
    isfinite(value) && value >= 0 ||
        throw(ArgumentError("mode tolerance must be finite and nonnegative"))
    return RejectOutsideUnitCircle(value)
end

"""
    DropOutsideUnitCircle(; tolerance=sqrt(eps(Float64)))

Drop nodes with magnitude greater than `1 + tolerance`. Retained nodes enter
the generic full-sample weight refit. Dropping every node is a structured
node-estimation failure.
"""
struct DropOutsideUnitCircle{T<:AbstractFloat} <: ModePolicy
    tolerance::T
    function DropOutsideUnitCircle(tolerance::T) where {T<:AbstractFloat}
        isfinite(tolerance) && tolerance >= 0 ||
            throw(ArgumentError(
                "mode tolerance must be finite and nonnegative",
            ))
        return new{T}(tolerance)
    end
end

function DropOutsideUnitCircle(;
    tolerance::Real=sqrt(eps(Float64)),
)
    value = Float64(tolerance)
    isfinite(value) && value >= 0 ||
        throw(ArgumentError("mode tolerance must be finite and nonnegative"))
    return DropOutsideUnitCircle(value)
end

struct ModeModification
    index::Int
    original::ComplexF64
    result::Union{Nothing,ComplexF64}
    action::Symbol
    reason::Symbol
end

"""
    ARLeastSquares(; rank=AutomaticRank(NumericalRank()),
                   reduction=AllComponents(),
                   zero=ExactZero(), regularization=0, modes=KeepModes())

Estimate discrete shift nodes from a shared autoregressive least-squares
recurrence. The recurrence coefficient solve is the same implementation used
by `linear_prediction`; node recovery, mode handling, and the generic
full-sample exponential refit are exclusive to this estimator path.
"""
struct ARLeastSquares{
    R<:AbstractRankPolicy,
    S<:SampleReductionPolicy,
    Z<:AbstractZeroPolicy,
    M<:ModePolicy,
} <: AbstractNodeEstimator
    rank::R
    reduction::S
    zero::Z
    regularization::Float64
    modes::M
    function ARLeastSquares(
        rank::R,
        reduction::S,
        zero::Z,
        regularization::Float64,
        modes::M,
    ) where {
        R<:AbstractRankPolicy,
        S<:SampleReductionPolicy,
        Z<:AbstractZeroPolicy,
        M<:ModePolicy,
    }
        isfinite(regularization) && regularization >= 0 ||
            throw(ArgumentError(
                "regularization must be finite and nonnegative",
            ))
        return new{R,S,Z,M}(
            rank,
            reduction,
            zero,
            regularization,
            modes,
        )
    end
end

function ARLeastSquares(;
    rank::AbstractRankPolicy=AutomaticRank(NumericalRank()),
    reduction::SampleReductionPolicy=AllComponents(),
    zero::AbstractZeroPolicy=ExactZero(),
    regularization::Real=0,
    modes::ModePolicy=KeepModes(),
)
    penalty = Float64(regularization)
    isfinite(penalty) && penalty >= 0 ||
        throw(ArgumentError("regularization must be finite and nonnegative"))
    return ARLeastSquares(rank, reduction, zero, penalty, modes)
end

struct ARLeastSquaresDiagnostics{M<:ModePolicy}
    evidence_order::Int
    evidence_singular_values::Vector{Float64}
    coefficients::Vector{ComplexF64}
    fit_singular_values::Vector{Float64}
    fit_rank::Int
    condition_number::Float64
    relative_residual::Float64
    mode_policy::M
    unstable_nodes::Vector{Int}
    modifications::Vector{ModeModification}
end

function _empty_ar_diagnostics(policy::ModePolicy)
    return ARLeastSquaresDiagnostics(
        0,
        Float64[],
        ComplexF64[],
        Float64[],
        0,
        1.0,
        0.0,
        policy,
        Int[],
        ModeModification[],
    )
end

function _ar_geometry(sample_count::Int, channel_count::Int)
    sample_count >= 2 ||
        throw(ArgumentError("autoregressive estimation needs at least two samples"))
    channel_count >= 1 ||
        throw(ArgumentError("autoregressive estimation needs at least one channel"))
    return min(sample_count - 1,
               fld(sample_count * channel_count, channel_count + 1))
end

function _ar_failure(
    reason::Symbol,
    message::String,
    rank_policy::AbstractRankPolicy,
    evidence::Int,
    geometry::Int,
    resolved::Int,
    clamped::Bool,
    reduction::ReductionDiagnostics,
    backend::ARLeastSquaresDiagnostics,
)
    common = NodeDiagnostics(
        _requested_rank(rank_policy),
        evidence,
        geometry,
        resolved,
        clamped,
        reduction,
    )
    return NodeEstimate(NodeEstimationFailure(reason, message), common, backend)
end

struct _ModePolicyApplication
    nodes::Vector{ComplexF64}
    unstable_nodes::Vector{Int}
    modifications::Vector{ModeModification}
    failure::Union{Nothing,NodeEstimationFailure}
end

function _apply_mode_policy(policy::KeepModes,
                            nodes::Vector{ComplexF64})
    unstable = findall(node -> abs(node) > 1 + policy.tolerance, nodes)
    return _ModePolicyApplication(
        copy(nodes),
        unstable,
        ModeModification[],
        nothing,
    )
end

function _apply_mode_policy(policy::ProjectUnitCircle,
                            nodes::Vector{ComplexF64})
    projected = copy(nodes)
    unstable = findall(node -> abs(node) > 1 + policy.tolerance, nodes)
    modifications = ModeModification[]
    for index in unstable
        original = nodes[index]
        result = original / abs(original)
        projected[index] = result
        push!(
            modifications,
            ModeModification(
                index,
                original,
                result,
                :projected,
                :outside_unit_circle,
            ),
        )
    end
    return _ModePolicyApplication(
        projected,
        unstable,
        modifications,
        nothing,
    )
end

function _apply_mode_policy(policy::RejectOutsideUnitCircle,
                            nodes::Vector{ComplexF64})
    unstable = findall(node -> abs(node) > 1 + policy.tolerance, nodes)
    isempty(unstable) && return _ModePolicyApplication(
        copy(nodes),
        unstable,
        ModeModification[],
        nothing,
    )
    modifications = [
        ModeModification(
            index,
            nodes[index],
            nothing,
            :rejected,
            :outside_unit_circle,
        )
        for index in unstable
    ]
    count = length(unstable)
    failure = NodeEstimationFailure(
        :mode_rejected,
        "RejectOutsideUnitCircle rejected $count node$(count == 1 ? "" : "s") " *
        "outside the unit circle tolerance",
    )
    return _ModePolicyApplication(
        ComplexF64[],
        unstable,
        modifications,
        failure,
    )
end

function _apply_mode_policy(policy::DropOutsideUnitCircle,
                            nodes::Vector{ComplexF64})
    unstable = findall(node -> abs(node) > 1 + policy.tolerance, nodes)
    isempty(unstable) && return _ModePolicyApplication(
        copy(nodes),
        unstable,
        ModeModification[],
        nothing,
    )
    modifications = [
        ModeModification(
            index,
            nodes[index],
            nothing,
            :dropped,
            :outside_unit_circle,
        )
        for index in unstable
    ]
    retained = nodes[setdiff(eachindex(nodes), unstable)]
    failure = isempty(retained) ? NodeEstimationFailure(
        :all_modes_dropped,
        "DropOutsideUnitCircle removed every identified node",
    ) : nothing
    return _ModePolicyApplication(
        retained,
        unstable,
        modifications,
        failure,
    )
end

function _companion_nodes(coefficients::Vector{ComplexF64})
    order = length(coefficients)
    order > 0 || throw(ArgumentError("companion polynomial needs positive order"))
    companion = zeros(ComplexF64, order, order)
    companion[1, :] .= coefficients
    for row in 2:order
        companion[row, row - 1] = 1
    end
    return ComplexF64.(eigvals(companion))
end

function estimate_nodes(estimator::ARLeastSquares,
                        sequence::UniformSequence)
    early, reduced, reduction = _pre_estimation(estimator, sequence)
    early === nothing || return early

    sample_count, channel_count = size(reduced)
    geometry = _ar_geometry(sample_count, channel_count)
    requested = _requested_rank(estimator.rank)

    evidence_order = geometry
    evidence_solution = try
        _solve_ar(reduced, evidence_order, estimator.regularization)
    catch error
        if error isa LinearAlgebra.LAPACKException ||
           error isa LinearAlgebra.SingularException ||
           error isa LinearAlgebra.PosDefException
            backend = _empty_ar_diagnostics(estimator.modes)
            return _ar_failure(
                :autoregressive_solve_failed,
                "autoregressive evidence solve failed: $(sprint(showerror, error))",
                estimator.rank,
                0,
                geometry,
                0,
                false,
                reduction,
                backend,
            )
        end
        rethrow()
    end
    evidence = _evidence_rank(
        evidence_solution.singular_values,
        _evidence_policy(estimator.rank),
        ((sample_count - evidence_order) * channel_count, evidence_order),
    )
    resolution = _resolve_rank(estimator.rank, evidence, geometry)
    if resolution === nothing
        reason = requested !== nothing && requested > geometry ?
            :rank_exceeds_geometry :
            evidence == 0 ? :zero_evidence_rank :
            requested !== nothing && requested > evidence ?
            :rank_exceeds_evidence : :rank_unavailable
        backend = ARLeastSquaresDiagnostics(
            evidence_order,
            evidence_solution.singular_values,
            ComplexF64[],
            Float64[],
            0,
            1.0,
            0.0,
            estimator.modes,
            Int[],
            ModeModification[],
        )
        return _ar_failure(
            reason,
            "requested rank $(repr(requested)) is incompatible with " *
            "autoregressive evidence rank $evidence and geometry bound $geometry",
            estimator.rank,
            evidence,
            geometry,
            0,
            false,
            reduction,
            backend,
        )
    end

    order = resolution.resolved
    solution = try
        _solve_ar(reduced, order, estimator.regularization)
    catch error
        if error isa LinearAlgebra.LAPACKException ||
           error isa LinearAlgebra.SingularException ||
           error isa LinearAlgebra.PosDefException
            backend = ARLeastSquaresDiagnostics(
                evidence_order,
                evidence_solution.singular_values,
                ComplexF64[],
                Float64[],
                0,
                1.0,
                0.0,
                estimator.modes,
                Int[],
                ModeModification[],
            )
            return _ar_failure(
                :autoregressive_solve_failed,
                "resolved autoregressive solve failed: $(sprint(showerror, error))",
                estimator.rank,
                evidence,
                geometry,
                order,
                resolution.clamped,
                reduction,
                backend,
            )
        end
        rethrow()
    end
    all(
        coefficient -> isfinite(real(coefficient)) &&
                       isfinite(imag(coefficient)),
        solution.coefficients,
    ) || begin
        backend = ARLeastSquaresDiagnostics(
            evidence_order,
            evidence_solution.singular_values,
            solution.coefficients,
            solution.singular_values,
            solution.design_rank,
            solution.condition_number,
            solution.relative_residual,
            estimator.modes,
            Int[],
            ModeModification[],
        )
        return _ar_failure(
            :nonfinite_coefficients,
            "autoregressive solve produced nonfinite coefficients",
            estimator.rank,
            evidence,
            geometry,
            order,
            resolution.clamped,
            reduction,
            backend,
        )
    end

    original_nodes = try
        _companion_nodes(solution.coefficients)
    catch error
        if error isa LinearAlgebra.LAPACKException
            backend = ARLeastSquaresDiagnostics(
                evidence_order,
                evidence_solution.singular_values,
                solution.coefficients,
                solution.singular_values,
                solution.design_rank,
                solution.condition_number,
                solution.relative_residual,
                estimator.modes,
                Int[],
                ModeModification[],
            )
            return _ar_failure(
                :companion_eigensolve_failed,
                "companion eigensolve failed: $(sprint(showerror, error))",
                estimator.rank,
                evidence,
                geometry,
                order,
                resolution.clamped,
                reduction,
                backend,
            )
        end
        rethrow()
    end
    all(
        node -> isfinite(real(node)) && isfinite(imag(node)),
        original_nodes,
    ) || begin
        backend = ARLeastSquaresDiagnostics(
            evidence_order,
            evidence_solution.singular_values,
            solution.coefficients,
            solution.singular_values,
            solution.design_rank,
            solution.condition_number,
            solution.relative_residual,
            estimator.modes,
            Int[],
            ModeModification[],
        )
        return _ar_failure(
            :nonfinite_nodes,
            "ARLeastSquares produced nonfinite companion nodes",
            estimator.rank,
            evidence,
            geometry,
            order,
            resolution.clamped,
            reduction,
            backend,
        )
    end

    mode_application = _apply_mode_policy(estimator.modes, original_nodes)
    backend = ARLeastSquaresDiagnostics(
        evidence_order,
        evidence_solution.singular_values,
        solution.coefficients,
        solution.singular_values,
        solution.design_rank,
        solution.condition_number,
        solution.relative_residual,
        estimator.modes,
        mode_application.unstable_nodes,
        mode_application.modifications,
    )
    mode_application.failure === nothing || return _ar_failure(
        mode_application.failure.reason,
        mode_application.failure.message,
        estimator.rank,
        evidence,
        geometry,
        order,
        resolution.clamped,
        reduction,
        backend,
    )
    common = _node_diagnostics(
        resolution, reduction, mode_application.nodes,
    )
    return NodeEstimate(IdentifiedNodes(mode_application.nodes), common, backend)
end
