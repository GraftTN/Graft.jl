"""
    UniformSequence(times, samples)

Validated uniformly sampled scalar, vector, or matrix sequence. Samples are
stored as `Nt × channels` values while `sample_shape` preserves the public
shape used by fitted weights and evaluation.
"""
function _uniform_grid_metadata(coordinates::Vector{Float64})
    length(coordinates) >= 3 ||
        throw(ArgumentError("at least three time samples are required"))
    all(isfinite, coordinates) || throw(ArgumentError("times must be finite"))
    increments = diff(coordinates)
    all(increment -> isfinite(increment) && increment > 0, increments) ||
        throw(ArgumentError(
            "times must be strictly increasing with finite increments",
        ))
    dt = first(increments)
    tolerance = 64 * eps(Float64) *
        max(maximum(abs, increments), abs(dt))
    all(
        isapprox(increment, dt; rtol=sqrt(eps(Float64)), atol=tolerance)
        for increment in increments
    ) || throw(ArgumentError("samples require a uniform time grid"))
    return first(coordinates), dt
end

struct UniformSequence{N}
    times::Vector{Float64}
    values::Matrix{ComplexF64}
    sample_shape::NTuple{N,Int}
    t0::Float64
    dt::Float64

    function UniformSequence(times::Vector{Float64},
                             values::Matrix{ComplexF64},
                             sample_shape::NTuple{N,Int},
                             t0::Float64,
                             dt::Float64) where {N}
        validated_t0, validated_dt = _uniform_grid_metadata(times)
        size(values, 1) == length(times) ||
            throw(ArgumentError(
                "sample rows must match the number of time coordinates",
            ))
        all(dimension -> dimension > 0, sample_shape) ||
            throw(ArgumentError("sample dimensions must be positive"))
        expected_channels = isempty(sample_shape) ? 1 : prod(sample_shape)
        size(values, 2) == expected_channels ||
            throw(ArgumentError(
                "sample_shape does not match the flattened channel count",
            ))
        all(
            value -> isfinite(real(value)) && isfinite(imag(value)),
            values,
        ) || throw(ArgumentError("samples must be finite"))
        isfinite(t0) && t0 == validated_t0 ||
            throw(ArgumentError("t0 must equal the first time coordinate"))
        isfinite(dt) && dt > 0 && dt == validated_dt ||
            throw(ArgumentError("dt must equal the uniform grid increment"))
        return new{N}(times, values, sample_shape, t0, dt)
    end
end

function _sample_matrix(samples)
    isempty(samples) && throw(ArgumentError("samples must be nonempty"))
    shape = size(first(samples))
    all(sample -> size(sample) == shape, samples) ||
        throw(ArgumentError("all samples must have the same shape"))
    rows = [vec(ComplexF64.(collect(sample))) for sample in samples]
    values = Matrix(transpose(reduce(hcat, rows)))
    all(value -> isfinite(real(value)) && isfinite(imag(value)), values) ||
        throw(ArgumentError("samples must be finite"))
    return values, shape
end

function _sample_matrix(samples::AbstractVector{<:Number})
    isempty(samples) && throw(ArgumentError("samples must be nonempty"))
    values = reshape(ComplexF64.(collect(samples)), :, 1)
    all(value -> isfinite(real(value)) && isfinite(imag(value)), values) ||
        throw(ArgumentError("samples must be finite"))
    return values, ()
end

function UniformSequence(times::AbstractVector{<:Real}, samples)
    length(times) == length(samples) ||
        throw(ArgumentError("times and samples must have equal length"))
    coordinates = Float64.(times)
    t0, dt = _uniform_grid_metadata(coordinates)
    values, shape = _sample_matrix(samples)
    return UniformSequence(coordinates, values, shape, t0, dt)
end

Base.length(sequence::UniformSequence) = length(sequence.times)

abstract type SampleReductionPolicy end

"""Use every flattened sample component as node-estimation evidence."""
struct AllComponents <: SampleReductionPolicy end

"""
    DeclaredDiagonal(; atol=0, rtol=0)

Use only diagonal matrix channels after validating the aggregate discarded
off-diagonal Frobenius norm against
`atol + rtol * unreduced_frobenius_norm`.
"""
struct DeclaredDiagonal{T<:AbstractFloat} <: SampleReductionPolicy
    atol::T
    rtol::T
end

function DeclaredDiagonal(; atol::Real=0.0, rtol::Real=0.0)
    absolute = Float64(atol)
    relative = Float64(rtol)
    isfinite(absolute) && absolute >= 0 ||
        throw(ArgumentError("DeclaredDiagonal atol must be finite and nonnegative"))
    isfinite(relative) && relative >= 0 ||
        throw(ArgumentError("DeclaredDiagonal rtol must be finite and nonnegative"))
    return DeclaredDiagonal(absolute, relative)
end

"""Use one scalar matrix-trace channel as node-estimation evidence."""
struct TraceReduction <: SampleReductionPolicy end

struct ReductionDiagnostics{N,P<:SampleReductionPolicy}
    policy::P
    sample_shape::NTuple{N,Int}
    flattening::Symbol
    original_channels::Int
    reduced_channels::Int
    discarded_norm::Float64
    unreduced_norm::Float64
end

function _square_dimension(sequence::UniformSequence)
    length(sequence.sample_shape) == 2 ||
        throw(ArgumentError("matrix reduction requires matrix-valued samples"))
    rows, columns = sequence.sample_shape
    rows == columns ||
        throw(ArgumentError("matrix reduction requires square samples"))
    return rows
end

function _reduce_samples(sequence::UniformSequence, policy::AllComponents)
    unreduced = norm(sequence.values)
    diagnostics = ReductionDiagnostics(
        policy, sequence.sample_shape, :column_major,
        size(sequence.values, 2), size(sequence.values, 2), 0.0, unreduced,
    )
    return copy(sequence.values), diagnostics
end

function _reduce_samples(sequence::UniformSequence, policy::DeclaredDiagonal)
    dimension = _square_dimension(sequence)
    diagonal_indices = [LinearIndices(sequence.sample_shape)[index, index]
                        for index in 1:dimension]
    offdiagonal_indices = setdiff(collect(axes(sequence.values, 2)), diagonal_indices)
    discarded = isempty(offdiagonal_indices) ? 0.0 :
        norm(view(sequence.values, :, offdiagonal_indices))
    unreduced = norm(sequence.values)
    discarded <= policy.atol + policy.rtol * unreduced ||
        return nothing, ReductionDiagnostics(
            policy, sequence.sample_shape, :column_major,
            size(sequence.values, 2), dimension, discarded, unreduced,
        )
    diagnostics = ReductionDiagnostics(
        policy, sequence.sample_shape, :column_major,
        size(sequence.values, 2), dimension, discarded, unreduced,
    )
    return copy(sequence.values[:, diagonal_indices]), diagnostics
end

function _reduce_samples(sequence::UniformSequence, policy::TraceReduction)
    dimension = _square_dimension(sequence)
    diagonal_indices = [LinearIndices(sequence.sample_shape)[index, index]
                        for index in 1:dimension]
    reduced = reshape(vec(sum(sequence.values[:, diagonal_indices]; dims=2)), :, 1)
    diagnostics = ReductionDiagnostics(
        policy, sequence.sample_shape, :column_major,
        size(sequence.values, 2), 1, 0.0, norm(sequence.values),
    )
    return reduced, diagnostics
end

abstract type AbstractZeroPolicy end
struct ExactZero <: AbstractZeroPolicy end

struct ToleranceZero{T<:AbstractFloat} <: AbstractZeroPolicy
    atol::T
    rtol::T
    reference_scale::T
end

function ToleranceZero(; atol::Real=0.0, rtol::Real=0.0,
                       reference_scale::Real)
    absolute = Float64(atol)
    relative = Float64(rtol)
    reference = Float64(reference_scale)
    isfinite(absolute) && absolute >= 0 ||
        throw(ArgumentError("ToleranceZero atol must be finite and nonnegative"))
    isfinite(relative) && relative >= 0 ||
        throw(ArgumentError("ToleranceZero rtol must be finite and nonnegative"))
    isfinite(reference) && reference >= 0 ||
        throw(ArgumentError(
            "ToleranceZero reference_scale must be finite and nonnegative",
        ))
    return ToleranceZero(absolute, relative, reference)
end

_zero_threshold(::ExactZero) = 0.0
_zero_threshold(policy::ToleranceZero) =
    policy.atol + policy.rtol * policy.reference_scale

abstract type AbstractEvidenceRankPolicy end
struct NumericalRank <: AbstractEvidenceRankPolicy end

struct AbsoluteThresholdRank{T<:AbstractFloat} <: AbstractEvidenceRankPolicy
    atol::T
end

function AbsoluteThresholdRank(atol::Real)
    value = Float64(atol)
    isfinite(value) && value >= 0 ||
        throw(ArgumentError(
            "AbsoluteThresholdRank threshold must be finite and nonnegative",
        ))
    return AbsoluteThresholdRank(value)
end

struct RelativeThresholdRank{T<:AbstractFloat} <: AbstractEvidenceRankPolicy
    rtol::T
end

function RelativeThresholdRank(rtol::Real)
    value = Float64(rtol)
    isfinite(value) && value >= 0 ||
        throw(ArgumentError(
            "RelativeThresholdRank threshold must be finite and nonnegative",
        ))
    return RelativeThresholdRank(value)
end

struct KneeRank{T<:AbstractFloat} <: AbstractEvidenceRankPolicy
    minimum_log_gap::T
end

function KneeRank(; minimum_log_gap::Real=0.5)
    gap = Float64(minimum_log_gap)
    isfinite(gap) && gap >= 0 ||
        throw(ArgumentError("KneeRank minimum_log_gap must be finite and nonnegative"))
    return KneeRank(gap)
end

abstract type AbstractRankPolicy end

struct AutomaticRank{E<:AbstractEvidenceRankPolicy} <: AbstractRankPolicy
    evidence::E
end

AutomaticRank() = AutomaticRank(KneeRank())

struct StrictRank{E<:AbstractEvidenceRankPolicy} <: AbstractRankPolicy
    requested::Int
    evidence::E
end

function StrictRank(requested::Integer,
                    evidence::AbstractEvidenceRankPolicy=NumericalRank())
    requested > 0 || throw(ArgumentError("StrictRank requested rank must be positive"))
    return StrictRank(Int(requested), evidence)
end

struct ClampedRank{E<:AbstractEvidenceRankPolicy} <: AbstractRankPolicy
    requested::Int
    evidence::E
end

function ClampedRank(requested::Integer,
                     evidence::AbstractEvidenceRankPolicy=NumericalRank())
    requested > 0 || throw(ArgumentError("ClampedRank requested rank must be positive"))
    return ClampedRank(Int(requested), evidence)
end

function _evidence_rank(singular_values::AbstractVector{<:Real},
                        policy::NumericalRank, matrix_size)
    isempty(singular_values) && return 0
    threshold = max(matrix_size...) * eps(Float64) * first(singular_values)
    return count(value -> value > threshold, singular_values)
end

function _evidence_rank(singular_values::AbstractVector{<:Real},
                        policy::AbsoluteThresholdRank, matrix_size)
    return count(value -> value > policy.atol, singular_values)
end

function _evidence_rank(singular_values::AbstractVector{<:Real},
                        policy::RelativeThresholdRank, matrix_size)
    isempty(singular_values) && return 0
    threshold = policy.rtol * first(singular_values)
    return count(value -> value > threshold, singular_values)
end

function _evidence_rank(singular_values::AbstractVector{<:Real},
                        policy::KneeRank, matrix_size)
    isempty(singular_values) && return 0
    numerical = _evidence_rank(singular_values, NumericalRank(), matrix_size)
    numerical <= 1 && return numerical
    floor_value = max(first(singular_values) * eps(Float64), floatmin(Float64))
    # Include the first numerically null direction when available.  Omitting
    # that boundary makes an exact rank-r spectrum choose an arbitrary gap
    # inside its retained subspace instead of the r/(r+1) evidence break.
    boundary = min(numerical + 1, length(singular_values))
    logs = log10.(max.(Float64.(singular_values[1:boundary]), floor_value))
    gaps = logs[1:(end - 1)] .- logs[2:end]
    index = argmax(gaps)
    return gaps[index] >= policy.minimum_log_gap ? min(index, numerical) :
        numerical
end

_requested_rank(::AutomaticRank) = nothing
_requested_rank(policy::StrictRank) = policy.requested
_requested_rank(policy::ClampedRank) = policy.requested
_evidence_policy(policy::AutomaticRank) = policy.evidence
_evidence_policy(policy::StrictRank) = policy.evidence
_evidence_policy(policy::ClampedRank) = policy.evidence

struct RankResolution
    requested::Union{Nothing,Int}
    evidence::Int
    geometry::Int
    resolved::Int
    clamped::Bool
    clamp_reason::Union{Nothing,Symbol}
end

function _resolve_rank(policy::AutomaticRank, evidence::Int, geometry::Int)
    resolved = min(evidence, geometry)
    resolved > 0 || return nothing
    return RankResolution(
        nothing, evidence, geometry, resolved, false, nothing,
    )
end

function _resolve_rank(policy::StrictRank, evidence::Int, geometry::Int)
    policy.requested <= evidence || return nothing
    policy.requested <= geometry || return nothing
    return RankResolution(
        policy.requested, evidence, geometry, policy.requested, false, nothing,
    )
end

function _resolve_rank(policy::ClampedRank, evidence::Int, geometry::Int)
    resolved = min(policy.requested, evidence, geometry)
    resolved > 0 || return nothing
    evidence_limited = policy.requested > evidence
    geometry_limited = policy.requested > geometry
    reason = evidence_limited && geometry_limited ? :evidence_and_geometry :
        evidence_limited ? :evidence :
        geometry_limited ? :geometry : nothing
    return RankResolution(
        policy.requested, evidence, geometry, resolved,
        resolved != policy.requested, reason,
    )
end

abstract type AbstractNodeEstimator end

struct HankelDMD{R<:AbstractRankPolicy,S<:SampleReductionPolicy,
                 Z<:AbstractZeroPolicy} <: AbstractNodeEstimator
    rank::R
    reduction::S
    zero::Z
    hankel_rows::Union{Nothing,Int}
end

function HankelDMD(; rank::AbstractRankPolicy=AutomaticRank(),
                   reduction::SampleReductionPolicy=AllComponents(),
                   zero::AbstractZeroPolicy=ExactZero(),
                   hankel_rows::Union{Nothing,Integer}=nothing)
    rows = hankel_rows === nothing ? nothing : Int(hankel_rows)
    rows === nothing || rows > 0 ||
        throw(ArgumentError("hankel_rows must be positive"))
    return HankelDMD(rank, reduction, zero, rows)
end

struct ESPRIT{R<:AbstractRankPolicy,S<:SampleReductionPolicy,
              Z<:AbstractZeroPolicy} <: AbstractNodeEstimator
    rank::R
    reduction::S
    zero::Z
    hankel_rows::Union{Nothing,Int}
end

function ESPRIT(; rank::AbstractRankPolicy=AutomaticRank(),
                reduction::SampleReductionPolicy=AllComponents(),
                zero::AbstractZeroPolicy=ExactZero(),
                hankel_rows::Union{Nothing,Integer}=nothing)
    rows = hankel_rows === nothing ? nothing : Int(hankel_rows)
    rows === nothing || rows > 0 ||
        throw(ArgumentError("hankel_rows must be positive"))
    return ESPRIT(rank, reduction, zero, rows)
end

"""
    LeftSubspaceESPRIT(; ...)

Explicit time-major block-Hankel left-subspace ESPRIT. Unlike the default
right-subspace `ESPRIT`, its shift relation advances by one complete channel
block and therefore requires the documented `:time_major_block_rows` layout.
It exists for consumers whose frozen noisy behavior demonstrates that the
left-subspace formulation is semantically relevant.
"""
struct LeftSubspaceESPRIT{
    R<:AbstractRankPolicy,
    S<:SampleReductionPolicy,
    Z<:AbstractZeroPolicy,
} <: AbstractNodeEstimator
    rank::R
    reduction::S
    zero::Z
    hankel_rows::Union{Nothing,Int}
end

function LeftSubspaceESPRIT(;
    rank::AbstractRankPolicy=AutomaticRank(),
    reduction::SampleReductionPolicy=AllComponents(),
    zero::AbstractZeroPolicy=ExactZero(),
    hankel_rows::Union{Nothing,Integer}=nothing,
)
    rows = hankel_rows === nothing ? nothing : Int(hankel_rows)
    rows === nothing || rows >= 2 ||
        throw(ArgumentError(
            "LeftSubspaceESPRIT hankel_rows must be at least two",
        ))
    return LeftSubspaceESPRIT(rank, reduction, zero, rows)
end

abstract type AbstractNodeOutcome end

struct IdentifiedNodes <: AbstractNodeOutcome
    nodes::Vector{ComplexF64}
end

struct ZeroSequence <: AbstractNodeOutcome
    norm::Float64
    threshold::Float64
end

abstract type AbstractNodeFailure <: AbstractNodeOutcome end

struct NodeEstimationFailure <: AbstractNodeFailure
    reason::Symbol
    message::String
end

struct ReductionErasedSignal <: AbstractNodeFailure
    original_norm::Float64
    reduced_norm::Float64
    threshold::Float64
end

struct NodeDiagnostics{R<:ReductionDiagnostics}
    requested_rank::Union{Nothing,Int}
    evidence_rank::Int
    geometry_rank::Int
    resolved_rank::Int
    clamped::Bool
    clamp_reason::Union{Nothing,Symbol}
    minimum_node_separation::Union{Nothing,Float64}
    unstable_nodes::Vector{Int}
    warnings::Vector{Symbol}
    reduction::R
end

struct HankelDMDDiagnostics
    hankel_rows::Int
    hankel_columns::Int
    singular_values::Vector{Float64}
    shift_condition_number::Union{Nothing,Float64}
    shift_residual::Union{Nothing,Float64}
    sample_layout::Symbol
    subspace_layout::Symbol
end

struct ESPRITDiagnostics
    hankel_rows::Int
    hankel_columns::Int
    singular_values::Vector{Float64}
    leading_singular_values::Vector{Float64}
    shift_condition_number::Union{Nothing,Float64}
    shift_residual::Union{Nothing,Float64}
    sample_layout::Symbol
    subspace_layout::Symbol
end

struct LeftSubspaceESPRITDiagnostics
    hankel_rows::Int
    hankel_columns::Int
    channel_count::Int
    singular_values::Vector{Float64}
    leading_singular_values::Vector{Float64}
    shift_condition_number::Union{Nothing,Float64}
    shift_residual::Union{Nothing,Float64}
    sample_layout::Symbol
    subspace_layout::Symbol
end

struct NodeEstimate{O<:AbstractNodeOutcome,C,B}
    outcome::O
    common::C
    backend::B
end

function NodeDiagnostics(requested::Union{Nothing,Int}, evidence::Int,
                         geometry::Int, resolved::Int, clamped::Bool,
                         reduction::ReductionDiagnostics)
    evidence_limited = requested !== nothing && requested > evidence
    geometry_limited = requested !== nothing && requested > geometry
    clamp_reason = !clamped ? nothing :
        evidence_limited && geometry_limited ? :evidence_and_geometry :
        evidence_limited ? :evidence : geometry_limited ? :geometry : nothing
    warnings = clamped ? Symbol[:rank_clamped] : Symbol[]
    return NodeDiagnostics(
        requested, evidence, geometry, resolved, clamped, clamp_reason,
        nothing, Int[], warnings, reduction,
    )
end

function _minimum_separation(values::AbstractVector{<:Number})
    length(values) < 2 && return Inf
    return minimum(
        abs(values[left] - values[right])
        for left in eachindex(values) for right in (left + 1):length(values)
    )
end

function _node_diagnostics(resolution::RankResolution,
                           reduction::ReductionDiagnostics,
                           nodes::Vector{ComplexF64})
    separation = _minimum_separation(nodes)
    unstable = findall(node -> abs(node) > 1 + sqrt(eps(Float64)), nodes)
    scale = max(maximum(abs, nodes; init=0.0), 1.0)
    warnings = Symbol[]
    resolution.clamped && push!(warnings, :rank_clamped)
    separation <= sqrt(eps(Float64)) * scale &&
        push!(warnings, :near_duplicate_nodes)
    !isempty(unstable) && push!(warnings, :unstable_nodes)
    return NodeDiagnostics(
        resolution.requested,
        resolution.evidence,
        resolution.geometry,
        resolution.resolved,
        resolution.clamped,
        resolution.clamp_reason,
        separation,
        unstable,
        warnings,
        reduction,
    )
end

function _empty_node_diagnostics(reduction::ReductionDiagnostics)
    return NodeDiagnostics(nothing, 0, 0, 0, false, reduction)
end

function _hankel_pair(values::AbstractMatrix, rows::Int)
    sample_count, channel_count = size(values)
    1 <= rows < sample_count ||
        throw(ArgumentError("hankel_rows must lie in 1:$(sample_count - 1)"))
    columns = sample_count - rows
    H0 = Matrix{ComplexF64}(undef, channel_count * rows, columns)
    H1 = similar(H0)
    for row_index in 0:(rows - 1), column_index in 0:(columns - 1)
        target = (row_index * channel_count + 1):((row_index + 1) * channel_count)
        H0[target, column_index + 1] .=
            values[row_index + column_index + 1, :]
        H1[target, column_index + 1] .=
            values[row_index + column_index + 2, :]
    end
    return H0, H1
end

function _hankel(values::AbstractMatrix, rows::Int)
    sample_count, channel_count = size(values)
    1 <= rows <= sample_count ||
        throw(ArgumentError("hankel_rows must lie in 1:$sample_count"))
    columns = sample_count - rows + 1
    H = Matrix{ComplexF64}(undef, channel_count * rows, columns)
    for row_index in 0:(rows - 1), column_index in 0:(columns - 1)
        target = (row_index * channel_count + 1):((row_index + 1) * channel_count)
        H[target, column_index + 1] .=
            values[row_index + column_index + 1, :]
    end
    return H
end

function _pre_estimation(estimator::AbstractNodeEstimator,
                         sequence::UniformSequence)
    original_norm = norm(sequence.values)
    threshold = _zero_threshold(estimator.zero)
    if original_norm <= threshold
        reduced, reduction = _reduce_samples(sequence, estimator.reduction)
        reduction === nothing && error("unreachable reduction diagnostics")
        outcome = ZeroSequence(original_norm, threshold)
        return NodeEstimate(
            outcome, _empty_node_diagnostics(reduction), nothing,
        ), nothing, reduction
    end
    reduced, reduction = _reduce_samples(sequence, estimator.reduction)
    if reduced === nothing
        outcome = NodeEstimationFailure(
            :diagonal_declaration_violated,
            "discarded off-diagonal Frobenius norm exceeds the declared tolerance",
        )
        return NodeEstimate(
            outcome, _empty_node_diagnostics(reduction), nothing,
        ), nothing, reduction
    end
    reduced_norm = norm(reduced)
    if reduced_norm <= threshold
        outcome = ReductionErasedSignal(
            original_norm, reduced_norm, threshold,
        )
        return NodeEstimate(
            outcome, _empty_node_diagnostics(reduction), nothing,
        ), nothing, reduction
    end
    return nothing, reduced, reduction
end

function _rank_failure(rank_policy::AbstractRankPolicy, evidence::Int,
                       geometry::Int, reduction::ReductionDiagnostics,
                       backend)
    requested = _requested_rank(rank_policy)
    reason = evidence == 0 ? :zero_evidence_rank :
        requested !== nothing && requested > evidence &&
        requested > geometry ? :rank_exceeds_evidence_and_geometry :
        requested !== nothing && requested > evidence ? :rank_exceeds_evidence :
        requested !== nothing && requested > geometry ? :rank_exceeds_geometry :
        :rank_unavailable
    message = "requested rank $(repr(requested)) is incompatible with " *
              "evidence rank $evidence and geometry bound $geometry"
    common = NodeDiagnostics(requested, evidence, geometry, 0, false, reduction)
    return NodeEstimate(NodeEstimationFailure(reason, message), common, backend)
end

function estimate_nodes(estimator::HankelDMD, sequence::UniformSequence)
    early, reduced, reduction = _pre_estimation(estimator, sequence)
    early === nothing || return early
    rows = estimator.hankel_rows === nothing ? length(sequence) ÷ 2 :
        estimator.hankel_rows
    H0, H1 = _hankel_pair(reduced, rows)
    decomposition = svd(H0; full=false)
    singular_values = Float64.(decomposition.S)
    evidence = _evidence_rank(
        singular_values, _evidence_policy(estimator.rank), size(H0),
    )
    geometry = min(size(H0)...)
    resolution = _resolve_rank(estimator.rank, evidence, geometry)
    backend = HankelDMDDiagnostics(
        rows, size(H0, 2), singular_values, nothing, nothing,
        :time_major_block_rows, :projected_reduced_operator,
    )
    resolution === nothing &&
        return _rank_failure(estimator.rank, evidence, geometry, reduction, backend)
    rank_value = resolution.resolved
    Ur = decomposition.U[:, 1:rank_value]
    Vr = decomposition.V[:, 1:rank_value]
    shift = Ur' * H1 * Vr *
            Diagonal(inv.(singular_values[1:rank_value]))
    shift_condition = first(singular_values) /
        singular_values[rank_value]
    shifted_target = H1 * Vr
    shift_residual = norm(
        shifted_target -
        Ur * shift * Diagonal(singular_values[1:rank_value]),
    ) / max(norm(shifted_target), eps(Float64))
    backend = HankelDMDDiagnostics(
        rows, size(H0, 2), singular_values, shift_condition, shift_residual,
        :time_major_block_rows, :projected_reduced_operator,
    )
    nodes = ComplexF64.(eigvals(shift))
    all(node -> isfinite(real(node)) && isfinite(imag(node)), nodes) ||
        return NodeEstimate(
            NodeEstimationFailure(:nonfinite_nodes, "HankelDMD produced nonfinite nodes"),
            NodeDiagnostics(
                resolution.requested, resolution.evidence, resolution.geometry,
                resolution.resolved, resolution.clamped, reduction,
            ),
            backend,
        )
    common = _node_diagnostics(resolution, reduction, nodes)
    return NodeEstimate(IdentifiedNodes(nodes), common, backend)
end

function estimate_nodes(estimator::ESPRIT, sequence::UniformSequence)
    early, reduced, reduction = _pre_estimation(estimator, sequence)
    early === nothing || return early
    rows = estimator.hankel_rows === nothing ? length(sequence) ÷ 2 :
        estimator.hankel_rows
    H = _hankel(reduced, rows)
    decomposition = svd(H; full=false)
    singular_values = Float64.(decomposition.S)
    evidence = _evidence_rank(
        singular_values, _evidence_policy(estimator.rank), size(H),
    )
    geometry = max(size(H, 2) - 1, 0)
    resolution = _resolve_rank(estimator.rank, evidence, geometry)
    empty_backend = ESPRITDiagnostics(
        rows, size(H, 2), singular_values, Float64[], nothing, nothing,
        :time_major_block_rows, :right_subspace_columns,
    )
    resolution === nothing &&
        return _rank_failure(
            estimator.rank, evidence, geometry, reduction, empty_backend,
        )
    rank_value = resolution.resolved
    subspace = decomposition.V[:, 1:rank_value]
    leading = @view subspace[1:(end - 1), :]
    trailing = @view subspace[2:end, :]
    leading_singular_values = Float64.(svd(leading; full=false).S)
    leading_rank = _evidence_rank(
        leading_singular_values, NumericalRank(), size(leading),
    )
    shift_condition = isempty(leading_singular_values) ? Inf :
        first(leading_singular_values) / last(leading_singular_values)
    backend = ESPRITDiagnostics(
        rows, size(H, 2), singular_values, leading_singular_values,
        shift_condition, nothing,
        :time_major_block_rows, :right_subspace_columns,
    )
    if leading_rank < rank_value
        common = NodeDiagnostics(
            resolution.requested, resolution.evidence, resolution.geometry,
            resolution.resolved, resolution.clamped, reduction,
        )
        return NodeEstimate(
            NodeEstimationFailure(
                :deficient_shift_solve,
                "ESPRIT leading right-subspace shift system is rank deficient",
            ),
            common,
            backend,
        )
    end
    shift = leading \ trailing
    shift_residual = norm(leading * shift - trailing) /
        max(norm(trailing), eps(Float64))
    backend = ESPRITDiagnostics(
        rows, size(H, 2), singular_values, leading_singular_values,
        shift_condition, shift_residual,
        :time_major_block_rows, :right_subspace_columns,
    )
    nodes = ComplexF64.(conj.(eigvals(shift)))
    all(node -> isfinite(real(node)) && isfinite(imag(node)), nodes) ||
        return NodeEstimate(
            NodeEstimationFailure(:nonfinite_nodes, "ESPRIT produced nonfinite nodes"),
            NodeDiagnostics(
                resolution.requested, resolution.evidence, resolution.geometry,
                resolution.resolved, resolution.clamped, reduction,
            ),
            backend,
        )
    common = _node_diagnostics(resolution, reduction, nodes)
    return NodeEstimate(IdentifiedNodes(nodes), common, backend)
end

function estimate_nodes(estimator::LeftSubspaceESPRIT,
                        sequence::UniformSequence)
    early, reduced, reduction = _pre_estimation(estimator, sequence)
    early === nothing || return early
    rows = estimator.hankel_rows === nothing ? length(sequence) ÷ 2 :
        estimator.hankel_rows
    rows >= 2 || throw(ArgumentError(
        "LeftSubspaceESPRIT needs at least two block-Hankel row groups",
    ))
    H = _hankel(reduced, rows)
    decomposition = svd(H; full=false)
    singular_values = Float64.(decomposition.S)
    evidence = _evidence_rank(
        singular_values, _evidence_policy(estimator.rank), size(H),
    )
    channel_count = size(reduced, 2)
    geometry = min(size(H, 2), channel_count * (rows - 1))
    resolution = _resolve_rank(estimator.rank, evidence, geometry)
    empty_backend = LeftSubspaceESPRITDiagnostics(
        rows, size(H, 2), channel_count, singular_values, Float64[],
        nothing, nothing,
        :time_major_block_rows, :left_time_shift_by_channel_block,
    )
    resolution === nothing && return _rank_failure(
        estimator.rank, evidence, geometry, reduction, empty_backend,
    )

    rank_value = resolution.resolved
    subspace = decomposition.U[:, 1:rank_value]
    leading = @view subspace[1:(end - channel_count), :]
    trailing = @view subspace[(channel_count + 1):end, :]
    leading_singular_values = Float64.(svd(leading; full=false).S)
    leading_rank = _evidence_rank(
        leading_singular_values, NumericalRank(), size(leading),
    )
    shift_condition = isempty(leading_singular_values) ? Inf :
        first(leading_singular_values) / last(leading_singular_values)
    backend = LeftSubspaceESPRITDiagnostics(
        rows, size(H, 2), channel_count, singular_values,
        leading_singular_values, shift_condition, nothing,
        :time_major_block_rows, :left_time_shift_by_channel_block,
    )
    if leading_rank < rank_value
        common = NodeDiagnostics(
            resolution.requested, resolution.evidence, resolution.geometry,
            resolution.resolved, resolution.clamped, reduction,
        )
        return NodeEstimate(
            NodeEstimationFailure(
                :deficient_shift_solve,
                "LeftSubspaceESPRIT leading shift system is rank deficient",
            ),
            common,
            backend,
        )
    end

    shift = leading \ trailing
    shift_residual = norm(leading * shift - trailing) /
        max(norm(trailing), eps(Float64))
    backend = LeftSubspaceESPRITDiagnostics(
        rows, size(H, 2), channel_count, singular_values,
        leading_singular_values, shift_condition, shift_residual,
        :time_major_block_rows, :left_time_shift_by_channel_block,
    )
    nodes = ComplexF64.(eigvals(shift))
    all(node -> isfinite(real(node)) && isfinite(imag(node)), nodes) ||
        return NodeEstimate(
            NodeEstimationFailure(
                :nonfinite_nodes,
                "LeftSubspaceESPRIT produced nonfinite nodes",
            ),
            NodeDiagnostics(
                resolution.requested, resolution.evidence,
                resolution.geometry, resolution.resolved,
                resolution.clamped, reduction,
            ),
            backend,
        )
    common = _node_diagnostics(resolution, reduction, nodes)
    return NodeEstimate(IdentifiedNodes(nodes), common, backend)
end

estimate_nodes(estimator::AbstractNodeEstimator,
               times::AbstractVector{<:Real}, samples) =
    estimate_nodes(estimator, UniformSequence(times, samples))

"""
    ExponentialSum(poles, weights)

Pure representation of `sum_k weights[k] * exp(-poles[k] * t)`. Diagnostics
belong to `ExponentialFit`, not to this value.
"""
struct ExponentialSum{P<:Number,W}
    poles::Vector{P}
    weights::Vector{W}
    function ExponentialSum(poles::AbstractVector{P},
                            weights::AbstractVector{W}) where {P<:Number,W}
        length(poles) == length(weights) ||
            throw(ArgumentError("ExponentialSum needs one weight per pole"))
        isempty(poles) &&
            throw(ArgumentError("zero fits are represented by ZeroFit"))
        all(pole -> isfinite(real(pole)) && isfinite(imag(pole)), poles) ||
            throw(ArgumentError("ExponentialSum poles must be finite"))
        shape = size(first(weights))
        all(weight -> size(weight) == shape, weights) ||
            throw(ArgumentError("ExponentialSum weights need one fixed shape"))
        all(weights) do weight
            values = weight isa Number ? (weight,) : weight
            all(
                value -> isfinite(real(value)) && isfinite(imag(value)),
                values,
            )
        end || throw(ArgumentError("ExponentialSum weights must be finite"))
        return new{P,W}(collect(poles), collect(weights))
    end
end

Base.length(sum_value::ExponentialSum) = length(sum_value.poles)
Base.isempty(::ExponentialSum) = false

function evaluate(sum_value::ExponentialSum, time::Number)
    value = sum_value.weights[1] * exp(-sum_value.poles[1] * time)
    for index in 2:length(sum_value)
        value += sum_value.weights[index] * exp(-sum_value.poles[index] * time)
    end
    return value
end

evaluate(sum_value::ExponentialSum, times) =
    [evaluate(sum_value, time) for time in times]
(sum_value::ExponentialSum)(time::Number) = evaluate(sum_value, time)

abstract type AbstractExponentialFitOutcome end

struct IdentifiedFit{S<:ExponentialSum,N<:NodeEstimate} <:
       AbstractExponentialFitOutcome
    value::S
    node_estimate::N
end

struct ZeroFit{N} <: AbstractExponentialFitOutcome
    sample_shape::NTuple{N,Int}
    element_type::DataType
    node_estimate::NodeEstimate
end

struct FailedFit{N<:NodeEstimate} <: AbstractExponentialFitOutcome
    node_estimate::N
end

struct FitDiagnostics{N}
    t0::Float64
    dt::Float64
    sample_count::Int
    sample_shape::NTuple{N,Int}
    estimator_rank::Int
    retained_order::Int
    least_squares_rank::Int
    condition_number::Float64
    maximum_error::Float64
    relative_l2::Float64
    channel_relative_errors::Vector{Float64}
    holdout_relative_l2::Union{Nothing,Float64}
    aliasing_warning::Bool
    alias_distance::Float64
    unstable_poles::Vector{Int}
    minimum_pole_separation::Float64
    residue_growth::Float64
end

struct ExponentialFit{O<:AbstractExponentialFitOutcome,D,A}
    outcome::O
    diagnostics::D
    attempts::A
end

function _restore_weights(weight_matrix::AbstractMatrix, shape)
    isempty(shape) && return vec(weight_matrix)
    return [
        reshape(copy(weight_matrix[index, :]), shape)
        for index in axes(weight_matrix, 1)
    ]
end

function _channel_relative_errors(reference::AbstractMatrix,
                                  fitted::AbstractMatrix)
    errors = Vector{Float64}(undef, size(reference, 2))
    for channel in axes(reference, 2)
        denominator = norm(view(reference, :, channel))
        errors[channel] =
            norm(view(reference, :, channel) - view(fitted, :, channel)) /
            max(denominator, eps(Float64))
    end
    return errors
end

function _empty_fit_diagnostics(sequence::UniformSequence)
    return FitDiagnostics(
        sequence.t0, sequence.dt, length(sequence), sequence.sample_shape,
        0, 0, 0, 1.0, 0.0, 0.0,
        zeros(Float64, size(sequence.values, 2)), nothing,
        false, Inf, Int[], Inf, 0.0,
    )
end

function _alias_distance(poles::AbstractVector{<:Number}, dt::Float64)
    isempty(poles) && return Inf
    limit = pi / dt
    return minimum(limit - abs(imag(pole)) for pole in poles)
end

function _residue_growth(weights::AbstractMatrix,
                         values::AbstractMatrix)
    return norm(weights) / max(norm(values), eps(Float64))
end

function _one_shot_fit_failure(sequence::UniformSequence,
                               estimate::NodeEstimate,
                               reason::Symbol,
                               message::String)
    failed = NodeEstimate(
        NodeEstimationFailure(reason, message),
        estimate.common,
        estimate.backend,
    )
    return ExponentialFit(
        FailedFit(failed), _empty_fit_diagnostics(sequence), nothing,
    )
end

function fit_exponential_sum(estimator::AbstractNodeEstimator,
                             sequence::UniformSequence)
    estimate = estimate_nodes(estimator, sequence)
    if estimate.outcome isa ZeroSequence
        outcome = ZeroFit(sequence.sample_shape, ComplexF64, estimate)
        return ExponentialFit(outcome, _empty_fit_diagnostics(sequence), nothing)
    elseif estimate.outcome isa AbstractNodeFailure
        outcome = FailedFit(estimate)
        return ExponentialFit(outcome, _empty_fit_diagnostics(sequence), nothing)
    end

    nodes = estimate.outcome.nodes
    isempty(nodes) && return _one_shot_fit_failure(
        sequence,
        estimate,
        :empty_node_set,
        "exponential weight fitting needs at least one node",
    )
    all(node -> isfinite(real(node)) && isfinite(imag(node)), nodes) ||
        return _one_shot_fit_failure(
            sequence,
            estimate,
            :nonfinite_nodes,
            "node estimation produced nonfinite nodes",
        )
    any(iszero, nodes) &&
        return _one_shot_fit_failure(
            sequence,
            estimate,
            :zero_node,
            "node-to-exponent mapping received a zero node",
        )

    poles = try
        ComplexF64.(-log.(nodes) ./ sequence.dt)
    catch error
        return _one_shot_fit_failure(
            sequence,
            estimate,
            :nonfinite_poles,
            "node-to-exponent mapping failed: $(sprint(showerror, error))",
        )
    end
    all(pole -> isfinite(real(pole)) && isfinite(imag(pole)), poles) ||
        return _one_shot_fit_failure(
            sequence,
            estimate,
            :nonfinite_poles,
            "node-to-exponent mapping produced nonfinite poles",
        )

    order = sortperm(poles; by=pole -> (real(pole), imag(pole)))
    poles = poles[order]
    relative_times = sequence.times .- sequence.t0
    design = exp.(-relative_times * transpose(poles))
    all(value -> isfinite(real(value)) && isfinite(imag(value)), design) ||
        return _one_shot_fit_failure(
            sequence,
            estimate,
            :nonfinite_design,
            "exponential design matrix contains nonfinite values",
        )
    relative_weights = try
        ComplexF64.(design \ sequence.values)
    catch error
        return _one_shot_fit_failure(
            sequence,
            estimate,
            :least_squares_failure,
            "exponential least-squares fitting failed: $(sprint(showerror, error))",
        )
    end
    all(
        weight -> isfinite(real(weight)) && isfinite(imag(weight)),
        relative_weights,
    ) || return _one_shot_fit_failure(
        sequence,
        estimate,
        :nonfinite_weights,
        "exponential least-squares fitting produced nonfinite weights",
    )
    absolute_weights = try
        relative_weights .*
        reshape(exp.(poles .* sequence.t0), :, 1)
    catch error
        return _one_shot_fit_failure(
            sequence,
            estimate,
            :nonfinite_weights,
            "shifted-origin weight conversion failed: $(sprint(showerror, error))",
        )
    end
    all(
        weight -> isfinite(real(weight)) && isfinite(imag(weight)),
        absolute_weights,
    ) || return _one_shot_fit_failure(
        sequence,
        estimate,
        :nonfinite_weights,
        "shifted-origin weight conversion produced nonfinite weights",
    )
    weights = _restore_weights(absolute_weights, sequence.sample_shape)
    model = try
        ExponentialSum(poles, weights)
    catch error
        return _one_shot_fit_failure(
            sequence,
            estimate,
            :model_construction_failure,
            "exponential-sum construction failed: $(sprint(showerror, error))",
        )
    end
    fitted = design * relative_weights
    residual = fitted - sequence.values
    scale = max(norm(sequence.values), eps(Float64))
    aliasing = any(
        abs(imag(pole)) > pi / sequence.dt * (1 + 100eps(Float64))
        for pole in poles
    )
    unstable = findall(pole -> real(pole) < -sqrt(eps(Float64)), poles)
    diagnostics = FitDiagnostics(
        sequence.t0,
        sequence.dt,
        length(sequence),
        sequence.sample_shape,
        estimate.common.resolved_rank,
        length(poles),
        rank(design),
        cond(design),
        maximum(abs, residual),
        norm(residual) / scale,
        _channel_relative_errors(sequence.values, fitted),
        nothing,
        aliasing,
        _alias_distance(poles, sequence.dt),
        unstable,
        _minimum_separation(poles),
        _residue_growth(absolute_weights, sequence.values),
    )
    return ExponentialFit(
        IdentifiedFit(model, estimate), diagnostics, nothing,
    )
end

fit_exponential_sum(estimator::AbstractNodeEstimator,
                    times::AbstractVector{<:Real}, samples) =
    fit_exponential_sum(estimator, UniformSequence(times, samples))

fit_exponential_sum(estimator::AbstractNodeEstimator,
                    series::CorrelatorSeries) =
    fit_exponential_sum(estimator, series.times, series.values)

function _zero_value(outcome::ZeroFit)
    isempty(outcome.sample_shape) && return zero(ComplexF64)
    return zeros(ComplexF64, outcome.sample_shape)
end

evaluate(fit::ExponentialFit{<:IdentifiedFit}, time::Number) =
    evaluate(fit.outcome.value, time)
evaluate(fit::ExponentialFit{<:IdentifiedFit}, times) =
    evaluate(fit.outcome.value, times)
evaluate(fit::ExponentialFit{<:ZeroFit}, time::Number) =
    _zero_value(fit.outcome)
evaluate(fit::ExponentialFit{<:ZeroFit}, times) =
    [_zero_value(fit.outcome) for _ in times]

function evaluate(fit::ExponentialFit{<:FailedFit}, time_or_times)
    failure = fit.outcome.node_estimate.outcome
    reason = failure isa NodeEstimationFailure ?
        failure.reason : :reduction_erased_signal
    message = failure isa NodeEstimationFailure ?
        failure.message : "sample reduction erased a nonzero input sequence"
    throw(ArgumentError(
        "cannot evaluate failed exponential fit: $reason: $message",
    ))
end
