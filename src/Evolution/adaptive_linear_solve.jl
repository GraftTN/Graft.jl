# Residual-controlled bond expansion for the one-site variational linear
# solver. The physical residual is always retained exactly for convergence;
# only a private, explicitly truncated copy is used to choose bond directions.

function _rde_nonnegative_finite(name::AbstractString, value::Real)
    x = Float64(value)
    isfinite(x) && x >= 0 ||
        throw(ArgumentError("$name must be finite and nonnegative"))
    return x
end

function _rde_positive_limit(name::AbstractString, value::Integer)
    x = Int(value)
    x >= 1 || throw(ArgumentError("$name must be positive"))
    return x
end

function _rde_nonnegative_limit(name::AbstractString, value::Integer)
    x = Int(value)
    x >= 0 || throw(ArgumentError("$name must be nonnegative"))
    return x
end

function _rde_check_truncation(
        trunc::TruncationScheme, field::AbstractString)
    trunc.maxdim >= 1 ||
        throw(ArgumentError(
            "ResidualDrivenExpansion $field.maxdim must be positive"))
    _rde_nonnegative_finite(
        "ResidualDrivenExpansion $field.atol", trunc.atol)
    _rde_nonnegative_finite(
        "ResidualDrivenExpansion $field.rtol", trunc.rtol)
    discarded = _rde_nonnegative_finite(
        "ResidualDrivenExpansion $field.discarded_weight",
        trunc.discarded_weight,
    )
    discarded <= 1 ||
        throw(ArgumentError(
            "ResidualDrivenExpansion $field.discarded_weight must not exceed one"))
    return trunc
end

"""
    ResidualDrivenExpansion(; trunc, residual_trunc, max_add, max_total_add,
                 max_edges, max_rounds, schedule, weight_atol, weight_rtol,
                 enrichment_atol, enrichment_rtol, compression_atol,
                 compression_rtol, residual_max_bond, residual_max_payload)

Validated policy for residual-controlled bond expansion. `max_add` is the
per-edge, per-round rank increase; `max_total_add` is the global increase over
one [`residual_driven_linsolve!`](@ref) call. `max_edges` bounds selected edges per round,
and `max_rounds` bounds expansion rounds. `trunc.maxdim` is the hard final rank
cap. The remaining `trunc` fields are validated and retained as the declared
state policy, while the state-preserving splice never truncates an existing
state direction.

`residual_trunc` independently controls a private residual surrogate used only
for edge scoring. Its actual global error is measured from exact overlaps and
must satisfy `max(compression_atol, compression_rtol * normres)`.
`residual_max_bond` and `residual_max_payload` guard construction of the
uncompressed physical residual.
"""
struct ResidualDrivenExpansion
    trunc::TruncationScheme
    residual_trunc::TruncationScheme
    max_add::Int
    max_total_add::Int
    max_edges::Int
    max_rounds::Int
    schedule::Symbol
    weight_atol::Float64
    weight_rtol::Float64
    enrichment_atol::Float64
    enrichment_rtol::Float64
    compression_atol::Float64
    compression_rtol::Float64
    residual_max_bond::Int
    residual_max_payload::Int

    function ResidualDrivenExpansion(
            trunc::TruncationScheme,
            residual_trunc::TruncationScheme,
            max_add::Int,
            max_total_add::Int,
            max_edges::Int,
            max_rounds::Int,
            schedule::Symbol,
            weight_atol::Float64,
            weight_rtol::Float64,
            enrichment_atol::Float64,
            enrichment_rtol::Float64,
            compression_atol::Float64,
            compression_rtol::Float64,
            residual_max_bond::Int,
            residual_max_payload::Int)
        _rde_check_truncation(trunc, "trunc")
        _rde_check_truncation(residual_trunc, "residual_trunc")
        _rde_nonnegative_limit("ResidualDrivenExpansion max_add", max_add)
        _rde_nonnegative_limit("ResidualDrivenExpansion max_total_add", max_total_add)
        _rde_nonnegative_limit("ResidualDrivenExpansion max_edges", max_edges)
        _rde_nonnegative_limit("ResidualDrivenExpansion max_rounds", max_rounds)
        schedule === :largest_uncovered ||
            throw(ArgumentError(
                "ResidualDrivenExpansion schedule must be :largest_uncovered"))
        _rde_nonnegative_finite("ResidualDrivenExpansion weight_atol", weight_atol)
        _rde_nonnegative_finite("ResidualDrivenExpansion weight_rtol", weight_rtol)
        _rde_nonnegative_finite(
            "ResidualDrivenExpansion enrichment_atol", enrichment_atol)
        _rde_nonnegative_finite(
            "ResidualDrivenExpansion enrichment_rtol", enrichment_rtol)
        _rde_nonnegative_finite(
            "ResidualDrivenExpansion compression_atol", compression_atol)
        _rde_nonnegative_finite(
            "ResidualDrivenExpansion compression_rtol", compression_rtol)
        _rde_positive_limit(
            "ResidualDrivenExpansion residual_max_bond", residual_max_bond)
        _rde_positive_limit(
            "ResidualDrivenExpansion residual_max_payload", residual_max_payload)
        return new(
            trunc,
            residual_trunc,
            max_add,
            max_total_add,
            max_edges,
            max_rounds,
            schedule,
            weight_atol,
            weight_rtol,
            enrichment_atol,
            enrichment_rtol,
            compression_atol,
            compression_rtol,
            residual_max_bond,
            residual_max_payload,
        )
    end
end

function ResidualDrivenExpansion(;
        trunc::TruncationScheme=TruncationScheme(; maxdim=100),
        residual_trunc::TruncationScheme=TruncationScheme(),
        max_add::Integer=8,
        max_total_add::Integer=32,
        max_edges::Integer=typemax(Int),
        max_rounds::Integer=4,
        schedule::Symbol=:largest_uncovered,
        weight_atol::Real=1e-12,
        weight_rtol::Real=1e-10,
        enrichment_atol::Real=1e-12,
        enrichment_rtol::Real=1e-10,
        compression_atol::Real=1e-12,
        compression_rtol::Real=1e-10,
        residual_max_bond::Integer=4096,
        residual_max_payload::Integer=100_000_000)
    return ResidualDrivenExpansion(
        trunc,
        residual_trunc,
        Int(max_add),
        Int(max_total_add),
        Int(max_edges),
        Int(max_rounds),
        schedule,
        Float64(weight_atol),
        Float64(weight_rtol),
        Float64(enrichment_atol),
        Float64(enrichment_rtol),
        Float64(compression_atol),
        Float64(compression_rtol),
        Int(residual_max_bond),
        Int(residual_max_payload),
    )
end

"""Diagnostics for one exact physical residual."""
struct LinearResidualReport
    normres::Float64
    compression_error::Float64
    edge_ranks::Vector{Pair{Symbol,Int}}
end

"""Residual score, selection, and state-preserving splice data for one edge."""
struct ResidualExpansionEdgeReport
    edge::Pair{Symbol,Symbol}
    rank_before::Int
    rank_after::Int
    uncovered_weight::Float64
    relative_weight::Float64
    selected::Bool
    requested_rank::Int
    added_rank::Int
    embedding_error::Float64
end

"""
Diagnostics for one residual-driven expansion round.

The round interleaves all-edge scoring with one state-preserving splice at a
time. Each edge report therefore aggregates the round: ranks are the initial
and final ranks, requested and added ranks are cumulative, and the reported
weight is the largest score at which a selected edge was chosen (or the final
score for an unselected edge). `weight_threshold` is the smallest deterministic
selection threshold encountered while rescoring.
"""
struct ResidualExpansionReport
    edges::Vector{ResidualExpansionEdgeReport}
    selected_edges::Int
    total_added::Int
    weight_threshold::Float64
    embedding_error::Float64
    remaining_add::Int
    stop_reason::Symbol
end

"""
Diagnostics for the complete adaptive linear solve.

`physical_residuals` is the authoritative uncompressed residual trajectory.
The local one-site solve's own status remains available in `solves`, but cannot
turn an exhausted expansion budget into convergence. A caught local-solver
failure leaves `committed == false` and records its rendered diagnostic in
`exception_message`.
"""
struct ResidualDrivenReport
    physical_residuals::Vector{Float64}
    converged::Bool
    stop_reason::Symbol
    solves::Vector{_LinInfo}
    residuals::Vector{LinearResidualReport}
    expansions::Vector{ResidualExpansionReport}
    total_added::Int
    committed::Bool
    exception_message::Union{Nothing,String}
end

function _rde_finite_number(x::Number)
    return isfinite(real(x)) && isfinite(imag(x))
end

function _rde_check_linear_system(
        ψ::TTNS, H::TTNO, rhs::TTNS, a0::Number, a1::Number)
    topology(ψ) == topology(H) == topology(rhs) ||
        throw(ArgumentError(
            "linear_residual: ψ, H, and rhs must share topology"))
    ψ.hasphys == H.hasphys == rhs.hasphys ||
        throw(ArgumentError(
            "linear_residual: ψ, H, and rhs must share physical layout"))
    spacetype(ψ) == spacetype(H) == spacetype(rhs) ||
        throw(ArgumentError(
            "linear_residual: ψ, H, and rhs must share spacetype"))
    eltype(ψ) == eltype(rhs) ||
        throw(ArgumentError(
            "linear_residual: ψ and rhs must have the same eltype"))
    _rde_finite_number(a0) ||
        throw(ArgumentError("linear_residual: a0 must be finite"))
    _rde_finite_number(a1) ||
        throw(ArgumentError("linear_residual: a1 must be finite"))
    T = eltype(ψ)
    if !(T <: Complex)
        ((a0 isa Complex && !isreal(a0)) ||
         (a1 isa Complex && !isreal(a1)) ||
         (eltype(H) <: Complex)) &&
            throw(ArgumentError(
                "linear_residual: a real ψ cannot represent a complex system"))
    end
    coefficients = try
        convert(T, a0), convert(T, a1)
    catch error
        (error isa InexactError || error isa MethodError) || rethrow()
        throw(ArgumentError(
            "linear_residual: coefficients are not representable by $T"))
    end
    return coefficients
end

"""
    linear_residual(ψ, H, rhs; a0=1, a1=1,
                    max_bond=4096, max_payload=100_000_000)
        -> (residual, report)

Construct the exact canonical residual
`rhs - (a0 * I + a1 * H) * ψ`. No variational fit or residual compression is
used. Zero-coefficient terms are omitted before the exact direct sum, so the
reported edge ranks describe the residual actually retained.
"""
function linear_residual(
        ψ::TTNS, H::TTNO, rhs::TTNS;
        a0::Number=one(eltype(ψ)),
        a1::Number=one(eltype(ψ)),
        max_bond::Integer=4096,
        max_payload::Integer=100_000_000)
    residual_max_bond = _rde_positive_limit(
        "linear_residual max_bond", max_bond)
    residual_max_payload = _rde_positive_limit(
        "linear_residual max_payload", max_payload)
    a0T, a1T = _rde_check_linear_system(ψ, H, rhs, a0, a1)

    states = TTNS[rhs]
    coefficients = eltype(ψ)[one(eltype(ψ))]
    if !iszero(a0T)
        push!(states, ψ)
        push!(coefficients, -a0T)
    end
    if !iszero(a1T)
        acted = apply(H, ψ; center=center(ψ))
        eltype(acted) == eltype(ψ) ||
            throw(ArgumentError(
                "linear_residual: H * ψ changes eltype; convert inputs explicitly"))
        push!(states, acted)
        push!(coefficients, -a1T)
    end
    residual = exact_linear_combination(
        states, coefficients;
        max_bond=residual_max_bond,
        max_payload=residual_max_payload)
    t = topology(residual)
    ranks = Pair{Symbol,Int}[
        nodeid(t, child) => dim(virtualspace(residual, child))
        for (child, _) in edges(t)
    ]
    report = LinearResidualReport(
        Float64(norm(residual)),
        0.0,
        ranks,
    )
    return residual, report
end

function _rde_no_truncation(trunc::TruncationScheme)
    return trunc.maxdim == typemax(Int) &&
        iszero(trunc.atol) &&
        iszero(trunc.rtol) &&
        iszero(trunc.discarded_weight)
end

function _rde_residual_ranks(residual::TTNS)
    t = topology(residual)
    return Pair{Symbol,Int}[
        nodeid(t, child) => dim(virtualspace(residual, child))
        for (child, _) in edges(t)
    ]
end

"""
Return an independently owned residual surrogate and a report whose norm is
still the exact physical residual norm. Compression proceeds edge by edge
through the ordinary two-site split, so every singular-spectrum decision uses
the declared `TruncationScheme`.
"""
function _rde_compress_residual(
        residual::TTNS,
        exact_report::LinearResidualReport,
        policy::ResidualDrivenExpansion)
    if _rde_no_truncation(policy.residual_trunc)
        return copy(residual), LinearResidualReport(
            exact_report.normres,
            0.0,
            copy(exact_report.edge_ranks),
        )
    end

    surrogate = copy(residual)
    t = topology(surrogate)
    original_center = center(surrogate)
    for child in postorder(t)
        parent = t.parent[child]
        parent == 0 && continue
        move_center!(surrogate, child)
        two_site = two_site_tensor(surrogate, child, parent)
        split_two_site!(
            surrogate,
            two_site,
            child,
            parent;
            trunc=policy.residual_trunc,
            center_on=:m,
        )
    end
    move_center!(surrogate, original_center)

    surrogate_norm2 = Float64(real(inner(surrogate, surrogate)))
    overlap = inner(residual, surrogate)
    error2 = exact_report.normres^2 + surrogate_norm2 -
        2 * Float64(real(overlap))
    roundoff_scale = max(
        exact_report.normres^2,
        abs(surrogate_norm2),
        2 * abs(overlap),
        1.0,
    )
    error2 >= -64eps(Float64) * roundoff_scale ||
        throw(ArgumentError(
            "residual compression produced an inconsistent overlap norm"))
    compression_error = sqrt(max(error2, 0.0))
    isfinite(compression_error) ||
        throw(ArgumentError(
            "residual compression produced a non-finite global error"))
    return surrogate, LinearResidualReport(
        exact_report.normres,
        compression_error,
        _rde_residual_ranks(surrogate),
    )
end

function _rde_compression_accurate(
        report::LinearResidualReport,
        policy::ResidualDrivenExpansion)
    tolerance = max(
        policy.compression_atol,
        policy.compression_rtol * report.normres,
    )
    return report.compression_error <= tolerance
end

function _rde_check_state_residual(ψ::TTNS, residual::TTNS)
    topology(ψ) == topology(residual) ||
        throw(ArgumentError(
            "residual_expand!: ψ and residual must share topology"))
    ψ.hasphys == residual.hasphys ||
        throw(ArgumentError(
            "residual_expand!: ψ and residual must share physical layout"))
    spacetype(ψ) == spacetype(residual) ||
        throw(ArgumentError(
            "residual_expand!: ψ and residual must share spacetype"))
    eltype(ψ) == eltype(residual) ||
        throw(ArgumentError(
            "residual_expand!: ψ and residual must have the same eltype"))
    t = topology(ψ)
    for n in 1:nnodes(t)
        hasphys(ψ, n) || continue
        physspace(ψ, n) == physspace(residual, n) ||
            throw(ArgumentError(
                "residual_expand!: physical space differs at $(nodeid(t, n))"))
    end
    domain(ψ.tensors[t.root])[1] ==
        domain(residual.tensors[t.root])[1] ||
        throw(ArgumentError(
            "residual_expand!: ψ and residual have different root charge spaces"))
    return nothing
end

function _rde_check_state_rank_cap(
        ψ::TTNS, policy::ResidualDrivenExpansion)
    t = topology(ψ)
    for (child, _) in edges(t)
        rank = dim(virtualspace(ψ, child))
        rank <= policy.trunc.maxdim ||
            throw(ArgumentError(
                "ResidualDrivenExpansion trunc.maxdim=$(policy.trunc.maxdim) " *
                "is below the current rank $rank on edge " *
                string(nodeid(t, child))))
    end
    return nothing
end

struct _RDECandidate
    child::Int
    parent::Int
    rank_before::Int
    rank_room::Int
    possible_add::Int
    weight::Float64
end

function _rde_local_residual_tensor(
        ψ::TTNS, residual::TTNS, child::Int, parent::Int)
    cache = Networks._FitCache(topology(ψ), nothing)
    window = Networks._fit_two_site_tensor(
        cache,
        ψ,
        residual,
        child,
        parent,
    )
    child_outputs = numout(ψ.tensors[child])
    nlegs = numind(window)
    return permute(
        window,
        (
            ntuple(identity, child_outputs),
            ntuple(
                offset -> child_outputs + offset,
                nlegs - child_outputs,
            ),
        ),
    )
end

function _rde_uncovered_weight(
        A::AbstractTensorMap, predictor::AbstractTensorMap)
    complement = left_null(A)
    dim(domain(complement)) == 0 && return 0.0
    uncovered_norm = Float64(norm(complement' * predictor))
    weight = abs2(uncovered_norm)
    isfinite(weight) ||
        throw(ArgumentError(
            "residual_expand!: non-finite uncovered residual weight"))
    return weight
end

function _rde_enrichment_predictor(
        A::AbstractTensorMap,
        predictor::AbstractTensorMap,
        room::Int,
        policy::ResidualDrivenExpansion)
    room > 0 || return nothing
    complement = left_null(A)
    dim(domain(complement)) > 0 || return nothing
    projected = complement' * predictor
    selected, _, _ = split_svd(
        projected,
        TruncationScheme(
            maxdim=room,
            atol=policy.enrichment_atol,
            rtol=policy.enrichment_rtol,
            discarded_weight=policy.trunc.discarded_weight,
        ),
    )
    dim(domain(selected)) > 0 || return nothing
    return complement * selected
end

function _rde_possible_add(
        A::AbstractTensorMap,
        predictor::AbstractTensorMap,
        room::Int,
        policy::ResidualDrivenExpansion)
    enrichment = _rde_enrichment_predictor(
        A, predictor, room, policy)
    enrichment === nothing && return 0
    return dim(domain(enrichment))
end

function _rde_score_edges!(
        ψ::TTNS,
        residual::TTNS,
        policy::ResidualDrivenExpansion,
        requested::Dict{Int,Int}=Dict{Int,Int}())
    t = topology(ψ)
    candidates = _RDECandidate[]
    for (child, parent) in edges(t)
        # Scoring must not gauge-sweep the staged state itself. A freshly
        # zero-padded bond is an enlarged variational manifold even though the
        # represented vector still has its old Schmidt rank; a thin QR center
        # move across that bond may legitimately remove the zero direction.
        # Gauge an edge-local copy so later candidates see every splice that
        # the round has already installed.
        edge_state = copy(ψ)
        move_center!(edge_state, child)
        A = edge_state.tensors[child]
        predictor = _rde_local_residual_tensor(
            edge_state, residual, child, parent)
        rank_before = dim(domain(A))
        room = min(
            max(policy.max_add - get(requested, child, 0), 0),
            max(policy.trunc.maxdim - rank_before, 0),
        )
        weight = _rde_uncovered_weight(A, predictor)
        possible = _rde_possible_add(A, predictor, room, policy)
        push!(candidates, _RDECandidate(
            child,
            parent,
            rank_before,
            room,
            possible,
            weight,
        ))
    end
    return candidates
end

function _rde_scoring_threshold(
        candidates::Vector{_RDECandidate},
        policy::ResidualDrivenExpansion)
    maximum_weight = maximum(
        (candidate.weight for candidate in candidates);
        init=0.0,
    )
    threshold = max(
        policy.weight_atol,
        policy.weight_rtol * maximum_weight,
    )
    return threshold, maximum_weight
end

function _rde_choose_candidate(
        candidates::Vector{_RDECandidate},
        policy::ResidualDrivenExpansion,
        remaining_add::Int,
        selected::Set{Int})
    threshold, maximum_weight = _rde_scoring_threshold(
        candidates, policy)
    remaining_add > 0 || return nothing, 0, threshold, maximum_weight
    eligible = _RDECandidate[
        candidate for candidate in candidates
        if candidate.weight > threshold &&
            candidate.possible_add > 0 &&
            !(candidate.child in selected) &&
            length(selected) < policy.max_edges
    ]
    sort!(eligible; by=candidate -> (-candidate.weight, candidate.child))
    isempty(eligible) &&
        return nothing, 0, threshold, maximum_weight
    candidate = first(eligible)
    request = min(candidate.possible_add, remaining_add)
    return candidate, request, threshold, maximum_weight
end

function _rde_splice_candidate!(
        ψ::TTNS,
        residual::TTNS,
        candidate::_RDECandidate,
        request::Int,
        policy::ResidualDrivenExpansion)
    t = topology(ψ)
    gauge_cache = EnvCache(t)
    child, parent = candidate.child, candidate.parent
    move_center!(ψ, child; cache=gauge_cache)
    A = ψ.tensors[child]
    predictor = _rde_local_residual_tensor(
        ψ, residual, child, parent)
    rank_before = dim(domain(A))
    request <= candidate.rank_room ||
        throw(ArgumentError(
            "residual_expand!: requested rank exceeds the current edge budget"))
    enrichment = _rde_enrichment_predictor(
        A, predictor, request, policy)
    enrichment === nothing && return 0, 0.0
    U, R = Contractions._expand_enrich_split(
        A,
        enrichment;
        maxdim=min(policy.trunc.maxdim, rank_before + request),
        max_add=request,
        enr_rtol=0.0,
        enr_atol=0.0,
    )
    rank_added = dim(domain(U)) - rank_before
    rank_added >= 0 ||
        throw(ArgumentError("residual_expand!: enrichment reduced a bond"))
    rank_added <= request ||
        throw(ArgumentError(
            "residual_expand!: enrichment exceeded the allocated rank"))
    embedding_error = Float64(norm(A - U * R))
    isfinite(embedding_error) ||
        throw(ArgumentError(
            "residual_expand!: non-finite state-embedding error"))
    rank_added == 0 && return 0, embedding_error

    ψ.tensors[child] = U
    link = Networks._pivotal_link(R)
    ψ.tensors[parent] = absorb_on_leg(
        ψ.tensors[parent],
        link,
        childslot(t, parent, child),
    )
    ψ.center = parent
    invalidate_edge!(gauge_cache, child, parent)
    return rank_added, embedding_error
end

mutable struct _RDEEdgeRoundAccumulator
    rank_before::Int
    final_weight::Float64
    final_relative_weight::Float64
    selected_weight::Float64
    selected_relative_weight::Float64
    requested_rank::Int
    added_rank::Int
    embedding_error::Float64
end

function _rde_round_accumulators(
        ψ::TTNS, candidates::Vector{_RDECandidate})
    accumulators = Dict{Int,_RDEEdgeRoundAccumulator}()
    for candidate in candidates
        accumulators[candidate.child] = _RDEEdgeRoundAccumulator(
            candidate.rank_before,
            candidate.weight,
            0.0,
            0.0,
            0.0,
            0,
            0,
            0.0,
        )
    end
    length(accumulators) == length(edges(topology(ψ))) ||
        throw(ArgumentError(
            "residual_expand!: scoring did not cover every tree edge"))
    return accumulators
end

function _rde_record_scores!(
        accumulators::Dict{Int,_RDEEdgeRoundAccumulator},
        candidates::Vector{_RDECandidate},
        maximum_weight::Float64)
    for candidate in candidates
        accumulator = accumulators[candidate.child]
        accumulator.final_weight = candidate.weight
        accumulator.final_relative_weight = iszero(maximum_weight) ? 0.0 :
            candidate.weight / maximum_weight
    end
    return accumulators
end

function _rde_expansion_stop_reason(
        candidates::Vector{_RDECandidate},
        requested::Dict{Int,Int},
        total_added::Int,
        threshold::Float64,
        policy::ResidualDrivenExpansion,
        remaining_add::Int)
    total_added > 0 && return :expanded
    isempty(candidates) && return :no_edges
    (policy.max_add == 0 || policy.max_edges == 0) &&
        return :expansion_disabled
    (remaining_add == 0 || policy.max_total_add == 0) &&
        return :budget_exhausted
    maximum_weight = maximum(
        (candidate.weight for candidate in candidates);
        init=0.0,
    )
    iszero(maximum_weight) && return :zero_weight
    maximum_weight <= threshold && return :weight_threshold
    all(candidate.rank_room == 0 for candidate in candidates) &&
        return :rank_capped
    all(candidate.possible_add == 0 for candidate in candidates) &&
        return :enrichment_threshold
    isempty(requested) && return :selection_budget_exhausted
    return :no_usable_directions
end

function _rde_commit!(destination::TTNS, source::TTNS)
    replacements = copy.(source.tensors)
    for n in eachindex(replacements)
        destination.tensors[n] = replacements[n]
    end
    destination.center = source.center
    return destination
end

"""
    residual_expand!(ψ, residual, policy;
                 remaining_add=policy.max_total_add) -> (ψ, report)

Score every tree edge by the norm of the exact residual component outside the
current child-side bond basis. Choose the largest eligible edge, splice it,
then rescore every edge before choosing again. This allows a state-preserving
splice to make an adjacent residual direction representable within the same
round. Ties use the child index, and the cumulative round remains bounded by
the per-edge, distinct-edge-count, global, and final-rank budgets.

Selected residual directions are inserted through the existing enrichment
split. The old tensor is factored exactly through the enlarged isometry, so
the represented state is unchanged at the instant of expansion. Work is
performed on a private copy and committed only after the complete round
succeeds.
"""
function residual_expand!(
        ψ::TTNS,
        residual::TTNS,
        policy::ResidualDrivenExpansion;
        remaining_add::Integer=policy.max_total_add)
    remaining = _rde_nonnegative_limit(
        "residual_expand! remaining_add", remaining_add)
    remaining <= policy.max_total_add ||
        throw(ArgumentError(
            "residual_expand!: remaining_add exceeds policy.max_total_add"))
    _rde_check_state_residual(ψ, residual)
    _rde_check_state_rank_cap(ψ, policy)

    stage = copy(ψ)
    original_center = center(stage)
    requested = Dict{Int,Int}()
    selected = Set{Int}()
    candidates = _rde_score_edges!(
        stage, residual, policy, requested)
    accumulators = _rde_round_accumulators(stage, candidates)
    thresholds = Float64[]
    remaining_round = remaining

    while true
        candidate, request, threshold, maximum_weight =
            _rde_choose_candidate(
                candidates,
                policy,
                remaining_round,
                selected,
            )
        push!(thresholds, threshold)
        _rde_record_scores!(
            accumulators, candidates, maximum_weight)
        candidate === nothing && break

        child = candidate.child
        accumulator = accumulators[child]
        push!(selected, child)
        requested[child] = get(requested, child, 0) + request
        accumulator.requested_rank += request
        candidate.weight > accumulator.selected_weight &&
            begin
                accumulator.selected_weight = candidate.weight
                accumulator.selected_relative_weight =
                    iszero(maximum_weight) ? 0.0 :
                    candidate.weight / maximum_weight
            end
        rank_added, embedding_error = _rde_splice_candidate!(
            stage,
            residual,
            candidate,
            request,
            policy,
        )
        accumulator.added_rank += rank_added
        accumulator.embedding_error = max(
            accumulator.embedding_error, embedding_error)
        remaining_round -= rank_added

        # A rescore is intentional even when a budget has just been exhausted:
        # it validates the final staged state and leaves coherent final scores.
        candidates = _rde_score_edges!(
            stage, residual, policy, requested)
    end
    move_center!(stage, original_center)

    reports = ResidualExpansionEdgeReport[]
    t = topology(stage)
    for (child, parent) in edges(t)
        accumulator = accumulators[child]
        was_selected = child in selected
        rank_after = dim(virtualspace(stage, child))
        actual_added = rank_after - accumulator.rank_before
        actual_added >= 0 ||
            throw(ArgumentError(
                "residual_expand!: staged gauge transport reduced an existing bond"))
        actual_added == accumulator.added_rank ||
            throw(ArgumentError(
                "residual_expand!: staged gauge transport did not preserve " *
                "the added rank on edge $(nodeid(t, child))"))
        push!(reports, ResidualExpansionEdgeReport(
            nodeid(t, child) => nodeid(t, parent),
            accumulator.rank_before,
            rank_after,
            was_selected ? accumulator.selected_weight :
                accumulator.final_weight,
            was_selected ? accumulator.selected_relative_weight :
                accumulator.final_relative_weight,
            was_selected,
            accumulator.requested_rank,
            actual_added,
            accumulator.embedding_error,
        ))
    end
    total_added = sum(edge.added_rank for edge in reports; init=0)
    embedding_error = maximum(
        accumulator.embedding_error for accumulator in values(accumulators);
        init=0.0,
    )
    threshold = minimum(thresholds; init=policy.weight_atol)
    final_threshold, _ = _rde_scoring_threshold(candidates, policy)
    stop_reason = _rde_expansion_stop_reason(
        candidates,
        requested,
        total_added,
        final_threshold,
        policy,
        remaining_round,
    )
    report = ResidualExpansionReport(
        reports,
        count(edge -> edge.selected, reports),
        total_added,
        threshold,
        embedding_error,
        remaining_round,
        stop_reason,
    )
    total_added > 0 && _rde_commit!(ψ, stage)
    return ψ, report
end

function _rde_linear_report(
        physical_residuals::Vector{Float64},
        converged::Bool,
        stop_reason::Symbol,
        solves::Vector{_LinInfo},
        residuals::Vector{LinearResidualReport},
        expansions::Vector{ResidualExpansionReport},
        total_added::Int,
        committed::Bool;
        exception_message::Union{Nothing,String}=nothing)
    return ResidualDrivenReport(
        physical_residuals,
        converged,
        stop_reason,
        solves,
        residuals,
        expansions,
        total_added,
        committed,
        exception_message,
    )
end

function _rde_local_failure_reason(error::_LocalLinearSolveFailure)
    return error.reason === :nonfinite_local_solution ?
        :nonfinite_local_solution : :local_solver_failed
end

"""
    residual_driven_linsolve!(ψ, H, rhs, policy; a0=1, a1=1,
                   krylovdim=30, maxiter=100, tol=1e-10,
                   fit_nsweeps=4, fit_tol=1e-10, fit_verbose=false)
        -> (ψ, report)

Run the controlled one-site
`solve -> exact physical residual -> residual expansion -> solve` loop.
Convergence is determined only by the exact physical residual after a solve.
Only the policy-truncated residual copy is used for edge scoring, and a global
exact-overlap error above its declared tolerance stops as
`:compression_inaccurate`.
Expansion-disabled, rank-capped, and exhausted-budget exits retain distinct
stop reasons and never report convergence.

The caller's `ψ` is not touched until a complete normal exit. Any exception
from residual construction, edge scoring, or a splice therefore leaves it
unchanged. The base solver's typed local failures become uncommitted
`:local_solver_failed` or `:nonfinite_local_solution` reports; unexpected
exceptions still propagate. Non-finite physical residual output is an
uncommitted `:nonfinite_residual`.
"""
function residual_driven_linsolve!(
        ψ::TTNS,
        H::TTNO,
        rhs::TTNS,
        policy::ResidualDrivenExpansion;
        a0::Number=one(eltype(ψ)),
        a1::Number=one(eltype(ψ)),
        krylovdim::Int=30,
        maxiter::Int=100,
        tol::Float64=1e-10,
        fit_nsweeps::Int=4,
        fit_tol::Float64=1e-10,
        fit_verbose::Bool=false)
    _check_linsolve_args(
        ψ,
        H,
        rhs,
        a0,
        a1,
        krylovdim,
        maxiter,
        tol,
        fit_nsweeps,
        fit_tol,
    )
    _rde_check_linear_system(ψ, H, rhs, a0, a1)
    _rde_check_state_rank_cap(ψ, policy)

    stage = copy(ψ)
    solves = _LinInfo[]
    residuals = LinearResidualReport[]
    physical_residuals = Float64[]
    expansions = ResidualExpansionReport[]
    total_added = 0
    expansion_round = 0

    while true
        solve_report = try
            _, result = linsolve!(
                stage,
                H,
                rhs;
                a0,
                a1,
                krylovdim,
                maxiter,
                tol,
                fit_nsweeps,
                fit_tol,
                fit_verbose,
                _fail_on_local=true,
            )
            result
        catch error
            error isa _LocalLinearSolveFailure || rethrow()
            reason = _rde_local_failure_reason(error)
            report = _rde_linear_report(
                physical_residuals,
                false,
                reason,
                solves,
                residuals,
                expansions,
                total_added,
                false;
                exception_message=sprint(showerror, error),
            )
            return ψ, report
        end
        push!(solves, solve_report)
        residual, residual_report = linear_residual(
            stage,
            H,
            rhs;
            a0,
            a1,
            max_bond=policy.residual_max_bond,
            max_payload=policy.residual_max_payload,
        )
        push!(physical_residuals, residual_report.normres)

        if !isfinite(residual_report.normres)
            push!(residuals, residual_report)
            report = _rde_linear_report(
                physical_residuals,
                false,
                :nonfinite_residual,
                solves,
                residuals,
                expansions,
                total_added,
                false,
            )
            return ψ, report
        end
        if residual_report.normres <= tol
            push!(residuals, residual_report)
            _rde_commit!(ψ, stage)
            report = _rde_linear_report(
                physical_residuals,
                true,
                :converged,
                solves,
                residuals,
                expansions,
                total_added,
                true,
            )
            return ψ, report
        end
        if expansion_round >= policy.max_rounds
            push!(residuals, residual_report)
            _rde_commit!(ψ, stage)
            report = _rde_linear_report(
                physical_residuals,
                false,
                :max_rounds_exhausted,
                solves,
                residuals,
                expansions,
                total_added,
                true,
            )
            return ψ, report
        end
        if total_added >= policy.max_total_add
            push!(residuals, residual_report)
            _rde_commit!(ψ, stage)
            report = _rde_linear_report(
                physical_residuals,
                false,
                :global_budget_exhausted,
                solves,
                residuals,
                expansions,
                total_added,
                true,
            )
            return ψ, report
        end

        scoring_residual, scoring_report = _rde_compress_residual(
            residual,
            residual_report,
            policy,
        )
        push!(residuals, scoring_report)
        if !_rde_compression_accurate(scoring_report, policy)
            report = _rde_linear_report(
                physical_residuals,
                false,
                :compression_inaccurate,
                solves,
                residuals,
                expansions,
                total_added,
                false,
            )
            return ψ, report
        end

        _, expansion_report = residual_expand!(
            stage,
            scoring_residual,
            policy;
            remaining_add=policy.max_total_add - total_added,
        )
        push!(expansions, expansion_report)
        total_added += expansion_report.total_added
        expansion_round += 1
        if expansion_report.total_added == 0
            _rde_commit!(ψ, stage)
            report = _rde_linear_report(
                physical_residuals,
                false,
                expansion_report.stop_reason,
                solves,
                residuals,
                expansions,
                total_added,
                true,
            )
            return ψ, report
        end
    end
end
