"""Per-edge diagnostics for paper-style global subspace expansion."""
struct GlobalSubspaceEdgeInfo
    edge::Pair{Symbol,Symbol}
    direction::Symbol
    rank_before::Int
    rank_after::Int
    rank_added::Int
    state_embedding_error::Float64
    ancillary_projection_error::Float64
end

"""Diagnostics for one complete synchronous global-subspace sweep."""
struct GlobalSubspaceExpansionInfo
    reverse::Bool
    edges::Vector{GlobalSubspaceEdgeInfo}
    state_embedding_error::Float64
    ancillary_projection_error::Float64
end

"""Diagnostics from ancillary construction, GSE enrichment, and TDVP1."""
struct GSEInfo{S<:Number}
    ancillary_order::Int
    ancillary_shift::S
    action_count::Int
    ancillary_truncation_errors::Vector{Float64}
    expansion::GlobalSubspaceExpansionInfo
    initial_bond_dimensions::Vector{Int}
    enriched_bond_dimensions::Vector{Int}
    final_bond_dimensions::Vector{Int}
    ancillary_generation_seconds::Float64
    enrichment_seconds::Float64
    propagation_seconds::Float64
end

"""
    TDVP1_GSE(; ancillary_shift, ancillary_order=2,
              ancillary_trunc=TruncationScheme(maxdim=100),
              trunc=TruncationScheme(maxdim=100), max_add=8, ...)

Paper-style Yang--White global-subspace-expansion TDVP.  Before each TDVP1
step, construct the literal global ancillary sequence
`(I - ancillary_shift * H)^n * psi`, select its bond complements in one
synchronous tree sweep, and embed `psi` in the common enlarged manifold.
`ancillary_shift` is an explicit caller-owned construction parameter and is
never inferred from the propagation step `dz`.

Ancillary recurrence steps use exact `apply` followed by
`truncated_linear_combination`; `max_exact_bond` and `max_exact_payload` bound
the intermediate direct sums before truncation.  The entire enrichment and
TDVP1 propagation are staged on a copy of the state and committed only after
both complete successfully.
"""
Base.@kwdef mutable struct TDVP1_GSE{S<:Number} <: Evolver
    ancillary_order::Int = 2
    ancillary_shift::S
    ancillary_trunc::TruncationScheme = TruncationScheme(; maxdim=100)
    trunc::TruncationScheme = TruncationScheme(; maxdim=100)
    max_add::Int = 8
    enrichment_atol::Float64 = 1e-12
    enrichment_rtol::Float64 = 1e-10
    reverse::Bool = false
    max_exact_bond::Int = 4096
    max_exact_payload::Int = 100_000_000
    optimize::Bool = true
    order::Int = 2
    krylovdim::Int = 30
    tol::Float64 = 1e-12
    threaded_channels::Bool = false
    channel_slices::Int = 2
    channel_minbatch::Int = 2
    channel_min_flops::Real = 1_000_000
    channel_memory_cap_bytes::Union{Nothing,Real} = nothing
    distributed::Union{Nothing,AbstractDistributedContext} = nothing
    verbose::Bool = true
    cache::Union{Nothing,EnvCache} = nothing
    last_info::Union{Nothing,GSEInfo} = nothing
end

"""
    gse_enrich!(ev, psi, H) -> (psi, info)

Construct global ancillary powers using `ev.ancillary_shift`, synchronously
enrich every tree edge, and return typed diagnostics.  This operation performs
no time propagation; [`step!`](@ref) follows it with TDVP1.
"""
function gse_enrich!(ev::TDVP1_GSE, psi::TTNS, H::TTNO)
    staged = copy(psi)
    info = _gse_enrich_staged!(ev, staged, H)
    _replace_state!(psi, staged)
    ev.last_info = info
    ev.cache = nothing
    return psi, info
end

function _gse_enrich_staged!(ev::TDVP1_GSE, psi::TTNS, H::TTNO)
    _check_gse(ev, psi, H)
    initial_bonds = _gse_bond_dimensions(psi)
    ancillary_started = time_ns()
    ancillaries, truncation_errors = _gse_ancillaries(ev, psi, H)
    ancillary_seconds = (time_ns() - ancillary_started) / 1.0e9
    enrichment_started = time_ns()
    _, raw_report = global_subspace_expand!(
        psi,
        ancillaries;
        trunc=ev.trunc,
        max_add=ev.max_add,
        atol=ev.enrichment_atol,
        rtol=ev.enrichment_rtol,
        reverse=ev.reverse,
    )
    enrichment_seconds = (time_ns() - enrichment_started) / 1.0e9
    expansion = _gse_expansion_info(topology(psi), raw_report)
    enriched_bonds = _gse_bond_dimensions(psi)
    info = GSEInfo(
        ev.ancillary_order,
        ev.ancillary_shift,
        ev.ancillary_order,
        truncation_errors,
        expansion,
        initial_bonds,
        enriched_bonds,
        copy(enriched_bonds),
        ancillary_seconds,
        enrichment_seconds,
        0.0,
    )
    return info
end

function step!(ev::TDVP1_GSE, psi::TTNS, H::TTNO, dz::Number)
    staged = copy(psi)
    enrichment_info = _gse_enrich_staged!(ev, staged, H)
    base = TDVP1(
        order=ev.order,
        krylovdim=ev.krylovdim,
        tol=ev.tol,
        threaded_channels=ev.threaded_channels,
        channel_slices=ev.channel_slices,
        channel_minbatch=ev.channel_minbatch,
        channel_min_flops=ev.channel_min_flops,
        channel_memory_cap_bytes=ev.channel_memory_cap_bytes,
        distributed=ev.distributed,
        verbose=ev.verbose,
        cache=nothing,
    )
    propagation_started = time_ns()
    step!(base, staged, H, dz)
    propagation_seconds = (time_ns() - propagation_started) / 1.0e9
    final_bonds = _gse_bond_dimensions(staged)
    final_info = GSEInfo(
        enrichment_info.ancillary_order,
        enrichment_info.ancillary_shift,
        enrichment_info.action_count,
        enrichment_info.ancillary_truncation_errors,
        enrichment_info.expansion,
        enrichment_info.initial_bond_dimensions,
        enrichment_info.enriched_bond_dimensions,
        final_bonds,
        enrichment_info.ancillary_generation_seconds,
        enrichment_info.enrichment_seconds,
        propagation_seconds,
    )

    _replace_state!(psi, staged)
    ev.cache = base.cache
    ev.last_info = final_info
    return psi
end

function _check_gse(ev::TDVP1_GSE, psi::TTNS, H::TTNO)
    topology(psi) == topology(H) ||
        throw(ArgumentError("TDVP1_GSE: H and psi have mismatched topologies"))
    psi.hasphys == H.hasphys ||
        throw(ArgumentError("TDVP1_GSE: H and psi have mismatched physical layout"))
    spacetype(psi) == spacetype(H) ||
        throw(ArgumentError("TDVP1_GSE: H and psi have mismatched spacetype"))
    promote_type(eltype(psi), eltype(H)) == eltype(psi) ||
        throw(ArgumentError(
            "TDVP1_GSE: exact H action would promote the TTNS eltype; " *
            "convert the input state explicitly"))
    ev.ancillary_order >= 1 ||
        throw(ArgumentError("TDVP1_GSE: ancillary_order must be positive"))
    ev.max_add >= 0 ||
        throw(ArgumentError("TDVP1_GSE: max_add must be nonnegative"))
    ev.enrichment_atol >= 0 ||
        throw(ArgumentError("TDVP1_GSE: enrichment_atol must be nonnegative"))
    ev.enrichment_rtol >= 0 ||
        throw(ArgumentError("TDVP1_GSE: enrichment_rtol must be nonnegative"))
    ev.max_exact_bond >= 1 ||
        throw(ArgumentError("TDVP1_GSE: max_exact_bond must be positive"))
    ev.max_exact_payload >= 1 ||
        throw(ArgumentError("TDVP1_GSE: max_exact_payload must be positive"))
    ev.order in (1, 2) ||
        throw(ArgumentError("TDVP1_GSE: order must be 1 or 2"))
    _gse_finite(ev.ancillary_shift) ||
        throw(ArgumentError("TDVP1_GSE: ancillary_shift must be finite"))
    T = eltype(psi)
    if !(T <: Complex) &&
       ((ev.ancillary_shift isa Complex && !isreal(ev.ancillary_shift)) ||
        eltype(H) <: Complex)
        throw(ArgumentError(
            "TDVP1_GSE: a real state cannot represent complex ancillaries"))
    end
    try
        convert(T, ev.ancillary_shift)
    catch error
        (error isa InexactError || error isa MethodError) || rethrow()
        throw(ArgumentError(
            "TDVP1_GSE: ancillary_shift is not representable by $T"))
    end
    return nothing
end

_gse_finite(value::Number) = isfinite(real(value)) && isfinite(imag(value))

function _gse_ancillaries(ev::TDVP1_GSE, psi::TTNS, H::TTNO)
    T = eltype(psi)
    shift = convert(T, ev.ancillary_shift)
    current = copy(psi)
    ancillaries = TTNS[]
    errors = Float64[]
    for _ in 1:ev.ancillary_order
        acted = apply(
            H, current;
            center=topology(current).root,
            optimize=ev.optimize,
        )
        eltype(acted) == T ||
            throw(ArgumentError(
                "TDVP1_GSE: H action changed the ancillary eltype"))
        current, error_bound = truncated_linear_combination(
            TTNS[current, acted],
            T[one(T), -shift];
            trunc=ev.ancillary_trunc,
            max_bond=ev.max_exact_bond,
            max_payload=ev.max_exact_payload,
        )
        push!(ancillaries, current)
        push!(errors, Float64(error_bound))
    end
    return ancillaries, errors
end

function _gse_expansion_info(t::TreeTopology, report)
    edges = GlobalSubspaceEdgeInfo[
        GlobalSubspaceEdgeInfo(
            nodeid(t, edge.child) => nodeid(t, edge.parent),
            edge.direction,
            edge.rank_before,
            edge.rank_after,
            edge.rank_added,
            edge.state_embedding_error,
            edge.ancillary_projection_error,
        )
        for edge in report.edges
    ]
    return GlobalSubspaceExpansionInfo(
        report.reverse,
        edges,
        report.state_embedding_error,
        report.ancillary_projection_error,
    )
end

function _gse_bond_dimensions(psi::TTNS)
    t = topology(psi)
    return Int[
        dim(virtualspace(psi, child))
        for child in 1:nnodes(t) if t.parent[child] != 0
    ]
end
