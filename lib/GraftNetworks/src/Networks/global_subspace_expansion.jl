"""
    global_subspace_expand!(psi, ancillaries;
                            trunc=TruncationScheme(maxdim=100),
                            max_add=8, atol=1e-12, rtol=1e-10,
                            reverse=false) -> (psi, report)

Enlarge every tree bond of `psi` with directions selected from global
`ancillaries`.  The state and internal copies of the ancillary TTNS objects are
canonicalized and moved synchronously.  At each crossed edge the ancillary active tensors are
projected through the left null space of the current state basis, direct-summed,
and truncated by one SVD.  The resulting state basis is installed in every
source before the corresponding link factors are absorbed on the other side of
the edge.

The state columns are retained without truncation, so enrichment changes its
TTNS manifold but not the represented vector (up to factorization roundoff).
The caller's ancillary objects are never mutated. Internal ancillary copies may
be projected when `max_add` or `trunc` limits the selected
complement.  `reverse=false` crosses every edge child-to-parent; `reverse=true`
uses the mirrored parent-to-child traversal.  The returned report is a named
tuple used by the higher-level evolution diagnostics.
"""
function global_subspace_expand!(
        psi::TTNS,
        ancillaries::AbstractVector{<:TTNS};
        trunc::TruncationScheme=TruncationScheme(; maxdim=100),
        max_add::Int=8,
        atol::Float64=1e-12,
        rtol::Float64=1e-10,
        reverse::Bool=false)
    max_add >= 0 ||
        throw(ArgumentError("global_subspace_expand!: max_add must be nonnegative"))
    atol >= 0 ||
        throw(ArgumentError("global_subspace_expand!: atol must be nonnegative"))
    rtol >= 0 ||
        throw(ArgumentError("global_subspace_expand!: rtol must be nonnegative"))
    _gse_check_sources(psi, ancillaries)
    _gse_check_rank_cap(psi, trunc)

    sources = TTNS[psi]
    append!(sources, copy.(ancillaries))
    t = topology(psi)
    order = reverse ? Base.reverse(postorder(t)) : postorder(t)
    _gse_move_sources!(sources, first(order))

    edge_reports = NamedTuple[]
    for index in 1:(length(order) - 1)
        segment = path_between(t, order[index], order[index + 1])
        for path_index in 2:length(segment)
            from, to = segment[path_index - 1], segment[path_index]
            crosses_selected_direction = (t.parent[from] == to) != reverse
            if crosses_selected_direction
                push!(edge_reports, _gse_cross_edge!(
                    sources, from, to;
                    trunc, max_add, atol, rtol, reverse,
                ))
            else
                _gse_move_sources!(sources, to)
            end
        end
    end

    state_embedding_error = maximum(
        (edge.state_embedding_error for edge in edge_reports); init=0.0)
    ancillary_projection_error = maximum(
        (edge.ancillary_projection_error for edge in edge_reports); init=0.0)
    report = (
        reverse,
        edges=edge_reports,
        state_embedding_error,
        ancillary_projection_error,
    )
    return psi, report
end

function _gse_check_sources(psi::TTNS, ancillaries::AbstractVector{<:TTNS})
    isempty(ancillaries) &&
        throw(ArgumentError("global_subspace_expand!: at least one ancillary is required"))
    t = topology(psi)
    for ancillary in ancillaries
        topology(ancillary) == t ||
            throw(ArgumentError(
                "global_subspace_expand!: source topologies differ"))
        ancillary.hasphys == psi.hasphys ||
            throw(ArgumentError(
                "global_subspace_expand!: source physical layouts differ"))
        spacetype(ancillary) == spacetype(psi) ||
            throw(ArgumentError(
                "global_subspace_expand!: source spacetypes differ"))
        eltype(ancillary) == eltype(psi) ||
            throw(ArgumentError(
                "global_subspace_expand!: source eltypes differ"))
        for node in 1:nnodes(t)
            hasphys(psi, node) || continue
            physspace(ancillary, node) == physspace(psi, node) ||
                throw(ArgumentError(
                    "global_subspace_expand!: physical spaces differ at " *
                    string(nodeid(t, node))))
        end
        domain(ancillary.tensors[t.root])[1] ==
            domain(psi.tensors[t.root])[1] ||
            throw(ArgumentError(
                "global_subspace_expand!: root charge spaces differ"))
    end
    return nothing
end

function _gse_check_rank_cap(psi::TTNS, trunc::TruncationScheme)
    t = topology(psi)
    for (child, _) in edges(t)
        rank = dim(virtualspace(psi, child))
        rank <= trunc.maxdim ||
            throw(ArgumentError(
                "global_subspace_expand!: trunc.maxdim=$(trunc.maxdim) is " *
                "below rank $rank on edge $(nodeid(t, child))"))
    end
    return nothing
end

function _gse_move_sources!(sources::Vector{TTNS}, target::Int)
    for source in sources
        move_center!(source, target)
    end
    return sources
end

function _gse_cross_edge!(
        sources::Vector{TTNS},
        from::Int,
        to::Int;
        trunc::TruncationScheme,
        max_add::Int,
        atol::Float64,
        rtol::Float64,
        reverse::Bool)
    state = first(sources)
    t = topology(state)
    all(center(source) == from for source in sources) ||
        throw(ArgumentError(
            "global_subspace_expand!: sources lost their synchronized center"))

    child = t.parent[from] == to ? from : to
    parent = t.parent[from] == to ? to : from
    frames = if t.parent[from] == to
        [source.tensors[from] for source in sources]
    else
        slot = childslot(t, from, to)
        [_gse_leg_frame(source.tensors[from], slot) for source in sources]
    end

    state_basis, _ = left_orth(first(frames); alg=:qr)
    rank_before = dim(domain(state_basis))
    room = min(max_add, max(trunc.maxdim - rank_before, 0))
    common_basis = _gse_common_basis(
        state_basis, @view(frames[2:end]), room;
        atol, rtol, discarded_weight=trunc.discarded_weight,
    )
    rank_after = dim(domain(common_basis))
    rank_added = rank_after - rank_before
    rank_added >= 0 ||
        throw(ArgumentError(
            "global_subspace_expand!: common basis reduced the state rank"))
    rank_added <= room ||
        throw(ArgumentError(
            "global_subspace_expand!: common basis exceeded the rank budget"))

    links = [common_basis' * active for active in frames]
    errors = Float64[
        norm(active - common_basis * link)
        for (active, link) in zip(frames, links)
    ]
    all(isfinite, errors) ||
        throw(ArgumentError(
            "global_subspace_expand!: non-finite embedding error"))

    if t.parent[from] == to
        _gse_install_up!(sources, from, to, common_basis, links)
    else
        _gse_install_down!(sources, from, to, common_basis, links)
    end
    all(center(source) == to for source in sources) ||
        throw(ArgumentError(
            "global_subspace_expand!: edge crossing desynchronized centers"))

    return (
        child,
        parent,
        direction=reverse ? :reverse : :forward,
        rank_before,
        rank_after,
        rank_added,
        state_embedding_error=first(errors),
        ancillary_projection_error=maximum(@view(errors[2:end]); init=0.0),
    )
end

function _gse_common_basis(
        state_basis::AbstractTensorMap,
        ancillary_frames,
        room::Int;
        atol::Float64,
        rtol::Float64,
        discarded_weight::Float64)
    room == 0 && return state_basis
    complement = left_null(state_basis)
    dim(domain(complement)) == 0 && return state_basis

    projected = nothing
    for active in ancillary_frames
        candidate = complement' * active
        projected = projected === nothing ? candidate :
            catdomain(projected, candidate)
    end
    projected === nothing && return state_basis
    selected, _, _ = split_svd(
        projected,
        TruncationScheme(;
            maxdim=room,
            atol,
            rtol,
            discarded_weight,
        ),
    )
    dim(domain(selected)) == 0 && return state_basis
    enrichment = complement * selected
    if isdual(domain(state_basis)[1]) != isdual(domain(enrichment)[1])
        enrichment = flip(enrichment, numind(enrichment))
    end
    return catdomain(state_basis, enrichment)
end

function _gse_leg_frame(tensor::AbstractTensorMap, slot::Int)
    rank = numind(tensor)
    return permute(tensor, (Backend._others(rank, slot), (slot,)))
end

function _gse_restore_leg_frame(
        tensor::AbstractTensorMap, rank::Int, outputs::Int, slot::Int)
    return permute(tensor, Backend._restore_perm(rank, outputs, slot))
end

function _gse_install_up!(
        sources::Vector{TTNS},
        child::Int,
        parent::Int,
        basis::AbstractTensorMap,
        links::Vector{<:AbstractTensorMap})
    t = topology(first(sources))
    slot = childslot(t, parent, child)
    for (source, link) in zip(sources, links)
        update_tensor!(source, child, copy(basis))
        transported = pivotal_link(link)
        source.tensors[parent] = absorb_on_leg(
            source.tensors[parent], transported, slot)
        source.center = parent
    end
    return sources
end

function _gse_install_down!(
        sources::Vector{TTNS},
        parent::Int,
        child::Int,
        basis::AbstractTensorMap,
        links::Vector{<:AbstractTensorMap})
    t = topology(first(sources))
    slot = childslot(t, parent, child)
    sample = first(sources).tensors[parent]
    rank, outputs = numind(sample), numout(sample)
    installed = _gse_restore_leg_frame(basis, rank, outputs, slot)
    for (source, link) in zip(sources, links)
        update_tensor!(source, parent, copy(installed))
        transported = pivotal_link(transpose(link))
        source.tensors[child] = source.tensors[child] * transported
        source.center = child
    end
    return sources
end
