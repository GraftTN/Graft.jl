"""
    OrientedTwoSiteFactorFrame

Planned, factorized Hamiltonian frame for the directed edge `source -> target`.
`source_action` and `target_action` are the two local ket--operator--environment
halves.  In each half the site-external bra legs are the codomain and the open
state and TTNO edge legs form the two-factor domain.  The type is deliberately
algorithm independent: it contains no expansion or truncation policy.
"""
struct OrientedTwoSiteFactorFrame{A<:AbstractTensorMap,B<:AbstractTensorMap}
    source::Int
    target::Int
    source_action::A
    target_action::B
end

"""
    contract_oriented_two_site(cache, frame)

Close the state and TTNO edge channels of an oriented factor frame.  This is
primarily the exact C1 identity for consumers and tests; it still contracts
the already factorized local halves and never constructs a two-site ket.
"""
function contract_oriented_two_site(cache::EnvCache,
                                    frame::OrientedTwoSiteFactorFrame;
                                    optimize::Bool=true)
    A, B = frame.source_action, frame.target_action
    ea, eb = numind(A) - 2, numind(B) - 2
    labels = Vector{Int}[
        [ntuple(i -> -i, ea)..., 1, 2],
        [ntuple(i -> -(ea + i), eb)..., 1, 2],
    ]
    spec = ContractionSpec(labels, Bool[false, false], ea + eb,
                           (ea + eb, 0), nothing;
                           preferred_slots=[1, 2])
    return _planned_execute!(cache, :oriented_two_site_close, spec, (A, B),
                             scalartype(A); optimize)
end

"""Contract a rank-two edge link into the source half, leaving state/TTNO channels open."""
function contract_source_factor(cache::EnvCache,
                                frame::OrientedTwoSiteFactorFrame,
                                link::AbstractTensorMap;
                                optimize::Bool=true)
    A = frame.source_action
    ea = numind(A) - 2
    numind(link) == 2 || throw(ArgumentError("edge link must have rank two"))
    labels = Vector{Int}[
        [ntuple(i -> -i, ea)..., 1, -(ea + 2)],
        [1, -(ea + 1)],
    ]
    spec = ContractionSpec(labels, Bool[false, false], ea + 2, (ea, 2),
                           nothing; preferred_slots=[1, 2])
    return _planned_execute!(cache, :oriented_source_factor, spec, (A, link),
                             scalartype(A); optimize)
end

"""Contract a rank-two bridge into the target half, leaving bridge/TTNO channels open."""
function contract_target_factor(cache::EnvCache,
                                frame::OrientedTwoSiteFactorFrame,
                                bridge::AbstractTensorMap;
                                optimize::Bool=true)
    B = frame.target_action
    eb = numind(B) - 2
    numind(bridge) == 2 || throw(ArgumentError("factor bridge must have rank two"))
    labels = Vector{Int}[
        [ntuple(i -> -i, eb)..., 1, -(eb + 2)],
        [-(eb + 1), 1],
    ]
    spec = ContractionSpec(labels, Bool[false, false], eb + 2, (eb, 2),
                           nothing; preferred_slots=[2, 1])
    return _planned_execute!(cache, :oriented_target_factor, spec, (B, bridge),
                             scalartype(B); optimize)
end

"""
Close a factorized two-site action after projecting the target external frame.
The result maps the projected target channel to the source external frame.
"""
function contract_projected_two_site(cache::EnvCache,
                                     frame::OrientedTwoSiteFactorFrame,
                                     link::AbstractTensorMap,
                                     target_basis::AbstractTensorMap;
                                     optimize::Bool=true)
    A, B = frame.source_action, frame.target_action
    ea = numind(A) - 2
    numind(link) == 2 || throw(ArgumentError("edge link must have rank two"))
    codomain(target_basis) == codomain(B) || throw(SpaceMismatch(
        "target basis does not span the target external frame"))
    projected = target_basis' * B
    labels = Vector{Int}[
        [ntuple(i -> -i, ea)..., 1, 3],
        [1, 2],
        [-(ea + 1), 2, 3],
    ]
    spec = ContractionSpec(labels, Bool[false, false, false], ea + 1,
                           (ea, 1), nothing; preferred_slots=[2, 3, 1])
    return _planned_execute!(cache, :oriented_projected_close, spec,
                             (A, link, projected), scalartype(A); optimize)
end

"""
    contract_biprojected_two_site(
        cache, frame, link, source_basis, target_basis; optimize=true,
    )

Contract the factorized two-site action directly inside source and target
frames, returning `source_basis' * H_eff * target_basis`.  The local
Hamiltonian halves are projected before the edge channels are closed, so the
unprojected two-site action is never materialized.  Both bases may themselves
be rank-small sketches, which makes this primitive suitable for matrix-free
range finding and projected-core construction. Threaded and distributed
channel slicing partitions the factor-frame TTNO edge directly; either mode
requires an explicit `channel_memory_cap_bytes` admission bound when slicing
is admitted. A one-dimensional or sub-threshold edge stays on the unsliced
factorized contraction and never restores a materialized two-site action.
"""
function contract_biprojected_two_site(
        cache::EnvCache,
        frame::OrientedTwoSiteFactorFrame,
        link::AbstractTensorMap,
        source_basis::AbstractTensorMap,
        target_basis::AbstractTensorMap;
        optimize::Bool=true,
        sector_aware::Bool=true,
        threaded_channels::Bool=false,
        channel_slices::Int=2,
        channel_minbatch::Int=2,
        channel_min_flops::Real=0,
        channel_memory_cap_bytes::Union{Nothing,Real}=nothing,
        distributed::Union{Nothing,AbstractDistributedContext}=nothing,
)
    A, B = frame.source_action, frame.target_action
    numind(link) == 2 || throw(ArgumentError("edge link must have rank two"))
    codomain(source_basis) == codomain(A) || throw(SpaceMismatch(
        "source basis does not span the source external frame"))
    codomain(target_basis) == codomain(B) || throw(SpaceMismatch(
        "target basis does not span the target external frame"))

    source_projected = source_basis' * A
    target_projected = target_basis' * B
    spec = _biprojected_two_site_spec(source_projected, target_projected)
    operands = (source_projected, link, target_projected)
    T = scalartype(A)
    full_plan = _cache_get_or_plan!(
        cache, :oriented_biprojected_close, spec, operands, T;
        optimize, sector_aware)

    operator_space = space(source_projected, numind(source_projected))
    use_channels = dim(operator_space) >= 2 &&
        _channel_plan_flops(full_plan) >= Float64(channel_min_flops) &&
        (distributed !== nothing ||
         (threaded_channels && Threads.nthreads() > 1))
    use_channels || return Planning.execute(full_plan, operands)
    return _channel_biprojected_two_site(
        cache, spec, operands;
        optimize, sector_aware, threaded_channels,
        channel_slices, channel_minbatch, channel_min_flops,
        channel_memory_cap_bytes, distributed)
end

function _biprojected_two_site_spec(source_projected::AbstractTensorMap,
                                    target_projected::AbstractTensorMap)
    es = numout(source_projected)
    et = numout(target_projected)
    labels = Vector{Int}[
        [ntuple(i -> -i, es)..., 1, 3],
        [1, 2],
        [ntuple(i -> -(es + i), et)..., 2, 3],
    ]
    return ContractionSpec(labels, Bool[false, false, false], es + et,
                           (es, et), nothing; preferred_slots=[1, 3, 2])
end

function _biprojected_channel_groups(V::ElementarySpace, requested::Int)
    requested >= 2 || throw(ArgumentError(
        "channel_slices must be at least 2"))
    entries = [(q, j) for q in sectors(V) for j in 1:dim(V, q)]
    isempty(entries) && throw(ArgumentError(
        "cannot slice an empty factor-frame operator channel"))
    nslices = min(requested, length(entries))
    bins = [Dict{Any,Vector{Int}}() for _ in 1:nslices]
    for (index, (q, j)) in enumerate(entries)
        push!(get!(bins[mod1(index, nslices)], q, Int[]), j)
    end
    return bins
end

function _biprojected_slice_operands(operands::Tuple, selected)
    source_projected, link, target_projected = operands
    source_op_leg = numind(source_projected)
    target_op_leg = numind(target_projected)
    source_slice = _restrict_channel_leg(
        source_projected, source_op_leg, selected)
    target_slice = _restrict_channel_leg(
        target_projected, target_op_leg, _dual_channel_groups(selected))
    # Closing a channel against its dual requires the pivotal twist on the
    # target half. It is the identity for ordinary dense spaces and carries
    # the fermionic sign for odd graded channels.
    target_slice = Backend.twist(target_slice, target_op_leg)
    return (source_slice, link, target_slice)
end

function _sum_biprojected_slices(plans, operands, indices;
                                 threaded::Bool, minbatch::Int)
    partials = Vector{Any}(undef, length(indices))
    threaded_foreach(eachindex(indices); threaded, minbatch) do local_index
        slice = indices[local_index]
        partials[local_index] = Planning.execute(
            plans[slice], operands[slice])
    end
    result = partials[1]::AbstractTensorMap
    for index in 2:length(partials)
        axpy!(1, partials[index]::AbstractTensorMap, result)
    end
    return result
end

function _channel_biprojected_two_site(
        cache::EnvCache,
        spec::ContractionSpec,
        operands::Tuple;
        optimize::Bool,
        sector_aware::Bool,
        threaded_channels::Bool,
        channel_slices::Int,
        channel_minbatch::Int,
        channel_min_flops::Real,
        channel_memory_cap_bytes::Union{Nothing,Real},
        distributed::Union{Nothing,AbstractDistributedContext},
)
    source_projected = operands[1]
    operator_space = space(source_projected, numind(source_projected))
    requested = distributed === nothing ? channel_slices :
        max(channel_slices,
            distributed_size(distributed) * max(Threads.nthreads(), 1))
    groups = _biprojected_channel_groups(operator_space, requested)
    if distributed !== nothing
        distributed_size(distributed) >= 2 || throw(ArgumentError(
            "distributed factorized channels require at least two ranks"))
        _require_nonempty_distributed_ranks(distributed, length(groups))
    end

    local_contraction = function ()
        channel_slices >= 2 || throw(ArgumentError(
            "channel_slices must be at least 2"))
        channel_minbatch >= 1 || throw(ArgumentError(
            "channel_minbatch must be positive"))
        min_flops = Float64(channel_min_flops)
        isfinite(min_flops) && min_flops >= 0 || throw(ArgumentError(
            "channel_min_flops must be a finite nonnegative number"))
        channel_memory_cap_bytes === nothing && throw(ArgumentError(
            "factorized channel execution requires channel_memory_cap_bytes"))
        cap = Float64(channel_memory_cap_bytes)
        isfinite(cap) && cap >= 0 || throw(ArgumentError(
            "channel_memory_cap_bytes must be a finite nonnegative number"))
        indices = distributed === nothing ? collect(eachindex(groups)) :
            collect((distributed_rank(distributed) + 1):
                    distributed_size(distributed):length(groups))
        sliced_operands = Vector{Any}(undef, length(groups))
        plans = Vector{Any}(undef, length(groups))
        for slice in indices
            sliced_operands[slice] = _biprojected_slice_operands(
                operands, groups[slice])
            plans[slice] = _cache_get_or_plan!(
                cache, Symbol("oriented_biprojected_channel_", slice),
                spec, sliced_operands[slice], scalartype(source_projected);
                optimize, sector_aware)
        end
        retained_bytes = sum(_env_payload_bytes, operands; init=0) + sum(
            _env_payload_bytes(sliced_operands[index][1]) +
            _env_payload_bytes(sliced_operands[index][3])
            for index in indices; init=0)
        live_bytes = ceil(Int, retained_bytes + sum(
            _slice_live_bytes(plans[index]::ContractionPlan)
            for index in indices; init=0.0))
        live_bytes <= cap || throw(ArgumentError(
            "factorized channel contraction requires approximately $live_bytes live bytes, " *
            "exceeding channel_memory_cap_bytes=$channel_memory_cap_bytes"))
        return _sum_biprojected_slices(
            plans, sliced_operands, indices;
            threaded=threaded_channels && Threads.nthreads() > 1,
            minbatch=channel_minbatch)
    end

    distributed === nothing && return local_contraction()
    result = _distributed_local_call(distributed) do
        local_contraction()
    end
    distributed_allreduce_sum!(distributed, result)
    return distributed_broadcast!(distributed, result)
end

function _oriented_site_tensor(psi::TTNS, site::Int, peer::Int)
    t = topology(psi)
    A = psi.tensors[site]
    N = numind(A)
    if t.parent[site] == peer
        return A, ntuple(identity, N)
    elseif t.parent[peer] == site
        k = childslot(t, site, peer)
        order = (Backend._others(N, k)..., k)
        return permute(A, (Tuple(order[1:(N - 1)]), (order[N],))), order
    end
    throw(ArgumentError("oriented factor-frame endpoints must be adjacent"))
end

function _site_factor_action(cache::EnvCache, psi::TTNS, H::TTNO,
                             site::Int, peer::Int,
                             X::AbstractTensorMap, order::Tuple;
                             optimize::Bool=true)
    t = topology(psi)
    hp = hasphys(psi, site)
    W = H.tensors[site]
    N = numind(X)
    N == length(order) || throw(ArgumentError(
        "oriented site factor has the wrong rank"))
    raw_to_oriented = invperm(collect(order))
    active_state = N
    active_state_raw = t.parent[site] == peer ? parentleg(psi, site) :
        childslot(t, site, peer)
    active_op_raw = _opleg(t, hp, site, peer)
    envlist = [(w, env!(cache, psi, H, w, site))
               for w in neighbors(t, site) if w != peer]

    xidx = zeros(Int, N)
    widx = zeros(Int, numind(W))
    labels = Vector{Int}[xidx, widx]
    conjs = Bool[false, false]
    operands = (X, W)
    children = Int[]
    parents = Int[]
    caps = Int[]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    # Keep the directed state and TTNO edge legs open as the two domain
    # factors, after all site-external output legs.
    xidx[active_state] = -N
    widx[active_op_raw] = -(N + 1)
    if hp
        raw_phys = physleg(psi, site)
        phys = raw_to_oriented[raw_phys]
        pin = fresh()
        xidx[phys] = pin
        widx[physleg(H, site) + 1] = pin
        widx[physleg(H, site)] = -phys
    end
    for (w, E) in envlist
        raw_state = _stateleg(t, hp, site, w)
        state = raw_to_oriented[raw_state]
        kk, oo = fresh(), fresh()
        xidx[state] = kk
        widx[_opleg(t, hp, site, w)] = oo
        push!(labels, [kk, oo, -state])
        push!(conjs, false)
        operands = (operands..., E)
        slot = length(labels)
        if t.parent[site] == w
            push!(parents, slot)
        else
            push!(children, slot)
        end
    end
    if t.parent[site] == 0
        raw_parent = numind(psi.tensors[site])
        state = raw_to_oriented[raw_parent]
        ka, ko = fresh(), fresh()
        xidx[state] = ka
        widx[end] = ko
        cap = _root_cap!(cache, scalartype(X),
                         domain(psi.tensors[site])[1] ⊗
                         domain(W)[numin(W)] ⊗
                         dual(domain(psi.tensors[site])[1]))
        push!(labels, [ka, ko, -state])
        push!(conjs, false)
        operands = (operands..., cap)
        push!(caps, length(labels))
    end

    preferred = vcat(children, [2], parents, caps)
    spec = ContractionSpec(labels, conjs, N + 1, (N - 1, 2), 1;
                           preferred_slots=preferred)
    result = _planned_execute!(cache, :oriented_site_factor, spec, operands,
                               scalartype(X); optimize)

    # The external output legs inherit exactly the same pivotal convention as
    # an h1 result.  The two central factor legs are ket/operator channels and
    # are never Euclidean-bra outputs.
    for raw_leg in _euclidean_output_legs(psi, site)
        raw_leg == active_state_raw && continue
        oriented_leg = raw_to_oriented[raw_leg]
        oriented_leg < N && (result = Backend.twist(result, oriented_leg))
    end
    return result
end

"""
    oriented_two_site_factor_frame(cache, psi, H, source, target)

Build the two planned local halves of the Hamiltonian sandwich on a directed
adjacent edge without materializing either a two-site ket or `eff_h2(Theta)`.
The returned halves are expressed in source/target edge-in-domain frames.
"""
function oriented_two_site_factor_frame(cache::EnvCache, psi::TTNS, H::TTNO,
                                        source::Int, target::Int;
                                        source_tensor::Union{Nothing,AbstractTensorMap}=nothing,
                                        target_tensor::Union{Nothing,AbstractTensorMap}=nothing,
                                        optimize::Bool=true)
    topology(psi) == topology(H) || throw(ArgumentError(
        "oriented factor frame requires matching TTNS and TTNO topologies"))
    source_default, source_order = _oriented_site_tensor(psi, source, target)
    target_default, target_order = _oriented_site_tensor(psi, target, source)
    Xs = source_tensor === nothing ? source_default : source_tensor
    Xt = target_tensor === nothing ? target_default : target_tensor
    codomain(Xs) == codomain(source_default) || throw(SpaceMismatch(
        "source tensor does not match the directed site frame"))
    codomain(Xt) == codomain(target_default) || throw(SpaceMismatch(
        "target tensor does not match the directed site frame"))
    source_action = _site_factor_action(
        cache, psi, H, source, target, Xs, source_order; optimize)
    target_action = _site_factor_action(
        cache, psi, H, target, source, Xt, target_order; optimize)
    return OrientedTwoSiteFactorFrame(source, target,
                                      source_action, target_action)
end
