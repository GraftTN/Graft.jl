# Effective Hamiltonians (PyTreeNet: contractions/effective_hamiltonians.py).
#
# All three return callable `EffectiveMap`s suitable for KrylovKit. Required
# environments, the root cap, and the labelled contraction specification are
# constructed once per local-map visit; the matvec itself executes a cached
# binary plan. The private `_ncon_effective_reference` remains deliberately
# available for A/B tests and benchmark validation.
#
# TODO(MPI extension, §8 level 2): operator-term-level MPI Allreduce lives
# exactly here. H_eff·x splits over TTNO virtual-bond blocks, and DMRG/TDVP
# should share it when the MPI extension milestone is opened.

# Open-leg label bookkeeping: result leg i gets label -i. Builders below retain
# the legacy ncon labels verbatim, but attach an explicit Phase-1 env-first
# static-slot order so Planning never mistakes the physical x–W leg for an
# environment edge.

"""
Opt-in Tier-3 effective map split over one TTNO virtual channel. Each partial
map owns reduced, immutable statics and is safe to execute in a fresh task.
Results are accumulated serially in slice order, so task scheduling does not
affect the reduction order. `concurrent_live_bytes` conservatively sums every
slice-plan live-memory peak because completed roots remain retained until
reduction.
"""
struct ChannelSlicedEffectiveMap{M<:Tuple}
    maps::M
    minbatch::Int
    concurrent_live_bytes::Int
end

function (f::ChannelSlicedEffectiveMap)(x::AbstractTensorMap)
    partials = Vector{Any}(undef, length(f.maps))
    threaded_foreach(eachindex(f.maps); threaded=true, minbatch=f.minbatch) do i
        # PlanWorkspace is deliberately not reused here: it is task-bound, and
        # @threads creates fresh tasks on each Krylov invocation.
        partials[i] = f.maps[i](x)
    end
    y = partials[1]::AbstractTensorMap
    for i in 2:length(partials)
        axpy!(1, partials[i]::AbstractTensorMap, y)
    end
    return y
end

function Base.show(io::IO, f::ChannelSlicedEffectiveMap)
    print(io, "ChannelSlicedEffectiveMap(slices=", length(f.maps),
          ", concurrent_live≈", f.concurrent_live_bytes, " B)")
end

# Keep the public generic free of background-task lifecycle obligations.
# Internal Krylov call sites use `_with_workspace_map` below to scope a
# persistent worker pool and close it deterministically after each solve.
Planning.workspace_map(f::ChannelSlicedEffectiveMap) = f

mutable struct _ChannelSlicedWorkspaceMap{F<:ChannelSlicedEffectiveMap}
    effective::F
    requests::Vector{Channel{Any}}
    responses::Vector{Channel{Any}}
    tasks::Vector{Task}
    owner::Union{Nothing,Task}
    busy::Bool
    closed::Bool
end

function _channel_worker(map_::EffectiveMap, request::Channel{Any},
                         response::Channel{Any})
    workspace = Planning.workspace_map(map_)
    while true
        x = take!(request)
        x === nothing && return nothing
        result = try
            (ok=true, value=workspace(x), error=nothing)
        catch err
            (ok=false, value=nothing,
             error=CapturedException(err, catch_backtrace()))
        end
        put!(response, result)
    end
end

function _channel_workspace_map(effective::ChannelSlicedEffectiveMap)
    requests = [Channel{Any}(1) for _ in effective.maps]
    responses = [Channel{Any}(1) for _ in effective.maps]
    tasks = map(eachindex(effective.maps)) do i
        Threads.@spawn _channel_worker(effective.maps[i], requests[i], responses[i])
    end
    return _ChannelSlicedWorkspaceMap(effective, requests, responses, tasks,
                                      nothing, false, false)
end

function (f::_ChannelSlicedWorkspaceMap)(x::AbstractTensorMap)
    f.closed && throw(ArgumentError("channel workspace map is closed"))
    task = current_task()
    if f.owner === nothing
        f.owner = task
    elseif f.owner !== task
        throw(ArgumentError("channel workspace map is task-local"))
    end
    f.busy && throw(ArgumentError("channel workspace map cannot be used reentrantly"))
    f.busy = true
    try
        foreach(request -> put!(request, x), f.requests)
        results = map(take!, f.responses)
        failed = findfirst(result -> !result.ok, results)
        failed === nothing || throw(results[failed].error)
        y = results[1].value::AbstractTensorMap
        for i in 2:length(results)
            axpy!(1, results[i].value::AbstractTensorMap, y)
        end
        return y
    finally
        f.busy = false
    end
end

function Base.close(f::_ChannelSlicedWorkspaceMap)
    f.closed && return nothing
    f.busy && throw(ArgumentError("cannot close a busy channel workspace map"))
    foreach(request -> put!(request, nothing), f.requests)
    foreach(fetch, f.tasks)
    f.closed = true
    return nothing
end

_with_workspace_map(body, effective) = body(Planning.workspace_map(effective))
function _with_workspace_map(body, effective::ChannelSlicedEffectiveMap)
    workspace = _channel_workspace_map(effective)
    try
        return body(workspace)
    finally
        close(workspace)
    end
end

_channel_slice_space(V::ComplexSpace, groups) =
    ComplexSpace(sum(length, values(groups)); dual=isdual(V))
_channel_slice_space(V::GradedSpace, groups) = typeof(V)(
    ((isdual(V) ? dual(q) : q) => length(groups[q])
     for q in sectors(V) if haskey(groups, q));
    dual=isdual(V))

function _channel_inclusion(::Type{T}, V::ElementarySpace, groups) where {T<:Number}
    all(q -> dim(q) == 1, keys(groups)) ||
        throw(ArgumentError("channel slicing requires one-dimensional abelian sectors"))
    Vs = _channel_slice_space(V, groups)
    inclusion = zeros(T, V ← Vs)
    for (q, block_) in blocks(inclusion)
        for (j, old) in enumerate(groups[q])
            block_[old, j] = one(T)
        end
    end
    return inclusion
end

function _restrict_channel_leg(A::AbstractTensorMap, leg::Int, groups)
    inclusion = _channel_inclusion(scalartype(A), space(A, leg), groups)
    return Backend.transform_leg(A, adjoint(inclusion), leg)
end

_dual_channel_groups(groups) =
    Dict{Any,Vector{Int}}(dual(q) => copy(indices) for (q, indices) in groups)

function _fixed_plan_flops(plan::ContractionPlan, operands::Tuple)
    ninputs = plan.nslots - length(plan.steps)
    length(operands) == ninputs ||
        throw(ArgumentError("channel cost model received the wrong operand count"))
    slots = Vector{Any}(undef, plan.nslots)
    for (i, operand) in enumerate(operands)
        slots[i] = Planning._prototype_space(operand)
    end
    total = 0.0
    for step in plan.steps
        A, B = slots[step.a], slots[step.b]
        metrics = Backend.pair_cost(A, (step.oindA, step.cindA), step.conjA,
                                    B, (step.cindB, step.oindB), step.conjB,
                                    step.out)
        isfinite(metrics.sector_flops) || return 1.0
        total += metrics.sector_flops
        slots[step.dst] = metrics.output
        slots[step.a] = nothing
        slots[step.b] = nothing
    end
    return total
end

function _restrict_channel_statics(statics::Tuple, targets, selected)
    sliced = collect(statics)
    for (static_slot, leg, use_dual_groups) in targets
        groups = use_dual_groups ? _dual_channel_groups(selected) : selected
        sliced[static_slot] = _restrict_channel_leg(statics[static_slot], leg, groups)
    end
    return Tuple(sliced)
end

function _restrict_channel_static_spaces(statics::Tuple, targets, selected)
    sliced = Any[Planning._prototype_space(static) for static in statics]
    for (static_slot, leg, use_dual_groups) in targets
        groups = use_dual_groups ? _dual_channel_groups(selected) : selected
        V = space(statics[static_slot], leg)
        Vs = _channel_slice_space(V, groups)
        sliced[static_slot] = Backend.transform_leg_space(
            sliced[static_slot], Vs, leg)
    end
    return Tuple(sliced)
end

function _channel_groups(V::ElementarySpace, requested::Int,
                         plan::ContractionPlan, protos, statics::Tuple,
                         targets)
    entries = [(q, j) for q in sectors(V) for j in 1:dim(V, q)]
    nslices = min(requested, length(entries))
    weights = map(entries) do (q, j)
        selected = Dict{Any,Vector{Int}}(q => [j])
        sliced = _restrict_channel_static_spaces(statics, targets, selected)
        _fixed_plan_flops(plan, (protos[1], sliced...))
    end

    # Deterministic longest-processing-time bin packing. Ties retain channel
    # order and then choose the lowest-index bin, fixing both plans and the
    # later reduction order independently of task scheduling.
    order = sortperm(eachindex(entries); by=i -> (-weights[i], i))
    bins = [Dict{Any,Vector{Int}}() for _ in 1:nslices]
    loads = zeros(Float64, nslices)
    for i in order
        bin = argmin(loads)
        q, j = entries[i]
        push!(get!(bins[bin], q, Int[]), j)
        loads[bin] += weights[i]
    end
    return bins
end

function _channel_group_flops(groups, plan::ContractionPlan, protos,
                              statics::Tuple, targets)
    return map(groups) do selected
        sliced = _restrict_channel_static_spaces(statics, targets, selected)
        _fixed_plan_flops(plan, (protos[1], sliced...))
    end
end

function _slice_live_bytes(plan::ContractionPlan)
    bytes = plan.sector_live_peak_bytes
    return isfinite(bytes) ? bytes : plan.live_peak_bytes
end

function _concurrent_slice_bytes(maps)
    # Completed roots remain live until the fixed-order reduction. Summing all
    # plan peaks is conservative even when slices outnumber Julia threads.
    return ceil(Int, sum(map_ -> _slice_live_bytes(map_.plan), maps))
end

function _channel_sliced_h1!(cache::EnvCache, full::EffectiveMap,
                             spec::ContractionSpec, statics::Tuple, protos,
                             ψ::TTNS, n::Int;
                             channel_slices::Int,
                             channel_minbatch::Int,
                             channel_memory_cap_bytes::Union{Nothing,Real},
                             optimize::Bool, memory_weight::Real,
                             sector_aware::Bool,
                             memory_cap_bytes::Union{Nothing,Real})
    channel_slices >= 2 || throw(ArgumentError("channel_slices must be at least 2"))
    channel_minbatch >= 1 || throw(ArgumentError("channel_minbatch must be positive"))
    Threads.nthreads() > 1 || return full
    channel_memory_cap_bytes === nothing && throw(ArgumentError(
        "threaded_channels=true requires an explicit channel_memory_cap_bytes"))
    cap = Float64(channel_memory_cap_bytes)
    isfinite(cap) && cap >= 0 || throw(ArgumentError(
        "channel_memory_cap_bytes must be a finite nonnegative number"))

    t = ψ.topo
    W = statics[1]
    hp = hasphys(ψ, n)
    neighbor_list = collect(neighbors(t, n))
    isempty(neighbor_list) && return full
    operator_legs = [_opleg(t, hp, n, neighbor) for neighbor in neighbor_list]
    channel_dims = [dim(space(W, leg)) for leg in operator_legs]
    edge_index = argmax(channel_dims)
    channel_dims[edge_index] >= 2 || return full
    operator_leg = operator_legs[edge_index]
    env_static_slot = 1 + edge_index
    targets = ((1, operator_leg, false), (env_static_slot, 2, true))

    groups = _channel_groups(space(W, operator_leg), channel_slices,
                             full.plan, protos, statics, targets)
    maps = map(enumerate(groups)) do (slice, selected)
        sliced_statics = _restrict_channel_statics(statics, targets, selected)
        sliced_protos = (protos[1], sliced_statics...)
        kind = Symbol("h1_channel_", operator_leg, "_", slice)
        _effective_map!(cache, kind, spec, sliced_protos, sliced_statics,
                        scalartype(ψ.tensors[n]);
                        optimize, memory_weight, sector_aware, memory_cap_bytes,
                        input_twists=full.input_twists,
                        output_twists=full.output_twists)
    end
    map_tuple = Tuple(maps)
    concurrent_live_bytes = _concurrent_slice_bytes(map_tuple)
    concurrent_live_bytes <= cap || throw(ArgumentError(
        "channel-sliced h1 requires approximately $concurrent_live_bytes live bytes, " *
        "exceeding channel_memory_cap_bytes=$channel_memory_cap_bytes"))
    return ChannelSlicedEffectiveMap(map_tuple, channel_minbatch,
                                     concurrent_live_bytes)
end

function _h2_channel_candidates(ψ::TTNS, n::Int, m::Int, statics::Tuple)
    t = ψ.topo
    Wn, Wm = statics[1], statics[2]
    k = childslot(t, m, n)
    candidates = Any[]

    # The crossed n-m TTNO bond is internal to the two-site block. Restricting
    # both ends partitions its exact contracted sum without involving an env.
    push!(candidates, (space=space(Wn, numind(Wn)),
                       targets=((1, numind(Wn), false), (2, k, true))))

    static_slot = 3
    for w in neighbors(t, n)
        w == m && continue
        leg = _opleg(t, hasphys(ψ, n), n, w)
        push!(candidates, (space=space(Wn, leg),
                           targets=((1, leg, false), (static_slot, 2, true))))
        static_slot += 1
    end
    for w in neighbors(t, m)
        w == n && continue
        leg = _opleg(t, hasphys(ψ, m), m, w)
        push!(candidates, (space=space(Wm, leg),
                           targets=((2, leg, false), (static_slot, 2, true))))
        static_slot += 1
    end
    return candidates
end

function _channel_sliced_h2!(cache::EnvCache, full::EffectiveMap,
                             spec::ContractionSpec, statics::Tuple, protos,
                             ψ::TTNS, n::Int, m::Int;
                             channel_slices::Int,
                             channel_minbatch::Int,
                             channel_memory_cap_bytes::Union{Nothing,Real},
                             optimize::Bool, memory_weight::Real,
                             sector_aware::Bool,
                             memory_cap_bytes::Union{Nothing,Real})
    channel_slices >= 2 || throw(ArgumentError("channel_slices must be at least 2"))
    channel_minbatch >= 1 || throw(ArgumentError("channel_minbatch must be positive"))
    Threads.nthreads() > 1 || return full
    channel_memory_cap_bytes === nothing && throw(ArgumentError(
        "threaded_channels=true requires an explicit channel_memory_cap_bytes"))
    cap = Float64(channel_memory_cap_bytes)
    isfinite(cap) && cap >= 0 || throw(ArgumentError(
        "channel_memory_cap_bytes must be a finite nonnegative number"))

    candidates = _h2_channel_candidates(ψ, n, m, statics)
    partitions = map(enumerate(candidates)) do (edge_index, candidate)
        dim(candidate.space) >= 2 || return nothing
        groups = _channel_groups(candidate.space, channel_slices, full.plan,
                                 protos, statics, candidate.targets)
        costs = _channel_group_flops(groups, full.plan, protos, statics,
                                     candidate.targets)
        return (; edge_index, candidate, groups, costs)
    end
    filter!(!isnothing, partitions)
    isempty(partitions) && return full
    scores = map(p -> (maximum(p.costs), sum(p.costs), p.edge_index), partitions)
    partition = partitions[argmin(scores)]
    edge_index = partition.edge_index
    candidate = partition.candidate
    groups = partition.groups
    maps = map(enumerate(groups)) do (slice, selected)
        sliced_statics = _restrict_channel_statics(
            statics, candidate.targets, selected)
        sliced_protos = (protos[1], sliced_statics...)
        kind = Symbol("h2_channel_", edge_index, "_", slice)
        _effective_map!(cache, kind, spec, sliced_protos, sliced_statics,
                        scalartype(ψ.tensors[n]);
                        optimize, memory_weight, sector_aware, memory_cap_bytes,
                        input_twists=full.input_twists,
                        output_twists=full.output_twists)
    end
    map_tuple = Tuple(maps)
    concurrent_live_bytes = _concurrent_slice_bytes(map_tuple)
    concurrent_live_bytes <= cap || throw(ArgumentError(
        "channel-sliced h2 requires approximately $concurrent_live_bytes live bytes, " *
        "exceeding channel_memory_cap_bytes=$channel_memory_cap_bytes"))
    return ChannelSlicedEffectiveMap(map_tuple, channel_minbatch,
                                     concurrent_live_bytes)
end

function _ncon_effective_reference(spec::ContractionSpec, x::AbstractTensorMap,
                                   statics::Tuple)
    return Planning.ncon_reference(spec, x, statics)
end

function _h1_spec(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int)
    t = ψ.topo
    x = ψ.tensors[n]
    hp = hasphys(ψ, n)
    W = H.tensors[n]
    envlist = [(w, env!(cache, ψ, H, w, n)) for w in neighbors(t, n)]
    isroot_ = t.parent[n] == 0
    K = nchildren(t, n)
    Nx = numind(x)

    xidx = zeros(Int, Nx)
    widx = zeros(Int, numind(W))
    labels = Vector{Int}[xidx, widx]
    conjs = Bool[false, false]
    statics = (W,)
    protos = (x, W)
    children = Int[]
    parents = Int[]
    caps = Int[]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    if hp
        pin = fresh()
        xidx[K + 1] = pin
        widx[K + 2] = pin
        widx[K + 1] = -(K + 1)       # open physical (out) leg
    end
    for (w, E) in envlist
        lx = _stateleg(t, hp, n, w)
        kk, oo = fresh(), fresh()
        xidx[lx] = kk
        widx[_opleg(t, hp, n, w)] = oo
        push!(labels, [kk, oo, -lx]); push!(conjs, false)
        statics = (statics..., E)
        protos = (protos..., E)
        slot = length(labels)
        if t.parent[n] == w
            push!(parents, slot)
        else
            push!(children, slot)
        end
    end
    if isroot_
        ka, ko = fresh(), fresh()
        xidx[end] = ka
        widx[end] = ko
        cap = _root_cap!(cache, scalartype(x),
                         domain(x)[1] ⊗ domain(W)[numin(W)] ⊗ dual(domain(x)[1]))
        push!(labels, [ka, ko, -Nx]); push!(conjs, false)
        statics = (statics..., cap)
        protos = (protos..., cap)
        push!(caps, length(labels))
    end
    # x → child envs → W → parent env/root cap
    preferred = vcat(children, [2], parents, caps)
    spec = ContractionSpec(labels, conjs, Nx, (Nx - 1, 1), 1;
                           preferred_slots=preferred)
    return spec, statics, protos
end

"""
    eff_h1(cache, ψ, H, n) -> EffectiveMap

One-site effective Hamiltonian at node `n`. The returned callable maps a tensor
with the structure of `ψ[n]` to the same `(N-1, 1)` TensorMap partition. The
plan cache is shape-only and safely survives ordinary environment invalidation.
Set `optimize=false` to force the Phase-1 env-first plan; `memory_weight`
selects the dense FLOP-plus-live-byte objective and is part of cache identity.
`memory_cap_bytes` is a hard conservative live-memory cap and is likewise
part of cache identity.  `sector_aware=true` (the default) uses the Phase-3 exact
unique-fusion block-GEMM objective when the TensorKit spaces support it;
non-unique fusion spaces retain the dense model.  Planar/anyonic execution is
outside the current regular TensorOperations backend surface.

`threaded_channels=true` enables the experimental Tier-3 split of the largest
TTNO virtual leg into `channel_slices` exact subspaces. It is off by
default, requires more than one Julia thread and an explicit
`channel_memory_cap_bytes`, and currently benefits only sufficiently large
branch-node maps. Slice outputs are accumulated serially in fixed order.
"""
function eff_h1(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int;
                optimize::Bool=true, memory_weight::Real=1,
                sector_aware::Bool=true,
                memory_cap_bytes::Union{Nothing,Real}=nothing,
                threaded_channels::Bool=false, channel_slices::Int=2,
                channel_minbatch::Int=2,
                channel_memory_cap_bytes::Union{Nothing,Real}=nothing)
    spec, statics, protos = _h1_spec(cache, ψ, H, n)
    full = _effective_map!(cache, :h1, spec, protos, statics,
                           scalartype(ψ.tensors[n]);
                           optimize=optimize, memory_weight=memory_weight,
                           sector_aware=sector_aware,
                           memory_cap_bytes=memory_cap_bytes,
                           output_twists=_euclidean_output_legs(ψ, n))
    threaded_channels || return full
    return _channel_sliced_h1!(cache, full, spec, statics, protos, ψ, n;
                               channel_slices, channel_minbatch,
                               channel_memory_cap_bytes, optimize, memory_weight,
                               sector_aware, memory_cap_bytes)
end

function _h0_input_space(En::AbstractTensorMap, Em::AbstractTensorMap)
    # C's first flat leg contracts env(n→m)'s ket leg; its second (domain)
    # leg contracts env(m→n)'s ket leg. This is the link shape produced by the
    # QR/CBE split seam without allocating a data-valued C just for planning.
    return dual(space(En, 1)) ← space(Em, 1)
end

function _h0_spec(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int, m::Int)
    ψ.topo.parent[n] == m ||
        throw(ArgumentError("eff_h0: m must be the parent of n"))
    En = env!(cache, ψ, H, n, m)
    Em = env!(cache, ψ, H, m, n)
    Cspace = _h0_input_space(En, Em)
    spec = ContractionSpec(Vector{Int}[[1, 2], [1, 3, -1], [2, 3, -2]],
                           Bool[false, false, false], 2, (1, 1), 1;
                           preferred_slots=[2, 3])
    return spec, (En, Em), (Cspace, En, Em)
end

"""
    eff_h0(cache, ψ, H, n, m) -> EffectiveMap

Zero-site (link) effective Hamiltonian on adjacent nodes `n, m`, acting on the
gauge link tensor used by the TDVP backward step. The returned map has direct
`(1, 1)` output partitioning; no trailing `repartition` copy is made. Planner
keywords have the same semantics as `eff_h1`. `m` must be the parent of `n`,
and the detached link must have left the orthogonality center on either `n`
(`_split_link_up`) or `m` (`_split_link_down`); the center side selects the
corresponding pivotal link coordinates.
"""
function eff_h0(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int, m::Int;
                optimize::Bool=true, memory_weight::Real=1,
                sector_aware::Bool=true,
                memory_cap_bytes::Union{Nothing,Real}=nothing)
    spec, statics, protos = _h0_spec(cache, ψ, H, n, m)
    ψ.center in (n, m) ||
        throw(ArgumentError("eff_h0: orthogonality center must lie on the active edge"))
    Cspace = first(protos)
    euclidean_twists = Int[]
    isdual(Cspace[1]) && _component_has_dual_physical(ψ, n, m) &&
        push!(euclidean_twists, 1)
    # The second flat output leg closes the parent-side environment.  As for
    # h1/h2, its supertrace residue depends on that actual leg orientation,
    # not on the physical carriers in the parent component.
    isdual(Cspace[2]) && push!(euclidean_twists, 2)

    # `_split_link_down` leaves the center on the parent and represents the
    # detached gauge link before the `_pivotal_link` used at absorption.  On
    # a mixed-duality down link the generator therefore acts by the
    # similarity P⁻¹ H₀ P, with P the codomain-leg ribbon twist.  A mixed link
    # produced by `_split_link_up` (center still on the child) is different:
    # its raw contraction already includes that bend, so applying P again
    # would corrupt the next forward sweep.
    down_split = ψ.center == m
    mixed_duality =
        isdual(codomain(Cspace)[1]) != isdual(domain(Cspace)[1])
    pivotal_twists = down_split && mixed_duality ? (1,) : ()
    output_twists = (Tuple(euclidean_twists)..., pivotal_twists...)
    return _effective_map!(cache, :h0, spec, protos, statics,
                           scalartype(ψ.tensors[n]);
                           optimize=optimize, memory_weight=memory_weight,
                           sector_aware=sector_aware,
                           memory_cap_bytes=memory_cap_bytes,
                           input_twists=pivotal_twists,
                           output_twists=output_twists)
end

"""
    two_site_space(ψ, n, m) -> TensorMapSpace

No-data prototype for `two_site_tensor(ψ, n, m)`. `m` must be the parent of
`n`; the all-codomain leg order is exactly the data contraction's order:
child-node codomain legs, followed by every parent-node flat leg except the
crossed child slot. It lets h2 planning discover Θ's shape without allocating
or contracting Θ.
"""
function two_site_space(ψ::TTNS, n::Int, m::Int)
    t = ψ.topo
    t.parent[n] == m || throw(ArgumentError("two_site_space: m must be the parent of n"))
    A, B = ψ.tensors[n], ψ.tensors[m]
    k = childslot(t, m, n)
    legs = [space(A, i) for i in 1:numout(A)]
    append!(legs, (space(B, j) for j in 1:numind(B) if j != k))
    cod = reduce(⊗, legs)
    return cod ← one(cod)
end

"""
    two_site_tensor(ψ, n, m) -> Θ

Contract `ψ[n]` and `ψ[m]` over their shared edge (`m` must be the parent of
`n`). Result is all-codomain, legs ordered: `A_n`'s codomain legs (slots,
physical), then `A_m`'s flat legs except the `n` slot (original order).
`split_two_site!` is the exact inverse bookkeeping.
"""
function two_site_tensor(ψ::TTNS, n::Int, m::Int)
    t = ψ.topo
    t.parent[n] == m || throw(ArgumentError("two_site_tensor: m must be the parent of n"))
    A, B = ψ.tensors[n], ψ.tensors[m]
    k = childslot(t, m, n)
    pn = numout(A)
    # This is a fixed binary contraction, so bypass `ncon`'s dynamic label
    # parser and call the TensorOperations expert API through L0. Besides
    # removing repeated label allocations, this keeps the large generic ncon
    # method out of TDVP2's first-use JIT path.
    pA = (Tuple(1:pn), (pn + 1,))
    bopen = Tuple(j for j in 1:numind(B) if j != k)
    pB = ((k,), bopen)
    nopen = pn + length(bopen)
    pAB = (Tuple(1:nopen), ())
    return Backend.contract_pair(A, pA, false, B, pB, false, pAB)
end

"""
    split_two_site!(ψ, Θ, n, m; trunc, center_on=:n) -> (ψ, info)

Truncated SVD of a two-site tensor back into `ψ[n]`, `ψ[m]` (inverse of
`two_site_tensor`), moving the orthogonality center onto `center_on ∈ (:n, :m)`.
All truncation passes through `TruncationScheme` (§9.5).
"""
function split_two_site!(ψ::TTNS, Θ::AbstractTensorMap, n::Int, m::Int;
                         trunc::TruncationScheme=Backend.NO_TRUNCATION, center_on::Symbol=:n)
    t = ψ.topo
    k = childslot(t, m, n)
    pn = numout(ψ.tensors[n])
    # (n legs) ← (m legs); explicit permute — repartition would REVERSE the
    # multi-leg domain order (planar bending), scrambling the m-leg bookkeeping
    N = numind(Θ)
    Θs = permute(Θ, (ntuple(identity, pn), ntuple(j -> pn + j, N - pn)))
    U, S, Vh = split_svd(Θs, trunc)
    if center_on === :n
        An = U * S
        Rm = Vh
    else
        An = U
        Rm = S * Vh
    end
    Km = numind(ψ.tensors[m]) - 1                     # m's codomain legs
    p1 = ntuple(j -> j == k ? 1 : 1 + (j < k ? j : j - 1), Km)
    Am = permute(Rm, (p1, (numind(Rm),)))
    ψ.tensors[n] = An
    ψ.tensors[m] = Am
    ψ.center = center_on === :n ? n : m
    return ψ
end

function _h2_spec(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int, m::Int)
    t = ψ.topo
    t.parent[n] == m || throw(ArgumentError("eff_h2: m must be the parent of n"))
    hpn, hpm = hasphys(ψ, n), hasphys(ψ, m)
    Wn, Wm = H.tensors[n], H.tensors[m]
    k = childslot(t, m, n)
    Kn, Km = nchildren(t, n), nchildren(t, m)
    envn = [(w, env!(cache, ψ, H, w, n)) for w in neighbors(t, n) if w != m]
    envm = [(w, env!(cache, ψ, H, w, m)) for w in neighbors(t, m) if w != n]
    isroot_ = t.parent[m] == 0
    pn = Kn + (hpn ? 1 : 0)
    xspace = two_site_space(ψ, n, m)
    Nx = numind(xspace)

    xidx = zeros(Int, Nx)
    wnidx = zeros(Int, numind(Wn))
    wmidx = zeros(Int, numind(Wm))
    labels = Vector{Int}[xidx, wnidx, wmidx]
    conjs = Bool[false, false, false]
    statics = (Wn, Wm)
    protos = (xspace, Wn, Wm)
    envnslots = Int[]
    envmchildren = Int[]
    envmparents = Int[]
    caps = Int[]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    # x leg positions: n part 1:pn, then m's flat legs except slot k.
    mpos(j) = pn + (j < k ? j : j - 1)

    # Operator virtual bond between Wn and Wm.
    ob = fresh()
    wnidx[end] = ob
    wmidx[k] = ob
    if hpn
        pin = fresh()
        xidx[Kn + 1] = pin
        wnidx[Kn + 2] = pin
        wnidx[Kn + 1] = -(Kn + 1)
    end
    if hpm
        pin = fresh()
        xidx[mpos(Km + 1)] = pin
        wmidx[Km + 2] = pin
        wmidx[Km + 1] = -mpos(Km + 1)
    end
    # Environments around n: all are children because m is n's parent.
    for (w, E) in envn
        lx = childslot(t, n, w)
        kk, oo = fresh(), fresh()
        xidx[lx] = kk
        wnidx[lx] = oo
        push!(labels, [kk, oo, -lx]); push!(conjs, false)
        statics = (statics..., E)
        protos = (protos..., E)
        push!(envnslots, length(labels))
    end
    # Environments around m: defer its parent until after Wm joins.
    for (w, E) in envm
        if t.parent[m] == w
            lw, lx = numind(Wm), Nx
        else
            lw = childslot(t, m, w)
            lx = mpos(lw)
        end
        kk, oo = fresh(), fresh()
        xidx[lx] = kk
        wmidx[lw] = oo
        push!(labels, [kk, oo, -lx]); push!(conjs, false)
        statics = (statics..., E)
        protos = (protos..., E)
        if t.parent[m] == w
            push!(envmparents, length(labels))
        else
            push!(envmchildren, length(labels))
        end
    end
    if isroot_
        ka, ko = fresh(), fresh()
        xidx[Nx] = ka
        wmidx[end] = ko
        # `xspace` is a no-data TensorMapSpace rather than an AbstractTensorMap;
        # TensorKit exposes its flat leg spaces through `getindex`, not `space`.
        # This is exactly the final leg of the Θ prototype used by the former
        # data-valued implementation.
        cap = _root_cap!(cache, scalartype(ψ.tensors[n]),
                         dual(xspace[Nx]) ⊗ domain(Wm)[numin(Wm)] ⊗ xspace[Nx])
        push!(labels, [ka, ko, -Nx]); push!(conjs, false)
        statics = (statics..., cap)
        protos = (protos..., cap)
        push!(caps, length(labels))
    end
    # Θ → n child envs → Wn → m child envs → Wm → m parent/root cap.
    preferred = vcat(envnslots, [2], envmchildren, [3], envmparents, caps)
    spec = ContractionSpec(labels, conjs, Nx, (Nx, 0), 1;
                           preferred_slots=preferred)
    return spec, statics, protos
end

"""
    eff_h2(cache, ψ, H, n, m) -> EffectiveMap

Two-site effective Hamiltonian on child-parent bond `(n, m)`, acting on the
all-codomain structure returned by `two_site_tensor(ψ, n, m)`. Planner
keywords and experimental channel-slicing controls have the same semantics as
`eff_h1`.
"""
function eff_h2(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int, m::Int;
                optimize::Bool=true, memory_weight::Real=1,
                sector_aware::Bool=true,
                memory_cap_bytes::Union{Nothing,Real}=nothing,
                threaded_channels::Bool=false, channel_slices::Int=2,
                channel_minbatch::Int=2,
                channel_memory_cap_bytes::Union{Nothing,Real}=nothing)
    spec, statics, protos = _h2_spec(cache, ψ, H, n, m)
    t = ψ.topo
    Kn, Km = nchildren(t, n), nchildren(t, m)
    k = childslot(t, m, n)
    pn = Kn + (hasphys(ψ, n) ? 1 : 0)
    mpos(j) = pn + (j < k ? j : j - 1)
    xspace = first(protos)
    twists = Int[]
    for (j, child) in enumerate(t.children[n])
        isdual(xspace[j]) && _component_has_dual_physical(ψ, child, n) &&
            push!(twists, j)
    end
    hasphys(ψ, n) && isdual(xspace[Kn + 1]) && push!(twists, Kn + 1)
    for (j, child) in enumerate(t.children[m])
        child == n && continue
        pos = mpos(j)
        isdual(xspace[pos]) && _component_has_dual_physical(ψ, child, m) &&
            push!(twists, pos)
    end
    hasphys(ψ, m) && isdual(xspace[mpos(Km + 1)]) &&
        push!(twists, mpos(Km + 1))
    if t.parent[m] != 0
        pos = mpos(parentleg(ψ, m))
        # As in the one-site map, this is a parent-side environment closure.
        # Its pivotal correction is determined by the actual open flat leg,
        # not by physical carriers elsewhere in that component.
        isdual(xspace[pos]) && push!(twists, pos)
    end
    full = _effective_map!(cache, :h2, spec, protos, statics,
                           scalartype(ψ.tensors[n]);
                           optimize=optimize, memory_weight=memory_weight,
                           sector_aware=sector_aware,
                           memory_cap_bytes=memory_cap_bytes,
                           output_twists=Tuple(twists))
    threaded_channels || return full
    return _channel_sliced_h2!(cache, full, spec, statics, protos, ψ, n, m;
                               channel_slices, channel_minbatch,
                               channel_memory_cap_bytes, optimize, memory_weight,
                               sector_aware, memory_cap_bytes)
end
