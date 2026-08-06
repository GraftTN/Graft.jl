"""
    expand!(ψ, H, edge; scheme=:exact, cache=nothing, rng=nothing,
            trunc, max_add=8, mixing=1, enr_rtol=1e-10, enr_atol=1e-12,
            rsvd_oversample=8, rsvd_poweriter=0,
            rsvd_threaded=Base.Threads.nthreads() > 1,
            rsvd_minbatch=max(2, Base.Threads.nthreads()),
            rsvd_memory_cap_bytes=nothing,
            rsvd_task_workspace_bytes=nothing,
            rsvd_fanout_diagnostics=nothing,
            factorized=false,
            contraction_optimize=true,
            contraction_sector_aware=true,
            threaded_channels=false, channel_slices=2,
            channel_minbatch=2, channel_min_flops=0,
            channel_memory_cap_bytes=nothing, distributed=nothing) -> ψ

Shared bond-expansion primitive (§5a/§11.7). `edge` is `(child, parent)` or
`child => parent` using node ids or indices. `scheme=:exact` forms the
predictor basis with a deterministic truncated SVD. `scheme=:directqr` forms
the complete predictor image with a deterministic sector-aware compact QR.
The randomized schemes use explicit-RNG blockwise Gaussian probes on the fused
rest space and never touch global randomness: `scheme=:rsvd` extracts the
sampled image with a truncated SVD, while `scheme=:rangefinder` uses compact QR
and QR-reorthogonalized power iterations. The `rsvd_*` keywords configure the
shared sketch generation and power policy for both randomized schemes.
Probe-block RNG seeds are derived serially from `rng`, so
`rsvd_threaded=true` is bitwise-equivalent to the serial probe generation.
Actual threaded probe filling requires both `rsvd_memory_cap_bytes` and the
measured conservative per-task scratch allowance
`rsvd_task_workspace_bytes`; omission selects an observable deterministic
serial fallback. `rsvd_fanout_diagnostics` may be a `Ref` used to inspect the
admitted batches and memory model without changing `expand!`'s return value.
`factorized=true` enables the biprojected range-finder path used by DMRG3S:
random probes pass directly through the local Hamiltonian halves and the
current child complement without forming a two-site ket or `H_eff(Θ)`.
Channel-sliced and distributed requests partition the factor-frame TTNO edge
channel and reduce the projected slice results without restoring a materialized
two-site action.
"""
function expand!(ψ::TTNS, H::TTNO, edge; scheme::Symbol=:exact,
                 cache::Union{Nothing,EnvCache}=nothing,
                 rng::Union{Nothing,AbstractRNG}=nothing,
                 trunc::TruncationScheme=TruncationScheme(; maxdim=100),
                 max_add::Int=8, mixing::Number=one(Float64),
                 enr_rtol::Float64=1e-10, enr_atol::Float64=1e-12,
                 rsvd_oversample::Int=8, rsvd_poweriter::Int=0,
                 rsvd_threaded::Bool=Base.Threads.nthreads() > 1,
                 rsvd_minbatch::Integer=max(2, Base.Threads.nthreads()),
                 rsvd_memory_cap_bytes::Union{Nothing,Integer}=nothing,
                 rsvd_task_workspace_bytes::Union{Nothing,Integer}=nothing,
                 rsvd_fanout_diagnostics::Union{Nothing,Base.RefValue}=nothing,
                 factorized::Bool=false,
                 contraction_optimize::Bool=true,
                 contraction_sector_aware::Bool=true,
                 threaded_channels::Bool=false, channel_slices::Int=2,
                 channel_minbatch::Int=2,
                 channel_min_flops::Real=0,
                 channel_memory_cap_bytes::Union{Nothing,Real}=nothing,
                 distributed::Union{Nothing,AbstractDistributedContext}=nothing)
    scheme in (:exact, :rsvd, :directqr, :rangefinder) ||
        throw(ArgumentError(
            "expand!: scheme must be :exact, :rsvd, :directqr, or :rangefinder"))
    max_add >= 0 || throw(ArgumentError("expand!: max_add must be nonnegative"))
    rsvd_oversample >= 0 || throw(ArgumentError("expand!: rsvd_oversample must be nonnegative"))
    rsvd_poweriter >= 0 || throw(ArgumentError("expand!: rsvd_poweriter must be nonnegative"))
    rsvd_minbatch >= 1 || throw(ArgumentError("expand!: rsvd_minbatch must be positive"))
    rsvd_memory_cap_bytes === nothing || rsvd_memory_cap_bytes >= 0 ||
        throw(ArgumentError("expand!: rsvd_memory_cap_bytes must be nonnegative"))
    rsvd_task_workspace_bytes === nothing ||
        rsvd_task_workspace_bytes >= 0 ||
        throw(ArgumentError(
            "expand!: rsvd_task_workspace_bytes must be nonnegative"))
    scheme in (:rsvd, :rangefinder) && rng === nothing &&
        throw(ArgumentError(
            "expand!: scheme=$scheme requires an explicit rng (§9.6)"))
    iszero(mixing) && return ψ
    t = ψ.topo
    n, m = _edge_child_parent(t, edge)
    olddim = dim(virtualspace(ψ, n))
    cap = min(trunc.maxdim, olddim + max_add)
    cap > olddim || return ψ

    c = cache === nothing ? EnvCache(t) : cache
    move_center!(ψ, n; cache=c)
    use_factor_frame = factorized && scheme === :rangefinder
    U, R = if use_factor_frame
        source_basis, link = left_orth(ψ.tensors[n])
        factor_frame = oriented_two_site_factor_frame(
            c, ψ, H, n, m; source_tensor=source_basis,
            optimize=contraction_optimize)
        _factorized_rangefinder_enrich_split(
            ψ.tensors[n], source_basis, c, factor_frame, link,
            cap, mixing;
            rng=rng::AbstractRNG,
            rsvd_oversample, rsvd_poweriter,
            rsvd_threaded, rsvd_minbatch,
            rsvd_memory_cap_bytes,
            rsvd_task_workspace_bytes,
            rsvd_fanout_diagnostics,
            enr_rtol, enr_atol,
            optimize=contraction_optimize,
            sector_aware=contraction_sector_aware,
            threaded_channels, channel_slices, channel_minbatch,
            channel_min_flops, channel_memory_cap_bytes, distributed)
    else
        Θ = two_site_tensor(ψ, n, m)
        h2 = eff_h2(c, ψ, H, n, m;
                    optimize=contraction_optimize,
                    sector_aware=contraction_sector_aware,
                    threaded_channels, channel_slices, channel_minbatch,
                    channel_min_flops,
                    channel_memory_cap_bytes, distributed)
        PΘ = mixing * h2(Θ)
        P = _child_predictor_basis(
            ψ, PΘ, n, cap; scheme, rng,
            rsvd_oversample, rsvd_poweriter,
            rsvd_threaded, rsvd_minbatch,
            rsvd_memory_cap_bytes,
            rsvd_task_workspace_bytes,
            rsvd_fanout_diagnostics)
        _expand_enrich_split(ψ.tensors[n], P; maxdim=cap,
                             max_add=cap - olddim,
                             enr_rtol, enr_atol)
    end
    dim(domain(U)) == olddim && return ψ
    ψ.tensors[n] = U
    R = Networks.pivotal_link(R)
    ψ.tensors[m] = absorb_on_leg(ψ.tensors[m], R, childslot(t, m, n))
    ψ.center = m
    invalidate_edge!(c, n, m)
    return ψ
end

function _factorized_rangefinder_enrich_split(
        active::AbstractTensorMap,
        kept::AbstractTensorMap,
        cache::EnvCache,
        frame::OrientedTwoSiteFactorFrame,
        link::AbstractTensorMap,
        maxdim::Int,
        mixing::Number;
        rng::AbstractRNG,
        rsvd_oversample::Int,
        rsvd_poweriter::Int,
        rsvd_threaded::Bool,
        rsvd_minbatch::Integer,
        rsvd_memory_cap_bytes::Union{Nothing,Integer},
        rsvd_task_workspace_bytes::Union{Nothing,Integer},
        rsvd_fanout_diagnostics::Union{Nothing,Base.RefValue},
        enr_rtol::Float64,
        enr_atol::Float64,
        optimize::Bool,
        sector_aware::Bool,
        threaded_channels::Bool,
        channel_slices::Int,
        channel_minbatch::Int,
        channel_min_flops::Real,
        channel_memory_cap_bytes::Union{Nothing,Real},
        distributed::Union{Nothing,AbstractDistributedContext},
)
    room = maxdim - dim(domain(active))
    complement = left_null(kept)
    (room > 0 && dim(domain(complement)) > 0) ||
        return kept, kept' * active
    target_frame = id(
        scalartype(frame.target_action), codomain(frame.target_action))
    apply = target_basis -> mixing * contract_biprojected_two_site(
        cache, frame, link, complement, target_basis;
        optimize, sector_aware, threaded_channels, channel_slices,
        channel_minbatch, channel_min_flops, channel_memory_cap_bytes,
        distributed)
    adjoint_apply = source_sketch -> conj(mixing) *
        _pivotal_target_basis(contract_biprojected_two_site(
            cache, frame, link, complement * source_sketch, target_frame;
            optimize, sector_aware, threaded_channels, channel_slices,
            channel_minbatch, channel_min_flops, channel_memory_cap_bytes,
            distributed)', codomain(frame.target_action))
    sketch = rangefinder(
        apply, adjoint_apply, domain(target_frame);
        maxrank=room,
        oversample=rsvd_oversample,
        poweriter=rsvd_poweriter,
        rng,
        probe_eltype=scalartype(frame.source_action),
        threaded=rsvd_threaded,
        minbatch=rsvd_minbatch,
        memory_cap_bytes=rsvd_memory_cap_bytes,
        task_workspace_memory_bytes=rsvd_task_workspace_bytes,
        fanout_diagnostics=rsvd_fanout_diagnostics,
        distributed)

    # QR removes the predictor scale. Recover it in a rank-small core before
    # applying the ordinary enrichment tolerances and final rank cap.
    small = mixing * contract_biprojected_two_site(
        cache, frame, link, complement * sketch, target_frame;
        optimize, sector_aware, threaded_channels, channel_slices,
        channel_minbatch, channel_min_flops, channel_memory_cap_bytes,
        distributed)
    local_directions, _, _ = split_svd(
        small,
        TruncationScheme(; maxdim=room, atol=enr_atol, rtol=enr_rtol))
    dim(domain(local_directions)) == 0 && return kept, kept' * active
    enrichment = complement * (sketch * local_directions)
    if isdual(domain(kept)[1]) != isdual(domain(enrichment)[1])
        enrichment = flip(enrichment, numind(enrichment))
    end
    expanded = catdomain(kept, enrichment)
    return expanded, expanded' * active
end

function _pivotal_target_basis(adjoint_core::AbstractTensorMap,
                               target_space::ProductSpace)
    target_basis = adjoint_core
    for leg in 1:numout(target_basis)
        target_basis = flip(target_basis, leg)
    end
    codomain(target_basis) == target_space || throw(SpaceMismatch(
        "pivotal adjoint did not restore the target factor frame"))
    return target_basis
end

function _edge_child_parent(t::TreeTopology, edge::Pair)
    a, b = nodeindex(t, edge.first), nodeindex(t, edge.second)
    return _orient_edge(t, a, b)
end
function _edge_child_parent(t::TreeTopology, edge::Tuple{Any,Any})
    a, b = nodeindex(t, edge[1]), nodeindex(t, edge[2])
    return _orient_edge(t, a, b)
end
function _orient_edge(t::TreeTopology, a::Int, b::Int)
    if t.parent[a] == b
        return a, b
    elseif t.parent[b] == a
        return b, a
    else
        throw(ArgumentError("expand!: edge endpoints $(nodeid(t, a)), $(nodeid(t, b)) are not adjacent"))
    end
end

function _child_predictor_basis(ψ::TTNS, PΘ::AbstractTensorMap, n::Int, maxdim::Int;
                                scheme::Symbol=:exact,
                                rng::Union{Nothing,AbstractRNG}=nothing,
                                rsvd_oversample::Int=8,
                                rsvd_poweriter::Int=0,
                                rsvd_threaded::Bool=Base.Threads.nthreads() > 1,
                                rsvd_minbatch::Integer=max(2, Base.Threads.nthreads()),
                                rsvd_memory_cap_bytes::Union{Nothing,Integer}=nothing,
                                rsvd_task_workspace_bytes::Union{Nothing,Integer}=nothing,
                                rsvd_fanout_diagnostics::Union{Nothing,Base.RefValue}=nothing)
    pn = numout(ψ.tensors[n])
    NP = numind(PΘ)
    Ps = permute(PΘ, (ntuple(identity, pn), ntuple(j -> pn + j, NP - pn)))
    if scheme === :rsvd
        return _rsvd_predictor_basis(Ps, maxdim; rng, rsvd_oversample,
                                     rsvd_poweriter, threaded=rsvd_threaded,
                                     minbatch=rsvd_minbatch,
                                     memory_cap_bytes=rsvd_memory_cap_bytes,
                                     task_workspace_memory_bytes=
                                         rsvd_task_workspace_bytes,
                                     fanout_diagnostics=rsvd_fanout_diagnostics)
    elseif scheme === :directqr
        return _directqr_predictor_basis(Ps)
    elseif scheme === :rangefinder
        return _rangefinder_predictor_basis(
            Ps, maxdim; rng, rsvd_oversample, rsvd_poweriter,
            threaded=rsvd_threaded, minbatch=rsvd_minbatch,
            memory_cap_bytes=rsvd_memory_cap_bytes,
            task_workspace_memory_bytes=rsvd_task_workspace_bytes,
            fanout_diagnostics=rsvd_fanout_diagnostics)
    end
    U, _, _ = split_svd(Ps, TruncationScheme(; maxdim))
    return U
end

function _directqr_predictor_basis(Ps::AbstractTensorMap)
    Q, _ = left_orth(Ps; alg=:qr)
    return Q
end

"""
    rangefinder(apply, adjoint_apply, domain;
                maxrank, oversample, poweriter, rng, probe_eltype, ...)

Construct an oversampled orthonormal basis for the range of a matrix-free
operator. `apply` and `adjoint_apply` act on TensorMap blocks, while `domain`
is the unfused input space used to construct sector-aware Gaussian probes.
The caller supplies `probe_eltype` explicitly because a TensorKit space does
not encode the numeric precision of the operator.

The primitive owns only randomized probing and QR stabilization. Any projected
small-matrix factorization or final rank truncation remains with the caller.
"""
function rangefinder(apply::F, adjoint_apply::A, probe_domain;
                     maxrank::Int,
                     oversample::Int,
                     poweriter::Int,
                     rng::AbstractRNG,
                     probe_eltype::Type{T},
                     threaded::Bool=Base.Threads.nthreads() > 1,
                     minbatch::Integer=max(2, Base.Threads.nthreads()),
                     memory_cap_bytes::Union{Nothing,Integer}=nothing,
                     task_workspace_memory_bytes::Union{Nothing,Integer}=nothing,
                     fanout_diagnostics::Union{Nothing,Base.RefValue}=nothing,
                     distributed::Union{Nothing,AbstractDistributedContext}=nothing) where
                    {F,A,T<:Number}
    maxrank > 0 || throw(ArgumentError("rangefinder maxrank must be positive"))
    oversample >= 0 ||
        throw(ArgumentError("rangefinder oversample must be nonnegative"))
    poweriter >= 0 ||
        throw(ArgumentError("rangefinder poweriter must be nonnegative"))

    Vrest = fuse(probe_domain)
    sketchdim = min(dim(Vrest), maxrank + oversample)
    K = _rsvd_probe_space(Vrest, sketchdim)
    Ω = _rsvd_random_probe(rng, T, probe_domain ← K;
                           threaded, minbatch, memory_cap_bytes,
                           task_workspace_memory_bytes,
                           fanout_diagnostics)
    distributed === nothing || distributed_broadcast!(distributed, Ω)
    Q, _ = left_orth(apply(Ω)::AbstractTensorMap; alg=:qr)
    for _ in 1:poweriter
        Z, _ = left_orth(adjoint_apply(Q)::AbstractTensorMap; alg=:qr)
        Q, _ = left_orth(apply(Z)::AbstractTensorMap; alg=:qr)
    end
    return Q
end

function _rangefinder_predictor_basis(Ps::AbstractTensorMap, maxdim::Int;
                                      rng::AbstractRNG,
                                      rsvd_oversample::Int,
                                      rsvd_poweriter::Int,
                                      threaded::Bool=Base.Threads.nthreads() > 1,
                                      minbatch::Integer=max(2, Base.Threads.nthreads()),
                                      memory_cap_bytes::Union{Nothing,Integer}=nothing,
                                      task_workspace_memory_bytes::Union{Nothing,Integer}=nothing,
                                      fanout_diagnostics::Union{Nothing,Base.RefValue}=nothing)
    return rangefinder(x -> Ps * x, x -> Ps' * x, domain(Ps);
                       maxrank=maxdim,
                       oversample=rsvd_oversample,
                       poweriter=rsvd_poweriter,
                       rng,
                       probe_eltype=scalartype(Ps),
                       threaded, minbatch, memory_cap_bytes,
                       task_workspace_memory_bytes,
                       fanout_diagnostics)
end

function _rsvd_predictor_basis(Ps::AbstractTensorMap, maxdim::Int;
                               rng::AbstractRNG,
                               rsvd_oversample::Int,
                               rsvd_poweriter::Int,
                               threaded::Bool=Base.Threads.nthreads() > 1,
                               minbatch::Integer=max(2, Base.Threads.nthreads()),
                               memory_cap_bytes::Union{Nothing,Integer}=nothing,
                               task_workspace_memory_bytes::Union{Nothing,Integer}=nothing,
                               fanout_diagnostics::Union{Nothing,Base.RefValue}=nothing)
    Vrest = fuse(domain(Ps))
    budget = min(dim(Vrest), maxdim + rsvd_oversample)
    K = _rsvd_probe_space(Vrest, budget)
    Ω = _rsvd_random_probe(rng, scalartype(Ps), domain(Ps) ← K;
                           threaded, minbatch, memory_cap_bytes,
                           task_workspace_memory_bytes,
                           fanout_diagnostics)
    Y = Ps * Ω
    for _ in 1:rsvd_poweriter
        Y = Ps * (Ps' * Y)
    end
    U, _, _ = split_svd(Y, TruncationScheme(; maxdim))
    return U
end

function _rsvd_random_probe(rng::AbstractRNG, ::Type{T}, target;
                            threaded::Bool=Base.Threads.nthreads() > 1,
                            minbatch::Integer=max(2, Base.Threads.nthreads()),
                            memory_cap_bytes::Union{Nothing,Integer}=nothing,
                            task_workspace_memory_bytes::Union{Nothing,Integer}=nothing,
                            fanout_diagnostics::Union{Nothing,Base.RefValue}=nothing,
                            block_fill=(rng, block_, _) -> randn!(rng, block_)) where {T<:Number}
    minbatch >= 1 || throw(ArgumentError("RSVD probe minbatch must be positive"))
    memory_cap_bytes === nothing || memory_cap_bytes >= 0 ||
        throw(ArgumentError("RSVD probe memory cap must be nonnegative"))
    task_workspace_memory_bytes === nothing ||
        task_workspace_memory_bytes >= 0 ||
        throw(ArgumentError(
            "RSVD probe task workspace bytes must be nonnegative"))
    Ω = zeros(T, target)
    probe_blocks = collect(blocks(Ω))
    # TensorMap block storage does not promise a stable iteration order.
    # Attach per-block seeds only after imposing a canonical sector order so
    # the caller-visible RNG contract and generated probe are independent of
    # allocation history and task scheduling.
    sort!(probe_blocks; by=item -> repr(item[1]))
    retained_memory_bytes = sum(
        block_ -> _rsvd_block_payload_bytes(block_[2]),
        probe_blocks;
        init=0,
    )
    rng_bytes = Base.summarysize(Xoshiro(zero(UInt64)))
    item_memory_bytes = [
        _rsvd_block_payload_bytes(block_) + rng_bytes +
            something(task_workspace_memory_bytes, 0)
        for (_, block_) in probe_blocks
    ]
    if memory_cap_bytes !== nothing
        cap = Int(memory_cap_bytes)
        retained_memory_bytes <= cap ||
            throw(BoundedFanoutAdmissionError(
                0, :retained_probe, 0, retained_memory_bytes, cap))
        for index in eachindex(probe_blocks)
            item_memory_bytes[index] <= cap - retained_memory_bytes ||
                throw(BoundedFanoutAdmissionError(
                    index,
                    (; probe_block_index=index,
                       sector=probe_blocks[index][1]),
                    item_memory_bytes[index],
                    retained_memory_bytes,
                    cap,
                ))
        end
    end
    seeds = rand(rng, UInt64, length(probe_blocks))
    probe_items = [
        (index, sector, block_, seeds[index])
        for (index, (sector, block_)) in enumerate(probe_blocks)
    ]
    missing_workspace_model = threaded &&
        task_workspace_memory_bytes === nothing
    diagnostics = bounded_threaded_foreach(
        probe_items;
        threaded=threaded && !missing_workspace_model,
        minbatch,
        retained_memory_bytes,
        memory_cap_bytes,
        item_memory_bytes=(_, item) -> item_memory_bytes[item[1]],
        item_id=(_, item) -> (; probe_block_index=item[1],
                              sector=item[2]),
    ) do item
        index, _, block_, seed = item
        block_fill(Xoshiro(seed), block_, index)
    end
    missing_workspace_model &&
        (diagnostics = _rsvd_fanout_fallback(
            diagnostics, :missing_task_workspace_memory))
    fanout_diagnostics === nothing || (fanout_diagnostics[] = diagnostics)
    return Ω
end

function _rsvd_fanout_fallback(
    diagnostics::BoundedFanoutDiagnostics,
    fallback::Symbol,
)
    return BoundedFanoutDiagnostics(
        diagnostics.mode,
        fallback,
        diagnostics.item_count,
        diagnostics.worker_limit,
        diagnostics.batch_count,
        diagnostics.max_batch_items,
        diagnostics.retained_memory_bytes,
        diagnostics.peak_admitted_bytes,
        diagnostics.memory_cap_bytes,
        diagnostics.completed_items,
        diagnostics.cancelled_items,
    )
end

function _rsvd_block_payload_bytes(block_)
    T = eltype(block_)
    return isbitstype(T) ? sizeof(T) * length(block_) :
        Base.summarysize(block_)
end

function _rsvd_probe_space(::ComplexSpace, budget::Int)
    return ℂ^budget
end

function _rsvd_probe_space(Vrest::S, budget::Int) where {S<:ElementarySpace}
    Q = sectortype(Vrest)
    dims = Pair{Q,Int}[]
    for q in sectors(Vrest)
        kq = min(dim(Vrest, q), budget)
        kq > 0 && push!(dims, q => kq)
    end
    isempty(dims) && throw(ArgumentError("expand!: randomized probe space is empty"))
    return Vect[Q](dims...)
end

function _expand_enrich_split(A::AbstractTensorMap, P::AbstractTensorMap;
                              maxdim::Int, max_add::Int,
                              enr_rtol::Float64, enr_atol::Float64)
    expanded = A
    room = min(max_add, maxdim - dim(domain(A)))
    if room > 0 && dim(codomain(A)) > dim(domain(A))
        N = left_null(A)
        if dim(domain(N)) > 0
            M = N' * P
            Um, _, _ = split_svd(M, TruncationScheme(; maxdim=room,
                                                       atol=enr_atol,
                                                       rtol=enr_rtol))
            if dim(domain(Um)) > 0
                E = N * Um
                if isdual(domain(A)[1]) != isdual(domain(E)[1])
                    E = flip(E, numind(E))
                end
                expanded = catdomain(A, E)
            end
        end
    end
    U, _, _ = split_svd(expanded, TruncationScheme(; maxdim))
    R = U' * A
    return U, R
end

"""
    _physless_root_growth_targets(ψ, trunc, max_add) -> Vector{Tuple{Int,Int}}

Per-edge target dimensions for the two children of a physical-leg-free binary
root.  A two-site predictor on either root edge is rank-limited by its sibling
edge: when both start at the same narrow dimension, neither one can provide the
other with a larger predictor basis.  Record their ordinary per-sweep caps
before any one-edge expansion so the paired bootstrap never exceeds `max_add`.
"""
function _physless_root_growth_targets(ψ::TTNS, trunc::TruncationScheme,
                                       max_add::Int)
    t = ψ.topo
    root = t.root
    (!hasphys(ψ, root) && length(t.children[root]) == 2) || return Tuple{Int,Int}[]
    return [(n, min(trunc.maxdim, dim(virtualspace(ψ, n)) + max_add))
            for n in t.children[root]]
end

"""
    _physless_root_two_site_targets(ψ, trunc) -> Vector{Tuple{Int,Int}}

Joint root-edge targets for two-site DMRG.  Unlike 3S, two-site DMRG has no
per-sweep `max_add`: its SVD may retain every Schmidt direction allowed by
`trunc`.  At a physical-leg-free binary root, open both child legs to that
ordinary two-site cap before the first root-edge update.  The common target is
also bounded by both child-side codomain dimensions, so an unbounded
`TruncationScheme()` never requests an artificial infinite virtual space.
"""
function _physless_root_two_site_targets(ψ::TTNS, trunc::TruncationScheme)
    t = ψ.topo
    root = t.root
    children = t.children[root]
    (!hasphys(ψ, root) && length(children) == 2) || return Tuple{Int,Int}[]
    target = min(trunc.maxdim, minimum(dim(codomain(ψ.tensors[n])) for n in children))
    return [(n, target) for n in children]
end

"""
    _null_enrich_split(A; maxdim) -> (U, R)

State-preserving completion of the column space of `A`: `A == U * R` while
`U` contains up to `maxdim` orthonormal columns.  This is only the bootstrap
fallback for the two sibling edges of a physical-leg-free binary root.  The
Hamiltonian-selected predictor remains the primary enrichment everywhere else;
without this paired completion its rank is bounded by the opposite root edge
and cannot start the growth at all.
"""
function _null_enrich_split(A::AbstractTensorMap; maxdim::Int)
    olddim = dim(domain(A))
    room = maxdim - olddim
    room > 0 && dim(codomain(A)) > olddim || return A, nothing

    N = left_null(A)
    dim(domain(N)) > 0 || return A, nothing
    E, _, _ = split_svd(N, TruncationScheme(; maxdim=room))
    if isdual(domain(A)[1]) != isdual(domain(E)[1])
        E = flip(E, numind(E))
    end
    U, _, _ = split_svd(catdomain(A, E), TruncationScheme(; maxdim))
    return U, U' * A
end

"""
    _bootstrap_physless_root!(ψ, cache, targets) -> ψ

Jointly complete the two root-child subspaces recorded in `targets`.  Each
completion is a gauge-preserving zero-weight embedding, so it cannot alter the
TTNS vector.  Performing both before the next one-site root update removes the
otherwise circular rank bound between the siblings.
"""
function _bootstrap_physless_root!(ψ::TTNS, cache::EnvCache,
                                   targets::Vector{Tuple{Int,Int}})
    isempty(targets) && return ψ
    t = ψ.topo
    root = t.root
    for (n, target) in targets
        t.parent[n] == root || throw(ArgumentError("root bootstrap target is not a root child"))
        dim(virtualspace(ψ, n)) >= target && continue
        move_center!(ψ, n; cache)
        U, R = _null_enrich_split(ψ.tensors[n]; maxdim=target)
        R === nothing && continue
        ψ.tensors[n] = U
        R = Networks.pivotal_link(R)
        ψ.tensors[root] = absorb_on_leg(ψ.tensors[root], R, childslot(t, root, n))
        ψ.center = root
        invalidate_edge!(cache, n, root)
    end
    return ψ
end
