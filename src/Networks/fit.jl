import ..Contractions: Planning

"""
    fit!(φ, ψ; nsweeps=4, tol=1e-10, normalize=false, verbose=false) -> (φ, errors)
    fit!(φ, sources; Hs=nothing, coeffs=nothing, kwargs...) -> (φ, errors)

Variationally fit `φ ≈ ψ` on the fixed TTNS manifold carried by `φ`.
This is the state-compression core of the architecture's public `fit!`
primitive (§3/§11.6). The source `ψ` is not mutated; `φ` is gauged and updated
in place. Because `φ` is canonicalized to the updated node, the one-site normal
matrix is the identity, so each ALS local solve is a direct projection of `ψ`
onto the current target environment.

The multi-source form fits `φ ≈ sum_i coeffs[i] * src_i`, or, when `Hs` is
provided, `φ ≈ sum_i coeffs[i] * apply(Hs[i], src_i)`. The operator-weighted
form is the compression surface used by GlobalKrylov/GSE-style algorithms.
"""
function fit!(φ::TTNS, ψ::TTNS; nsweeps::Int=4, tol::Float64=1e-10,
              normalize::Bool=false, verbose::Bool=false)
    return fit!(φ, (ψ,); nsweeps, tol, normalize, verbose)
end

function fit!(φ::TTNS, sources; Hs=nothing, coeffs=nothing,
              nsweeps::Int=4, tol::Float64=1e-10,
              normalize::Bool=false, verbose::Bool=false)
    srcs = _fit_sources(φ, sources, Hs)
    coeffv = _fit_coeffs(φ, srcs, coeffs)
    nsweeps >= 0 || throw(ArgumentError("fit!: nsweeps must be nonnegative"))
    target_center = center(φ)
    caches = [_FitCache(φ.topo) for _ in srcs]
    errors = Float64[]
    order = postorder(φ.topo)
    for sweep in 1:nsweeps
        for n in Iterators.flatten((order, Iterators.reverse(order)))
            move_center!(φ, n)
            for c in caches
                _invalidate_fit_node!(c, n)
            end
            A = _fit_local_tensor(caches, φ, srcs, coeffv, n)
            update_tensor!(φ, n, A; gauge=true)
            for c in caches
                _invalidate_fit_node!(c, n)
            end
        end
        normalize && normalize!(φ)
        err = _fit_error(φ, srcs, coeffv)
        push!(errors, err)
        verbose && @info "fit! sweep $sweep" err
        length(errors) > 1 && abs(errors[end] - errors[end - 1]) < tol && break
    end
    move_center!(φ, target_center)
    return φ, errors
end

function _fit_sources(φ::TTNS, sources, Hs)
    srcs0 = collect(sources)
    isempty(srcs0) && throw(ArgumentError("fit!: at least one source is required"))
    srcs = if Hs === nothing
        srcs0
    else
        ops = collect(Hs)
        length(ops) == length(srcs0) ||
            throw(ArgumentError("fit!: Hs and sources must have the same length"))
        [op === nothing ? src : apply(op, src; center=center(φ)) for (op, src) in zip(ops, srcs0)]
    end
    for src in srcs
        src isa TTNS || throw(ArgumentError("fit!: every source must be a TTNS"))
        φ.topo == src.topo || throw(ArgumentError("fit!: source and target topologies differ"))
        φ.hasphys == src.hasphys || throw(ArgumentError("fit!: physical-leg layout mismatch"))
        spacetype(φ) == spacetype(src) || throw(ArgumentError("fit!: source and target spacetype mismatch"))
    end
    return srcs
end

function _fit_coeffs(φ::TTNS, sources, coeffs)
    T = eltype(φ)
    coeffv = if coeffs === nothing
        fill(one(T), length(sources))
    else
        c = collect(coeffs)
        length(c) == length(sources) ||
            throw(ArgumentError("fit!: coeffs and sources must have the same length"))
        c
    end
    if T <: Real
        any(src -> eltype(src) <: Complex, sources) &&
            throw(ArgumentError("fit!: real target cannot fit complex-eltype sources without explicit complex target"))
        any(c -> c isa Complex && !isreal(c), coeffv) &&
            throw(ArgumentError("fit!: real target cannot use complex coefficients without explicit complex target"))
    end
    return convert(Vector{T}, coeffv)
end

"""
Value environments, shape-only plans, and immutable root caps for one source
inside a fit sweep.  Invalidation deliberately removes only `envs`: tensor
values do not change exact TensorKit spaces, so plans and caps remain valid.
"""
struct _FitCache
    topo::TreeTopology
    envs::Dict{Tuple{Int,Int},AbstractTensorMap}
    plans::Dict{Planning.PlanKey,Planning.ContractionPlan}
    rootcaps::Dict{Tuple,AbstractTensorMap}
end
_FitCache(topo::TreeTopology) =
    _FitCache(topo, Dict{Tuple{Int,Int},AbstractTensorMap}(),
              Dict{Planning.PlanKey,Planning.ContractionPlan}(),
              Dict{Tuple,AbstractTensorMap}())
_FitCache(topo::TreeTopology, envs::Dict{Tuple{Int,Int},AbstractTensorMap}) =
    _FitCache(topo, envs, Dict{Planning.PlanKey,Planning.ContractionPlan}(),
              Dict{Tuple,AbstractTensorMap}())

function _invalidate_fit_node!(c::_FitCache, n::Int)
    filter!(p -> !_fit_on_side(c.topo, n, p.first[1], p.first[2]), c.envs)
    return c
end

function _fit_on_side(t::TreeTopology, n::Int, u::Int, v::Int)
    n == u && return true
    n == v && return false
    return u in path_between(t, n, v)
end

_fit_scalar_type(φ::TTNS, ψ::TTNS) = promote_type(eltype(φ), eltype(ψ))

"""Return a root cap owned by the fit cache and keyed only by type and space."""
function _fit_root_cap!(c::_FitCache, T::DataType, capspace)
    return get!(c.rootcaps, (T, capspace)) do
        ones_tensor(T, capspace)
    end
end

"""Execute a complete-tuple fit network through its independent plan cache."""
function _fit_planned_execute!(c::_FitCache, kind::Symbol,
                               spec::Planning.ContractionSpec,
                               operands::Tuple, T::DataType)
    plan, _ = Planning.get_or_plan!(c.plans, kind, spec, operands, T)
    return Planning.execute(plan, operands)
end

function _fit_env!(c::_FitCache, φ::TTNS, ψ::TTNS, u::Int, v::Int)
    return get!(c.envs, (u, v)) do
        for w in neighbors(c.topo, u)
            w == v && continue
            _fit_env!(c, φ, ψ, w, u)
        end
        _fit_build_env(c, φ, ψ, u, v)
    end
end

function _fit_local_tensor(c::_FitCache, φ::TTNS, ψ::TTNS, n::Int)
    t = φ.topo
    for w in neighbors(t, n)
        _fit_env!(c, φ, ψ, w, n)
    end
    return _fit_project_tensor(c, φ, ψ, n)
end

function _fit_local_tensor(caches::Vector{_FitCache}, φ::TTNS, sources,
                           coeffs::AbstractVector, n::Int)
    length(caches) == length(sources) == length(coeffs) ||
        throw(ArgumentError("fit local projection: cache/source/coefficient length mismatch"))
    # The destination is fresh for `update_tensor!`; every source writes its
    # final contraction into it, so no full per-source `Ai` is retained.
    A = zeros(eltype(φ), space(φ.tensors[n]))
    for (c, src, α) in zip(caches, sources, coeffs)
        _fit_project_tensor!(A, c, φ, src, n, α)
    end
    return A
end

"""
    _fit_build_env_ncon_reference(φ, ψ, u, v, envs)

Retained direct `ncon` transfer environment for small planned-versus-legacy
tests.  The planned lowering keeps this operand order and labels unchanged.
"""
function _fit_build_env_ncon_reference(φ::TTNS, ψ::TTNS, u::Int, v::Int,
                                       envs::Dict{Tuple{Int,Int},AbstractTensorMap})
    t = φ.topo
    A = ψ.tensors[u]
    B = φ.tensors[u]
    hp = hasphys(ψ, u)
    aidx = zeros(Int, numind(A))
    bidx = zeros(Int, numind(B))
    tensors = Any[A]
    indices = Vector{Int}[aidx]
    conjs = Bool[false]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    if hp
        p = physleg(ψ, u)
        lbl = fresh()
        aidx[p] = lbl
        bidx[p] = v == 0 ? -p : lbl
    end
    for w in neighbors(t, u)
        la = _fit_stateleg(t, hp, u, w)
        if w == v
            aidx[la] = -1
            bidx[la] = -2
        else
            E = envs[(w, u)]
            eidx = [fresh(), fresh()]
            aidx[la] = eidx[1]
            bidx[la] = eidx[2]
            push!(tensors, E); push!(indices, eidx); push!(conjs, false)
        end
    end
    if t.parent[u] == 0
        ka, kb = fresh(), fresh()
        aidx[end] = ka
        bidx[end] = kb
        cap = ones_tensor(promote_type(eltype(φ), eltype(ψ)),
                          dual(domain(B)[1]) ⊗ domain(A)[1])
        push!(tensors, cap); push!(indices, [kb, ka]); push!(conjs, false)
    end
    if v == 0
        for i in eachindex(bidx)
            iszero(bidx[i]) && (bidx[i] = -i)
        end
    end
    push!(tensors, B); push!(indices, bidx); push!(conjs, true)
    y = ncon(tensors, indices, conjs)
    return v == 0 ? repartition(y, numout(B), 1) : y
end

"""Lower a fit transfer environment into a complete-operand contraction spec."""
function _fit_build_env_spec(c::_FitCache, φ::TTNS, ψ::TTNS, u::Int, v::Int)
    t = φ.topo
    A = ψ.tensors[u]
    B = φ.tensors[u]
    hp = hasphys(ψ, u)
    aidx = zeros(Int, numind(A))
    bidx = zeros(Int, numind(B))
    operands = Any[A]
    labels = Vector{Int}[aidx]
    conjs = Bool[false]
    envslots = Int[]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    if hp
        p = physleg(ψ, u)
        lbl = fresh()
        aidx[p] = lbl
        bidx[p] = v == 0 ? -p : lbl
    end
    for w in neighbors(t, u)
        la = _fit_stateleg(t, hp, u, w)
        if w == v
            aidx[la] = -1
            bidx[la] = -2
        else
            E = c.envs[(w, u)]
            eidx = [fresh(), fresh()]
            aidx[la] = eidx[1]
            bidx[la] = eidx[2]
            push!(operands, E); push!(labels, eidx); push!(conjs, false)
            push!(envslots, length(labels))
        end
    end
    caps = Int[]
    if t.parent[u] == 0
        ka, kb = fresh(), fresh()
        aidx[end] = ka
        bidx[end] = kb
        T = _fit_scalar_type(φ, ψ)
        cap = _fit_root_cap!(c, T, dual(domain(B)[1]) ⊗ domain(A)[1])
        push!(operands, cap); push!(labels, [kb, ka]); push!(conjs, false)
        push!(caps, length(labels))
    end
    if v == 0
        for i in eachindex(bidx)
            iszero(bidx[i]) && (bidx[i] = -i)
        end
    end
    push!(operands, B); push!(labels, bidx); push!(conjs, true)
    preferred = Int[1]
    append!(preferred, envslots)
    append!(preferred, caps)
    push!(preferred, length(labels))
    nopen = v == 0 ? numind(B) : 2
    partition = v == 0 ? (numout(B), 1) : (2, 0)
    spec = Planning.ContractionSpec(labels, conjs, nopen, partition, nothing;
                                    preferred_slots=preferred)
    return spec, Tuple(operands)
end

function _fit_build_env(c::_FitCache, φ::TTNS, ψ::TTNS, u::Int, v::Int)
    spec, operands = _fit_build_env_spec(c, φ, ψ, u, v)
    return _fit_planned_execute!(c, :fit_env, spec, operands,
                                 _fit_scalar_type(φ, ψ))
end

"""
    _fit_project_tensor_ncon_reference(φ, ψ, n, envs)

Retained direct `ncon` local projection.  Its flat target-leg labels are the
authoritative reference for the planned projection below.
"""
function _fit_project_tensor_ncon_reference(
    φ::TTNS, ψ::TTNS, n::Int,
    envs::Dict{Tuple{Int,Int},AbstractTensorMap},
)
    t = φ.topo
    A = ψ.tensors[n]
    B = φ.tensors[n]
    T = promote_type(eltype(φ), eltype(ψ))
    hp = hasphys(ψ, n)
    aidx = zeros(Int, numind(A))
    tensors = Any[A]
    indices = Vector{Int}[aidx]
    conjs = Bool[false]
    nxt = Ref(0)
    fresh() = (nxt[] += 1)

    if hp
        aidx[physleg(ψ, n)] = -(nchildren(t, n) + 1)
    end
    for w in neighbors(t, n)
        la = _fit_stateleg(t, hp, n, w)
        E = envs[(w, n)]
        eidx = [fresh(), -la]
        aidx[la] = eidx[1]
        push!(tensors, E); push!(indices, eidx); push!(conjs, false)
    end
    if t.parent[n] == 0
        ka = fresh()
        aidx[end] = ka
        cap = ones_tensor(T, dual(domain(B)[1]) ⊗ domain(A)[1])
        push!(tensors, cap); push!(indices, [-numind(B), ka]); push!(conjs, false)
    end
    y = ncon(tensors, indices, conjs)
    return repartition(y, numout(B), 1)
end

"""Lower one local fit projection into a complete-operand planned spec."""
function _fit_project_spec(c::_FitCache, φ::TTNS, ψ::TTNS, n::Int)
    t = φ.topo
    A = ψ.tensors[n]
    B = φ.tensors[n]
    hp = hasphys(ψ, n)
    aidx = zeros(Int, numind(A))
    operands = Any[A]
    labels = Vector{Int}[aidx]
    conjs = Bool[false]
    envslots = Int[]
    nxt = Ref(0)
    fresh() = (nxt[] += 1)

    if hp
        aidx[physleg(ψ, n)] = -(nchildren(t, n) + 1)
    end
    for w in neighbors(t, n)
        la = _fit_stateleg(t, hp, n, w)
        E = c.envs[(w, n)]
        eidx = [fresh(), -la]
        aidx[la] = eidx[1]
        push!(operands, E); push!(labels, eidx); push!(conjs, false)
        push!(envslots, length(labels))
    end
    caps = Int[]
    if t.parent[n] == 0
        ka = fresh()
        aidx[end] = ka
        T = _fit_scalar_type(φ, ψ)
        cap = _fit_root_cap!(c, T, dual(domain(B)[1]) ⊗ domain(A)[1])
        push!(operands, cap); push!(labels, [-numind(B), ka]); push!(conjs, false)
        push!(caps, length(labels))
    end
    preferred = Int[1]
    append!(preferred, envslots)
    append!(preferred, caps)
    spec = Planning.ContractionSpec(labels, conjs, numind(B),
                                    (numout(B), 1), nothing;
                                    preferred_slots=preferred)
    return spec, Tuple(operands)
end

function _fit_project_plan(c::_FitCache, φ::TTNS, ψ::TTNS, n::Int)
    spec, operands = _fit_project_spec(c, φ, ψ, n)
    plan, _ = Planning.get_or_plan!(c.plans, :fit_project, spec, operands,
                                    _fit_scalar_type(φ, ψ))
    return plan, operands
end

function _fit_project_tensor(c::_FitCache, φ::TTNS, ψ::TTNS, n::Int)
    plan, operands = _fit_project_plan(c, φ, ψ, n)
    return Planning.execute(plan, operands)
end

"""Accumulate one source projection directly into a fresh target tensor."""
function _fit_project_tensor!(dest::AbstractTensorMap, c::_FitCache,
                              φ::TTNS, ψ::TTNS, n::Int, α::Number)
    for w in neighbors(φ.topo, n)
        _fit_env!(c, φ, ψ, w, n)
    end
    plan, operands = _fit_project_plan(c, φ, ψ, n)
    return Planning.execute_accumulate!(dest, plan, operands;
                                        α=α, β=one(eltype(φ)))
end

_fit_stateleg(t::TreeTopology, hasphys_u::Bool, u::Int, w::Int) =
    t.parent[u] == w ? nchildren(t, u) + (hasphys_u ? 1 : 0) + 1 : childslot(t, u, w)

function _fit_overlap(φ::TTNS, ψ::TTNS)
    φ.topo == ψ.topo || throw(ArgumentError("fit overlap: topologies differ"))
    c = _FitCache(φ.topo)
    r = φ.topo.root
    for w in neighbors(φ.topo, r)
        _fit_env!(c, φ, ψ, w, r)
    end
    return _fit_scalar(c, φ, ψ, r)
end

"""
    _fit_scalar_ncon_reference(φ, ψ, u, envs)

Retained direct `ncon` overlap closure for small fixture A/B tests.
"""
function _fit_scalar_ncon_reference(φ::TTNS, ψ::TTNS, u::Int,
                                    envs::Dict{Tuple{Int,Int},AbstractTensorMap})
    t = φ.topo
    A = ψ.tensors[u]
    B = φ.tensors[u]
    hp = hasphys(ψ, u)
    aidx = zeros(Int, numind(A))
    bidx = zeros(Int, numind(B))
    tensors = Any[A]
    indices = Vector{Int}[aidx]
    conjs = Bool[false]
    nxt = Ref(0)
    fresh() = (nxt[] += 1)
    if hp
        lbl = fresh()
        aidx[physleg(ψ, u)] = lbl
        bidx[physleg(φ, u)] = lbl
    end
    for w in neighbors(t, u)
        la = _fit_stateleg(t, hp, u, w)
        E = envs[(w, u)]
        eidx = [fresh(), fresh()]
        aidx[la] = eidx[1]
        bidx[la] = eidx[2]
        push!(tensors, E); push!(indices, eidx); push!(conjs, false)
    end
    if t.parent[u] == 0
        ka, kb = fresh(), fresh()
        aidx[end] = ka
        bidx[end] = kb
        cap = ones_tensor(promote_type(eltype(φ), eltype(ψ)),
                          dual(domain(B)[1]) ⊗ domain(A)[1])
        push!(tensors, cap); push!(indices, [kb, ka]); push!(conjs, false)
    end
    push!(tensors, B); push!(indices, bidx); push!(conjs, true)
    return ncon(tensors, indices, conjs)
end

"""Lower a fully contracted one-node fit overlap into a planned spec."""
function _fit_scalar_spec(c::_FitCache, φ::TTNS, ψ::TTNS, u::Int)
    t = φ.topo
    A = ψ.tensors[u]
    B = φ.tensors[u]
    hp = hasphys(ψ, u)
    aidx = zeros(Int, numind(A))
    bidx = zeros(Int, numind(B))
    operands = Any[A]
    labels = Vector{Int}[aidx]
    conjs = Bool[false]
    envslots = Int[]
    nxt = Ref(0)
    fresh() = (nxt[] += 1)
    if hp
        lbl = fresh()
        aidx[physleg(ψ, u)] = lbl
        bidx[physleg(φ, u)] = lbl
    end
    for w in neighbors(t, u)
        la = _fit_stateleg(t, hp, u, w)
        E = c.envs[(w, u)]
        eidx = [fresh(), fresh()]
        aidx[la] = eidx[1]
        bidx[la] = eidx[2]
        push!(operands, E); push!(labels, eidx); push!(conjs, false)
        push!(envslots, length(labels))
    end
    caps = Int[]
    if t.parent[u] == 0
        ka, kb = fresh(), fresh()
        aidx[end] = ka
        bidx[end] = kb
        T = _fit_scalar_type(φ, ψ)
        cap = _fit_root_cap!(c, T, dual(domain(B)[1]) ⊗ domain(A)[1])
        push!(operands, cap); push!(labels, [kb, ka]); push!(conjs, false)
        push!(caps, length(labels))
    end
    push!(operands, B); push!(labels, bidx); push!(conjs, true)
    preferred = Int[1]
    append!(preferred, envslots)
    append!(preferred, caps)
    push!(preferred, length(labels))
    spec = Planning.ContractionSpec(labels, conjs, 0, (0, 0), nothing;
                                    preferred_slots=preferred)
    return spec, Tuple(operands)
end

function _fit_scalar(c::_FitCache, φ::TTNS, ψ::TTNS, u::Int)
    spec, operands = _fit_scalar_spec(c, φ, ψ, u)
    return _fit_planned_execute!(c, :fit_scalar, spec, operands,
                                 _fit_scalar_type(φ, ψ))
end

function _fit_error(φ::TTNS, ψ::TTNS)
    nφ = real(_fit_overlap(φ, φ))
    nψ = real(_fit_overlap(ψ, ψ))
    ov = _fit_overlap(φ, ψ)
    return sqrt(max(nφ + nψ - 2 * real(ov), 0))
end

function _fit_error(φ::TTNS, sources, coeffs)
    nφ = _fit_overlap(φ, φ)
    ntarget = zero(nφ)
    cross = zero(nφ)
    for i in eachindex(sources)
        cross += coeffs[i] * _fit_overlap(φ, sources[i])
        for j in eachindex(sources)
            ntarget += conj(coeffs[i]) * coeffs[j] * _fit_overlap(sources[i], sources[j])
        end
    end
    return sqrt(max(real(nφ + ntarget - 2 * real(cross)), 0))
end
