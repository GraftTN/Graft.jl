import ..Contractions: Planning, _euclidean_bra_tensor, _euclidean_output_legs

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
provided, `φ ≈ sum_i coeffs[i] * Hs[i] * src_i` through direct
target-bra/operator/source-ket contractions.  The operator-weighted form is
the compression surface used by GlobalKrylov/GSE-style algorithms and does
not materialize a bond-expanded exact TTNO application.
"""
function fit!(φ::TTNS, ψ::TTNS; nsweeps::Int=4, tol::Float64=1e-10,
              normalize::Bool=false, verbose::Bool=false)
    return fit!(φ, (ψ,); nsweeps, tol, normalize, verbose)
end

function fit!(φ::TTNS, sources; Hs=nothing, coeffs=nothing,
              nsweeps::Int=4, tol::Float64=1e-10,
              normalize::Bool=false, verbose::Bool=false)
    srcs, ops = _fit_sources(φ, sources, Hs)
    coeffv = _fit_coeffs(φ, srcs, ops, coeffs)
    nsweeps >= 0 || throw(ArgumentError("fit!: nsweeps must be nonnegative"))
    target_center = center(φ)
    caches = [_FitCache(φ.topo, op) for op in ops]
    errors = Float64[]
    # Match the existing zero-sweep contract: validate arguments and return
    # without performing any target contraction when no ALS update was asked.
    target_norm = nsweeps == 0 ? nothing : _fit_target_norm(srcs, ops, coeffv)
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
        err = _fit_error(φ, srcs, ops, coeffv; target_norm)
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
    ops = if Hs === nothing
        fill(nothing, length(srcs0))
    else
        rawops = collect(Hs)
        length(rawops) == length(srcs0) ||
            throw(ArgumentError("fit!: Hs and sources must have the same length"))
        rawops
    end
    for (src, op) in zip(srcs0, ops)
        src isa TTNS || throw(ArgumentError("fit!: every source must be a TTNS"))
        φ.topo == src.topo || throw(ArgumentError("fit!: source and target topologies differ"))
        φ.hasphys == src.hasphys || throw(ArgumentError("fit!: physical-leg layout mismatch"))
        spacetype(φ) == spacetype(src) || throw(ArgumentError("fit!: source and target spacetype mismatch"))
        op === nothing && continue
        op isa TTNO || throw(ArgumentError("fit!: every H must be a TTNO or nothing"))
        φ.topo == op.topo || throw(ArgumentError("fit!: operator and target topologies differ"))
        φ.hasphys == op.hasphys || throw(ArgumentError("fit!: operator/target physical-leg layout mismatch"))
        spacetype(φ) == spacetype(op) || throw(ArgumentError("fit!: operator and target spacetype mismatch"))
    end
    return srcs0, ops
end

_fit_action_eltype(src::TTNS, ::Nothing) = eltype(src)
_fit_action_eltype(src::TTNS, op::TTNO) = promote_type(eltype(src), eltype(op))

function _fit_coeffs(φ::TTNS, sources, ops, coeffs)
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
        any(_fit_action_eltype(src, op) <: Complex for (src, op) in zip(sources, ops)) &&
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
    operator::Union{Nothing,TTNO}
    envs::Dict{Tuple{Int,Int},AbstractTensorMap}
    plans::Dict{Planning.PlanKey,Planning.ContractionPlan}
    rootcaps::Dict{Tuple,AbstractTensorMap}
end
_FitCache(topo::TreeTopology, operator::Union{Nothing,TTNO}=nothing) =
    _FitCache(topo, operator, Dict{Tuple{Int,Int},AbstractTensorMap}(),
              Dict{Planning.PlanKey,Planning.ContractionPlan}(),
              Dict{Tuple,AbstractTensorMap}())
_FitCache(topo::TreeTopology, envs::Dict{Tuple{Int,Int},AbstractTensorMap}) =
    _FitCache(topo, nothing, envs, Dict{Planning.PlanKey,Planning.ContractionPlan}(),
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
_fit_scalar_type(φ::TTNS, ψ::TTNS, O::TTNO) =
    promote_type(eltype(φ), eltype(ψ), eltype(O))
_fit_scalar_type(c::_FitCache, φ::TTNS, ψ::TTNS) =
    c.operator === nothing ? _fit_scalar_type(φ, ψ) :
                          _fit_scalar_type(φ, ψ, c.operator)

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
    twists = _euclidean_output_legs(φ, n)
    isempty(twists) || twist!(A, twists)
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
    B = _euclidean_bra_tensor(φ, nothing, u)
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
    B = _euclidean_bra_tensor(φ, nothing, u)
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
    c.operator === nothing ||
        return _fit_operator_build_env(c, φ, ψ, c.operator, u, v)
    spec, operands = _fit_build_env_spec(c, φ, ψ, u, v)
    return _fit_planned_execute!(c, :fit_env, spec, operands,
                                 _fit_scalar_type(c, φ, ψ))
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
    y = repartition(ncon(tensors, indices, conjs), numout(B), 1)
    twists = _euclidean_output_legs(φ, n)
    isempty(twists) || twist!(y, twists)
    return y
end

"""Lower one local fit projection into a complete-operand planned spec."""
function _fit_project_spec(c::_FitCache, φ::TTNS, ψ::TTNS, n::Int)
    c.operator === nothing ||
        return _fit_operator_project_spec(c, φ, ψ, c.operator, n)
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
    kind = c.operator === nothing ? :fit_project : :fit_operator_project
    plan, _ = Planning.get_or_plan!(c.plans, kind, spec, operands,
                                    _fit_scalar_type(c, φ, ψ))
    return plan, operands
end

function _fit_project_tensor(c::_FitCache, φ::TTNS, ψ::TTNS, n::Int)
    plan, operands = _fit_project_plan(c, φ, ψ, n)
    y = Planning.execute(plan, operands)
    twists = _euclidean_output_legs(φ, n)
    isempty(twists) || twist!(y, twists)
    return y
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

_fit_opleg(t::TreeTopology, hasphys_u::Bool, u::Int, w::Int) =
    t.parent[u] == w ? nchildren(t, u) + (hasphys_u ? 2 : 0) + 1 : childslot(t, u, w)

"""
    _fit_operator_build_env_ncon_reference(φ, ψ, O, u, v, envs)

Retained direct target-bra/operator/source-ket environment reference.  Its
directed edge legs are ordered `(source ket, operator, target bra)`.
"""
function _fit_operator_build_env_ncon_reference(
    φ::TTNS, ψ::TTNS, O::TTNO, u::Int, v::Int,
    envs::Dict{Tuple{Int,Int},AbstractTensorMap},
)
    t = φ.topo
    A, W, B = ψ.tensors[u], O.tensors[u], φ.tensors[u]
    hp = hasphys(ψ, u)
    hasphys(O, u) == hp ||
        throw(ArgumentError("fit operator/TTNS physical-leg mismatch at node $(nodeid(t, u))"))
    aidx = zeros(Int, numind(A))
    widx = zeros(Int, numind(W))
    bidx = zeros(Int, numind(B))
    tensors = Any[A]
    indices = Vector{Int}[aidx]
    conjs = Bool[false]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    if hp
        K = nchildren(t, u)
        pin, pout = fresh(), fresh()
        aidx[physleg(ψ, u)] = pin
        widx[K + 2] = pin
        widx[K + 1] = pout
        bidx[physleg(φ, u)] = pout
    end
    for w in neighbors(t, u)
        la = _fit_stateleg(t, hp, u, w)
        lw = _fit_opleg(t, hp, u, w)
        if w == v
            aidx[la] = -1
            widx[lw] = -2
            bidx[la] = -3
        else
            E = envs[(w, u)]
            eidx = [fresh(), fresh(), fresh()]
            aidx[la] = eidx[1]
            widx[lw] = eidx[2]
            bidx[la] = eidx[3]
            push!(tensors, E); push!(indices, eidx); push!(conjs, false)
        end
    end
    if t.parent[u] == 0
        ka, ko, kb = fresh(), fresh(), fresh()
        aidx[end] = ka
        widx[end] = ko
        bidx[end] = kb
        T = _fit_scalar_type(φ, ψ, O)
        cap = ones_tensor(T, dual(domain(B)[1]) ⊗
                              domain(W)[numin(W)] ⊗ domain(A)[1])
        push!(tensors, cap); push!(indices, [kb, ko, ka]); push!(conjs, false)
    end
    push!(tensors, W); push!(indices, widx); push!(conjs, false)
    push!(tensors, B); push!(indices, bidx); push!(conjs, true)
    return ncon(tensors, indices, conjs)
end

"""Lower a directed target-bra/operator/source-ket environment to a plan."""
function _fit_operator_build_env_spec(c::_FitCache, φ::TTNS, ψ::TTNS,
                                      O::TTNO, u::Int, v::Int)
    t = φ.topo
    A, W, B = ψ.tensors[u], O.tensors[u], φ.tensors[u]
    hp = hasphys(ψ, u)
    hasphys(O, u) == hp ||
        throw(ArgumentError("fit operator/TTNS physical-leg mismatch at node $(nodeid(t, u))"))
    aidx = zeros(Int, numind(A))
    widx = zeros(Int, numind(W))
    bidx = zeros(Int, numind(B))
    operands = Any[A]
    labels = Vector{Int}[aidx]
    conjs = Bool[false]
    envslots = Int[]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    if hp
        K = nchildren(t, u)
        pin, pout = fresh(), fresh()
        aidx[physleg(ψ, u)] = pin
        widx[K + 2] = pin
        widx[K + 1] = pout
        bidx[physleg(φ, u)] = pout
    end
    for w in neighbors(t, u)
        la = _fit_stateleg(t, hp, u, w)
        lw = _fit_opleg(t, hp, u, w)
        if w == v
            aidx[la] = -1
            widx[lw] = -2
            bidx[la] = -3
        else
            E = c.envs[(w, u)]
            eidx = [fresh(), fresh(), fresh()]
            aidx[la] = eidx[1]
            widx[lw] = eidx[2]
            bidx[la] = eidx[3]
            push!(operands, E); push!(labels, eidx); push!(conjs, false)
            push!(envslots, length(labels))
        end
    end
    caps = Int[]
    if t.parent[u] == 0
        ka, ko, kb = fresh(), fresh(), fresh()
        aidx[end] = ka
        widx[end] = ko
        bidx[end] = kb
        T = _fit_scalar_type(c, φ, ψ)
        cap = _fit_root_cap!(c, T, dual(domain(B)[1]) ⊗
                                  domain(W)[numin(W)] ⊗ domain(A)[1])
        push!(operands, cap); push!(labels, [kb, ko, ka]); push!(conjs, false)
        push!(caps, length(labels))
    end
    push!(operands, W); push!(labels, widx); push!(conjs, false)
    wslot = length(labels)
    push!(operands, B); push!(labels, bidx); push!(conjs, true)
    preferred = Int[1]
    append!(preferred, envslots)
    push!(preferred, wslot)
    append!(preferred, caps)
    push!(preferred, length(labels))
    spec = Planning.ContractionSpec(labels, conjs, 3, (3, 0), nothing;
                                    preferred_slots=preferred)
    return spec, Tuple(operands)
end

function _fit_operator_build_env(c::_FitCache, φ::TTNS, ψ::TTNS,
                                 O::TTNO, u::Int, v::Int)
    spec, operands = _fit_operator_build_env_spec(c, φ, ψ, O, u, v)
    return _fit_planned_execute!(c, :fit_operator_env, spec, operands,
                                 _fit_scalar_type(c, φ, ψ))
end

"""
    _fit_operator_project_tensor_ncon_reference(φ, ψ, O, n, envs)

Direct reference for the operator-aware local projection into `φ`'s tensor
space.  It intentionally does not materialize `apply(O, ψ)`.
"""
function _fit_operator_project_tensor_ncon_reference(
    φ::TTNS, ψ::TTNS, O::TTNO, n::Int,
    envs::Dict{Tuple{Int,Int},AbstractTensorMap},
)
    t = φ.topo
    A, W, B = ψ.tensors[n], O.tensors[n], φ.tensors[n]
    hp = hasphys(ψ, n)
    hasphys(O, n) == hp ||
        throw(ArgumentError("fit operator/TTNS physical-leg mismatch at node $(nodeid(t, n))"))
    aidx = zeros(Int, numind(A))
    widx = zeros(Int, numind(W))
    tensors = Any[A]
    indices = Vector{Int}[aidx]
    conjs = Bool[false]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    if hp
        K = nchildren(t, n)
        pin = fresh()
        aidx[physleg(ψ, n)] = pin
        widx[K + 2] = pin
        widx[K + 1] = -(K + 1)
    end
    for w in neighbors(t, n)
        la = _fit_stateleg(t, hp, n, w)
        lw = _fit_opleg(t, hp, n, w)
        E = envs[(w, n)]
        eidx = [fresh(), fresh(), -la]
        aidx[la] = eidx[1]
        widx[lw] = eidx[2]
        push!(tensors, E); push!(indices, eidx); push!(conjs, false)
    end
    if t.parent[n] == 0
        ka, ko = fresh(), fresh()
        aidx[end] = ka
        widx[end] = ko
        T = _fit_scalar_type(φ, ψ, O)
        cap = ones_tensor(T, dual(domain(B)[1]) ⊗
                              domain(W)[numin(W)] ⊗ domain(A)[1])
        push!(tensors, cap); push!(indices, [-numind(B), ko, ka]); push!(conjs, false)
    end
    push!(tensors, W); push!(indices, widx); push!(conjs, false)
    return repartition(ncon(tensors, indices, conjs), numout(B), 1)
end

"""Lower an operator-aware local projection to a complete-operand plan."""
function _fit_operator_project_spec(c::_FitCache, φ::TTNS, ψ::TTNS,
                                    O::TTNO, n::Int)
    t = φ.topo
    A, W, B = ψ.tensors[n], O.tensors[n], φ.tensors[n]
    hp = hasphys(ψ, n)
    hasphys(O, n) == hp ||
        throw(ArgumentError("fit operator/TTNS physical-leg mismatch at node $(nodeid(t, n))"))
    aidx = zeros(Int, numind(A))
    widx = zeros(Int, numind(W))
    operands = Any[A]
    labels = Vector{Int}[aidx]
    conjs = Bool[false]
    envslots = Int[]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    if hp
        K = nchildren(t, n)
        pin = fresh()
        aidx[physleg(ψ, n)] = pin
        widx[K + 2] = pin
        widx[K + 1] = -(K + 1)
    end
    for w in neighbors(t, n)
        la = _fit_stateleg(t, hp, n, w)
        lw = _fit_opleg(t, hp, n, w)
        E = c.envs[(w, n)]
        eidx = [fresh(), fresh(), -la]
        aidx[la] = eidx[1]
        widx[lw] = eidx[2]
        push!(operands, E); push!(labels, eidx); push!(conjs, false)
        push!(envslots, length(labels))
    end
    caps = Int[]
    if t.parent[n] == 0
        ka, ko = fresh(), fresh()
        aidx[end] = ka
        widx[end] = ko
        T = _fit_scalar_type(c, φ, ψ)
        cap = _fit_root_cap!(c, T, dual(domain(B)[1]) ⊗
                                  domain(W)[numin(W)] ⊗ domain(A)[1])
        push!(operands, cap); push!(labels, [-numind(B), ko, ka]); push!(conjs, false)
        push!(caps, length(labels))
    end
    push!(operands, W); push!(labels, widx); push!(conjs, false)
    wslot = length(labels)
    preferred = Int[1]
    append!(preferred, envslots)
    push!(preferred, wslot)
    append!(preferred, caps)
    spec = Planning.ContractionSpec(labels, conjs, numind(B), (numout(B), 1), nothing;
                                    preferred_slots=preferred)
    return spec, Tuple(operands)
end

"""
    _fit_operator_scalar_ncon_reference(φ, ψ, O, u, envs)

Direct scalar closure for `⟨φ|O|ψ⟩`, retained for A/B testing.
"""
function _fit_operator_scalar_ncon_reference(
    φ::TTNS, ψ::TTNS, O::TTNO, u::Int,
    envs::Dict{Tuple{Int,Int},AbstractTensorMap},
)
    t = φ.topo
    A, W, B = ψ.tensors[u], O.tensors[u], φ.tensors[u]
    hp = hasphys(ψ, u)
    hasphys(O, u) == hp ||
        throw(ArgumentError("fit operator/TTNS physical-leg mismatch at node $(nodeid(t, u))"))
    aidx = zeros(Int, numind(A))
    widx = zeros(Int, numind(W))
    bidx = zeros(Int, numind(B))
    tensors = Any[A]
    indices = Vector{Int}[aidx]
    conjs = Bool[false]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])
    if hp
        K = nchildren(t, u)
        pin, pout = fresh(), fresh()
        aidx[physleg(ψ, u)] = pin
        widx[K + 2] = pin
        widx[K + 1] = pout
        bidx[physleg(φ, u)] = pout
    end
    for w in neighbors(t, u)
        la = _fit_stateleg(t, hp, u, w)
        lw = _fit_opleg(t, hp, u, w)
        E = envs[(w, u)]
        eidx = [fresh(), fresh(), fresh()]
        aidx[la] = eidx[1]
        widx[lw] = eidx[2]
        bidx[la] = eidx[3]
        push!(tensors, E); push!(indices, eidx); push!(conjs, false)
    end
    if t.parent[u] == 0
        ka, ko, kb = fresh(), fresh(), fresh()
        aidx[end] = ka
        widx[end] = ko
        bidx[end] = kb
        T = _fit_scalar_type(φ, ψ, O)
        cap = ones_tensor(T, dual(domain(B)[1]) ⊗
                              domain(W)[numin(W)] ⊗ domain(A)[1])
        push!(tensors, cap); push!(indices, [kb, ko, ka]); push!(conjs, false)
    end
    push!(tensors, W); push!(indices, widx); push!(conjs, false)
    push!(tensors, B); push!(indices, bidx); push!(conjs, true)
    return ncon(tensors, indices, conjs)
end

"""Lower the fully contracted `⟨φ|O|ψ⟩` closure to a plan."""
function _fit_operator_scalar_spec(c::_FitCache, φ::TTNS, ψ::TTNS,
                                   O::TTNO, u::Int)
    t = φ.topo
    A, W, B = ψ.tensors[u], O.tensors[u], φ.tensors[u]
    hp = hasphys(ψ, u)
    hasphys(O, u) == hp ||
        throw(ArgumentError("fit operator/TTNS physical-leg mismatch at node $(nodeid(t, u))"))
    aidx = zeros(Int, numind(A))
    widx = zeros(Int, numind(W))
    bidx = zeros(Int, numind(B))
    operands = Any[A]
    labels = Vector{Int}[aidx]
    conjs = Bool[false]
    envslots = Int[]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])
    if hp
        K = nchildren(t, u)
        pin, pout = fresh(), fresh()
        aidx[physleg(ψ, u)] = pin
        widx[K + 2] = pin
        widx[K + 1] = pout
        bidx[physleg(φ, u)] = pout
    end
    for w in neighbors(t, u)
        la = _fit_stateleg(t, hp, u, w)
        lw = _fit_opleg(t, hp, u, w)
        E = c.envs[(w, u)]
        eidx = [fresh(), fresh(), fresh()]
        aidx[la] = eidx[1]
        widx[lw] = eidx[2]
        bidx[la] = eidx[3]
        push!(operands, E); push!(labels, eidx); push!(conjs, false)
        push!(envslots, length(labels))
    end
    caps = Int[]
    if t.parent[u] == 0
        ka, ko, kb = fresh(), fresh(), fresh()
        aidx[end] = ka
        widx[end] = ko
        bidx[end] = kb
        T = _fit_scalar_type(c, φ, ψ)
        cap = _fit_root_cap!(c, T, dual(domain(B)[1]) ⊗
                                  domain(W)[numin(W)] ⊗ domain(A)[1])
        push!(operands, cap); push!(labels, [kb, ko, ka]); push!(conjs, false)
        push!(caps, length(labels))
    end
    push!(operands, W); push!(labels, widx); push!(conjs, false)
    wslot = length(labels)
    push!(operands, B); push!(labels, bidx); push!(conjs, true)
    preferred = Int[1]
    append!(preferred, envslots)
    push!(preferred, wslot)
    append!(preferred, caps)
    push!(preferred, length(labels))
    spec = Planning.ContractionSpec(labels, conjs, 0, (0, 0), nothing;
                                    preferred_slots=preferred)
    return spec, Tuple(operands)
end

function _fit_operator_overlap(φ::TTNS, O::TTNO, ψ::TTNS)
    φ.topo == ψ.topo == O.topo ||
        throw(ArgumentError("fit operator overlap: topologies differ"))
    c = _FitCache(φ.topo, O)
    r = φ.topo.root
    for w in neighbors(φ.topo, r)
        _fit_env!(c, φ, ψ, w, r)
    end
    return _fit_scalar(c, φ, ψ, r)
end

"""
Shape/value cache for the direct rank-four contraction
`⟨left|leftop† * rightop|right⟩`.  Source states and operators are immutable
through a fit sweep, but plans and root caps are still owned locally so no
mutable workspace is shared with another solver task.
"""
struct _FitDoubleCache
    topo::TreeTopology
    leftop::TTNO
    rightop::TTNO
    envs::Dict{Tuple{Int,Int},AbstractTensorMap}
    plans::Dict{Planning.PlanKey,Planning.ContractionPlan}
    rootcaps::Dict{Tuple,AbstractTensorMap}
end
_FitDoubleCache(topo::TreeTopology, leftop::TTNO, rightop::TTNO) =
    _FitDoubleCache(topo, leftop, rightop,
                    Dict{Tuple{Int,Int},AbstractTensorMap}(),
                    Dict{Planning.PlanKey,Planning.ContractionPlan}(),
                    Dict{Tuple,AbstractTensorMap}())

_fit_double_scalar_type(left::TTNS, leftop::TTNO,
                        right::TTNS, rightop::TTNO) =
    promote_type(eltype(left), eltype(leftop), eltype(right), eltype(rightop))

function _fit_double_root_cap!(c::_FitDoubleCache, T::DataType, capspace)
    return get!(c.rootcaps, (T, capspace)) do
        ones_tensor(T, capspace)
    end
end

function _fit_double_execute!(c::_FitDoubleCache, kind::Symbol,
                              spec::Planning.ContractionSpec,
                              operands::Tuple, T::DataType)
    plan, _ = Planning.get_or_plan!(c.plans, kind, spec, operands, T)
    return Planning.execute(plan, operands)
end

"""
    _fit_double_spec(cache, left, right, u, v) -> (spec, operands)

Lower `⟨left|leftop† * rightop|right⟩` at node `u`.  A directed edge keeps
the four legs ordered `(right state, right operator, left operator, left
state)`; `v == 0` closes all legs for the scalar overlap.
"""
function _fit_double_spec(c::_FitDoubleCache, left::TTNS, right::TTNS,
                          u::Int, v::Int)
    t = c.topo
    AR, WR = right.tensors[u], c.rightop.tensors[u]
    WL, AL = c.leftop.tensors[u], left.tensors[u]
    hp = hasphys(right, u)
    hasphys(left, u) == hp == hasphys(c.rightop, u) == hasphys(c.leftop, u) ||
        throw(ArgumentError("fit double overlap: physical-leg layout mismatch at node $(nodeid(t, u))"))
    ridx = zeros(Int, numind(AR))
    wridx = zeros(Int, numind(WR))
    wlidx = zeros(Int, numind(WL))
    lidx = zeros(Int, numind(AL))
    operands = Any[AR]
    labels = Vector{Int}[ridx]
    conjs = Bool[false]
    envslots = Int[]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    if hp
        K = nchildren(t, u)
        qr, qmid, ql = fresh(), fresh(), fresh()
        ridx[physleg(right, u)] = qr
        wridx[K + 2] = qr
        wridx[K + 1] = qmid
        wlidx[K + 1] = qmid
        wlidx[K + 2] = ql
        lidx[physleg(left, u)] = ql
    end
    for w in neighbors(t, u)
        lr = _fit_stateleg(t, hp, u, w)
        lo = _fit_opleg(t, hp, u, w)
        if w == v
            ridx[lr] = -1
            wridx[lo] = -2
            wlidx[lo] = -3
            lidx[lr] = -4
        else
            E = c.envs[(w, u)]
            eidx = [fresh(), fresh(), fresh(), fresh()]
            ridx[lr] = eidx[1]
            wridx[lo] = eidx[2]
            wlidx[lo] = eidx[3]
            lidx[lr] = eidx[4]
            push!(operands, E); push!(labels, eidx); push!(conjs, false)
            push!(envslots, length(labels))
        end
    end
    caps = Int[]
    if t.parent[u] == 0
        kr, kor, kol, kl = fresh(), fresh(), fresh(), fresh()
        ridx[end] = kr
        wridx[end] = kor
        wlidx[end] = kol
        lidx[end] = kl
        T = _fit_double_scalar_type(left, c.leftop, right, c.rightop)
        capspace = dual(domain(AL)[1]) ⊗ dual(domain(WL)[numin(WL)]) ⊗
                   domain(WR)[numin(WR)] ⊗ domain(AR)[1]
        cap = _fit_double_root_cap!(c, T, capspace)
        push!(operands, cap); push!(labels, [kl, kol, kor, kr]); push!(conjs, false)
        push!(caps, length(labels))
    end
    push!(operands, WR); push!(labels, wridx); push!(conjs, false)
    wrslot = length(labels)
    push!(operands, WL); push!(labels, wlidx); push!(conjs, true)
    wlslot = length(labels)
    push!(operands, AL); push!(labels, lidx); push!(conjs, true)
    preferred = Int[1]
    append!(preferred, envslots)
    push!(preferred, wrslot, wlslot)
    append!(preferred, caps)
    push!(preferred, length(labels))
    nopen = v == 0 ? 0 : 4
    spec = Planning.ContractionSpec(labels, conjs, nopen,
                                    nopen == 0 ? (0, 0) : (4, 0), nothing;
                                    preferred_slots=preferred)
    return spec, Tuple(operands)
end

"""Retained direct `ncon` reference for the rank-four overlap network."""
function _fit_double_ncon_reference(c::_FitDoubleCache, left::TTNS,
                                    right::TTNS, u::Int, v::Int)
    spec, operands = _fit_double_spec(c, left, right, u, v)
    return Planning.ncon_reference(spec, operands)
end

function _fit_double_env!(c::_FitDoubleCache, left::TTNS, right::TTNS,
                          u::Int, v::Int)
    return get!(c.envs, (u, v)) do
        for w in neighbors(c.topo, u)
            w == v && continue
            _fit_double_env!(c, left, right, w, u)
        end
        spec, operands = _fit_double_spec(c, left, right, u, v)
        _fit_double_execute!(c, :fit_double_env, spec, operands,
                             _fit_double_scalar_type(left, c.leftop,
                                                     right, c.rightop))
    end
end

function _fit_double_overlap(left::TTNS, leftop::TTNO,
                             right::TTNS, rightop::TTNO)
    left.topo == right.topo == leftop.topo == rightop.topo ||
        throw(ArgumentError("fit double overlap: topologies differ"))
    c = _FitDoubleCache(left.topo, leftop, rightop)
    r = left.topo.root
    for w in neighbors(c.topo, r)
        _fit_double_env!(c, left, right, w, r)
    end
    spec, operands = _fit_double_spec(c, left, right, r, 0)
    return _fit_double_execute!(c, :fit_double_scalar, spec, operands,
                                 _fit_double_scalar_type(left, leftop,
                                                         right, rightop))
end

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
    B = _euclidean_bra_tensor(φ, nothing, u)
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
    c.operator === nothing ||
        return _fit_operator_scalar_spec(c, φ, ψ, c.operator, u)
    t = φ.topo
    A = ψ.tensors[u]
    B = _euclidean_bra_tensor(φ, nothing, u)
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
    kind = c.operator === nothing ? :fit_scalar : :fit_operator_scalar
    return _fit_planned_execute!(c, kind, spec, operands,
                                 _fit_scalar_type(c, φ, ψ))
end

function _fit_error(φ::TTNS, ψ::TTNS)
    return _fit_error(φ, (ψ,), (nothing,), (one(eltype(φ)),))
end

function _fit_error(φ::TTNS, sources, coeffs)
    return _fit_error(φ, sources, fill(nothing, length(sources)), coeffs)
end

"""
    _fit_action_overlap(left, leftop, right, rightop)

Exact overlap of two unmaterialized fit targets, with `leftop`/`rightop`
optionally absent.  In the two-operator case it contracts the direct
rank-four double layer, retaining non-Hermitian semantics rather than assuming
`H†H == H^2`.
"""
function _fit_action_overlap(left::TTNS, leftop::Union{Nothing,TTNO},
                             right::TTNS, rightop::Union{Nothing,TTNO})
    if leftop === nothing
        return rightop === nothing ? _fit_overlap(left, right) :
                                     _fit_operator_overlap(left, rightop, right)
    elseif rightop === nothing
        return conj(_fit_operator_overlap(right, leftop, left))
    end
    return _fit_double_overlap(left, leftop, right, rightop)
end

"""Exact norm of `sum_i coeffs[i] * ops[i] * sources[i]` without `apply`."""
function _fit_target_norm(sources, ops, coeffs)
    length(sources) == length(ops) == length(coeffs) ||
        throw(ArgumentError("fit target norm: source/operator/coefficient length mismatch"))
    total = nothing
    for i in eachindex(sources), j in eachindex(sources)
        term = conj(coeffs[i]) * coeffs[j] *
               _fit_action_overlap(sources[i], ops[i], sources[j], ops[j])
        total = total === nothing ? term : total + term
    end
    return total
end

function _fit_error(φ::TTNS, sources, ops, coeffs; target_norm=nothing)
    length(sources) == length(ops) == length(coeffs) ||
        throw(ArgumentError("fit error: source/operator/coefficient length mismatch"))
    nφ = _fit_overlap(φ, φ)
    ntarget = target_norm === nothing ? _fit_target_norm(sources, ops, coeffs) : target_norm
    cross = zero(promote_type(typeof(nφ), typeof(ntarget), eltype(coeffs)))
    for i in eachindex(sources)
        cross += coeffs[i] * _fit_action_overlap(φ, nothing, sources[i], ops[i])
    end
    return sqrt(max(real(nφ + ntarget - 2 * real(cross)), 0))
end
