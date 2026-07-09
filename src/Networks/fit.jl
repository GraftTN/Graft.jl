"""
    fit!(φ, ψ; nsweeps=4, tol=1e-10, normalize=false, verbose=false) -> (φ, errors)

Variationally fit `φ ≈ ψ` on the fixed TTNS manifold carried by `φ`.
This is the state-compression core of the architecture's public `fit!`
primitive (§3/§11.6). The source `ψ` is not mutated; `φ` is gauged and updated
in place. Because `φ` is canonicalized to the updated node, the one-site normal
matrix is the identity, so each ALS local solve is a direct projection of `ψ`
onto the current target environment.
"""
function fit!(φ::TTNS, ψ::TTNS; nsweeps::Int=4, tol::Float64=1e-10,
              normalize::Bool=false, verbose::Bool=false)
    φ.topo == ψ.topo || throw(ArgumentError("fit!: source and target topologies differ"))
    φ.hasphys == ψ.hasphys || throw(ArgumentError("fit!: physical-leg layout mismatch"))
    spacetype(φ) == spacetype(ψ) || throw(ArgumentError("fit!: source and target spacetype mismatch"))
    nsweeps >= 0 || throw(ArgumentError("fit!: nsweeps must be nonnegative"))
    target_center = center(φ)
    cache = _FitCache(φ.topo)
    errors = Float64[]
    order = postorder(φ.topo)
    for sweep in 1:nsweeps
        for n in Iterators.flatten((order, Iterators.reverse(order)))
            move_center!(φ, n)
            _invalidate_fit_node!(cache, n)
            A = _fit_local_tensor(cache, φ, ψ, n)
            update_tensor!(φ, n, A; gauge=true)
            _invalidate_fit_node!(cache, n)
        end
        normalize && normalize!(φ)
        err = _fit_error(φ, ψ)
        push!(errors, err)
        verbose && @info "fit! sweep $sweep" err
        length(errors) > 1 && abs(errors[end] - errors[end - 1]) < tol && break
    end
    move_center!(φ, target_center)
    return φ, errors
end

struct _FitCache
    topo::TreeTopology
    envs::Dict{Tuple{Int,Int},AbstractTensorMap}
end
_FitCache(topo::TreeTopology) = _FitCache(topo, Dict{Tuple{Int,Int},AbstractTensorMap}())

function _invalidate_fit_node!(c::_FitCache, n::Int)
    filter!(p -> !_fit_on_side(c.topo, n, p.first[1], p.first[2]), c.envs)
    return c
end

function _fit_on_side(t::TreeTopology, n::Int, u::Int, v::Int)
    n == u && return true
    n == v && return false
    return u in path_between(t, n, v)
end

function _fit_env!(c::_FitCache, φ::TTNS, ψ::TTNS, u::Int, v::Int)
    return get!(c.envs, (u, v)) do
        for w in neighbors(c.topo, u)
            w == v && continue
            _fit_env!(c, φ, ψ, w, u)
        end
        _fit_build_env(φ, ψ, u, v, c.envs)
    end
end

function _fit_local_tensor(c::_FitCache, φ::TTNS, ψ::TTNS, n::Int)
    t = φ.topo
    for w in neighbors(t, n)
        _fit_env!(c, φ, ψ, w, n)
    end
    return _fit_project_tensor(φ, ψ, n, c.envs)
end

function _fit_build_env(φ::TTNS, ψ::TTNS, u::Int, v::Int,
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

function _fit_project_tensor(φ::TTNS, ψ::TTNS, n::Int,
                             envs::Dict{Tuple{Int,Int},AbstractTensorMap})
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

_fit_stateleg(t::TreeTopology, hasphys_u::Bool, u::Int, w::Int) =
    t.parent[u] == w ? nchildren(t, u) + (hasphys_u ? 1 : 0) + 1 : childslot(t, u, w)

function _fit_overlap(φ::TTNS, ψ::TTNS)
    φ.topo == ψ.topo || throw(ArgumentError("fit overlap: topologies differ"))
    c = _FitCache(φ.topo)
    r = φ.topo.root
    for w in neighbors(φ.topo, r)
        _fit_env!(c, φ, ψ, w, r)
    end
    return _fit_scalar(φ, ψ, r, c.envs)
end

function _fit_scalar(φ::TTNS, ψ::TTNS, u::Int,
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

function _fit_error(φ::TTNS, ψ::TTNS)
    nφ = real(_fit_overlap(φ, φ))
    nψ = real(_fit_overlap(ψ, ψ))
    ov = _fit_overlap(φ, ψ)
    return sqrt(max(nφ + nψ - 2 * real(ov), 0))
end
