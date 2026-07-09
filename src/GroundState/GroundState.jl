"""
L5a — Ground-state kernels (architecture §5a). PyTreeNet: dmrg/dmrg.py.

Implemented: `dmrg1!` (single-site, benchmark/warm-up), `dmrg2!` (two-site,
small systems & hard initializations), and `dmrg1_3s!` (single-site sweeps plus
the shared `expand!` primitive for 3S-style bond growth). CBE is deliberately
*not* planned for ground states (§5a: RSVD post-expansion supersedes it for the
long-range/star-geometry Hamiltonians we target) — the CBE code lives in
Evolution (`TDVP1_CBE`) where the local PyTreeNet fork provides the reference.

DMRG requires a hermitian TTNO — enforced via the `ishermitian` trait (§9.8).
"""
module GroundState

using KrylovKit: eigsolve
using ..Backend
using ..Trees
using ..Networks
using ..Contractions

export dmrg1!, dmrg2!, dmrg1_3s!, expand!

"""
    expand!(ψ, H, edge; scheme=:exact, cache=nothing, rng=nothing,
            trunc, max_add=8, mixing=1, enr_rtol=1e-10, enr_atol=1e-12) -> ψ

Shared bond-expansion primitive (§5a/§11.7). `edge` is `(child, parent)` or
`child => parent` using node ids or indices. The current implementation uses a
deterministic exact two-site predictor and orthogonal-complement selection.
TODO(M0): add `scheme=:rsvd` blockwise randomized probes with explicit `rng`;
until then `:rsvd` is intentionally rejected rather than silently using global
randomness or a different algorithm.
"""
function expand!(ψ::TTNS, H::TTNO, edge; scheme::Symbol=:exact,
                 cache::Union{Nothing,EnvCache}=nothing, rng=nothing,
                 trunc::TruncationScheme=TruncationScheme(; maxdim=100),
                 max_add::Int=8, mixing::Number=one(Float64),
                 enr_rtol::Float64=1e-10, enr_atol::Float64=1e-12)
    scheme === :exact ||
        throw(ArgumentError("expand!: only deterministic scheme=:exact is implemented; TODO(M0) add scheme=:rsvd with explicit rng"))
    max_add >= 0 || throw(ArgumentError("expand!: max_add must be nonnegative"))
    iszero(mixing) && return ψ
    t = ψ.topo
    n, m = _edge_child_parent(t, edge)
    olddim = dim(virtualspace(ψ, n))
    cap = min(trunc.maxdim, olddim + max_add)
    cap > olddim || return ψ

    c = cache === nothing ? EnvCache(t) : cache
    move_center!(ψ, n; cache=c)
    Θ = two_site_tensor(ψ, n, m)
    h2 = eff_h2(c, ψ, H, n, m)
    PΘ = mixing * h2(Θ)
    P = _child_predictor_basis(ψ, PΘ, n, cap)
    U, R = _expand_enrich_split(ψ.tensors[n], P; maxdim=cap,
                                max_add=cap - olddim,
                                enr_rtol, enr_atol)
    dim(domain(U)) == olddim && return ψ
    ψ.tensors[n] = U
    ψ.tensors[m] = absorb_on_leg(ψ.tensors[m], R, childslot(t, m, n))
    ψ.center = m
    invalidate_edge!(c, n, m)
    return ψ
end

"""
    dmrg1!(ψ, H; nsweeps=10, tol=1e-10, krylovdim=20, verbose=false) -> (ψ, energies)

Single-site DMRG: post-order + reverse sweeps, local Lanczos ground state of
the one-site effective Hamiltonian at every node. Bond dimensions are fixed —
start from a state with the target bond spaces (or wait for `dmrg1_3s!`).
Returns the per-half-sweep energy trace; stops early when the energy change
drops below `tol`.
"""
function dmrg1!(ψ::TTNS, H::TTNO; nsweeps::Int=10, tol::Float64=1e-10,
                krylovdim::Int=20, verbose::Bool=false)
    ishermitian(H) || throw(ArgumentError("dmrg1!: DMRG requires ishermitian(H) == true (§9.8)"))
    cache = EnvCache(ψ.topo)
    energies = Float64[]
    order = postorder(ψ.topo)
    for sweep in 1:nsweeps
        E = NaN
        for n in Iterators.flatten((order, Iterators.reverse(order)))
            move_center!(ψ, n; cache)
            h1 = eff_h1(cache, ψ, H, n)
            vals, vecs, _ = eigsolve(h1, ψ.tensors[n], 1, :SR;
                                     ishermitian=true, krylovdim)
            E = real(vals[1])
            update_tensor!(ψ, n, vecs[1]; caches=(cache,))
        end
        push!(energies, E)
        verbose && @info "dmrg1! sweep $sweep" E
        length(energies) > 1 && abs(energies[end] - energies[end - 1]) < tol && break
    end
    return ψ, energies
end

"""
    dmrg2!(ψ, H; trunc, nsweeps=10, tol=1e-10, krylovdim=20, verbose=false) -> (ψ, energies)

Two-site DMRG: sweeps every edge (post-order and reverse), Lanczos on the
bond's two-site block, truncated split through `TruncationScheme` (§9.5).
Grows bond dimensions up to `trunc.maxdim`.
"""
function dmrg2!(ψ::TTNS, H::TTNO; trunc::TruncationScheme=TruncationScheme(),
                nsweeps::Int=10, tol::Float64=1e-10, krylovdim::Int=20,
                verbose::Bool=false)
    ishermitian(H) || throw(ArgumentError("dmrg2!: DMRG requires ishermitian(H) == true (§9.8)"))
    t = ψ.topo
    cache = EnvCache(t)
    energies = Float64[]
    bonds = [n for n in postorder(t) if t.parent[n] != 0]   # edge ≡ its child node
    for sweep in 1:nsweeps
        E = NaN
        for (n, center_on) in Iterators.flatten(
                (((n, :m) for n in bonds), ((n, :n) for n in Iterators.reverse(bonds))))
            m = t.parent[n]
            move_center!(ψ, n; cache)
            Θ = two_site_tensor(ψ, n, m)
            h2 = eff_h2(cache, ψ, H, n, m)
            vals, vecs, _ = eigsolve(h2, Θ, 1, :SR; ishermitian=true, krylovdim)
            E = real(vals[1])
            invalidate_edge!(cache, n, m)
            split_two_site!(ψ, vecs[1], n, m; trunc, center_on)
        end
        push!(energies, E)
        verbose && @info "dmrg2! sweep $sweep" E
        length(energies) > 1 && abs(energies[end] - energies[end - 1]) < tol && break
    end
    return ψ, energies
end

"""
    dmrg1_3s!(ψ, H; trunc, nsweeps=10, mixing=1.0, max_add=8, kwargs...)
        -> (ψ, energies)

Single-site DMRG with 3S-style subspace expansion between sweeps. The local
optimization is `dmrg1!`'s one-site Lanczos update; the bond growth step is the
shared [`expand!`](@ref) primitive and therefore uses `TruncationScheme` as the
single truncation entry point. `mixing` may be a number, vector, or function
`sweep -> α`; `α == 0` skips expansion for that sweep.
"""
function dmrg1_3s!(ψ::TTNS, H::TTNO; trunc::TruncationScheme=TruncationScheme(; maxdim=100),
                   nsweeps::Int=10, tol::Float64=1e-10, krylovdim::Int=20,
                   mixing=1.0, max_add::Int=8, expand_scheme::Symbol=:exact,
                   enr_rtol::Float64=1e-10, enr_atol::Float64=1e-12,
                   verbose::Bool=false)
    ishermitian(H) || throw(ArgumentError("dmrg1_3s!: DMRG requires ishermitian(H) == true (§9.8)"))
    t = ψ.topo
    cache = EnvCache(t)
    energies = Float64[]
    order = postorder(t)
    bonds = [n for n in order if t.parent[n] != 0]
    for sweep in 1:nsweeps
        E = NaN
        for n in Iterators.flatten((order, Iterators.reverse(order)))
            move_center!(ψ, n; cache)
            h1 = eff_h1(cache, ψ, H, n)
            vals, vecs, _ = eigsolve(h1, ψ.tensors[n], 1, :SR;
                                     ishermitian=true, krylovdim)
            E = real(vals[1])
            update_tensor!(ψ, n, vecs[1]; caches=(cache,))
        end
        α = _mixing_value(mixing, sweep)
        if !iszero(α)
            for n in bonds
                expand!(ψ, H, (n, t.parent[n]); scheme=expand_scheme, cache,
                        trunc, max_add, mixing=α, enr_rtol, enr_atol)
            end
        end
        push!(energies, E)
        verbose && @info "dmrg1_3s! sweep $sweep" E α maxbond=maximum(_bond_dims(ψ))
        length(energies) > 1 && abs(energies[end] - energies[end - 1]) < tol && break
    end
    return ψ, energies
end

# TODO: als!/lobpcg! ports (PyTreeNet dmrg/als.py, lobpcg.py) as alternative
#   local eigensolvers; als.py doubles as the starting point for `linsolve!`.
# Variational fitting lives in Networks.fit! (§11.6), where it can be shared by
# GK/GSE/METTS-style compression without creating a GroundState dependency path.

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

function _child_predictor_basis(ψ::TTNS, PΘ::AbstractTensorMap, n::Int, maxdim::Int)
    pn = numout(ψ.tensors[n])
    NP = numind(PΘ)
    Ps = permute(PΘ, (ntuple(identity, pn), ntuple(j -> pn + j, NP - pn)))
    U, _, _ = split_svd(Ps, TruncationScheme(; maxdim))
    return U
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

_mixing_value(α::Number, ::Int) = α
_mixing_value(αs::AbstractVector, sweep::Int) = αs[min(sweep, lastindex(αs))]
_mixing_value(f, sweep::Int) = f(sweep)

_bond_dims(ψ::TTNS) = [dim(virtualspace(ψ, n)) for n in 1:nnodes(ψ.topo) if ψ.topo.parent[n] != 0]

end # module GroundState
