# TDVP family (PyTreeNet: time_evolution/tdvp.py, tdvp_algorithms/*).
#
# Tree projector-splitting integrator (Ceruti–Lubich–Walach; PyTreeNet's
# OneSiteTDVP). Sweep formulation used here, equivalent to the recursive
# integrator:
#   * update path = post-order (children before parents, root last);
#   * every node is forward-evolved exactly once per sweep;
#   * a link (edge) is backward-evolved exactly when the center crosses it
#     child→parent during the forward sweep (crossings *into* fresh subtrees
#     are plain gauge moves) — so every edge is backward-evolved exactly once;
#   * the reverse sweep mirrors this: backward evolution on parent→child
#     crossings.
# First order: one forward sweep with dz. Second order (symmetric): forward
# sweep with dz/2, reverse sweep with dz/2.
#
# Krylov exponentials via KrylovKit.exponentiate; the Lanczos path is taken
# only when the TTNO carries `ishermitian == true` (§9.8), Arnoldi otherwise —
# so nothing here assumes hermiticity or a purely imaginary step (§0.2).

"""
    TDVP1(; order=2, krylovdim=30, tol=1e-12)

Single-site TDVP evolver. Constant bond dimension (the tangent-space
projection never grows bonds) — pair with `TDVP1_CBE` when the state needs to
grow. An instance owns its `EnvCache` and is bound to one evolution run: all
mutations of `ψ` between its `step!` calls must go through
`update_tensor!`/`move_center!` with that cache, or the cache goes stale.
"""
Base.@kwdef mutable struct TDVP1 <: Evolver
    order::Int = 2
    krylovdim::Int = 30
    tol::Float64 = 1e-12
    cache::Union{Nothing,EnvCache} = nothing
end

function step!(ev::TDVP1, ψ::TTNS, H::TTNO, dz::Number)
    if ev.cache === nothing || ev.cache.topo != ψ.topo
        ev.cache = EnvCache(ψ.topo)
    end
    if ev.order == 1
        _tdvp1_sweep!(ev, ψ, H, dz; rev=false)
    elseif ev.order == 2
        _tdvp1_sweep!(ev, ψ, H, dz / 2; rev=false)
        _tdvp1_sweep!(ev, ψ, H, dz / 2; rev=true)
    else
        throw(ArgumentError("TDVP1: order must be 1 or 2"))
    end
    return ψ
end

function _tdvp1_sweep!(ev::TDVP1, ψ::TTNS, H::TTNO, dz::Number; rev::Bool)
    t = ψ.topo
    cache = ev.cache::EnvCache
    order = rev ? reverse(postorder(t)) : postorder(t)
    herm = ishermitian(H)
    move_center!(ψ, order[1]; cache)
    for i in eachindex(order)
        n = order[i]
        @assert ψ.center == n
        # forward-evolve the site
        h1 = eff_h1(cache, ψ, H, n)
        A, _ = exponentiate(h1, dz, ψ.tensors[n];
                            ishermitian=herm, krylovdim=ev.krylovdim, tol=ev.tol)
        update_tensor!(ψ, n, A; caches=(cache,))
        # walk to the next update site, backward-evolving the links that the
        # splitting assigns to this sweep direction
        i == lastindex(order) && break
        seg = path_between(t, n, order[i + 1])
        for j in 2:length(seg)
            u, v = seg[j - 1], seg[j]
            if (t.parent[u] == v) != rev       # child→parent in fwd, parent→child in rev
                _evolve_link_and_move!(ev, ψ, H, u, v, dz; herm)
            else
                move_center!(ψ, v; cache)      # plain gauge move, no link evolution
            end
        end
    end
    return ψ
end

# split the center tensor towards `v`, backward-evolve the link tensor with
# the zero-site Hamiltonian, absorb it into `v`
function _evolve_link_and_move!(ev::TDVP1, ψ::TTNS, H::TTNO, u::Int, v::Int,
                                dz::Number; herm::Bool)
    t = ψ.topo
    cache = ev.cache::EnvCache
    @assert ψ.center == u
    if t.parent[u] == v
        Q, C = left_orth(ψ.tensors[u])              # C :: V_new ← V_e
        ψ.tensors[u] = Q
        invalidate_node!(cache, u)
        k0 = eff_h0(cache, ψ, H, u, v)              # env(u→v) rebuilt from Q
        C, _ = exponentiate(k0, -dz, C;
                            ishermitian=herm, krylovdim=ev.krylovdim, tol=ev.tol)
        ψ.tensors[v] = absorb_on_leg(ψ.tensors[v], C, childslot(t, v, u))
    else
        k = childslot(t, u, v)
        Q, Cd = orth_factor_leg(ψ.tensors[u], k)    # Cd :: Y ← dual(V_e)
        ψ.tensors[u] = Q
        invalidate_node!(cache, u)
        # the edge is (v, u) with v the child; the link tensor in that edge's
        # (below ← above) orientation is C :: V_e ← dual(Y)
        C = transpose(Cd)
        k0 = eff_h0(cache, ψ, H, v, u)              # env(v→u) untouched, env(u→v) rebuilt from Q
        C, _ = exponentiate(k0, -dz, C;
                            ishermitian=herm, krylovdim=ev.krylovdim, tol=ev.tol)
        ψ.tensors[v] = ψ.tensors[v] * C
    end
    ψ.center = v
    invalidate_node!(cache, v)
    return ψ
end

# ---------------------------------------------------------------------------

"""
    TDVP2(; order=2, trunc=TruncationScheme(), krylovdim=30, tol=1e-12)

Two-site TDVP (benchmark kernel, §5b). TODO: sweep implementation pending the
PyTreeNet TwoSiteTDVP port — forward-evolve each bond's two-site block once
(post-order edge sweep, truncated split through `TruncationScheme`),
backward-evolve the single-site tensor between consecutive bonds.
"""
Base.@kwdef mutable struct TDVP2 <: Evolver
    order::Int = 2
    trunc::TruncationScheme = TruncationScheme()
    krylovdim::Int = 30
    tol::Float64 = 1e-12
    cache::Union{Nothing,EnvCache} = nothing
end

"""
    TDVP1_CBE(; trunc, expansion..., krylovdim, tol)

Single-site TDVP with controlled bond expansion (the local PyTreeNet fork's
1TDVP-CBE). TODO: port pending — before each site/link update the bond towards
the update direction is expanded with the truncated complement projection
(`expand!` primitive, §11.7: shared with 3S/GSE/LSE), then the sweep proceeds
as TDVP1 and the bond is truncated back through `TruncationScheme`.
"""
Base.@kwdef mutable struct TDVP1_CBE <: Evolver
    trunc::TruncationScheme = TruncationScheme()
    krylovdim::Int = 30
    tol::Float64 = 1e-12
    cache::Union{Nothing,EnvCache} = nothing
end
