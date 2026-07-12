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
    TDVP1(; order=2, krylovdim=30, tol=1e-12, verbose=true)

Single-site TDVP evolver. Constant bond dimension (the tangent-space
projection never grows bonds) — pair with `TDVP1_CBE` when the state needs to
grow. An instance owns its `EnvCache` and is bound to one evolution run: all
mutations of `ψ` between its `step!` calls must go through
`update_tensor!`/`move_center!` with that cache, or the cache goes stale. With
`verbose=true`, emits step and half-sweep `@info` records with topology,
solver, step size, direction, center, update counts, and bond statistics.
"""
Base.@kwdef mutable struct TDVP1 <: Evolver
    order::Int = 2
    krylovdim::Int = 30
    tol::Float64 = 1e-12
    verbose::Bool = true
    cache::Union{Nothing,EnvCache} = nothing
end

"""
    TDVP2(; order=2, trunc=TruncationScheme(), krylovdim=30, tol=1e-12,
          verbose=true)

Two-site TDVP (benchmark kernel, §5b; PyTreeNet twositetdvp.py +
secondordertwosite.py). Every bond's two-site block is forward-evolved once
per sweep (post-order edge sweep, truncated split through `TruncationScheme`),
with a single-site *backward* evolution at the connecting parent between
consecutive bond updates. Bond dimensions adapt up to `trunc.maxdim`. With
`verbose=true`, emits step and half-sweep `@info` records including the
truncation policy and bond growth.
"""
Base.@kwdef mutable struct TDVP2 <: Evolver
    order::Int = 2
    trunc::TruncationScheme = TruncationScheme()
    krylovdim::Int = 30
    tol::Float64 = 1e-12
    verbose::Bool = true
    cache::Union{Nothing,EnvCache} = nothing
end

"""
    TDVP1_CBE(; order=2, trunc, d_tilde_max=32, enr_rtol=1e-10, enr_atol=1e-12,
              enabled=true, krylovdim=30, tol=1e-12, verbose=true)

Single-site TDVP with controlled bond expansion (local PyTreeNet fork's
1TDVP-CBE; see the implementation notes further down). `trunc` is the main
`TruncationScheme` (its `maxdim` caps the final bond); `d_tilde_max` caps how
many *new* directions are proposed per bond per step; `enr_rtol`/`enr_atol`
are the enrichment singular-value tolerances (PyTreeNet
`enrichment_rel_tol`/`enrichment_total_tol`). With `verbose=true`, emits the
TDVP1 step and half-sweep records together with the CBE enrichment policy and
observed bond growth.
"""
Base.@kwdef mutable struct TDVP1_CBE <: Evolver
    order::Int = 2
    trunc::TruncationScheme = TruncationScheme(; maxdim=100)
    d_tilde_max::Int = 32
    enr_rtol::Float64 = 1e-10
    enr_atol::Float64 = 1e-12
    enabled::Bool = true
    krylovdim::Int = 30
    tol::Float64 = 1e-12
    verbose::Bool = true
    cache::Union{Nothing,EnvCache} = nothing
end

_tdvp_name(::TDVP1) = "TDVP1"
_tdvp_name(::TDVP2) = "TDVP2"
_tdvp_name(::TDVP1_CBE) = "TDVP1_CBE"

function _tdvp_max_bond_dim(ψ::TTNS)
    ds = [dim(virtualspace(ψ, n))
          for n in 1:nnodes(ψ.topo) if ψ.topo.parent[n] != 0]
    return isempty(ds) ? 1 : maximum(ds)
end

_tdvp_center_site(ψ::TTNS) = nodeid(ψ.topo, center(ψ))

function _log_tdvp_step_start(ev::TDVP1, ψ::TTNS, H::TTNO, dz::Number;
                              cache_reused::Bool)
    t = ψ.topo
    nodes = nnodes(t)
    physical_sites = count(identity, ψ.hasphys)
    bonds = nodes - 1
    center_site = _tdvp_center_site(ψ)
    initial_maxbond = _tdvp_max_bond_dim(ψ)
    order = ev.order
    krylovdim = ev.krylovdim
    tol = ev.tol
    hermitian = ishermitian(H)
    @info "TDVP1 step start" dz order nodes physical_sites bonds center_site initial_maxbond krylovdim tol hermitian cache_reused
    return nothing
end

function _log_tdvp_step_start(ev::TDVP2, ψ::TTNS, H::TTNO, dz::Number;
                              cache_reused::Bool)
    t = ψ.topo
    nodes = nnodes(t)
    physical_sites = count(identity, ψ.hasphys)
    bonds = nodes - 1
    center_site = _tdvp_center_site(ψ)
    initial_maxbond = _tdvp_max_bond_dim(ψ)
    order = ev.order
    krylovdim = ev.krylovdim
    tol = ev.tol
    hermitian = ishermitian(H)
    trunc_maxdim = ev.trunc.maxdim
    trunc_atol = ev.trunc.atol
    trunc_rtol = ev.trunc.rtol
    trunc_discarded_weight = ev.trunc.discarded_weight
    @info "TDVP2 step start" dz order nodes physical_sites bonds center_site initial_maxbond krylovdim tol hermitian cache_reused trunc_maxdim trunc_atol trunc_rtol trunc_discarded_weight
    return nothing
end

function _log_tdvp_step_start(ev::TDVP1_CBE, ψ::TTNS, H::TTNO, dz::Number;
                              cache_reused::Bool)
    t = ψ.topo
    nodes = nnodes(t)
    physical_sites = count(identity, ψ.hasphys)
    bonds = nodes - 1
    center_site = _tdvp_center_site(ψ)
    initial_maxbond = _tdvp_max_bond_dim(ψ)
    order = ev.order
    krylovdim = ev.krylovdim
    tol = ev.tol
    hermitian = ishermitian(H)
    enabled = ev.enabled
    d_tilde_max = ev.d_tilde_max
    enr_rtol = ev.enr_rtol
    enr_atol = ev.enr_atol
    trunc_maxdim = ev.trunc.maxdim
    trunc_atol = ev.trunc.atol
    trunc_rtol = ev.trunc.rtol
    trunc_discarded_weight = ev.trunc.discarded_weight
    @info "TDVP1_CBE step start" dz order nodes physical_sites bonds center_site initial_maxbond krylovdim tol hermitian cache_reused enabled d_tilde_max enr_rtol enr_atol trunc_maxdim trunc_atol trunc_rtol trunc_discarded_weight
    return nothing
end

function _log_tdvp_step_complete(ev, ψ::TTNS; initial_maxbond::Int)
    name = _tdvp_name(ev)
    center_site = _tdvp_center_site(ψ)
    final_maxbond = _tdvp_max_bond_dim(ψ)
    @info "$name step complete" center_site initial_maxbond final_maxbond
    return nothing
end

function step!(ev::Union{TDVP1,TDVP1_CBE}, ψ::TTNS, H::TTNO, dz::Number)
    cache_reused = ev.cache !== nothing && ev.cache.topo == ψ.topo
    if !cache_reused
        ev.cache = EnvCache(ψ.topo)
    end
    initial_maxbond = ev.verbose ? _tdvp_max_bond_dim(ψ) : 0
    ev.verbose && _log_tdvp_step_start(ev, ψ, H, dz; cache_reused)
    if ev.order == 1
        _tdvp1_sweep!(ev, ψ, H, dz; rev=false)
    elseif ev.order == 2
        _tdvp1_sweep!(ev, ψ, H, dz / 2; rev=false)
        _tdvp1_sweep!(ev, ψ, H, dz / 2; rev=true)
    else
        throw(ArgumentError("order must be 1 or 2"))
    end
    ev.verbose && _log_tdvp_step_complete(ev, ψ; initial_maxbond)
    return ψ
end

function _tdvp1_sweep!(ev::Union{TDVP1,TDVP1_CBE}, ψ::TTNS, H::TTNO, dz::Number; rev::Bool)
    t = ψ.topo
    cache = ev.cache::EnvCache
    order = rev ? reverse(postorder(t)) : postorder(t)
    herm = ishermitian(H)
    maxbond_before = ev.verbose ? _tdvp_max_bond_dim(ψ) : 0
    move_center!(ψ, order[1]; cache)
    for i in eachindex(order)
        n = order[i]
        @assert ψ.center == n
        # forward-evolve the site
        h1 = eff_h1(cache, ψ, H, n)
        A, _ = exponentiate(workspace_map(h1), dz, ψ.tensors[n];
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
    if ev.verbose
        name = _tdvp_name(ev)
        direction = rev ? :reverse : :forward
        site_updates = length(order)
        link_updates = nnodes(t) - 1
        center_site = _tdvp_center_site(ψ)
        maxbond_after = _tdvp_max_bond_dim(ψ)
        @info "$name sweep complete" direction dz site_updates link_updates center_site maxbond_before maxbond_after
    end
    return ψ
end

# split the center tensor towards `v`, backward-evolve the link tensor with
# the zero-site Hamiltonian, absorb it into `v`. `_split_link_up`/`_split_link_down`
# are the single seam CBE overrides (PyTreeNet: CBEOneSiteTDVPMixin only
# replaces `_update_link`'s split — the sweep skeleton is shared verbatim).
function _evolve_link_and_move!(ev::Union{TDVP1,TDVP1_CBE}, ψ::TTNS, H::TTNO,
                                u::Int, v::Int, dz::Number; herm::Bool)
    t = ψ.topo
    cache = ev.cache::EnvCache
    @assert ψ.center == u
    if t.parent[u] == v
        C = _split_link_up(ev, ψ, H, u, v, dz)      # installs isometry at u; C :: V_new ← V_e
        invalidate_node!(cache, u)
        k0 = eff_h0(cache, ψ, H, u, v)              # env(u→v) rebuilt from the new isometry
        C, _ = exponentiate(workspace_map(k0), -dz, C;
                            ishermitian=herm, krylovdim=ev.krylovdim, tol=ev.tol)
        ψ.tensors[v] = absorb_on_leg(ψ.tensors[v], C, childslot(t, v, u))
    else
        # the edge is (v, u) with v the child; the link tensor in that edge's
        # (below ← above) orientation is C :: V_e ← V_new'
        C = _split_link_down(ev, ψ, H, u, v, dz)
        invalidate_node!(cache, u)
        k0 = eff_h0(cache, ψ, H, v, u)              # env(v→u) untouched, env(u→v) rebuilt
        C, _ = exponentiate(workspace_map(k0), -dz, C;
                            ishermitian=herm, krylovdim=ev.krylovdim, tol=ev.tol)
        ψ.tensors[v] = ψ.tensors[v] * C
    end
    ψ.center = v
    invalidate_node!(cache, v)
    return ψ
end

# vanilla QR splits (TDVP1)
function _split_link_up(::TDVP1, ψ::TTNS, ::TTNO, u::Int, ::Int, ::Number)
    Q, C = left_orth(ψ.tensors[u])                  # C :: V_new ← V_e
    ψ.tensors[u] = Q
    return C
end
function _split_link_down(::TDVP1, ψ::TTNS, ::TTNO, u::Int, v::Int, ::Number)
    k = childslot(ψ.topo, u, v)
    Q, Cd = orth_factor_leg(ψ.tensors[u], k)        # Cd :: Y ← dual(V_e)
    ψ.tensors[u] = Q
    return transpose(Cd)                            # :: V_e ← dual(Y)
end

# ---------------------------------------------------------------------------

function step!(ev::TDVP2, ψ::TTNS, H::TTNO, dz::Number)
    cache_reused = ev.cache !== nothing && ev.cache.topo == ψ.topo
    if !cache_reused
        ev.cache = EnvCache(ψ.topo)
    end
    initial_maxbond = ev.verbose ? _tdvp_max_bond_dim(ψ) : 0
    ev.verbose && _log_tdvp_step_start(ev, ψ, H, dz; cache_reused)
    if ev.order == 1
        _tdvp2_sweep!(ev, ψ, H, dz; rev=false)
    elseif ev.order == 2
        _tdvp2_sweep!(ev, ψ, H, dz / 2; rev=false)
        _tdvp2_sweep!(ev, ψ, H, dz / 2; rev=true)
    else
        throw(ArgumentError("order must be 1 or 2"))
    end
    ev.verbose && _log_tdvp_step_complete(ev, ψ; initial_maxbond)
    return ψ
end

# One half-sweep of two-site TDVP over the post-order bond list (bond ≡ its
# child node). Forward: two-site(+dz) at each bond, single-site(-dz) at the
# parent in between. Reverse: mirror — single-site(-dz) at the parent *before*
# the two-site(+dz) update. The turning-point bond (last in forward, first in
# reverse) gets no intervening single-site backward step, matching the Strang
# structure of PyTreeNet's secondordertwosite.py.
function _tdvp2_sweep!(ev::TDVP2, ψ::TTNS, H::TTNO, dz::Number; rev::Bool)
    t = ψ.topo
    cache = ev.cache::EnvCache
    herm = ishermitian(H)
    bonds = [n for n in postorder(t) if t.parent[n] != 0]
    B = lastindex(bonds)
    maxbond_before = ev.verbose ? _tdvp_max_bond_dim(ψ) : 0
    for j in (rev ? reverse(eachindex(bonds)) : eachindex(bonds))
        n = bonds[j]
        m = t.parent[n]
        if rev
            move_center!(ψ, m; cache)
            j == B || _site_backward!(ev, ψ, H, m, dz; herm)
            _bond_forward!(ev, ψ, H, n, m, dz; herm, center_on=:n)
        else
            # Centering on the parent already makes `(n, m)` the active
            # two-site block. Avoid a redundant parent-to-child gauge move:
            # bending its link factor through a dual fermionic ancilla can
            # insert a pivotal sign before the block is merged again.
            move_center!(ψ, m; cache)
            _bond_forward!(ev, ψ, H, n, m, dz; herm, center_on=:m)
            j == B || _site_backward!(ev, ψ, H, m, dz; herm)
        end
    end
    if ev.verbose
        direction = rev ? :reverse : :forward
        bond_updates = length(bonds)
        backward_site_updates = max(bond_updates - 1, 0)
        center_site = _tdvp_center_site(ψ)
        maxbond_after = _tdvp_max_bond_dim(ψ)
        @info "TDVP2 sweep complete" direction dz bond_updates backward_site_updates center_site maxbond_before maxbond_after
    end
    return ψ
end

function _bond_forward!(ev::TDVP2, ψ::TTNS, H::TTNO, n::Int, m::Int, dz::Number;
                        herm::Bool, center_on::Symbol)
    cache = ev.cache::EnvCache
    Θ = two_site_tensor(ψ, n, m)
    h2 = eff_h2(cache, ψ, H, n, m)
    Θ, _ = exponentiate(workspace_map(h2), dz, Θ;
                         ishermitian=herm, krylovdim=ev.krylovdim, tol=ev.tol)
    invalidate_edge!(cache, n, m)
    split_two_site!(ψ, Θ, n, m; trunc=ev.trunc, center_on)
    return ψ
end

function _site_backward!(ev::Union{TDVP2,TDVP1_CBE}, ψ::TTNS, H::TTNO, m::Int,
                         dz::Number; herm::Bool)
    @assert ψ.center == m
    cache = ev.cache::EnvCache
    h1 = eff_h1(cache, ψ, H, m)
    A, _ = exponentiate(workspace_map(h1), -dz, ψ.tensors[m];
                        ishermitian=herm, krylovdim=ev.krylovdim, tol=ev.tol)
    update_tensor!(ψ, m, A; caches=(cache,))
    return ψ
end

# ---------------------------------------------------------------------------
# TDVP1-CBE — controlled bond expansion (port of the local PyTreeNet fork:
# cbe_onesitetdvp.py + cbe_util.py). The sweep is TDVP1's, verbatim; only the
# link split is replaced:
#   1. two-site predictor: forward-evolve the bond's two-site block (throwaway
#      probe; the sweep state is untouched), SVD-split it back with the *main*
#      TruncationScheme — its node-side isometry P spans the bond directions
#      the true dynamics wants;
#   2. "shrewd selection": project P onto the orthogonal complement of the
#      current site tensor's bond space (left_null), SVD the projection, keep
#      the top `d_tilde_max` directions above the enrichment tolerances;
#   3. split [A | enrichment]: SVD of the concatenation, truncated by the main
#      TruncationScheme (hard `maxdim` cap), new site isometry U, link
#      R = U† A on the *old* edge space (not square — this is what grows the
#      bond); backward link evolution and absorption proceed as in TDVP1.
# With `enabled=false` this reproduces TDVP1 exactly (QR split path).
# ---------------------------------------------------------------------------

function _split_link_up(ev::TDVP1_CBE, ψ::TTNS, H::TTNO, u::Int, v::Int, dz::Number)
    ev.enabled || return _split_link_up(TDVP1(), ψ, H, u, v, dz)
    A = ψ.tensors[u]                                # :: cod ← V_e (bond = domain)
    P = _cbe_predictor(ev, ψ, H, u, v, dz)          # :: cod ← V_pred
    U, R = _cbe_enrich_split(ev, A, P)              # U :: cod ← V_new, R :: V_new ← V_e
    ψ.tensors[u] = U
    return R
end

function _split_link_down(ev::TDVP1_CBE, ψ::TTNS, H::TTNO, u::Int, v::Int, dz::Number)
    ev.enabled || return _split_link_down(TDVP1(), ψ, H, u, v, dz)
    t = ψ.topo
    k = childslot(t, u, v)
    A = ψ.tensors[u]
    P = _cbe_predictor(ev, ψ, H, u, v, dz)
    # work in the bond-leg frame: permute slot k to the domain
    N, No = numind(A), numout(A)
    frame = (Backend._others(N, k), (k,))
    U, R = _cbe_enrich_split(ev, permute(A, frame), permute(P, frame))
    ψ.tensors[u] = permute(U, Backend._restore_perm(N, No, k))
    return transpose(R)                             # :: V_e ← dual(V_new)
end

# Two-site predictor (PyTreeNet _predict_site_tensor_two_site): forward-evolve
# the (child, parent) block across the crossed edge, split back with the main
# truncation, return the u-side isometry (S is contracted away from u, matching
# PyTreeNet's ContractionMode.VCONTR / u_identifier = node_id convention).
# Our eff_h2/two_site_tensor are non-mutating, so no scratch copies are needed.
function _cbe_predictor(ev::TDVP1_CBE, ψ::TTNS, H::TTNO, u::Int, v::Int, dz::Number)
    t = ψ.topo
    n, m = t.parent[u] == v ? (u, v) : (v, u)       # (child, parent) of the edge
    cache = ev.cache::EnvCache
    Θ = two_site_tensor(ψ, n, m)
    h2 = eff_h2(cache, ψ, H, n, m)
    Θ, _ = exponentiate(workspace_map(h2), dz, Θ;
                        ishermitian=ishermitian(H), krylovdim=ev.krylovdim, tol=ev.tol)
    pn = numout(ψ.tensors[n])
    NΘ = numind(Θ)
    Θs = permute(Θ, (ntuple(identity, pn), ntuple(j -> pn + j, NΘ - pn)))
    U, S, Vh = split_svd(Θs, ev.trunc)
    if u == n
        return U                                    # child side: (n legs) ← V_pred
    else
        # parent side: rebuild the m-layout tensor from Vh (slot k = V_pred)
        k = childslot(t, m, n)
        Km = numind(ψ.tensors[m]) - 1
        p1 = ntuple(j -> j == k ? 1 : 1 + (j < k ? j : j - 1), Km)
        return permute(Vh, (p1, (numind(Vh),)))
    end
end

# Shrewd selection + enriched split, in the bond-leg frame:
# tA :: rest ← X (current site, bond X), tP :: rest ← X_pred (predictor).
# Returns (U :: rest ← X_new, R :: X_new ← X). Falls back to the plain
# SVD-split of tA alone when no enrichment directions survive (PyTreeNet
# returns enrichment=None but still SVD-splits with the main truncation).
function _cbe_enrich_split(ev::TDVP1_CBE, tA::AbstractTensorMap, tP::AbstractTensorMap)
    expanded = tA
    room = min(ev.d_tilde_max, ev.trunc.maxdim - dim(domain(tA)))
    if room > 0 && dim(codomain(tA)) > dim(domain(tA))
        N = left_null(tA)                           # rest ← Y⊥,  N†·tA = 0
        if dim(domain(N)) > 0
            M = N' * tP                             # project predictor on the complement
            Um, _, _ = svd_trunc(M; trunc=truncrank(room) &
                                        trunctol(; atol=ev.enr_atol, rtol=ev.enr_rtol))
            if dim(domain(Um)) > 0
                E = N * Um                          # rest ← V_add (orthonormal, ⊥ tA)
                if isdual(domain(tA)[1]) != isdual(domain(E)[1])
                    E = flip(E, numind(E))
                end
                expanded = catdomain(tA, E)         # [A | enrichment]
            end
        end
    end
    U, _, _ = split_svd(expanded, ev.trunc)         # main truncation: hard maxdim cap
    R = U' * tA                                     # link on the OLD edge space
    return U, R
end
