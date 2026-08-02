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
    TDVP1(; order=2, krylovdim=30, tol=1e-12, eager=true, verbose=true)

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
    eager::Bool = true
    threaded_channels::Bool = false
    channel_slices::Int = 2
    channel_minbatch::Int = 2
    channel_min_flops::Real = 1_000_000
    channel_memory_cap_bytes::Union{Nothing,Real} = nothing
    distributed::Union{Nothing,AbstractDistributedContext} = nothing
    verbose::Bool = true
    cache::Union{Nothing,EnvCache} = nothing
end

"""
    TDVP2(; order=2, trunc=TruncationScheme(), krylovdim=30, tol=1e-12,
          eager=true, fuse_turning=true,
          contraction_optimize=true, contraction_sector_aware=true,
          verbose=true)

Two-site TDVP (benchmark kernel, §5b; PyTreeNet twositetdvp.py +
secondordertwosite.py). Every bond's two-site block is forward-evolved once
per sweep (post-order edge sweep, truncated split through `TruncationScheme`),
with a single-site *backward* evolution at the connecting parent between
consecutive bond updates. Bond dimensions adapt up to `trunc.maxdim`. With
`verbose=true`, emits step and half-sweep `@info` records including the
truncation policy and bond growth. `contraction_optimize=false` selects the
deterministic environment-first plan; `contraction_sector_aware=false` keeps
optimization enabled but skips the exact sector-aware dynamic program.
`fuse_turning=true` replaces the consecutive order-two turning-bond half
steps and their intermediate split by one full step and one split. With active
truncation this removes one truncation event and therefore is not bit-identical
to the unfused algorithm; set `fuse_turning=false` to preserve the two
half-step splits.
"""
Base.@kwdef mutable struct TDVP2 <: Evolver
    order::Int = 2
    trunc::TruncationScheme = TruncationScheme()
    krylovdim::Int = 30
    tol::Float64 = 1e-12
    eager::Bool = true
    fuse_turning::Bool = true
    contraction_optimize::Bool = true
    contraction_sector_aware::Bool = true
    threaded_channels::Bool = false
    channel_slices::Int = 2
    channel_minbatch::Int = 2
    channel_min_flops::Real = 1_000_000
    channel_memory_cap_bytes::Union{Nothing,Real} = nothing
    distributed::Union{Nothing,AbstractDistributedContext} = nothing
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
    eager = ev.eager
    hermitian = ishermitian(H)
    @info "TDVP1 step start" dz order nodes physical_sites bonds center_site initial_maxbond krylovdim tol eager hermitian cache_reused
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
    eager = ev.eager
    fuse_turning = ev.fuse_turning
    hermitian = ishermitian(H)
    trunc_maxdim = ev.trunc.maxdim
    trunc_atol = ev.trunc.atol
    trunc_rtol = ev.trunc.rtol
    trunc_discarded_weight = ev.trunc.discarded_weight
    contraction_optimize = ev.contraction_optimize
    contraction_sector_aware = ev.contraction_sector_aware
    @info "TDVP2 step start" dz order nodes physical_sites bonds center_site initial_maxbond krylovdim tol eager fuse_turning hermitian cache_reused trunc_maxdim trunc_atol trunc_rtol trunc_discarded_weight contraction_optimize contraction_sector_aware
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
    eager = ev.eager
    hermitian = ishermitian(H)
    cbe = nameof(typeof(ev.cbe))
    enabled = ev.enabled
    max_add = _cbe_max_add(ev.cbe)
    trunc_maxdim = ev.trunc.maxdim
    trunc_atol = ev.trunc.atol
    trunc_rtol = ev.trunc.rtol
    trunc_discarded_weight = ev.trunc.discarded_weight
    @info "TDVP1_CBE step start" dz order nodes physical_sites bonds center_site initial_maxbond krylovdim tol eager hermitian cache_reused cbe enabled max_add trunc_maxdim trunc_atol trunc_rtol trunc_discarded_weight
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
        fuse_turning = ev isa TDVP1 || !ev.enabled || !(ev.cbe isa LGVDCBE)
        if fuse_turning
            _tdvp1_sweep!(ev, ψ, H, dz / 2; rev=false, terminal_dz=dz)
            _tdvp1_sweep!(ev, ψ, H, dz / 2; rev=true,
                          skip_initial_site=true)
        else
            _tdvp1_sweep!(ev, ψ, H, dz / 2; rev=false)
            _tdvp1_sweep!(ev, ψ, H, dz / 2; rev=true)
        end
    else
        throw(ArgumentError("order must be 1 or 2"))
    end
    ev.verbose && _log_tdvp_step_complete(ev, ψ; initial_maxbond)
    return ψ
end

function _tdvp1_sweep!(
        ev::Union{TDVP1,TDVP1_CBE}, ψ::TTNS, H::TTNO, dz::Number;
        rev::Bool, terminal_dz::Union{Nothing,Number}=nothing,
        skip_initial_site::Bool=false)
    t = ψ.topo
    cache = ev.cache::EnvCache
    order = rev ? reverse(postorder(t)) : postorder(t)
    herm = ishermitian(H)
    maxbond_before = ev.verbose ? _tdvp_max_bond_dim(ψ) : 0
    move_center!(ψ, order[1]; cache)
    for i in eachindex(order)
        n = order[i]
        @assert ψ.center == n
        if !(skip_initial_site && i == firstindex(order))
            # The order-two public step replaces the two consecutive
            # turning-site half steps by one full step on the forward sweep.
            site_dz = terminal_dz !== nothing && i == lastindex(order) ?
                terminal_dz : dz
            h1 = eff_h1(cache, ψ, H, n;
                        threaded_channels=ev.threaded_channels,
                        channel_slices=ev.channel_slices,
                        channel_minbatch=ev.channel_minbatch,
                        channel_min_flops=ev.channel_min_flops,
                        channel_memory_cap_bytes=ev.channel_memory_cap_bytes,
                        distributed=ev.distributed)
            A, _ = _effective_exponentiate(
                ev.distributed, h1, site_dz, ψ.tensors[n];
                ishermitian=herm, krylovdim=ev.krylovdim, tol=ev.tol,
                eager=ev.eager)
            update_tensor!(ψ, n, A; caches=(cache,))
        end
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
        site_updates = length(order) - Int(skip_initial_site)
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
        C, _ = evolution_exponentiate_backend(
            workspace_map(k0), -dz, C;
            ishermitian=herm, krylovdim=ev.krylovdim, tol=ev.tol,
            eager=ev.eager)
        C = Networks.pivotal_link(C)
        ψ.tensors[v] = absorb_on_leg(ψ.tensors[v], C, childslot(t, v, u))
    else
        # the edge is (v, u) with v the child; the link tensor in that edge's
        # (below ← above) orientation is C :: V_e ← V_new'
        C = _split_link_down(ev, ψ, H, u, v, dz)
        invalidate_node!(cache, u)
        k0 = eff_h0(cache, ψ, H, v, u)              # env(v→u) untouched, env(u→v) rebuilt
        C, _ = evolution_exponentiate_backend(
            workspace_map(k0), -dz, C;
            ishermitian=herm, krylovdim=ev.krylovdim, tol=ev.tol,
            eager=ev.eager)
        C = Networks.pivotal_link(C)
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
    elseif ev.order == 2 && ev.fuse_turning
        _tdvp2_sweep!(ev, ψ, H, dz / 2; rev=false, terminal_dz=dz)
        _tdvp2_sweep!(ev, ψ, H, dz / 2; rev=true,
                      skip_initial_bond=true)
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
# structure of PyTreeNet's secondordertwosite.py. The order-two public step
# supplies `terminal_dz` and skips that same bond on the reverse sweep, fusing
# the consecutive turning exponentials and their intermediate split. Default
# keyword values retain the unfused half-sweep as a test/reference oracle.
function _tdvp2_sweep!(
        ev::TDVP2, ψ::TTNS, H::TTNO, dz::Number;
        rev::Bool, terminal_dz::Union{Nothing,Number}=nothing,
        skip_initial_bond::Bool=false)
    t = ψ.topo
    cache = ev.cache::EnvCache
    herm = ishermitian(H)
    bonds = [n for n in postorder(t) if t.parent[n] != 0]
    B = lastindex(bonds)
    maxbond_before = ev.verbose ? _tdvp_max_bond_dim(ψ) : 0
    for j in (rev ? reverse(eachindex(bonds)) : eachindex(bonds))
        rev && skip_initial_bond && j == B && continue
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
            bond_dz = terminal_dz !== nothing && j == B ? terminal_dz : dz
            center_on = terminal_dz !== nothing && j == B ? :n : :m
            _bond_forward!(ev, ψ, H, n, m, bond_dz; herm, center_on)
            j == B || _site_backward!(ev, ψ, H, m, dz; herm)
        end
    end
    if ev.verbose
        direction = rev ? :reverse : :forward
        bond_updates = length(bonds) - Int(rev && skip_initial_bond && B > 0)
        backward_site_updates = max(length(bonds) - 1, 0)
        center_site = _tdvp_center_site(ψ)
        maxbond_after = _tdvp_max_bond_dim(ψ)
        @info "TDVP2 sweep complete" direction dz bond_updates backward_site_updates center_site maxbond_before maxbond_after
    end
    return ψ
end

Base.@noinline function _bond_forward!(
        ev::TDVP2, ψ::TTNS, H::TTNO, n::Int, m::Int, dz::Number;
        herm::Bool, center_on::Symbol)
    cache = ev.cache::EnvCache
    Θ = two_site_tensor(ψ, n, m)
    h2 = eff_h2(cache, ψ, H, n, m;
                optimize=ev.contraction_optimize,
                sector_aware=ev.contraction_sector_aware,
                threaded_channels=ev.threaded_channels,
                channel_slices=ev.channel_slices,
                channel_minbatch=ev.channel_minbatch,
                channel_min_flops=ev.channel_min_flops,
                channel_memory_cap_bytes=ev.channel_memory_cap_bytes,
                distributed=ev.distributed)
    Θ, _ = _effective_exponentiate(
        ev.distributed, h2, dz, Θ;
        ishermitian=herm, krylovdim=ev.krylovdim, tol=ev.tol,
        eager=ev.eager)
    invalidate_edge!(cache, n, m)
    split_two_site!(ψ, Θ, n, m; trunc=ev.trunc, center_on)
    return ψ
end

Base.@noinline function _site_backward!(
        ev::Union{TDVP2,TDVP1_CBE}, ψ::TTNS, H::TTNO, m::Int,
        dz::Number; herm::Bool)
    @assert ψ.center == m
    cache = ev.cache::EnvCache
    h1 = if ev isa TDVP2
        eff_h1(cache, ψ, H, m;
               optimize=ev.contraction_optimize,
               sector_aware=ev.contraction_sector_aware,
               threaded_channels=ev.threaded_channels,
               channel_slices=ev.channel_slices,
               channel_minbatch=ev.channel_minbatch,
               channel_min_flops=ev.channel_min_flops,
               channel_memory_cap_bytes=ev.channel_memory_cap_bytes,
               distributed=ev.distributed)
    else
        eff_h1(cache, ψ, H, m;
               threaded_channels=ev.threaded_channels,
               channel_slices=ev.channel_slices,
               channel_minbatch=ev.channel_minbatch,
               channel_min_flops=ev.channel_min_flops,
               channel_memory_cap_bytes=ev.channel_memory_cap_bytes,
               distributed=ev.distributed)
    end
    A, _ = _effective_exponentiate(
        ev.distributed, h1, -dz, ψ.tensors[m];
        ishermitian=herm, krylovdim=ev.krylovdim, tol=ev.tol,
        eager=ev.eager)
    update_tensor!(ψ, m, A; caches=(cache,))
    return ψ
end
