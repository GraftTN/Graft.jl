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
using Random: AbstractRNG
using ..Backend
using ..Trees
using ..Networks
using ..Contractions

export dmrg1!, dmrg2!, dmrg1_3s!, expand!

"""
    dmrg1!(ψ, H; nsweeps=10, tol=1e-10, krylovdim=20, verbose=false) -> (ψ, energies)

Single-site DMRG: post-order + reverse sweeps, local Lanczos ground state of
the one-site effective Hamiltonian at every node. Bond dimensions are fixed —
start from a state with the target bond spaces (or wait for `dmrg1_3s!`).
Returns the per-half-sweep energy trace; stops early when the energy change
drops below `tol`. With `verbose=true`, emits setup and per-sweep `@info`
records with topology, solver, energy, convergence, center, and bond statistics.
"""
function dmrg1!(ψ::TTNS, H::TTNO; nsweeps::Int=10, tol::Float64=1e-10,
                krylovdim::Int=20, verbose::Bool=true)
    ishermitian(H) || throw(ArgumentError("dmrg1!: DMRG requires ishermitian(H) == true (§9.8)"))
    cache = EnvCache(ψ.topo)
    energies = Float64[]
    order = postorder(ψ.topo)
    verbose && _log_dmrg_start("dmrg1!", ψ; nsweeps, tol, krylovdim,
                               updates_per_sweep=2 * length(order),
                               fixed_bonds=true)
    for sweep in 1:nsweeps
        E = NaN
        for n in Iterators.flatten((order, Iterators.reverse(order)))
            move_center!(ψ, n; cache)
            h1 = eff_h1(cache, ψ, H, n)
            vals, vecs, _ = eigsolve(workspace_map(h1), ψ.tensors[n], 1, :SR;
                                     ishermitian=true, krylovdim)
            E = real(vals[1])
            update_tensor!(ψ, n, vecs[1]; caches=(cache,))
        end
        push!(energies, E)
        converged = _energy_converged(energies, tol)
        verbose && _log_dmrg_sweep("dmrg1!", ψ, sweep, energies; converged,
                                   updates=2 * length(order))
        converged && break
    end
    return ψ, energies
end

"""
    dmrg2!(ψ, H; trunc, nsweeps=10, tol=1e-10, krylovdim=20, verbose=false) -> (ψ, energies)

Two-site DMRG: sweeps every edge (post-order and reverse), Lanczos on the
bond's two-site block, truncated split through `TruncationScheme` (§9.5).
Grows bond dimensions up to `trunc.maxdim`. With `verbose=true`, emits setup
and per-sweep `@info` records with topology, solver, truncation, energy,
convergence, center, and bond statistics.
"""
function dmrg2!(ψ::TTNS, H::TTNO; trunc::TruncationScheme=TruncationScheme(),
                nsweeps::Int=10, tol::Float64=1e-10, krylovdim::Int=20,
                verbose::Bool=true)
    ishermitian(H) || throw(ArgumentError("dmrg2!: DMRG requires ishermitian(H) == true (§9.8)"))
    t = ψ.topo
    cache = EnvCache(t)
    energies = Float64[]
    bonds = [n for n in postorder(t) if t.parent[n] != 0]   # edge ≡ its child node
    verbose && _log_dmrg_trunc_start("dmrg2!", ψ; nsweeps, tol, krylovdim,
                                     updates_per_sweep=2 * length(bonds),
                                     trunc)
    for sweep in 1:nsweeps
        E = NaN
        for (n, center_on) in Iterators.flatten(
                (((n, :m) for n in bonds), ((n, :n) for n in Iterators.reverse(bonds))))
            m = t.parent[n]
            move_center!(ψ, n; cache)
            Θ = two_site_tensor(ψ, n, m)
            h2 = eff_h2(cache, ψ, H, n, m)
            vals, vecs, _ = eigsolve(workspace_map(h2), Θ, 1, :SR;
                                     ishermitian=true, krylovdim)
            E = real(vals[1])
            invalidate_edge!(cache, n, m)
            split_two_site!(ψ, vecs[1], n, m; trunc, center_on)
        end
        push!(energies, E)
        converged = _energy_converged(energies, tol)
        verbose && _log_dmrg_sweep("dmrg2!", ψ, sweep, energies; converged,
                                   updates=2 * length(bonds))
        converged && break
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
`sweep -> α`; `α == 0` skips expansion for that sweep. If
`expand_scheme=:rsvd`, pass an explicit `rng`. With `verbose=true`, emits setup
and per-sweep `@info` records with topology, solver, expansion, truncation,
energy, convergence, center, and bond statistics.
"""
function dmrg1_3s!(ψ::TTNS, H::TTNO; trunc::TruncationScheme=TruncationScheme(; maxdim=100),
                   nsweeps::Int=10, tol::Float64=1e-10, krylovdim::Int=20,
                   mixing=1.0, max_add::Int=8, expand_scheme::Symbol=:exact,
                   rng::Union{Nothing,AbstractRNG}=nothing,
                   rsvd_oversample::Int=8, rsvd_poweriter::Int=0,
                   enr_rtol::Float64=1e-10, enr_atol::Float64=1e-12,
                   verbose::Bool=true)
    ishermitian(H) || throw(ArgumentError("dmrg1_3s!: DMRG requires ishermitian(H) == true (§9.8)"))
    t = ψ.topo
    cache = EnvCache(t)
    energies = Float64[]
    order = postorder(t)
    bonds = [n for n in order if t.parent[n] != 0]
    verbose && _log_dmrg_expansion_start("dmrg1_3s!", ψ; nsweeps, tol, krylovdim,
                                         updates_per_sweep=2 * length(order),
                                         trunc, expand_scheme, max_add,
                                         rsvd_oversample, rsvd_poweriter,
                                         enr_rtol, enr_atol)
    for sweep in 1:nsweeps
        E = NaN
        for n in Iterators.flatten((order, Iterators.reverse(order)))
            move_center!(ψ, n; cache)
            h1 = eff_h1(cache, ψ, H, n)
            vals, vecs, _ = eigsolve(workspace_map(h1), ψ.tensors[n], 1, :SR;
                                     ishermitian=true, krylovdim)
            E = real(vals[1])
            update_tensor!(ψ, n, vecs[1]; caches=(cache,))
        end
        α = _mixing_value(mixing, sweep)
        if !iszero(α)
            root_targets = Contractions._physless_root_growth_targets(ψ, trunc, max_add)
            for n in bonds
                expand!(ψ, H, (n, t.parent[n]); scheme=expand_scheme, cache,
                        rng, trunc, max_add, mixing=α, enr_rtol, enr_atol,
                        rsvd_oversample, rsvd_poweriter)
            end
            Contractions._bootstrap_physless_root!(ψ, cache, root_targets)
        end
        push!(energies, E)
        converged = _energy_converged(energies, tol)
        verbose && _log_dmrg_expansion_sweep("dmrg1_3s!", ψ, sweep, energies;
                                             converged, updates=2 * length(order),
                                             alpha=α,
                                             expanded_bonds=iszero(α) ? 0 : length(bonds))
        converged && break
    end
    return ψ, energies
end

# TODO: als!/lobpcg! ports (PyTreeNet dmrg/als.py, lobpcg.py) as alternative
#   local eigensolvers; als.py doubles as the starting point for `linsolve!`.
# Variational fitting lives in Networks.fit! (§11.6), where it can be shared by
# GK/GSE/METTS-style compression without creating a GroundState dependency path.

_mixing_value(α::Number, ::Int) = α
_mixing_value(αs::AbstractVector, sweep::Int) = αs[min(sweep, lastindex(αs))]
_mixing_value(f, sweep::Int) = f(sweep)

_bond_dims(ψ::TTNS) = [dim(virtualspace(ψ, n)) for n in 1:nnodes(ψ.topo) if ψ.topo.parent[n] != 0]
function _max_bond_dim(ψ::TTNS)
    ds = _bond_dims(ψ)
    return isempty(ds) ? 1 : maximum(ds)
end
_delta_energy(energies::Vector{Float64}) =
    length(energies) > 1 ? energies[end] - energies[end - 1] : NaN
_energy_converged(energies::Vector{Float64}, tol::Float64) =
    length(energies) > 1 && abs(_delta_energy(energies)) < tol

function _log_dmrg_start(name::String, ψ::TTNS; nsweeps::Int, tol::Float64,
                         krylovdim::Int, updates_per_sweep::Int,
                         fixed_bonds::Bool=false)
    t = ψ.topo
    nodes = nnodes(t)
    physical_sites = count(identity, ψ.hasphys)
    bonds = nnodes(t) - 1
    center_site = nodeid(t, center(ψ))
    initial_maxbond = _max_bond_dim(ψ)
    @info "$name start" nodes physical_sites bonds center_site nsweeps tol krylovdim updates_per_sweep initial_maxbond fixed_bonds
    return nothing
end

function _log_dmrg_trunc_start(name::String, ψ::TTNS; nsweeps::Int, tol::Float64,
                               krylovdim::Int, updates_per_sweep::Int,
                               trunc::TruncationScheme)
    t = ψ.topo
    nodes = nnodes(t)
    physical_sites = count(identity, ψ.hasphys)
    bonds = nnodes(t) - 1
    center_site = nodeid(t, center(ψ))
    initial_maxbond = _max_bond_dim(ψ)
    trunc_maxdim = trunc.maxdim
    trunc_atol = trunc.atol
    trunc_rtol = trunc.rtol
    trunc_discarded_weight = trunc.discarded_weight
    @info "$name start" nodes physical_sites bonds center_site nsweeps tol krylovdim updates_per_sweep initial_maxbond trunc_maxdim trunc_atol trunc_rtol trunc_discarded_weight
    return nothing
end

function _log_dmrg_expansion_start(name::String, ψ::TTNS; nsweeps::Int, tol::Float64,
                                   krylovdim::Int, updates_per_sweep::Int,
                                   trunc::TruncationScheme, expand_scheme::Symbol,
                                   max_add::Int, rsvd_oversample::Int,
                                   rsvd_poweriter::Int, enr_rtol::Float64,
                                   enr_atol::Float64)
    t = ψ.topo
    nodes = nnodes(t)
    physical_sites = count(identity, ψ.hasphys)
    bonds = nnodes(t) - 1
    center_site = nodeid(t, center(ψ))
    initial_maxbond = _max_bond_dim(ψ)
    trunc_maxdim = trunc.maxdim
    trunc_atol = trunc.atol
    trunc_rtol = trunc.rtol
    trunc_discarded_weight = trunc.discarded_weight
    @info "$name start" nodes physical_sites bonds center_site nsweeps tol krylovdim updates_per_sweep initial_maxbond trunc_maxdim trunc_atol trunc_rtol trunc_discarded_weight expand_scheme max_add rsvd_oversample rsvd_poweriter enr_rtol enr_atol
    return nothing
end

function _log_dmrg_sweep(name::String, ψ::TTNS, sweep::Int,
                         energies::Vector{Float64};
                         converged::Bool, updates::Int)
    energy = energies[end]
    delta_energy = _delta_energy(energies)
    center_site = nodeid(ψ.topo, center(ψ))
    maxbond = _max_bond_dim(ψ)
    @info "$name sweep complete" sweep energy delta_energy converged updates center_site maxbond
    return nothing
end

function _log_dmrg_expansion_sweep(name::String, ψ::TTNS, sweep::Int,
                                   energies::Vector{Float64};
                                   converged::Bool, updates::Int,
                                   alpha, expanded_bonds::Int)
    energy = energies[end]
    delta_energy = _delta_energy(energies)
    center_site = nodeid(ψ.topo, center(ψ))
    maxbond = _max_bond_dim(ψ)
    @info "$name sweep complete" sweep energy delta_energy converged updates center_site maxbond alpha expanded_bonds
    return nothing
end

end # module GroundState
