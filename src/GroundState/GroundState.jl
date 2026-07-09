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
`sweep -> α`; `α == 0` skips expansion for that sweep. If
`expand_scheme=:rsvd`, pass an explicit `rng`.
"""
function dmrg1_3s!(ψ::TTNS, H::TTNO; trunc::TruncationScheme=TruncationScheme(; maxdim=100),
                   nsweeps::Int=10, tol::Float64=1e-10, krylovdim::Int=20,
                   mixing=1.0, max_add::Int=8, expand_scheme::Symbol=:exact,
                   rng::Union{Nothing,AbstractRNG}=nothing,
                   rsvd_oversample::Int=8, rsvd_poweriter::Int=0,
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
                        rng, trunc, max_add, mixing=α, enr_rtol, enr_atol,
                        rsvd_oversample, rsvd_poweriter)
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

_mixing_value(α::Number, ::Int) = α
_mixing_value(αs::AbstractVector, sweep::Int) = αs[min(sweep, lastindex(αs))]
_mixing_value(f, sweep::Int) = f(sweep)

_bond_dims(ψ::TTNS) = [dim(virtualspace(ψ, n)) for n in 1:nnodes(ψ.topo) if ψ.topo.parent[n] != 0]

end # module GroundState
