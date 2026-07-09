"""
L5a — Ground-state kernels (architecture §5a). PyTreeNet: dmrg/dmrg.py.

Implemented: `dmrg1!` (single-site, benchmark/warm-up), `dmrg2!` (two-site,
small systems & hard initializations). The production kernel `dmrg1_3s!`
(3S mixing + RSVD post-expansion, McCulloch–Osborne arXiv:2403.00562) is TODO
and blocked only on the shared `expand!` primitive. CBE is deliberately *not*
planned for ground states (§5a: RSVD post-expansion supersedes it for the
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

export dmrg1!, dmrg2!, expand!

# TODO(M0, §5a/§11.7): expand!(ψ, H, edge; scheme=:rsvd, cache) — the shared
# subspace-expansion primitive on an edge (the fiddly part: mixing/expansion
# terms on branching tensors with ≥3 virtual legs — written once, reused by
# 3S/RSVD, CBE, GSE, LSE). RSVD probes are drawn blockwise per sector
# (TensorKit-compatible); RNG explicit (§9.6).
"""
    expand!(ψ, H, edge; scheme=:rsvd, cache, rng, kwargs...) -> ψ

Shared bond-expansion primitive (§5a). TODO — no methods yet; `TDVP1_CBE` and
`dmrg1_3s!` both hang on this.
"""
function expand! end

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

# TODO(M0): dmrg1_3s!(ψ, H; trunc, mixing α schedule) — production kernel:
#   single-site update + 3S mixing + RSVD post-expansion via `expand!`
#   (arXiv:2403.00562: one QR (+ optional SVD), O(dwkD²), no five-fold SVD).
# TODO: als!/lobpcg! ports (PyTreeNet dmrg/als.py, lobpcg.py) as alternative
#   local eigensolvers; als.py doubles as the starting point for `linsolve!`.
# TODO(M0): Networks.fit! (variational fitting) — PyTreeNet
#   dmrg/variational_fitting.py; shared with GK/GSE/METTS (§11.6).

end # module GroundState
