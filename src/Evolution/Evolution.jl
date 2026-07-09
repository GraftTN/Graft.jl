"""
L5b — Time evolution kernels (architecture §5b).

The universal contract: any `Evolver` accepts a **complex step** `dz` —
imaginary time, real time and complex time are three special cases of one
interface (design principle §0.2):

    step!(ev, ψ, H, dz)   evolves   ψ ← exp(dz·H)·ψ   (up to the scheme's order)

so real time is `dz = -im*δt`, imaginary time is `dz = -δτ`. (The architecture
document writes the same convention as "dz = -iδt / δτ"; the sign of the
imaginary-time step is explicit here.) No Evolver may assume `H` hermitian or
`dz` purely imaginary (§9.8) — complex-time Krylov impurity schemes must work
without touching this layer (§11.13). The single documented exception is
`ImplicitLogTime`, whose A-stability argument only holds on the imaginary
axis; that restriction is a *trait*, not a runtime throw (§10.3).

eltype rule (§9.7): a real TTNS is never silently promoted — real-time steps
on a real-eltype state are a caller-side explicit conversion.
"""
module Evolution

using KrylovKit: KrylovKit, exponentiate
using ..Backend
using ..Trees
using ..Networks
using ..Contractions

export Evolver, step!, evolve!, CorrelatorSeries, correlator, correlator_series,
    supports_complex_step,
    TDVP1, TDVP2, TDVP1_CBE, GlobalKrylov, GSE_TDVP, LSE_TDVP, TEBD, BUG,
    FixedBUG, ImplicitLogTime

abstract type Evolver end

"""
    step!(ev::Evolver, ψ::TTNS, H::TTNO, dz::Number) -> ψ

One propagation step `ψ ← exp(dz·H)·ψ`. See module docstring for the `dz`
convention. Implementations mutate `ψ` (tensors, gauge, and their `EnvCache`)
in place and must keep the single-center invariant (§9.1) on exit.
"""
function step! end

"""
    evolve!(ev, ψ, H, dz, nsteps; callback=nothing) -> ψ

Drive `nsteps` equal steps, invoking `callback(ψ, i)` after each. TODO(§10.8):
replace with the iterator-protocol sweep skeleton (`Base.iterate` +
`with_checkpoint` combinators) so checkpointing/monitoring wrap non-invasively.
"""
function evolve!(ev::Evolver, ψ::TTNS, H::TTNO, dz::Number, nsteps::Int;
                 callback=nothing)
    for i in 1:nsteps
        step!(ev, ψ, H, dz)
        callback === nothing || callback(ψ, i)
    end
    return ψ
end

"""
    correlator(ψ0, E0, A, B, ts; H, evolver) -> Vector

Single-evolution zero-temperature real-time correlator

`⟨ψ0| A exp(-im * (H - E0) * t) B |ψ0⟩`

for neutral local insertions `A = site => op`, `B = site => op`. The initial
state is assumed to be an eigenstate with energy `E0`; for non-eigenstate
snapshots the two-evolution form `⟨Aψ0(t)|Bψ0(t)⟩` should be implemented as a
separate driver. `ts` must be real, nonnegative, and nondecreasing. `H` is a
required keyword to avoid hidden solver state (§9.9), and the supplied evolver
is deep-copied with its cache reset when it owns one.
"""
function correlator(ψ0::TTNS, E0::Number, A, B, ts; H::TTNO, evolver::Evolver)
    eltype(ψ0) <: Complex ||
        throw(ArgumentError("correlator real-time evolution requires a complex-eltype TTNS; convert explicitly"))
    topology(H) == topology(ψ0) || throw(ArgumentError("correlator: H and ψ0 have mismatched topologies"))
    Asite, Aop = _local_insertion(A)
    Bsite, Bop = _local_insertion(B)
    bra = apply_local(ψ0, Aop', Asite)
    ket = apply_local(ψ0, Bop, Bsite)
    times = collect(ts)
    valtype = typeof(inner(bra, ket))
    isempty(times) && return valtype[]

    ev = _fresh_evolver(evolver)
    τprev = zero(real(times[1]))
    vals = Vector{typeof(exp(im * E0 * τprev) * inner(bra, ket))}(undef, length(times))
    for i in eachindex(times)
        isreal(times[i]) || throw(ArgumentError("correlator times must be real"))
        τ = real(times[i])
        τ >= zero(τ) || throw(ArgumentError("correlator times must be nonnegative"))
        τ >= τprev || throw(ArgumentError("correlator times must be nondecreasing"))
        dt = τ - τprev
        iszero(dt) || step!(ev, ket, H, -im * dt)
        vals[i] = exp(im * E0 * τ) * inner(bra, ket)
        τprev = τ
    end
    return vals
end

"""
    CorrelatorSeries(times, values, metadata)

Discrete correlator snapshot series for M1 spectral post-processing:
iteration yields `(t, value)` pairs and `metadata` is a typed `NamedTuple`.
"""
struct CorrelatorSeries{R<:Real,V,M<:NamedTuple}
    times::Vector{R}
    values::Vector{V}
    metadata::M
    function CorrelatorSeries(times::AbstractVector{R}, values::AbstractVector{V},
                              metadata::M) where {R<:Real,V,M<:NamedTuple}
        length(times) == length(values) ||
            throw(ArgumentError("CorrelatorSeries needs one value per time"))
        return new{R,V,M}(collect(times), collect(values), metadata)
    end
end
Base.length(s::CorrelatorSeries) = length(s.times)
Base.getindex(s::CorrelatorSeries, i::Int) = (s.times[i], s.values[i])
Base.iterate(s::CorrelatorSeries, state...) = iterate(zip(s.times, s.values), state...)

"""
    correlator_series(ψ0, E0, A, B, ts; H, evolver, metadata=(;)) -> CorrelatorSeries

Typed snapshot wrapper around [`correlator`](@ref). `metadata` is merged with
the insertion sites and `E0`; operator tensors stay in the call, not in the
metadata payload.
"""
function correlator_series(ψ0::TTNS, E0::Number, A, B, ts; H::TTNO,
                           evolver::Evolver, metadata::NamedTuple=(;))
    times = collect(ts)
    vals = correlator(ψ0, E0, A, B, times; H, evolver)
    Asite, _ = _local_insertion(A)
    Bsite, _ = _local_insertion(B)
    meta = merge((; E0, Asite, Bsite), metadata)
    return CorrelatorSeries(real.(times), vals, meta)
end

_local_insertion(x) =
    throw(ArgumentError("local insertion must be `site::Symbol => op::AbstractTensorMap`"))
function _local_insertion(x::Pair)
    x.first isa Symbol || throw(ArgumentError("local insertion site must be a Symbol"))
    x.second isa AbstractTensorMap || throw(ArgumentError("local insertion operator must be an AbstractTensorMap"))
    return x.first, x.second
end

function _fresh_evolver(ev::Evolver)
    evrun = deepcopy(ev)
    if hasproperty(evrun, :cache)
        setproperty!(evrun, :cache, nothing)
    end
    return evrun
end

"""
    supports_complex_step(::Type{<:Evolver}) -> Bool

Trait (§10.3): whether the scheme is valid for arbitrary complex `dz`.
`false` only for imaginary-time-only schemes (`ImplicitLogTime`).
"""
supports_complex_step(::Type{<:Evolver}) = true

include("tdvp.jl")

# ---------------------------------------------------------------------------
# TODO stubs — declared types so dispatch surfaces exist; no methods yet.
# ---------------------------------------------------------------------------

# TODO(M1): GlobalKrylov (GK) — KrylovKit exponentiate on the full state via
# apply(H, ψ) + fit! compression; non-hermitian safe (Arnoldi path). Blocked on
# Networks.fit! (variational fitting primitive, §11.6).
"""Global Krylov evolver. TODO(M1) — requires `Networks.fit!`."""
struct GlobalKrylov <: Evolver end

# TODO(M1+): GSE_TDVP — global subspace expansion (Yang–White) + 1TDVP;
# expansion step goes through the shared `expand!` primitive (§11.7).
"""Global-subspace-expansion TDVP. TODO(M1+) — requires `expand!(scheme=:gse)`."""
struct GSE_TDVP <: Evolver end

# TODO(M1+): LSE_TDVP — local subspace expansion variant.
"""Local-subspace-expansion TDVP. TODO(M1+)."""
struct LSE_TDVP <: Evolver end

# TODO(M0/M1): TEBD — nearest-neighbour gates on tree edges only (valid inside
# bath chains); port PyTreeNet time_evolution/tebd.py + trotter.py (Trotter
# layers are already tree-aware there).
"""Tree-edge TEBD. TODO — PyTreeNet tebd.py/trotter.py port pending."""
struct TEBD <: Evolver end

# TODO(M1+): BUG / FixedBUG — rank-adaptive Basis-Update & Galerkin
# (PyTreeNet specialty: time_evolution/bug.py, fixed_bug.py); friendly to
# branching nodes.
"""Rank-adaptive Basis-Update & Galerkin. TODO — PyTreeNet bug.py port pending."""
struct BUG <: Evolver end
"""Fixed-rank BUG. TODO — PyTreeNet fixed_bug.py port pending."""
struct FixedBUG <: Evolver end

# TODO(M1): ImplicitLogTime — A-stable implicit stepping on a logarithmic
# τ-grid (Zima–Stoudenmire–White–Parcollet–Kaye, arXiv:2606.02930). Needs the
# `linsolve!` primitive (tree ALS outer loop + KrylovKit.linsolve inner —
# PyTreeNet dmrg/als.py is the starting point); acceptance test: AIM Kondo
# scaling of 2606.02930. Imaginary-axis only, enforced via the trait below and
# a `Re(dz) > 0, Im(dz) = 0` type restriction on its `step!`.
"""Implicit log-time-grid imaginary-time evolver. TODO(M1) — requires `linsolve!`."""
struct ImplicitLogTime <: Evolver end
supports_complex_step(::Type{ImplicitLogTime}) = false

# TODO(M1): linsolve!(ψ, A, rhs) — variational linear solve on trees (ALS
# sweep), the second public primitive demanded by ImplicitLogTime (§5b);
# also reused by TaSK / correction-vector methods later. Lives here or in
# Networks once written.

end # module Evolution
