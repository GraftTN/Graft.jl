"""
L5c — Finite-temperature representation layer (architecture §5c). **All TODO (M2).**

Orthogonal to Evolution by design: this layer only defines how thermal states
are *represented and sampled*; imaginary/real-time propagation is always
delegated to an arbitrary `Evolution.Evolver` through its complex-step
interface.

Scope note (§5c): finite-T *baths* default to thermofield star encoding fitted
by the companion `GraftImpurity.jl` package (temperature absorbed into the fit,
vacuum product initial state) — `Purified`/`METTS` are for Matsubara G(τ), local
thermalization, and lattice problems.
"""
module Thermal

using ..Networks

export ThermalRep, Purified, METTS, HybridMETTS, thermalize

abstract type ThermalRep end

"""
    Purified(; aux_evolution=:none)

Ancilla-leg purification. `aux_evolution` is a first-class knob (§11.4):
`:none | :backward | :custom(H_aux)` — Karrasch–Barthel backward evolution of
the auxiliary legs; half of the real-time entanglement budget lives here.
TODO(M2). TODO(M4): `infinite_T_state(::Type{SU2Irrep})` Feiguin–White singlet
structure via symmetry dispatch.
"""
Base.@kwdef struct Purified <: ThermalRep
    aux_evolution::Any = :none
end

"""
    METTS(; rng, collapse_basis=:alternating)

Minimally entangled typical thermal states: sample stream with alternating
collapse bases (ergodicity); collapse on tree leaves ≡ cutting subtrees —
naturally cheap. Collapse-basis/sector compatibility (U(1): occupation basis
vs its dual) lives in the sampler. TODO(M2). RNG is explicit (§9.6).
"""
struct METTS <: ThermalRep end

"""Impurity/bath may each pick sampling or purification (§5c). TODO(M2)."""
struct HybridMETTS <: ThermalRep end

# TODO(M2): implement the driver — preparation (e^{-βH/2} delegated to any
# Evolver's imaginary-time step) + sampling bookkeeping; β-scheduling pluggable
# (`τ_grid = :uniform | :log | Vector`), required by ImplicitLogTime (§5b).
"""
    thermalize(rep::ThermalRep, H, β; evolver, τ_grid=:uniform) -> state/samples

Thermal-state preparation/sampling driver. TODO(M2) — no methods yet.
"""
function thermalize end

end # module Thermal
