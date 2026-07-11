"""
L5c — Finite-temperature representation layer (architecture §5c).

Implements the canonical purification route (§05 plan): a grand-canonical
thermal state is represented as `(e^{-βK/2} ⊗ I_a)|I⟩` on a doubled topology,
where `|I⟩` is the maximally entangled coevaluation state and `K = H - μN` is
the supplied thermal generator. All propagation is delegated to an
`Evolution.Evolver` through its complex-step interface — Thermal owns no
propagation kernel.

Scope note (§5c): finite-T *baths* default to thermofield star encoding fitted
by the companion `GraftImpurity.jl` package (temperature absorbed into the fit,
vacuum product initial state) — `Purified`/`METTS` are for Matsubara G(τ), local
thermalization, and lattice problems.

## Scope (v1)

- `Purified(; aux_evolution=:none)` is the only implemented representation.
  `METTS` and `HybridMETTS` remain declared future representations.
- Equilibrium imaginary time only: `thermalize(Purified, ...)` prepares
  `|Ψ_β⟩` and supports `thermal_expect` and `thermal_correlator`.
- Abelian symmetry only (trivial, fermion parity, U(1)).
- PP-dressed bosons: the `P + B_PP + B_thermal` cluster is supported.
"""
module Thermal

using ..Backend
using ..Trees
using ..Networks
using ..Contractions
using ..Symbolic
using ..TTNOBuild
using ..Evolution

using ..Backend: ℂ, ComplexSpace, ⊗, ←, dual, oneunit, dim, space, id,
    numind, numout, numin, codomain, domain, sectors, sectortype, spacetype,
    U1Space, U1Irrep, Vect, FermionParity, Trivial, TensorMap,
    AbstractTensorMap, ProductSpace, permute, blocks, block, norm, dot,
    isdual, fuse, ones_tensor

using ..Trees: TreeTopology, nnodes, nodeid, nodeindex, isleaf, leaves,
    nchildren, edges, postorder, preorder,
    path_to_root, path_between, mount_chain, childslot, neighbors

using ..Networks: TTNS, TTNO, topology, center, move_center!, update_tensor!,
    normalize!, check_arrows, physspace, virtualspace, apply_local,
    hasphys, physleg, parentleg

using ..Contractions: expect, inner, EnvCache

using ..Symbolic: OpSum, Term, SiteOp, charge

using ..TTNOBuild: ttno_from_opsum

using ..Evolution: Evolver, step!, supports_complex_step

export ThermalRep, Purified, METTS, HybridMETTS, thermalize,
    PurificationProblem, purification_problem, physical_ttno,
    PurifiedState, PurificationTrajectory, ScaledTTNS,
    infinite_temperature_state, thermal_expect, thermal_correlator,
    state_at, logZ

abstract type ThermalRep end

"""
    Purified(; aux_evolution=:none)

Ancilla-leg purification. `aux_evolution` is a first-class knob (§11.4):
`:none | :backward | :custom(H_aux)` — Karrasch–Barthel backward evolution of
the auxiliary legs; half of the real-time entanglement budget lives here.
v1 implements only `:none` (imaginary-time equilibrium). `:backward` and
`:custom` belong to the future finite-T real-time driver.
TODO(M4): `infinite_T_state(::Type{SU2Irrep})` Feiguin–White singlet
structure via symmetry dispatch.
"""
Base.@kwdef struct Purified <: ThermalRep
    aux_evolution::Any = :none
end

"""
    METTS(; rng, collapse_basis=:alternating)

Minimally entangled typical thermal states. TODO(M2) — no methods yet.
"""
struct METTS <: ThermalRep end

"""Impurity/bath may each pick sampling or purification (§5c). TODO(M2)."""
struct HybridMETTS <: ThermalRep end

"""
    PurificationProblem{S}

Problem container for thermal purification. Built by `purification_problem`.
"""
struct PurificationProblem{S<:ElementarySpace}
    topo_orig::TreeTopology
    topo_doubled::TreeTopology
    phys_orig::Dict{Symbol,S}
    phys_doubled::Dict{Symbol,S}
    ancilla_of::Dict{Symbol,Symbol}         # physical site => thermal ancilla
    physical_of::Dict{Symbol,Symbol}        # thermal ancilla => physical site
    pp_ancilla_of::Dict{Symbol,Symbol}      # P site => B_PP leaf (ppdress)
    thermal_ancilla_of::Dict{Symbol,Symbol} # P site => B_thermal leaf
    logical_groups::Vector{Vector{Symbol}}  # each group is one logical degree
    K::TTNO{S}                              # lifted thermal generator
    log_hilbert_dim::Float64
    hermitian::Bool
    elt::Type{<:Number}
    metadata::NamedTuple
end

"""
    ScaledTTNS

Internal normalization carrier: `psi` is always normalized, and the raw state
is `exp(log_amplitude) * psi`.
"""
struct ScaledTTNS{S<:ElementarySpace,T<:Number}
    psi::TTNS{S,T}
    log_amplitude::Float64
end

"""
    PurifiedState{S,T}

Thermal state at a given inverse temperature.
"""
struct PurifiedState{S<:ElementarySpace,T<:Number}
    psi::TTNS{S,T}
    beta::Float64
    log_amplitude::Float64
    logZ::Float64
    metadata::NamedTuple
end

"""
    PurificationTrajectory

Result of `thermalize(Purified, ...)`: final state plus checkpoints at
requested inverse temperatures.
"""
struct PurificationTrajectory{S<:ElementarySpace,T<:Number}
    final::PurifiedState{S,T}
    checkpoints::Dict{Float64,PurifiedState{S,T}}
    tau_grid::Vector{Float64}
    metadata::NamedTuple
end

# Accessor for checkpoint lookup by inverse temperature.
function state_at(traj::PurificationTrajectory, b::Real; atol::Float64=0.0)
    for k in keys(traj.checkpoints)
        if abs(k - Float64(b)) <= atol
            return traj.checkpoints[k]
        end
    end
    throw(KeyError("no checkpoint at β=$b (atol=$atol)"))
end

# logZ accessor
Base.log(p::PurifiedState) = p.logZ
logZ(p::PurifiedState) = p.logZ
logZ(t::PurificationTrajectory) = t.final.logZ

include("problem.jl")
include("state.jl")
include("driver.jl")

end # module Thermal
