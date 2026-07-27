"""
L5c — Finite-temperature representation layer (architecture §5c).

Implements the canonical purification route (§05 plan): a grand-canonical
thermal state is represented as `(e^{-βK/2} ⊗ I_a)|I⟩` on a doubled topology,
where `|I⟩` is the maximally entangled coevaluation state and `K = H - μN` is
the supplied thermal generator. All propagation is delegated to an
`Evolution.Evolver` through its complex-step interface — Thermal owns no
propagation kernel.

Scope note (§5c, 2026-07-14): complete finite-temperature Hamiltonians use the
ordinary `Purified` route implemented here. The disabled companion-package
thermofield code is a frozen EDMFT bosonic-bath layout experiment: it would
organize purified bath modes into emission/absorption branches, not prepare the
thermal state. Its public fit/mount calls warn and return `nothing`.

## Implemented scope

- `Purified` equilibrium preparation, imaginary-time correlators, and
  finite-temperature real-time correlators with
  `aux_evolution=:none|:backward|custom_ttno`.
- serial `METTS` and partial-purification `HybridMETTS`, including explicit
  RNGs, conditional Born collapse, burn-in/thinning, autocorrelation-aware
  errors, and restartable trajectories.
- Abelian symmetry only (trivial, fermion parity, U(1)).
- PP-dressed bosons: the `P + B_PP + B_thermal` cluster is supported.

Impurity-domain utilities that used to live here (Matsubara transforms,
finite-mode Anderson-Holstein benchmarks, CTSEG CSV exchange, Kondo/bath
scaling analysis) moved to the companion `GraftImpurity.jl` package; Thermal
keeps only the representation layer definable from `(ψ, K, operators, Evolver)`.
"""
module Thermal

using ..Backend
using ..Trees
using ..Networks
using ..Contractions
using ..Symbolic
using ..TTNOBuild
using ..Evolution
using ..Parallel: threaded_foreach
using Random: AbstractRNG

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
    METTSSample, METTSTrajectory, METTSStatistics, metts_statistics,
    infinite_temperature_state, thermal_expect, thermal_correlator,
    thermal_realtime_correlator, state_at, logZ

abstract type ThermalRep end

"""
    Purified(; aux_evolution=:none)

Ancilla-leg purification. `aux_evolution` is a first-class knob (§11.4):
`:none | :backward | :custom(H_aux)` — Karrasch–Barthel backward evolution of
the auxiliary legs; half of the real-time entanglement budget lives here.
Equilibrium preparation is independent of this gauge choice; the real-time
driver applies the requested auxiliary evolution.
TODO(M4): `infinite_T_state(::Type{SU2Irrep})` Feiguin–White singlet
structure via symmetry dispatch.
"""
Base.@kwdef struct Purified <: ThermalRep
    aux_evolution::Any = :none
end

"""
    METTS(; rng, collapse_basis=:alternating)

Minimally entangled typical thermal states with explicit RNG and sampling
schedule. See `thermalize(::METTS, ...)`.
"""
struct METTS{R<:AbstractRNG,B} <: ThermalRep
    rng::R
    collapse_basis::B
    burnin::Int
    nsamples::Int
    thin::Int
    function METTS(rng::R, collapse_basis::B, burnin::Integer,
                   nsamples::Integer, thin::Integer) where {R<:AbstractRNG,B}
        burnin >= 0 || throw(ArgumentError("METTS burnin must be nonnegative"))
        nsamples >= 1 || throw(ArgumentError("METTS nsamples must be positive"))
        thin >= 1 || throw(ArgumentError("METTS thin must be positive"))
        return new{R,B}(rng, collapse_basis, Int(burnin), Int(nsamples), Int(thin))
    end
end

METTS(; rng::AbstractRNG,
      collapse_basis=:alternating,
      burnin::Integer=10,
      nsamples::Integer=100,
      thin::Integer=1) =
    METTS(rng, collapse_basis, burnin, nsamples, thin)

"""Impurity/bath logical groups may independently use sampling or purification."""
struct HybridMETTS{R<:AbstractRNG,B} <: ThermalRep
    rng::R
    sampled_sites::Vector{Symbol}
    collapse_basis::B
    burnin::Int
    nsamples::Int
    thin::Int
    function HybridMETTS(rng::R, sampled_sites, collapse_basis::B,
                         burnin::Integer, nsamples::Integer,
                         thin::Integer) where {R<:AbstractRNG,B}
        sites = unique(Symbol.(collect(sampled_sites)))
        isempty(sites) &&
            throw(ArgumentError("HybridMETTS needs at least one sampled site"))
        burnin >= 0 || throw(ArgumentError("HybridMETTS burnin must be nonnegative"))
        nsamples >= 1 || throw(ArgumentError("HybridMETTS nsamples must be positive"))
        thin >= 1 || throw(ArgumentError("HybridMETTS thin must be positive"))
        return new{R,B}(rng, sites, collapse_basis,
                        Int(burnin), Int(nsamples), Int(thin))
    end
end

HybridMETTS(; rng::AbstractRNG,
            sampled_sites,
            collapse_basis=:alternating,
            burnin::Integer=10,
            nsamples::Integer=100,
            thin::Integer=1) =
    HybridMETTS(rng, sampled_sites, collapse_basis, burnin, nsamples, thin)

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
    generator::OpSum                        # source for auxiliary transpose
    K_orig::TTNO{S}                         # generator on original topology
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
    checkpoints::Dict{Float64,<:PurifiedState}
    tau_grid::Vector{Float64}
    metadata::NamedTuple
    function PurificationTrajectory(final::PurifiedState{S,T},
                                    checkpoints::Dict{Float64,<:PurifiedState},
                                    tau_grid::Vector{Float64},
                                    metadata::NamedTuple) where {S<:ElementarySpace,T<:Number}
        return new{S,T}(final, checkpoints, tau_grid, metadata)
    end
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
include("metts.jl")

end # module Thermal
