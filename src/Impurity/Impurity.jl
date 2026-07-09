"""
L6 — GRAFT.Impurity: embedded impurity-solver module (architecture §6).
**Backbone types + signatures only; implementations TODO (M0 bath fitting →
M2 thermofield → M5 EDMFT).**

Never referenced by any lower layer (§9.10). Owns *no* private geometry code
(§0.1): geometry builders emit plain `Trees.TreeTopology`.
"""
module Impurity

using ..Trees
using ..Networks
using ..Symbolic

export Partition, audit_partition, BathParametrization, RealPoles, ComplexPoles,
    couplings, mount_bath, fit_bath, BosonBath, solve

# ---------------------------------------------------------------------------
# §6.2 Partition: a *user declaration* on the impurity orbitals; H_bath never
# partitions independently — it inherits the block structure through Δ(ω).
# Dependency chain (one-way):
#   physics priors → Partition P → block structure of Δ → blockwise bath fit
#   → modes mounted on the block's branch.
# ---------------------------------------------------------------------------

"""
    Partition(blocks::Vector{Vector{Symbol}})

Immutable grouping of impurity orbitals into blocks (eg/t2g, j_eff, d+ligand…).
First-class *input*: fixed partition ⇒ fixed topology ⇒ warm starts across the
self-consistency loop stay valid (`==`/`hash` are value-based, §9.4/§10.9).
Automatic partitioning is deliberately **not** offered — `audit_partition` is
the diagnostic ("人分区、库验收").
"""
struct Partition
    blocks::Vector{Vector{Symbol}}
    function Partition(blocks::Vector{Vector{Symbol}})
        orbs = reduce(vcat, blocks; init=Symbol[])
        allunique(orbs) || throw(ArgumentError("orbitals appear in more than one block"))
        return new(blocks)
    end
end
Base.:(==)(a::Partition, b::Partition) = a.blocks == b.blocks
Base.hash(p::Partition, h::UInt) = hash(p.blocks, hash(:Partition, h))

# TODO(M5): cross-block vs in-block mutual-information audit; warn when
# cross-block MI ≳ in-block MI ("the partition may be cut wrong"). Rare-event
# handling stays manual: re-partition + loop restart (§6.2).
"""
    audit_partition(ψ_converged, P::Partition) -> report

Partition diagnostic ("人分区、库验收"). TODO(M5) — no methods yet.
"""
function audit_partition end

# ---------------------------------------------------------------------------
# §6.3 bath discretization / hybridization fitting
# ---------------------------------------------------------------------------

abstract type BathParametrization end

"""
    RealPoles(poles, residues, blocks, block_ranges, diagnostics)

Real-pole bath parametrization. For bosons, `poles[k] = ω_k > 0` and
`residues[k] = g_k^2 >= 0`, so the mode coupling is `g_k = sqrt(residues[k])`.
`block_ranges[i]` indexes the modes fitted for `blocks[i]`. Diagnostics are a
typed `NamedTuple` produced by [`fit_bath`](@ref).
"""
struct RealPoles{D<:NamedTuple} <: BathParametrization
    poles::Vector{Float64}
    residues::Vector{Float64}
    blocks::Vector{Vector{Symbol}}
    block_ranges::Vector{UnitRange{Int}}
    diagnostics::D

    function RealPoles(poles::AbstractVector{<:Real}, residues::AbstractVector{<:Real},
                       blocks::Vector{Vector{Symbol}},
                       block_ranges::Vector{UnitRange{Int}},
                       diagnostics::D) where {D<:NamedTuple}
        length(poles) == length(residues) ||
            throw(ArgumentError("RealPoles needs one residue per pole"))
        length(blocks) == length(block_ranges) ||
            throw(ArgumentError("RealPoles needs one mode range per partition block"))
        p = Float64.(poles)
        r = Float64.(residues)
        all(isfinite, p) && all(x -> x > 0, p) ||
            throw(ArgumentError("RealPoles poles must be finite and positive"))
        all(isfinite, r) && all(x -> x >= 0, r) ||
            throw(ArgumentError("RealPoles residues must be finite and nonnegative"))
        expected = _contiguous_ranges(block_ranges)
        expected == block_ranges ||
            throw(ArgumentError("RealPoles block ranges must be contiguous and ordered"))
        return new{D}(p, r, deepcopy(blocks), copy(block_ranges), diagnostics)
    end
end
Base.length(b::RealPoles) = length(b.poles)
couplings(b::RealPoles) = sqrt.(b.residues)

"""
    ComplexPoles

Type slot ONLY (§6.3): quasi-Lindblad pseudomode baths (complex poles). Kept so
the fitter interface doesn't change when the TTNDO route lands; deliberately
unimplemented.
"""
struct ComplexPoles <: BathParametrization end

"""
    fit_bath(J, P::Partition; T=0, kind=:boson, nmodes, ωmin, ωmax,
             grid=:linear, method=:midpoint, crossblock=:highmount) -> RealPoles

Blockwise T=0 real-pole fitting of a continuous boson spectral density `J(ω)`.
The M0 implementation is deterministic midpoint quadrature: each bin produces
one pole at the bin midpoint and a residue `g_k^2 = ∫_bin J(ω)dω` approximated
by the midpoint rule. `J` may be one function shared by all blocks or a vector
of one function per partition block. The partition argument is in the signature
from day one (§6.3).

* `T = 0`: Δ(iωₙ)/Δ(ω) → real-pole fit per block.
* `T > 0` (TODO M2): thermofield star encoding — fit Γf and Γ(1−f) (fermions) /
  absorption & emission parts (bosons) separately; vacuum product initial state.
* `crossblock = :highmount | :rotate` (§6.2): high mounting near the tree
  center, or a pre-fit single-particle rotation (returned with results).

Mandatory self-checks to implement with it (§6.3): (1) β·δε ≪ 1 resolution
check; (2) loop-bath vs final-bath both projected back to Δ(iωₙ) and compared.
Global fitting across the whole Δ matrix while ignoring `P` is a forbidden
path — the interface does not offer it. TODO(M0+): replace/augment midpoint
with adapol-style AAA initialization plus nonlinear refinement.
"""
function fit_bath(J, P::Partition; T::Real=0, kind::Symbol=:boson,
                  nmodes::Integer=8, wmin=nothing, wmax=nothing,
                  ωmin=nothing, ωmax=nothing, grid::Symbol=:linear,
                  method::Symbol=:midpoint, crossblock::Symbol=:highmount)
    T == 0 || throw(ArgumentError("fit_bath T > 0 is TODO(M2): thermofield star fitting is not part of the forwarded T=0 boson path"))
    kind == :boson || throw(ArgumentError("fit_bath currently implements only kind=:boson for the forwarded B4 path"))
    method == :midpoint || throw(ArgumentError("fit_bath method=$method is unavailable; TODO(M0+) adapol/AAA refinement"))
    crossblock in (:highmount, :rotate) ||
        throw(ArgumentError("crossblock must be :highmount or :rotate"))
    nmodes > 0 || throw(ArgumentError("nmodes must be positive"))
    lo = _bound("ωmin", wmin, ωmin)
    hi = _bound("ωmax", wmax, ωmax)
    lo < hi || throw(ArgumentError("ωmin must be smaller than ωmax"))

    targets = _block_targets(J, P)
    poles = Float64[]
    residues = Float64[]
    ranges = UnitRange{Int}[]
    block_diags = map(enumerate(targets)) do (i, target)
        start = length(poles) + 1
        p, r = _midpoint_modes(target, lo, hi, Int(nmodes), grid)
        append!(poles, p)
        append!(residues, r)
        stop = length(poles)
        push!(ranges, start:stop)
        _fit_diagnostics(target, p, r, lo, hi, Int(nmodes), grid, i)
    end
    diagnostics = (;
        kind,
        T = 0.0,
        method,
        grid,
        nmodes = Int(nmodes),
        ωmin = lo,
        ωmax = hi,
        crossblock,
        block_diagnostics = block_diags,
    )
    return RealPoles(poles, residues, deepcopy(P.blocks), ranges, diagnostics)
end

"""
    mount_bath(topo, bath::RealPoles, P::Partition; mode=:star, prefix=:bath, attach=nothing)

Return a named tuple with a new topology and the mode-site labels created for
`bath`. `mode=:star` mounts one boson leaf per fitted mode under the block
anchor; `mode=:chain` mounts one chain per block for cross-validation only.
`attach` may be omitted (first orbital in each block), a `Symbol` for a
single-block partition, a vector of symbols, a dictionary keyed by block index,
or a function `(block, i) -> site`.
"""
function mount_bath(topo::TreeTopology, bath::RealPoles, P::Partition;
                    mode::Symbol=:star, prefix::Symbol=:bath, attach=nothing)
    P.blocks == bath.blocks ||
        throw(ArgumentError("mount_bath: partition does not match RealPoles blocks"))
    mode in (:star, :chain) || throw(ArgumentError("mount_bath mode must be :star or :chain"))
    top = topo
    sites = Symbol[]
    anchors = Symbol[]
    block_sites = Vector{Vector{Symbol}}()
    for (bi, block) in enumerate(P.blocks)
        anchor = _block_anchor(block, bi, attach)
        haskey(top.index, anchor) ||
            throw(ArgumentError("mount_bath anchor $anchor is not present in the topology"))
        r = bath.block_ranges[bi]
        local_sites = Symbol[]
        if mode == :chain
            pref = Symbol(prefix, bi, :_)
            top = mount_chain(top, anchor, length(r); prefix=pref)
            for j in eachindex(r)
                site = Symbol(pref, j)
                push!(sites, site); push!(anchors, anchor); push!(local_sites, site)
            end
        else
            for j in eachindex(r)
                pref = Symbol(prefix, bi, :_, j, :_)
                top = mount_chain(top, anchor, 1; prefix=pref)
                site = Symbol(pref, 1)
                push!(sites, site); push!(anchors, anchor); push!(local_sites, site)
            end
        end
        push!(block_sites, local_sites)
    end
    return (; topology=top, sites, anchors, block_sites)
end

"""
    BosonBath(J; partition, topology, matter_ops, boson_ops, mode=:star, kwargs...)

Continuous T=0 boson-bath entry point. Fits `J(ω)` with [`fit_bath`](@ref),
mounts explicit boson sites with [`mount_bath`](@ref), and emits ordinary
symbolic terms via `boson_modes` and `BosonCoupling`. Returns
`(; bath, topology, sites, anchors, phys, H)`, where `phys` contains the new
bath-site physical spaces and `H` is an `OpSum`.
"""
function BosonBath(J; partition::Partition, topology::TreeTopology, matter_ops,
                   boson_ops, mode::Symbol=:star, density::Symbol=:N,
                   prefix::Symbol=:bath, attach=nothing, kwargs...)
    bath = fit_bath(J, partition; kwargs...)
    mounted = mount_bath(topology, bath, partition; mode, prefix, attach)
    g = couplings(bath)
    modes = [site => bath.poles[k] for (k, site) in enumerate(mounted.sites)]
    coupls = [(mounted.anchors[k], mounted.sites[k]) => g[k] for k in eachindex(g)]
    H = boson_modes(modes; ops=boson_ops) +
        BosonCoupling(coupls, :density; matter_ops, boson_ops, density)
    phys = Dict(site => boson_ops.P for site in mounted.sites)
    return (; bath, topology=mounted.topology, sites=mounted.sites,
            anchors=mounted.anchors, phys, H)
end

# ---------------------------------------------------------------------------
# §6.1 geometry constructors — thin wrappers over Trees.Geometries that mount
# bath branches according to the Partition (mechanical expansion, §6.2).
# TODO(M0): star/chain/fork assembly from (Partition, BathParametrization).
# Trees already provides: mps/star/binary/fork topologies, is_t3ns predicate.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# §6.4 measurements — G(τ) on sparse IR/DLR grids, G(t), F(t) improved
# estimators (Σ = F·G⁻¹), χ_ch(τ) two-particle (EDMFT W-loop), TRIQS BlockGf
# round-trip. TODO(M0–M2). §6.5 Spectral post-processing (LP/ESPRIT +
# complex-time Krylov Gram matrices) is milestone-1 scope but consumes evolver
# snapshots only — lands as a separate file once Evolution is validated. TODO.
# ---------------------------------------------------------------------------

# TODO(M0+): the single self-consistency-facing entry point (§6.6). GRAFT does
# NOT implement the DMFT/EDMFT loop itself. Contract: `ψ0` warm starts are
# first-class (topology hash validated — refuse silently rebuilt geometry);
# basis rotations `U` are returned with the results, loop side stays oblivious.
"""
    solve(bath, H_loc; partition, T, observables, ψ0=nothing) -> (; G, Σ, χ, ψ, U)

Impurity-solver entry point for self-consistency loops. TODO — no methods yet.
"""
function solve end

function _contiguous_ranges(rs::Vector{UnitRange{Int}})
    out = UnitRange{Int}[]
    next = 1
    for r in rs
        isempty(r) && throw(ArgumentError("RealPoles block ranges may not be empty"))
        first(r) == next || return out
        push!(out, r)
        next = last(r) + 1
    end
    return out
end

function _bound(name::String, ascii, unicode)
    v = unicode === nothing ? ascii : unicode
    ascii_name = name == "ωmin" ? "wmin" : "wmax"
    v === nothing && throw(ArgumentError("fit_bath requires `$name`/`$ascii_name`"))
    x = Float64(v)
    isfinite(x) || throw(ArgumentError("$name must be finite"))
    return x
end

function _block_targets(J, P::Partition)
    n = length(P.blocks)
    if J isa Function
        return [J for _ in 1:n]
    elseif J isa AbstractVector
        length(J) == n || throw(ArgumentError("vector-valued spectral density needs one function per partition block"))
        all(f -> f isa Function, J) || throw(ArgumentError("spectral-density vector entries must be functions"))
        return collect(J)
    else
        throw(ArgumentError("fit_bath expects a spectral-density function or one function per partition block"))
    end
end

function _grid_edges(lo::Float64, hi::Float64, nmodes::Int, grid::Symbol)
    if grid == :linear
        return collect(range(lo, hi; length=nmodes + 1))
    elseif grid == :log
        lo > 0 || throw(ArgumentError("log bath grid requires ωmin > 0"))
        return exp.(range(log(lo), log(hi); length=nmodes + 1))
    else
        throw(ArgumentError("unknown bath grid $grid; expected :linear or :log"))
    end
end

function _midpoint_modes(J::Function, lo::Float64, hi::Float64, nmodes::Int, grid::Symbol)
    edges = _grid_edges(lo, hi, nmodes, grid)
    poles = Float64[]
    residues = Float64[]
    for k in 1:nmodes
        a, b = edges[k], edges[k + 1]
        ω = grid == :log ? sqrt(a * b) : (a + b) / 2
        weight = Float64(real(J(ω))) * (b - a)
        weight >= -100eps(Float64) ||
            throw(ArgumentError("boson spectral density must be nonnegative; got J($ω) = $(J(ω))"))
        push!(poles, ω)
        push!(residues, max(weight, 0.0))
    end
    return poles, residues
end

function _moment(poles, residues, power::Int)
    s = 0.0
    for (ω, r) in zip(poles, residues)
        s += r * ω^power
    end
    return s
end

_reldiff(a, b) = abs(a - b) / max(abs(b), eps(Float64))

function _fit_diagnostics(J::Function, poles, residues, lo::Float64, hi::Float64,
                          nmodes::Int, grid::Symbol, block_index::Int)
    p2, r2 = _midpoint_modes(J, lo, hi, max(2nmodes, nmodes + 1), grid)
    m0 = _moment(poles, residues, 0)
    m1 = _moment(poles, residues, 1)
    ref0 = _moment(p2, r2, 0)
    ref1 = _moment(p2, r2, 1)
    return (;
        block_index,
        spectral_weight = m0,
        first_moment = m1,
        reference_nmodes = length(p2),
        rel_weight_change = _reldiff(m0, ref0),
        rel_first_moment_change = _reldiff(m1, ref1),
    )
end

function _block_anchor(block::Vector{Symbol}, i::Int, attach)
    if attach === nothing
        return first(block)
    elseif attach isa Symbol
        i == 1 || throw(ArgumentError("single Symbol attach is valid only for one block"))
        return attach
    elseif attach isa AbstractVector
        length(attach) >= i || throw(ArgumentError("attach vector has no entry for block $i"))
        attach[i] isa Symbol || throw(ArgumentError("attach entries must be Symbols"))
        return attach[i]
    elseif attach isa AbstractDict
        site = haskey(attach, i) ? attach[i] : get(attach, first(block), nothing)
        site isa Symbol || throw(ArgumentError("attach dictionary has no Symbol anchor for block $i"))
        return site
    elseif attach isa Function
        site = attach(block, i)
        site isa Symbol || throw(ArgumentError("attach function must return a Symbol"))
        return site
    else
        throw(ArgumentError("unsupported attach specification"))
    end
end

end # module Impurity
