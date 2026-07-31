# SD1/CAT0 — category semantics, capability matrix, and the two narrow
# category services (state-diagram-compiler plan v2, "Category services";
# ADR-0003 decisions 1 and 3).
#
# StateDiagram core never imports TensorKit fusion internals. Its complete
# coupling surface to the categorical backend is:
#   * the equivalence and basis service:      canonicalize_channel,
#     boundary_signature, channel_basis_relation, mixing_block_key,
#     validate_basis_proof;
#   * the TensorKit materialization service:  tensor_unit, dual_object,
#     materialize_virtual_space, materialize_morphism, validate_morphism.
# Anything a later CAT1a-CAT4 backend needs must arrive as an additive method
# on these ten operations (or an additive certificate kind), never as a
# change to the IR channel-identity axes.

# ---------------------------------------------------------------------------
# Category semantics trait product (CAT4 seam: fusion × braiding)
# ---------------------------------------------------------------------------

"Fusion axis of the category profile product. See [`CategorySemantics`](@ref)."
abstract type FusionProfile end

"Abelian fusion: every sector pair has exactly one fusion outcome."
struct UniqueFusionProfile <: FusionProfile end

"Non-abelian multiplicity-free fusion (CAT1a). Nameable but unsupported here."
struct MultiplicityFreeFusionProfile <: FusionProfile end

"Fusion with explicit vertex multiplicities (CAT2). Nameable but unsupported here."
struct GenericFusionProfile <: FusionProfile end

"Braiding axis of the category profile product. See [`CategorySemantics`](@ref)."
abstract type BraidingProfile end

"Symmetric braiding (bosonic or fermionic): the braiding squares to identity."
struct SymmetricBraidingProfile <: BraidingProfile end

"Nontrivial (anyonic) braiding (CAT3). Nameable but unsupported here."
struct AnyonicBraidingProfile <: BraidingProfile end

"No braiding available. Nameable but unsupported here."
struct NoBraidingProfile <: BraidingProfile end

"""
    CategorySemantics(fusion, braiding)

The category profile of one TTNO build as a fusion × braiding trait product.
This product spans exactly the CAT1-CAT4 profile table of the (frozen)
categorical-backends plan; no field or planned type is named after SU(2), a
concrete anyon theory, or an operator label, so composing the axes never
requires a named-category branch. The first (and only currently supported)
production profile is `UniqueFusion + SymmetricBraiding`.
"""
struct CategorySemantics{F<:FusionProfile,B<:BraidingProfile}
    fusion::F
    braiding::B
end

"""
    category_semantics(::Type{S}) -> CategorySemantics

Derive the category profile of an elementary-space type through the Backend
adapter. `ComplexSpace` (trivial sector) is unique-fusion symmetric-braiding.
"""
function category_semantics(::Type{S}) where {S<:ElementarySpace}
    Q = sectortype(S)
    fusion = _fusion_profile(sector_fusion_symbol(Q))
    braiding = _braiding_profile(sector_braiding_symbol(Q))
    return CategorySemantics(fusion, braiding)
end

_fusion_profile(sym::Symbol) =
    sym === :unique ? UniqueFusionProfile() :
    sym === :multiplicity_free ? MultiplicityFreeFusionProfile() :
    GenericFusionProfile()

_braiding_profile(sym::Symbol) =
    sym === :symmetric ? SymmetricBraidingProfile() :
    sym === :anyonic ? AnyonicBraidingProfile() :
    NoBraidingProfile()

"Whether this profile is the supported Abelian production profile."
is_supported_profile(cs::CategorySemantics) =
    cs.fusion isa UniqueFusionProfile && cs.braiding isa SymmetricBraidingProfile

# ---------------------------------------------------------------------------
# Fail-closed capability results
# ---------------------------------------------------------------------------

"""
    MissingCategoryCapability(service, operation, requirement, detail)

Typed fail-closed result naming exactly which capability an unsupported
category profile lacks. `service` is one of `:lowering`, `:equivalence`,
`:merge`, or `:materialization`; `operation` names the specific kernel or
service operation; `requirement` names the missing profile capability (for
example `:UniqueFusion`). Raised as an exception before any densification and
recorded verbatim in capability reports.
"""
struct MissingCategoryCapability <: Exception
    service::Symbol
    operation::Symbol
    requirement::Symbol
    detail::String
end

function Base.showerror(io::IO, err::MissingCategoryCapability)
    print(io, "MissingCategoryCapability: service=", err.service,
          " operation=", err.operation, " missing=", err.requirement,
          ": ", err.detail)
end

"""
    CategoryCapabilityReport

The supported-capability matrix evaluated for one category profile: every
missing capability appears as a [`MissingCategoryCapability`](@ref) entry.
`supported` is true iff `missing` is empty. This report is data; kernels
additionally throw the first missing entry when asked to run anyway.
"""
struct CategoryCapabilityReport{CS<:CategorySemantics}
    semantics::CS
    missing::Vector{MissingCategoryCapability}
end

is_supported(report::CategoryCapabilityReport) = isempty(report.missing)

_profile_name(::UniqueFusionProfile) = :UniqueFusion
_profile_name(::MultiplicityFreeFusionProfile) = :MultiplicityFreeFusion
_profile_name(::GenericFusionProfile) = :GenericFusion
_profile_name(::SymmetricBraidingProfile) = :SymmetricBraiding
_profile_name(::AnyonicBraidingProfile) = :AnyonicBraiding
_profile_name(::NoBraidingProfile) = :NoBraiding

"""
    capability_report(semantics::CategorySemantics) -> CategoryCapabilityReport

Evaluate the supported Abelian capability matrix. The supported profile is
`UniqueFusion + SymmetricBraiding`; every other fusion or braiding axis value
is nameable and fails closed with the owning service and operation that first
lacks it. The matrix rows are additive: a future CAT backend extends support
by adding service methods, never by weakening a row here.
"""
function capability_report(semantics::CategorySemantics)
    missing = MissingCategoryCapability[]
    if !(semantics.fusion isa UniqueFusionProfile)
        requirement = :UniqueFusion
        got = _profile_name(semantics.fusion)
        push!(missing, MissingCategoryCapability(
            :lowering, :lower_terms, requirement,
            "fusion profile $got requires a route-aware lowering kernel; " *
            "only the UniqueFusion abelian lowering exists",
        ))
        push!(missing, MissingCategoryCapability(
            :equivalence, :channel_basis_relation, requirement,
            "fusion profile $got requires F-move span basis relations; " *
            "only identity span relations exist",
        ))
        push!(missing, MissingCategoryCapability(
            :materialization, :materialize_morphism, requirement,
            "fusion profile $got requires fusion-tree morphism " *
            "materialization; only abelian block materialization exists",
        ))
    end
    if !(semantics.braiding isa SymmetricBraidingProfile)
        requirement = :SymmetricBraiding
        got = _profile_name(semantics.braiding)
        push!(missing, MissingCategoryCapability(
            :lowering, :lower_terms, requirement,
            "braiding profile $got requires braid-word frame certificates " *
            "beyond the symmetric fermionic frame",
        ))
        push!(missing, MissingCategoryCapability(
            :merge, :merge_channels, requirement,
            "braiding profile $got requires braided merge-compatibility " *
            "evidence; only symmetric-braiding merges exist",
        ))
    end
    return CategoryCapabilityReport(semantics, missing)
end

"""
    require_capability(semantics, service, operation)

Fail closed before densification: throw the first
[`MissingCategoryCapability`](@ref) charged to `service`/`operation` (or the
first missing capability at all) when `semantics` is unsupported.
"""
function require_capability(semantics::CategorySemantics, service::Symbol,
                            operation::Symbol)
    report = capability_report(semantics)
    is_supported(report) && return report
    for entry in report.missing
        if entry.service === service && entry.operation === operation
            throw(entry)
        end
    end
    throw(first(report.missing))
end

# ---------------------------------------------------------------------------
# Equivalence and basis service (5 operations)
# ---------------------------------------------------------------------------

"""
Equivalence and basis service seam. It canonicalizes channels, provides
complete boundary signatures and mixing-block keys, returns span-level basis
relations, and validates their proofs. A basis isomorphism only transports an
entire compatible span into a common basis; it never reduces dimension, and
every actual merge additionally requires an exact structural or exact
linear-dependence witness.
"""
abstract type AbstractEquivalenceService end

"""
    AbelianEquivalenceService()

The equivalence service for the supported `UniqueFusion + SymmetricBraiding`
profile. All route data is determined by ordered leaves under unique fusion;
span basis relations are identity relations.
"""
struct AbelianEquivalenceService <: AbstractEquivalenceService end

"""
    canonicalize_channel(svc, channel) -> channel

Return the canonical form of a channel identity. The canonical route ordering
convention is owned by this operation, not by IR field layout (audit finding
F3): consumers must treat a stored route as canonical only because this
operation produced or validated it, and any later backend may canonicalize
differently without an IR field change. The abelian method validates that the
stored route is the unique-fusion completion of its ordered leaves and
returns the channel unchanged.
"""
function canonicalize_channel end

"""
    boundary_signature(svc, channel) -> BoundarySignature

The complete merge-evidence signature of one channel: oriented sector, full
fusion route, fusion multiplicity labels, channel-copy span, and braid/frame
certificate — everything except the copy index inside the span. Same charge
or same outer sector is never sufficient evidence for a merge; two channels
may share Gamma mixing blocks yet have different boundary signatures.
"""
function boundary_signature end

"""
    channel_basis_relation(svc, lhs_span, rhs_span) -> AbstractSpanRelation

Span-level basis relation between two channel-copy spans. A relation
transports an entire compatible span into a common basis and never reduces
dimension. The abelian method knows exactly one relation: the identity
relation on a span equal to itself; every other pair yields
[`NoKnownSpanRelation`](@ref). Nontrivial F-move transports are additive
future methods (CAT1b+).
"""
function channel_basis_relation end

"""
    mixing_block_key(svc, channel) -> MixingBlockKey

The Gamma mixing-block key of a channel: the complete compatible cut
hom-space identity (sector, route, multiplicity, orientation, frame) without
the channel-copy span. Channels sharing a mixing-block key may participate in
one Gamma block; reducing among them still requires exact witnesses.
"""
function mixing_block_key end

"""
    validate_basis_proof(svc, relation) -> Bool

Validate a span basis relation proof. The abelian method accepts exactly the
identity relation between equal spans and rejects everything else.
"""
function validate_basis_proof end

# ---------------------------------------------------------------------------
# TensorKit materialization service (5 operations)
# ---------------------------------------------------------------------------

"""
Materialization service seam: the only path from IR data to TensorKit
objects. Provides the tensor unit, dual objects, virtual-space construction,
morphism materialization, and morphism validation.
"""
abstract type AbstractMaterializationService end

"""
    TensorKitMaterializationService()

Materialization through the Backend TensorKit adapter for the supported
Abelian profile.
"""
struct TensorKitMaterializationService <: AbstractMaterializationService end

"""
    tensor_unit(svc, ::Type{S}) -> ElementarySpace

The category tensor unit as an elementary space of kind `S`. The realized
TTNO root boundary must equal this object.
"""
function tensor_unit end

tensor_unit(::TensorKitMaterializationService, ::Type{S}) where {S<:ElementarySpace} =
    trivialspace(S)

"""
    dual_object(svc, V) -> ElementarySpace

The dual object of `V` under the backend's duality convention.
"""
function dual_object end

dual_object(::TensorKitMaterializationService, V::ElementarySpace) = dual(V)

"""
    materialize_virtual_space(svc, ::Type{S}, channels) -> (space, coords)

Materialize the oriented virtual edge space carrying an ordered channel list,
returning the backend space plus the dense block coordinate of every channel
(index into its sector block, in the given deterministic order). The channel
order is a contract of the caller; this operation never reorders channels
within a sector. Implemented with the direct-sum realization (SD2).
"""
function materialize_virtual_space end

"""
    materialize_morphism(svc, spec) -> AbstractTensorMap

Materialize one invariant local morphism from its typed specification.
Implemented with the direct-sum realization (SD2).
"""
function materialize_morphism end

"""
    validate_morphism(svc, morphism, spec) -> Bool

Validate a materialized morphism against its typed specification.
Implemented with the direct-sum realization (SD2).
"""
function validate_morphism end
