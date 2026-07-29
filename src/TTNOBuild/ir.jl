# SD1/CAT0 — canonical categorical IR (state-diagram-compiler plan v2,
# "Canonical IR"; ADR-0003 decisions 2-3; representability audit
# evd-20260729T044231Z-1bda5fc82025 findings F1-F3).
#
# The edge-state channel identity tuple is
#     s_e = (q_e, r_e, mu_e, d_e, o_e, f_e)
# — sector, canonical fusion route, fusion-vertex multiplicity labels,
# channel-copy degeneracy label, orientation/dual, and braid/boundary frame
# certificate. Every axis is load-bearing from SD1 onward: unit-valued
# route, multiplicity, orientation, frame, and degeneracy fields participate
# in equality, hashing, and serialization, and channels differing in exactly
# one axis are unequal. Fusion multiplicity (mu_e) and channel-copy
# degeneracy (d_e) are different axes and never share one integer field.

# ---------------------------------------------------------------------------
# Route and multiplicity axes
# ---------------------------------------------------------------------------

"""
    FusionRoute(leaves, intermediates)

Canonical fusion route of one channel: the ordered charged-leaf sectors below
the edge and every intermediate sector along the route
(`intermediates[k]` is the fusion of `leaves[1:k]`; the last intermediate is
the channel sector). The canonical ordering convention of `leaves` is owned
by [`canonicalize_channel`](@ref), not by this field layout (audit finding
F3): a stored route is canonical only because the equivalence service
produced or validated it. In the Abelian profile the route is determined by
its leaves under unique fusion and is stored explicitly anyway — the field
is load-bearing for equality, hashing, and serialization.
"""
struct FusionRoute{Q}
    leaves::Vector{Q}
    intermediates::Vector{Q}
end

FusionRoute{Q}() where {Q} = FusionRoute(Q[], Q[])

"""
    MultiplicityLabels(labels)

Fusion-vertex multiplicity indices along a route (CAT2 axis). The Abelian
degenerate value is the empty label vector; the field and its interface are
retained so an explicit-multiplicity backend extends this axis without an IR
change. Never used to count channel copies — that is the separate
[`DegeneracyLabel`](@ref) axis.
"""
struct MultiplicityLabels
    labels::Vector{Int}
end

MultiplicityLabels() = MultiplicityLabels(Int[])

# ---------------------------------------------------------------------------
# Channel-copy degeneracy axis
# ---------------------------------------------------------------------------

"""
Semantic transport class of one channel in the Abelian state-diagram profile.
Explicit identity and omitted identity remain distinct classes: an
`ACTIVE` channel whose factors are explicit `:I` operators never collapses
into transport unless a proved canonicalization step merges them.
"""
@enum ChannelClass begin
    PLAIN_IDLE = 0
    FRAMED_IDLE = 1
    DONE_TRANSPORT = 2
    ACTIVE = 3
end

"""
    ChannelSpan(class, content)

Identity of one channel-copy degeneracy span: the transport class plus the
canonical span payload (the restriction identity for `ACTIVE` channels, the
frame payload for `FRAMED_IDLE`, empty for pure transport). Channels sharing
a span are copies of one semantic channel family; merge proofs address spans,
never bare copy indices.
"""
struct ChannelSpan
    class::ChannelClass
    content::Tuple
end

"""
    DegeneracyLabel(span, copy)

Channel-copy degeneracy label (audit axis A4): which span this channel
belongs to and which copy of that span it is. Lowering emits one copy per
originating term (`copy` = term ordinal), so the direct-sum plan mechanically
retains all term channels; structural and Gamma merges reduce copies only
with typed proofs. Distinct from fusion multiplicity (audit axis A3) by
construction.
"""
struct DegeneracyLabel
    span::ChannelSpan
    copy::Int
end

# ---------------------------------------------------------------------------
# Orientation and frame axes
# ---------------------------------------------------------------------------

"""
    ChannelOrientation(toward_root, dual)

Orientation/dual axis of one channel (audit axis A1): whether the channel is
read in the child-to-parent (toward-root) direction and whether the
underlying object is dualized. The Abelian degenerate value is
`ChannelOrientation(true, false)`; the axis is load-bearing in equality,
hashing, and serialization from SD1 onward.
"""
struct ChannelOrientation
    toward_root::Bool
    dual::Bool
end

ChannelOrientation() = ChannelOrientation(true, false)

"""
Braid/boundary frame certificate of one channel (audit axis A5). The SD0
braided-term certificate is the sole source of the existing fermionic sign
and frame semantics; new IR or assembly code never re-derives those signs.
This type is deliberately extensible (audit finding F2): a later braided
backend adds certificate kinds carrying twist data in addition to
braid/bend/boundary data as new subtypes — never by widening the Abelian
carrier below.
"""
abstract type AbstractFrameCertificate end

"""
    AbelianFrameCertificate(crossings)

The Abelian degenerate frame certificate: the SD0 fermionic crossing frame
only — sorted `(node => crossing_charge)` pairs recording which physical
inputs the canonical/native traversal reordering crosses fermionically. An
empty tuple is the trivial frame and still participates in equality and
hashing.
"""
struct AbelianFrameCertificate{A<:Tuple} <: AbstractFrameCertificate
    crossings::A
end

AbelianFrameCertificate() = AbelianFrameCertificate(())

# ---------------------------------------------------------------------------
# Channel identity tuple
# ---------------------------------------------------------------------------

"""
    ChannelIdentity(sector, route, multiplicity, degeneracy, orientation, frame)

The complete edge-state channel identity tuple
`s_e = (q_e, r_e, mu_e, d_e, o_e, f_e)`. Every axis participates in
equality, hashing, and serialization; two channels differing in exactly one
axis are unequal, and same charge or same outer sector is never sufficient
merge evidence. The materialized oriented edge space is *not* stored here:
it is produced behind [`materialize_virtual_space`](@ref) so that TensorKit
fusion internals never leak into IR identity.
"""
struct ChannelIdentity{Q}
    sector::Q
    route::FusionRoute{Q}
    multiplicity::MultiplicityLabels
    degeneracy::DegeneracyLabel
    orientation::ChannelOrientation
    frame::AbstractFrameCertificate
end

"Root boundary marker: the parent side of the root hyperedge is the category tensor unit."
struct RootBoundary end

"""
    channel_order_key(channel) -> Tuple

Stable semantic ordering key for channels on one edge: transport class,
canonical span payload, copy index, then the remaining identity axes.
Dictionary iteration order is never an ordering contract; every deterministic
layout sorts with this key.
"""
function channel_order_key(ch::ChannelIdentity)
    d = ch.degeneracy
    return (Int(d.span.class), string(d.span.content), d.copy,
            string(ch.sector), string(ch.route.leaves),
            string(ch.multiplicity.labels),
            ch.orientation.toward_root, ch.orientation.dual,
            string(ch.frame))
end

# ---------------------------------------------------------------------------
# Typed local transitions and morphism certificates
# ---------------------------------------------------------------------------

"""
Typed local transition of one term hyperedge. Explicit and omitted identity
are distinct transition classes: identity padding introduced by the state
diagram ([`OmittedIdentityTransition`](@ref)) never merges with an explicit
labelled factor — even an explicit `:I` — unless a proved canonicalization
step does it.
"""
abstract type AbstractLocalTransition end

"Identity padding introduced by the state diagram, not a Term factor."
struct OmittedIdentityTransition <: AbstractLocalTransition end

"""
    ExplicitLocalTransition(key)

One explicit labelled local factor, identified by its fail-closed
[`LocalOpKey`](@ref) (name + physical-space signature + charge).
"""
struct ExplicitLocalTransition{Q} <: AbstractLocalTransition
    key::LocalOpKey{Q}
end

"""
    LocalMorphismCertificate(crossing, scalar)

Exact typed morphism certificate of one hyperedge in the Abelian scalar
profile: the fused native crossing charge braided past the local physical
input, and the exact certificate scalar (braid word, branch duality,
completion bend, and CG-009 exit orientation contributions) owned by this
node under the SD0 braided-term plan. Realization consumes this certificate;
it never re-derives fermionic signs. Certificate scalars are exact values
(roots of unity in the supported profile) stored in the coefficient scalar
type.
"""
struct LocalMorphismCertificate{Q,C<:Number}
    crossing::Q
    scalar::C
end

# ---------------------------------------------------------------------------
# Term hyperedges and expansions
# ---------------------------------------------------------------------------

"""
    TermHyperedge(node, transition, certificate, coeff)

One hyperedge of a term expansion: the topology node, its typed local
transition, its exact morphism certificate, and its typed coefficient slot.
Its ordered child channel references are `topology.children[node]` into the
expansion's `edge_channels`, and its parent channel reference is
`edge_channels[node]` (the [`RootBoundary`](@ref) at the root).
"""
struct TermHyperedge{Q,C<:Number}
    node::Int
    transition::AbstractLocalTransition
    certificate::LocalMorphismCertificate{Q,C}
    coeff::AbstractCoeffSlot
end

"""
    LoweringProvenance(term_ordinal, canonical_word, native_word,
                       uses_certificate, certificate_scale)

Lowering provenance of one term expansion, connecting the SD0 braided-term
certificate: the canonical and native braid words over canonical node
indices, whether the exact certificate lowering (rather than the legacy
parity fallback) was used, and the exact global certificate scale. Recorded
without changing assembly — the permanent braided-term plan remains the sole
sign source.
"""
struct LoweringProvenance{C<:Number}
    term_ordinal::Int
    canonical_word::Vector{Int}
    native_word::Vector{Int}
    uses_certificate::Bool
    certificate_scale::C
end

"""
    TermTTNOExpansion

Canonical lowering output for one Hamiltonian term: one channel identity per
oriented tree edge (indexed by child node; the root slot is the
[`RootBoundary`](@ref)), one [`TermHyperedge`](@ref) per topology node, the
coefficient atom with its unique anchor node, and the lowering provenance.
`lower_terms` has exactly this output family for every lowering kernel.
"""
struct TermTTNOExpansion{Q,C<:Number}
    term_ordinal::Int
    atom::CoeffAtom
    anchor_node::Int
    edge_channels::Vector{Union{RootBoundary,ChannelIdentity{Q}}}
    hyperedges::Vector{TermHyperedge{Q,C}}
    provenance::LoweringProvenance{C}
end

# ---------------------------------------------------------------------------
# Boundary signatures, mixing keys, and span relations
# ---------------------------------------------------------------------------

"""
    BoundarySignature

Complete merge-evidence signature of one channel: everything in the identity
tuple except the copy index — oriented sector, full route, multiplicity
labels, channel-copy span, orientation, and frame certificate. Two channels
are structurally merge-compatible only when their boundary signatures are
equal; the actual merge still requires an exact witness.
"""
struct BoundarySignature{Q}
    sector::Q
    route::FusionRoute{Q}
    multiplicity::MultiplicityLabels
    span::ChannelSpan
    orientation::ChannelOrientation
    frame::AbstractFrameCertificate
end

"""
    MixingBlockKey

Gamma mixing-block key of one channel: the compatible cut hom-space identity
(sector, route, multiplicity, orientation, frame) without the channel-copy
span. Channels sharing this key may enter one Gamma block; same-charge or
same-outer-sector alone never suffices because route, multiplicity,
orientation, and frame all participate.
"""
struct MixingBlockKey{Q}
    sector::Q
    route::FusionRoute{Q}
    multiplicity::MultiplicityLabels
    orientation::ChannelOrientation
    frame::AbstractFrameCertificate
end

"""
Span-level basis relation between channel-copy spans. A relation transports
an entire compatible span into a common basis; it never reduces dimension.
The opaque span convention (audit finding F1): a span may address either the
channel-copy degeneracy axis or the fusion-multiplicity axis; in the Abelian
degenerate profile every span addresses the degeneracy axis only.
"""
abstract type AbstractSpanRelation end

"The identity basis relation of a span with itself."
struct IdentitySpanRelation <: AbstractSpanRelation
    span::ChannelSpan
end

"No basis relation is known between two spans; they must not be mixed."
struct NoKnownSpanRelation <: AbstractSpanRelation
    lhs::ChannelSpan
    rhs::ChannelSpan
end

# ---------------------------------------------------------------------------
# Value semantics registration
# ---------------------------------------------------------------------------

for T in (:CategorySemantics, :MissingCategoryCapability,
          :CategoryCapabilityReport, :CoeffAtom, :CoeffTable, :ExactUnitSlot,
          :CoeffAtomSlot, :FusionRoute, :MultiplicityLabels, :ChannelSpan,
          :DegeneracyLabel, :ChannelOrientation, :AbelianFrameCertificate,
          :ChannelIdentity, :RootBoundary, :OmittedIdentityTransition,
          :ExplicitLocalTransition, :LocalMorphismCertificate, :TermHyperedge,
          :LoweringProvenance, :TermTTNOExpansion, :BoundarySignature,
          :MixingBlockKey, :IdentitySpanRelation, :NoKnownSpanRelation)
    @eval begin
        Base.:(==)(a::$T, b::$T) = _ir_isequal(a, b)
        Base.isequal(a::$T, b::$T) = _ir_isequal(a, b)
        Base.hash(x::$T, h::UInt) = _ir_hash(x, h)
    end
end

# ---------------------------------------------------------------------------
# Abelian equivalence-service methods
# ---------------------------------------------------------------------------

function canonicalize_channel(svc::AbelianEquivalenceService,
                              ch::ChannelIdentity{Q}) where {Q}
    isempty(ch.multiplicity.labels) || throw(ArgumentError(
        "abelian channels carry no fusion-vertex multiplicity labels; " *
        "got $(ch.multiplicity.labels)"))
    expected = Q[]
    acc = one(Q)
    for leaf in ch.route.leaves
        acc = _fuse_charge(acc, leaf)
        push!(expected, acc)
    end
    ch.route.intermediates == expected || throw(ArgumentError(
        "channel route intermediates $(ch.route.intermediates) are not the " *
        "unique-fusion completion $(expected) of leaves $(ch.route.leaves); " *
        "canonical route ordering is owned by canonicalize_channel"))
    final = isempty(expected) ? one(Q) : last(expected)
    final == ch.sector || throw(ArgumentError(
        "channel route fuses to $final but the channel sector is $(ch.sector)"))
    return ch
end

boundary_signature(::AbelianEquivalenceService, ch::ChannelIdentity) =
    BoundarySignature(ch.sector, ch.route, ch.multiplicity,
                      ch.degeneracy.span, ch.orientation, ch.frame)

mixing_block_key(::AbelianEquivalenceService, ch::ChannelIdentity) =
    MixingBlockKey(ch.sector, ch.route, ch.multiplicity, ch.orientation,
                   ch.frame)

channel_basis_relation(::AbelianEquivalenceService, lhs::ChannelSpan,
                       rhs::ChannelSpan) =
    lhs == rhs ? IdentitySpanRelation(lhs) : NoKnownSpanRelation(lhs, rhs)

validate_basis_proof(::AbelianEquivalenceService, ::IdentitySpanRelation) = true
validate_basis_proof(::AbelianEquivalenceService, ::NoKnownSpanRelation) = false
