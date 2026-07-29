# SD1/CAT0 — immutable typed build input (state-diagram-compiler plan v2,
# "Ownership and kernel separation").
#
# `TTNOBuildInput` is data only: the normalized Hamiltonian, topology,
# physical spaces, category context, typed Hermiticity contract, immutable
# coefficient table, and stable provenance metadata. It contains no mutable
# optimizer state, and it is consumed read-only by every lowering and merge
# kernel.

# ---------------------------------------------------------------------------
# Structural equality/hash helpers for the IR value types
# ---------------------------------------------------------------------------

"""
Field-wise structural equality. IR equality must never depend on mutable
object identity or TensorKit object identity; every IR type opts in through
`_register_ir_value_semantics`.
"""
function _ir_isequal(a::T, b::U) where {T,U}
    nameof(T) === nameof(U) || return false
    fieldnames(T) === fieldnames(U) || return false
    for name in fieldnames(T)
        isequal(getfield(a, name), getfield(b, name)) || return false
    end
    return true
end

function _ir_hash(x::T, h::UInt) where {T}
    h = hash(nameof(T), h)
    for name in fieldnames(T)
        h = hash(getfield(x, name), h)
    end
    return h
end

# ---------------------------------------------------------------------------
# Typed Hermiticity contract
# ---------------------------------------------------------------------------

"""
Typed Hermiticity contract of one build input. The contract is caller-owned
input data (§9.8): realization propagates it into the TTNO `ishermitian`
trait and records it in the build report. A wrong [`AssertedHermitian`](@ref)
is a caller bug; [`NoHermiticityAssertion`](@ref) is always safe and only
costs performance downstream.
"""
abstract type AbstractHermiticityContract end

"The caller asserts the realized operator is Hermitian."
struct AssertedHermitian <: AbstractHermiticityContract end

"No Hermiticity assertion; kernels must use conservative (Arnoldi) paths."
struct NoHermiticityAssertion <: AbstractHermiticityContract end

# ---------------------------------------------------------------------------
# Immutable coefficient table and typed coefficient slots
# ---------------------------------------------------------------------------

"""
    CoeffAtom(index)

Opaque reference to one Hamiltonian coefficient in the immutable
[`CoeffTable`](@ref). Floating coefficients are never interpreted by the
compiler: they remain opaque atoms combined with exact scalar multipliers
(restricted exact algebra, plan v2 "Exact coefficient and morphism algebra").
"""
struct CoeffAtom
    index::Int
end

"""
    CoeffTable(values)

Immutable coefficient table of one build input. Entry `i` is the coefficient
of the `i`-th normalized term; [`CoeffAtom`](@ref) indices resolve here at
realization time. The table is never mutated after construction.
"""
struct CoeffTable{C<:Number}
    values::Vector{C}
end

Base.length(table::CoeffTable) = length(table.values)

function coefficient_value(table::CoeffTable, atom::CoeffAtom)
    1 <= atom.index <= length(table.values) ||
        throw(ArgumentError("coefficient atom $(atom.index) is outside the table (1:$(length(table.values)))"))
    return table.values[atom.index]
end

"""
One typed coefficient slot on a term hyperedge. Exactly one hyperedge per
term expansion — the anchor — carries a [`CoeffAtomSlot`](@ref); every other
hyperedge carries an [`ExactUnitSlot`](@ref). All exact braid/frame scalars
live in the hyperedge's morphism certificate, never in the slot, so the slot
axis stays a pure coefficient-ownership contract.
"""
abstract type AbstractCoeffSlot end

"""
    ExactUnitSlot()

The exact scalar one. Carried by every non-anchor hyperedge of a term.
"""
struct ExactUnitSlot <: AbstractCoeffSlot end

"""
    CoeffAtomSlot(atom, scale)

The coefficient-owning anchor slot: opaque `atom` times the exact scalar
`scale`. Lowering emits `scale == one(C)`; later exact merge steps may fold
exact-proportionality factors into `scale`, never into the atom. For a
nonempty term the anchor hyperedge is the completion/LCA hyperedge; for a
constant or all-identity term it is the root hyperedge.
"""
struct CoeffAtomSlot{C<:Number} <: AbstractCoeffSlot
    atom::CoeffAtom
    scale::C
end

is_anchor_slot(::AbstractCoeffSlot) = false
is_anchor_slot(::CoeffAtomSlot) = true

# ---------------------------------------------------------------------------
# Local operator keys (fail-closed collision contract)
# ---------------------------------------------------------------------------

"""
    LocalOpKey(name, space_signature, charge)

Merge identity of one local operator factor. `SiteOp.name` alone is not
sufficient merge identity: the key combines the symbolic name with the
deterministic physical-space signature and the factor's injected charge.
Registering the same key for a different local map fails closed at input
construction ([`TTNOBuildInput`](@ref)); no silent first-wins registry
exists on this path.
"""
struct LocalOpKey{Q}
    name::Symbol
    space_signature::String
    charge::Q
end

Base.:(==)(a::LocalOpKey, b::LocalOpKey) = _ir_isequal(a, b)
Base.isequal(a::LocalOpKey, b::LocalOpKey) = _ir_isequal(a, b)
Base.hash(x::LocalOpKey, h::UInt) = _ir_hash(x, h)

_space_signature_string(P::ElementarySpace) = sprint(show, P)

local_op_key(so::SiteOp) = LocalOpKey(
    so.name, _space_signature_string(codomain(so.op)[1]), charge(so),
)

"Exact structural identity of two local maps: equal spaces and equal blocks."
function _same_local_map(a::AbstractTensorMap, b::AbstractTensorMap)
    codomain(a) == codomain(b) || return false
    domain(a) == domain(b) || return false
    ablocks = Dict(string(q) => bl for (q, bl) in blocks(a))
    bblocks = Dict(string(q) => bl for (q, bl) in blocks(b))
    keys(ablocks) == keys(bblocks) || return false
    for (q, bl) in ablocks
        isequal(bl, bblocks[q]) || return false
    end
    return true
end

"Deterministic content hash of one local map (spaces plus block values)."
function _local_map_hash(op::AbstractTensorMap, h::UInt)
    h = hash(_space_signature_string(codomain(op)[1]), h)
    h = hash(sprint(show, domain(op)), h)
    entries = sort!([(string(q), bl) for (q, bl) in blocks(op)]; by=first)
    for (q, bl) in entries
        h = hash(q, h)
        for v in bl
            h = hash(v, h)
        end
    end
    return h
end

"Whether a factor is an exact identity: neutral with an exact identity map."
function _is_exact_identity_op(so::SiteOp)
    q = charge(so)
    q === nothing || q == one(q) || return false
    op = so.op
    numout(op) == 1 && numin(op) == 1 || return false
    codomain(op)[1] == domain(op)[1] || return false
    for (_, bl) in blocks(op)
        for i in axes(bl, 1), j in axes(bl, 2)
            expected = i == j ? one(eltype(bl)) : zero(eltype(bl))
            bl[i, j] == expected || return false
        end
    end
    return true
end

# ---------------------------------------------------------------------------
# Normalized Hamiltonian terms
# ---------------------------------------------------------------------------

"""
    NormalizedFactor(node, key, op)

One local factor of a normalized term: canonical node index, fail-closed
[`LocalOpKey`](@ref), and the invariant local operator. Equality compares the
key and the exact map content, never TensorKit object identity.
"""
struct NormalizedFactor{Q}
    node::Int
    key::LocalOpKey{Q}
    op::SiteOp
end

Base.:(==)(a::NormalizedFactor, b::NormalizedFactor) =
    a.node == b.node && a.key == b.key && _same_local_map(a.op.op, b.op.op)
Base.isequal(a::NormalizedFactor, b::NormalizedFactor) = a == b
Base.hash(x::NormalizedFactor, h::UInt) =
    _local_map_hash(x.op.op, hash(x.key, hash(x.node, hash(:NormalizedFactor, h))))

"""
    NormalizedTerm(atom, factors, identity_only)

One normalized Hamiltonian term: its coefficient atom and its factors sorted
by canonical node index. `identity_only` records whether the term is a
constant (no factors) or every factor is an exact identity; such terms anchor
their coefficient at the root hyperedge.
"""
struct NormalizedTerm
    atom::CoeffAtom
    factors::Vector{NormalizedFactor}
    identity_only::Bool
end

Base.:(==)(a::NormalizedTerm, b::NormalizedTerm) = _ir_isequal(a, b)
Base.isequal(a::NormalizedTerm, b::NormalizedTerm) = _ir_isequal(a, b)
Base.hash(x::NormalizedTerm, h::UInt) = _ir_hash(x, h)

# ---------------------------------------------------------------------------
# Stable provenance
# ---------------------------------------------------------------------------

"""
    BuildInputProvenance(schema, digest)

Stable provenance identity of one build input: the IR schema version this
input was normalized under and the FNV-1a content digest of its canonical
serialization (topology, physical-space signatures, operator keys and exact
map bits, coefficient bits, Hermiticity contract, category profile). Plans
and reports carry this identity so realized TTNOs are traceable to their
exact input.
"""
struct BuildInputProvenance
    schema::Int
    digest::UInt64
end

Base.:(==)(a::BuildInputProvenance, b::BuildInputProvenance) = _ir_isequal(a, b)
Base.isequal(a::BuildInputProvenance, b::BuildInputProvenance) = _ir_isequal(a, b)
Base.hash(x::BuildInputProvenance, h::UInt) = _ir_hash(x, h)

provenance_hex(p::BuildInputProvenance) = string(p.digest; base=16, pad=16)

# ---------------------------------------------------------------------------
# TTNOBuildInput
# ---------------------------------------------------------------------------

"""
    TTNOBuildInput(H::OpSum, topo::TreeTopology, phys; hermiticity=NoHermiticityAssertion())

Immutable, data-only build input for the typed StateDiagram compiler. It
normalizes the Hamiltonian (canonical node-indexed factors, fail-closed
[`LocalOpKey`](@ref) collision checks, one immutable [`CoeffTable`](@ref)),
snapshots the physical spaces deterministically, derives the
[`CategorySemantics`](@ref) profile, and computes stable provenance. It
performs no lowering, no merging, and no category-capability gating: kernels
own capability failure so that every kernel sees the same input data.

The public `Term` vector is never an ordering signal: term factors are
re-keyed by canonical topology node index, and physical spaces are stored in
deterministic site order.
"""
struct TTNOBuildInput{S<:ElementarySpace,C<:Number,
                      H<:AbstractHermiticityContract,CS<:CategorySemantics}
    topology::TreeTopology
    phys::Vector{Pair{Symbol,S}}          # deterministic site order
    phys_lookup::Dict{Symbol,S}           # derived lookup; not identity-bearing
    terms::Vector{NormalizedTerm}
    coefficients::CoeffTable{C}
    operator_table::Vector{Pair{LocalOpKey,SiteOp}}  # deterministic key order
    hermiticity::H
    category::CS
    provenance::BuildInputProvenance
end

function Base.:(==)(a::TTNOBuildInput, b::TTNOBuildInput)
    a.topology == b.topology || return false
    a.phys == b.phys || return false
    a.terms == b.terms || return false
    isequal(a.coefficients.values, b.coefficients.values) || return false
    a.hermiticity == b.hermiticity || return false
    a.category == b.category || return false
    return a.provenance == b.provenance
end
Base.isequal(a::TTNOBuildInput, b::TTNOBuildInput) = a == b
Base.hash(x::TTNOBuildInput, h::UInt) =
    hash(x.provenance, hash(x.phys, hash(x.topology, hash(:TTNOBuildInput, h))))

function TTNOBuildInput(H::OpSum, topo::TreeTopology,
                        phys::Dict{Symbol,<:ElementarySpace};
                        hermiticity::AbstractHermiticityContract=NoHermiticityAssertion())
    isempty(H.terms) && throw(ArgumentError("empty OpSum"))
    isempty(phys) && throw(ArgumentError("phys must contain at least one physical space"))
    S = spacetype(first(values(phys)))
    all(P -> spacetype(P) === S, values(phys)) ||
        throw(ArgumentError("TTNO builder requires all physical spaces to share one concrete spacetype"))
    for site in keys(phys)
        haskey(topo.index, site) ||
            throw(ArgumentError("physical space declared on $site, which is not a topology node"))
    end

    phys_sorted = Pair{Symbol,S}[site => phys[site]
                                 for site in sort!(collect(keys(phys)); by=string)]
    phys_lookup = Dict{Symbol,S}(phys_sorted)

    C = mapreduce(term -> typeof(term.coeff), promote_type, H.terms)
    coeff_values = C[C(term.coeff) for term in H.terms]

    registry = Dict{LocalOpKey,SiteOp}()
    terms = NormalizedTerm[]
    for (i, term) in enumerate(H.terms)
        factors = NormalizedFactor[]
        for so in term.ops
            haskey(phys_lookup, so.site) ||
                throw(ArgumentError("term factor on $(so.site), which has no physical space"))
            spacetype(codomain(so.op)[1]) === S ||
                throw(ArgumentError("term factor on $(so.site) uses a physical-space symmetry incompatible with `phys`"))
            codomain(so.op)[1] == phys_lookup[so.site] ||
                throw(ArgumentError("term factor on $(so.site) acts on $(codomain(so.op)[1]), but phys declares $(phys_lookup[so.site])"))
            key = local_op_key(so)
            if haskey(registry, key)
                _same_local_map(registry[key].op, so.op) || throw(ArgumentError(
                    "local operator key collision: `$(key.name)` on space " *
                    "$(key.space_signature) with charge $(key.charge) is " *
                    "registered with a different local map; operator-key " *
                    "collisions fail closed (plan v2, canonical IR)",
                ))
            else
                registry[key] = so
            end
            push!(factors, NormalizedFactor(nodeindex(topo, so.site), key, so))
        end
        sort!(factors; by=f -> f.node)
        identity_only = all(_is_exact_identity_op(f.op) for f in factors)
        push!(terms, NormalizedTerm(CoeffAtom(i), factors, identity_only))
    end

    operator_table = Pair{LocalOpKey,SiteOp}[
        key => registry[key]
        for key in sort!(collect(keys(registry));
                         by=k -> (string(k.name), k.space_signature, string(k.charge)))
    ]

    category = category_semantics(S)
    input = TTNOBuildInput{S,C,typeof(hermiticity),typeof(category)}(
        topo, phys_sorted, phys_lookup, terms, CoeffTable(coeff_values),
        operator_table, hermiticity, category,
        BuildInputProvenance(0, UInt64(0)),
    )
    return _with_provenance(input)
end

input_physspace(input::TTNOBuildInput, site::Symbol) = input.phys_lookup[site]

input_spacetype(::TTNOBuildInput{S}) where {S} = S
input_coefftype(::TTNOBuildInput{S,C}) where {S,C} = C

"Sector type of the build input's category (Trivial for dense spaces)."
input_sectortype(input::TTNOBuildInput) = sectortype(input_spacetype(input))

# `_with_provenance` recomputes the stable digest through the canonical IR
# serializer (ir_serialization.jl); it is resolved at call time.
