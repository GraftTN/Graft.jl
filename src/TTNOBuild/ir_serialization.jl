# SD1/CAT0 — versioned deterministic IR serialization (ADR-0003 decision 4).
#
# The IR serialization schema carries an explicit schema version and golden
# fixtures from SD1 onward, so any later CAT-driven field change is a visible
# schema bump reviewed as a CAT0 contract gap rather than applied silently.
# The format is a canonical S-expression text: no dictionary iteration, no
# object identity, exact bit-level floating-point rendering.

"""
Explicit IR serialization schema version. Any field addition, removal, or
rendering change of a serialized IR type must bump this constant and
regenerate the golden fixtures; the bump is reviewed as a CAT0 contract gap
(ADR-0003 decision 4).
"""
const IR_SCHEMA_VERSION = 1

"FNV-1a 64-bit content hash; process- and version-independent."
function _fnv1a64(data::AbstractVector{UInt8})
    h = 0xcbf29ce484222325
    for b in data
        h = (h ⊻ b) * 0x00000100000001b3
    end
    return h
end

# ---------------------------------------------------------------------------
# Canonical value writers
# ---------------------------------------------------------------------------

_ir_ser(io::IO, x::Int) = print(io, "i", x)
_ir_ser(io::IO, x::Bool) = print(io, x ? "#t" : "#f")
_ir_ser(io::IO, x::UInt64) = print(io, "u", string(x; base=16, pad=16))
_ir_ser(io::IO, x::Symbol) = print(io, "y", x)
_ir_ser(io::IO, x::String) = print(io, repr(x))
_ir_ser(io::IO, ::Nothing) = print(io, "nil")
_ir_ser(io::IO, x::Float64) =
    print(io, "f", string(reinterpret(UInt64, x); base=16, pad=16))
_ir_ser(io::IO, x::Real) = _ir_ser(io, Float64(x))
function _ir_ser(io::IO, x::Complex)
    print(io, "(c ")
    _ir_ser(io, Float64(real(x)))
    print(io, " ")
    _ir_ser(io, Float64(imag(x)))
    print(io, ")")
end
_ir_ser(io::IO, x::ChannelClass) = print(io, "e", Int(x))

function _ir_ser(io::IO, xs::Union{Tuple,AbstractVector})
    print(io, xs isa Tuple ? "(t" : "(v")
    for x in xs
        print(io, " ")
        _ir_ser(io, x)
    end
    print(io, ")")
end

function _ir_ser(io::IO, x::Pair)
    print(io, "(p ")
    _ir_ser(io, x.first)
    print(io, " ")
    _ir_ser(io, x.second)
    print(io, ")")
end

# Sectors, category profiles, Hermiticity singletons, and other leaf values
# render through `repr`. A rendering change (for example a TensorKit sector
# `show` change) alters fixture bytes and therefore surfaces as a schema
# review, which is the intended visibility.
_ir_ser(io::IO, x) = print(io, "{", repr(x), "}")

function _ir_ser_struct(io::IO, x)
    T = typeof(x)
    print(io, "(", nameof(T))
    for name in fieldnames(T)
        print(io, " (", name, " ")
        _ir_ser(io, getfield(x, name))
        print(io, ")")
    end
    print(io, ")")
end

for T in (:CategorySemantics, :MissingCategoryCapability, :CoeffAtom,
          :CoeffTable, :ExactUnitSlot, :CoeffAtomSlot, :LocalOpKey,
          :BuildInputProvenance, :FusionRoute, :MultiplicityLabels,
          :ChannelSpan, :DegeneracyLabel, :ChannelOrientation,
          :AbelianFrameCertificate, :ChannelIdentity, :RootBoundary,
          :OmittedIdentityTransition, :ExplicitLocalTransition,
          :LocalMorphismCertificate, :TermHyperedge, :LoweringProvenance,
          :TermTTNOExpansion, :BoundarySignature, :MixingBlockKey,
          :IdentitySpanRelation, :NoKnownSpanRelation,
          :StructuralIdentityWitness, :GammaCoverWitness, :MergeProofStep,
          :OptimizerLogEntry, :OptimizerLog, :NormalizedTerm)
    @eval _ir_ser(io::IO, x::$T) = _ir_ser_struct(io, x)
end

function _ir_ser(io::IO, x::TreeTopology)
    print(io, "(TreeTopology (ids ")
    _ir_ser(io, x.ids)
    print(io, ") (parent ")
    _ir_ser(io, x.parent)
    print(io, ") (children ")
    _ir_ser(io, x.children)
    print(io, "))")
end

"Exact content rendering of one local map: spaces plus sorted block bits."
function _ir_ser_localmap(io::IO, op::AbstractTensorMap)
    print(io, "(localmap (cod ")
    _ir_ser(io, sprint(show, codomain(op)))
    print(io, ") (dom ")
    _ir_ser(io, sprint(show, domain(op)))
    print(io, ") (blocks")
    entries = sort!([(string(q), bl) for (q, bl) in blocks(op)]; by=first)
    for (q, bl) in entries
        print(io, " (", "sector ")
        _ir_ser(io, q)
        print(io, " ")
        _ir_ser(io, ComplexF64.(vec(collect(bl))))
        print(io, ")")
    end
    print(io, "))")
end

function _ir_ser(io::IO, x::SiteOp)
    print(io, "(SiteOp (site ")
    _ir_ser(io, x.site)
    print(io, ") (name ")
    _ir_ser(io, x.name)
    print(io, ") (charge ")
    _ir_ser(io, charge(x))
    print(io, ") ")
    _ir_ser_localmap(io, x.op)
    print(io, ")")
end

function _ir_ser(io::IO, x::NormalizedFactor)
    print(io, "(NormalizedFactor (node ")
    _ir_ser(io, x.node)
    print(io, ") (key ")
    _ir_ser(io, x.key)
    print(io, ") (op ")
    _ir_ser(io, x.op)
    print(io, "))")
end

function _ir_ser(io::IO, x::TTNOBuildInput)
    print(io, "(TTNOBuildInput (schema ", IR_SCHEMA_VERSION, ") ")
    _ir_ser(io, x.topology)
    print(io, " (phys ")
    _ir_ser(io, x.phys)
    print(io, ") (terms ")
    _ir_ser(io, x.terms)
    print(io, ") (coefficients ")
    _ir_ser(io, x.coefficients)
    print(io, ") (hermiticity ")
    _ir_ser(io, x.hermiticity)
    print(io, ") (category ")
    _ir_ser(io, x.category)
    print(io, "))")
end

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

"""
    serialize_ir(x) -> String

Canonical deterministic serialization of one IR value, wrapped in the
versioned envelope. Equal IR values serialize to identical bytes; unequal
values differ. Golden fixtures pin these bytes from SD1 onward.
"""
function serialize_ir(x)
    buf = IOBuffer()
    print(buf, "(graft-ttno-ir (schema ", IR_SCHEMA_VERSION, ")\n ")
    _ir_ser(buf, x)
    print(buf, "\n)\n")
    return String(take!(buf))
end

"""
    serialize_expansions(input, expansions) -> String

Canonical serialization of a lowering result: the versioned envelope, the
input provenance, and every term expansion in term order.
"""
function serialize_expansions(input::TTNOBuildInput,
                              expansions::Vector{<:TermTTNOExpansion})
    buf = IOBuffer()
    print(buf, "(graft-ttno-ir (schema ", IR_SCHEMA_VERSION, ") (input ")
    _ir_ser(buf, input.provenance)
    print(buf, ")\n")
    for exp in expansions
        print(buf, " ")
        _ir_ser(buf, exp)
        print(buf, "\n")
    end
    print(buf, ")\n")
    return String(take!(buf))
end

"Recompute the stable input provenance from the canonical serialization."
function _with_provenance(input::TTNOBuildInput{S,C,H,CS}) where {S,C,H,CS}
    buf = IOBuffer()
    _ir_ser(buf, input)
    digest = _fnv1a64(take!(buf))
    return TTNOBuildInput{S,C,H,CS}(
        input.topology, input.phys, input.phys_lookup, input.terms,
        input.coefficients, input.operator_table, input.hermiticity,
        input.category, BuildInputProvenance(IR_SCHEMA_VERSION, digest),
    )
end
