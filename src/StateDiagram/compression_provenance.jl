# CP1d — compiler-side exact channel provenance for TTNO compression
# (ttno-compression plan v2, CP1a/CP1d integration seam).
#
# The emitter walks a direct-sum plan and certifies exact proportional
# channel relations from build knowledge: two channels whose below-cut
# structure is identical except the coefficient anchor realize columns that
# are algebraically proportional with the ratio of their evaluated anchor
# values. Compression consumes the result through the narrow
# `TTNOExactProvenance` type without seeing any diagram IR.

"Below-content key with the anchor opaque and channel copies normalized."
function _provenance_below_key(exp::TermTTNOExpansion{Q}, t::TreeTopology,
                               E::_Euler, c::Int) where {Q}
    buf = IOBuffer()
    for m in 1:nnodes(t)
        _insub(E, m, c) || continue
        he = exp.hyperedges[m]
        print(buf, "(n", m, " ")
        _ir_ser(buf, he.transition)
        _ir_ser(buf, he.certificate)
        he.coeff isa CoeffAtomSlot ? print(buf, "(anchor)") :
            _ir_ser(buf, he.coeff)
        if m != c
            ch = exp.edge_channels[m]::ChannelIdentity{Q}
            # A direct-sum plan labels every copy with its term ordinal;
            # the structural comparison is modulo that copy index.
            _ir_ser(buf, ChannelIdentity{Q}(
                ch.sector, ch.route, ch.multiplicity,
                DegeneracyLabel(ch.degeneracy.span, 0), ch.orientation,
                ch.frame))
        end
        print(buf, ")")
    end
    return String(take!(buf))
end

"""
    compiler_exact_provenance(input, plan::DirectSumPlan) -> TTNOExactProvenance

Certify exact proportional channel relations of the realized direct-sum
TTNO from build knowledge: channels owned by completed terms whose
below-cut content is identical except the coefficient anchor are
proportional columns with factor `value_removed / value_kept` (evaluated
anchor slots). Column indices address the per-sector block columns in the
deterministic realization order. The result feeds `compress!`'s exact
Stage 1; compression still validates every relation against the data and
falls back to retention on disagreement.
"""
function compiler_exact_provenance(input::TTNOBuildInput,
                                   plan::DirectSumPlan{Q,C}) where {Q,C}
    plan.provenance == input.provenance || throw(ArgumentError(
        "plan provenance does not match the build input"))
    plan_view = realization_view(plan)
    t = plan.topology
    E = _Euler(t)
    by_ordinal = Dict(exp.term_ordinal => exp for exp in plan.expansions)
    relations = TTNOExactChannelRelation{Q}[]
    for c in 1:nnodes(t)
        c == t.root && continue
        channels = plan_view.edge_channels[c]
        sector_count = Dict{Q,Int}()
        groups = Dict{Tuple{String,String},Vector{Tuple{Int,Number}}}()
        for ch in channels
            local_index = (sector_count[ch.sector] =
                get(sector_count, ch.sector, 0) + 1)
            exp = get(by_ordinal, ch.degeneracy.copy, nothing)
            exp === nothing && continue
            _insub(E, exp.anchor_node, c) || continue
            slot = exp.hyperedges[exp.anchor_node].coeff::CoeffAtomSlot
            value = _slot_value(input.coefficients, slot)
            iszero(value) && continue
            key = (string(ch.sector), _provenance_below_key(exp, t, E, c))
            push!(get!(groups, key, Tuple{Int,Number}[]),
                  (local_index, value))
        end
        for key in sort!(collect(keys(groups)))
            members = groups[key]
            length(members) > 1 || continue
            sort!(members; by=first)
            kept, kept_value = first(members)
            sector = first([ch.sector for ch in channels
                            if string(ch.sector) == key[1]])
            for (removed, value) in members[2:end]
                push!(relations, TTNOExactChannelRelation{Q}(
                    nodeid(t, c), sector, kept, removed,
                    ComplexF64(value / kept_value)))
            end
        end
    end
    return TTNOExactProvenance{Q}(relations)
end
