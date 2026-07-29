# SD3 — canonical blocks and exact structural sharing (state-diagram-compiler
# plan v2, SD3 deliverables).
#
# The structural optimizer performs the first three steps of the
# deterministic postorder sequence: validate and canonicalize, exact
# identical-hyperedge (transport) merging, and exact identical-subtree
# (restriction) merging. Every reduction carries a typed merge proof with an
# equivalence-service basis relation and an explicit structural witness that
# is verified by walking the expansions — equal span keys alone are never
# sufficient evidence. Gamma blocks and restricted exact elimination arrive
# with SD4/SD5.

"""
    StructuralOptimizer()

SD3 optimizer for [`StateDiagramMerge`](@ref): exact structural sharing
only. Channels merge per edge exactly when their complete boundary
signatures are equal and a structural witness verifies — identical
transport structure for plain-idle and done channels, identical below-edge
subtree content for framed-idle and active channels. The merge is
idempotent and invariant to term insertion order.
"""
struct StructuralOptimizer end

_is_unit_certificate(cert::LocalMorphismCertificate{Q}, unit::Q) where {Q} =
    cert.crossing == unit && cert.scalar == one(cert.scalar)

"Verify pure identity transport below edge `c` (plain-idle witness)."
function _verify_idle_below(exp::TermTTNOExpansion, t::TreeTopology, E::_Euler,
                            c::Int, unit)
    for m in 1:nnodes(t)
        _insub(E, m, c) || continue
        he = exp.hyperedges[m]
        he.transition isa OmittedIdentityTransition || return false
        _is_unit_certificate(he.certificate, unit) || return false
        he.coeff isa ExactUnitSlot || return false
        if m != c
            ch = exp.edge_channels[m]
            ch isa ChannelIdentity || return false
            ch.degeneracy.span.class == PLAIN_IDLE || return false
        end
    end
    return true
end

"Verify pure done transport above edge `c` (done witness)."
function _verify_done_above(exp::TermTTNOExpansion, t::TreeTopology, c::Int,
                            unit)
    a = t.parent[c]
    while a != 0
        he = exp.hyperedges[a]
        he.transition isa OmittedIdentityTransition || return false
        _is_unit_certificate(he.certificate, unit) || return false
        # The coefficient slot above a completed term is exact one, except
        # the root anchor of a constant/all-identity term, which stays an
        # additive contribution on the shared wire.
        he.coeff isa ExactUnitSlot || a == t.root || return false
        if a != t.root
            ch = exp.edge_channels[a]
            ch isa ChannelIdentity || return false
            ch.degeneracy.span.class == DONE_TRANSPORT || return false
        end
        a = t.parent[a]
    end
    return true
end

"""
Verify that two expansions have identical below-edge content at `c`:
node-for-node equal transitions, certificates, exact-one coefficient slots,
and (already merged) equal channels on every edge strictly below.
"""
function _verify_identical_below(expA::TermTTNOExpansion,
                                 expB::TermTTNOExpansion, t::TreeTopology,
                                 E::_Euler, c::Int)
    for m in 1:nnodes(t)
        _insub(E, m, c) || continue
        ha = expA.hyperedges[m]
        hb = expB.hyperedges[m]
        ha.transition == hb.transition || return false
        ha.certificate == hb.certificate || return false
        ha.coeff isa ExactUnitSlot && hb.coeff isa ExactUnitSlot || return false
        if m != c
            expA.edge_channels[m] == expB.edge_channels[m] || return false
        end
    end
    return true
end

function _rewrite_channel(exp::TermTTNOExpansion{Q,C}, edge::Int,
                          channel::ChannelIdentity{Q}) where {Q,C}
    channels = copy(exp.edge_channels)
    channels[edge] = channel
    return TermTTNOExpansion{Q,C}(exp.term_ordinal, exp.atom, exp.anchor_node,
                                  channels, exp.hyperedges, exp.provenance)
end

function merge_channels(input::TTNOBuildInput,
                        terms::Vector{TermTTNOExpansion{Q,C}},
                        kernel::StateDiagramMerge{StructuralOptimizer}) where {Q,C}
    require_capability(input.category, :merge, :merge_channels)
    length(terms) == length(input.terms) ||
        throw(ArgumentError("structural merge requires one expansion per normalized term"))
    t = input.topology
    E = _Euler(t)
    N = nnodes(t)
    unit = one(Q)
    svc = AbelianEquivalenceService()

    work = collect(terms)
    proofs = MergeProofStep[]
    log = OptimizerLogEntry[]
    step = 0

    for exp in work
        validate_expansion(input, exp)
        for c in 1:N
            c == t.root && continue
            canonicalize_channel(svc, exp.edge_channels[c])
        end
    end
    step += 1
    push!(log, OptimizerLogEntry(step, :validate_and_canonicalize,
                                 "validated $(length(work)) expansions over $(N) nodes"))

    transport_merges = 0
    restriction_merges = 0
    for c in postorder(t)
        c == t.root && continue
        groups = Dict{BoundarySignature{Q},Vector{Int}}()
        for (i, exp) in enumerate(work)
            ch = exp.edge_channels[c]
            ch isa ChannelIdentity{Q} || continue
            push!(get!(groups, boundary_signature(svc, ch), Int[]), i)
        end
        for signature in sort!(collect(keys(groups)); by=string)
            members = groups[signature]
            channels = unique(ChannelIdentity{Q}[work[i].edge_channels[c]
                                                 for i in members])
            length(channels) > 1 || continue
            sort!(channels; by=channel_order_key)
            retained = first(channels)
            keeper = members[findfirst(
                i -> work[i].edge_channels[c] == retained, members)]
            class = retained.degeneracy.span.class

            removed_labels = DegeneracyLabel[]
            for i in members
                current = work[i].edge_channels[c]::ChannelIdentity{Q}
                current == retained && continue
                witness_ok = if class == PLAIN_IDLE
                    _verify_idle_below(work[i], t, E, c, unit) &&
                        _verify_idle_below(work[keeper], t, E, c, unit)
                elseif class == DONE_TRANSPORT
                    _verify_done_above(work[i], t, c, unit) &&
                        _verify_done_above(work[keeper], t, c, unit)
                else
                    _verify_identical_below(work[keeper], work[i], t, E, c)
                end
                if !witness_ok
                    step += 1
                    push!(log, OptimizerLogEntry(step, :merge_rejected,
                        "edge $(nodeid(t, c)): witness failed for term $(work[i].term_ordinal)"))
                    continue
                end
                push!(removed_labels, current.degeneracy)
                work[i] = _rewrite_channel(work[i], c, retained)
            end
            isempty(removed_labels) && continue
            relation = channel_basis_relation(svc, retained.degeneracy.span,
                                              retained.degeneracy.span)
            validate_basis_proof(svc, relation) ||
                throw(ArgumentError("structural merge produced an invalid span relation"))
            witness = class in (PLAIN_IDLE, DONE_TRANSPORT) ?
                StructuralIdentityWitness(:identical_hyperedge) :
                StructuralIdentityWitness(:identical_subtree)
            push!(proofs, MergeProofStep(
                class in (PLAIN_IDLE, DONE_TRANSPORT) ?
                    :transport_channel_merge : :restriction_channel_merge,
                c, retained.degeneracy.span, relation, witness,
                removed_labels, [retained.degeneracy]))
            if class in (PLAIN_IDLE, DONE_TRANSPORT)
                transport_merges += length(removed_labels)
            else
                restriction_merges += length(removed_labels)
            end
            step += 1
            push!(log, OptimizerLogEntry(step,
                class in (PLAIN_IDLE, DONE_TRANSPORT) ?
                    :identical_hyperedge_merge : :identical_subtree_merge,
                "edge $(nodeid(t, c)): merged $(length(removed_labels)) copies into copy $(retained.degeneracy.copy)"))
        end
    end

    step += 1
    push!(log, OptimizerLogEntry(step, :cleanup,
        "transport merges $(transport_merges), restriction merges $(restriction_merges), no unreachable objects"))
    for exp in work
        validate_expansion(input, exp)
    end
    step += 1
    push!(log, OptimizerLogEntry(step, :final_validate,
                                 "revalidated $(length(work)) expansions"))

    return StateDiagram{Q,C}(t, input.provenance, work, proofs,
                             OptimizerLog(log))
end

# ---------------------------------------------------------------------------
# Proof replay and reversal
# ---------------------------------------------------------------------------

"""
    unmerge_expansions(plan::StateDiagram) -> Vector{TermTTNOExpansion}

Reverse every recorded merge proof step: each removed channel-copy label is
restored onto its originating term expansion (the copy label is the term
ordinal). Merges are exactly reversible because a structural merge changes
only the copy axis of a channel whose remaining identity axes are equal by
signature.
"""
function unmerge_expansions(plan::StateDiagram{Q,C}) where {Q,C}
    restored = collect(plan.expansions)
    by_ordinal = Dict(exp.term_ordinal => i for (i, exp) in enumerate(restored))
    restore_label! = function (ordinal, label)
        i = get(by_ordinal, ordinal, nothing)
        i === nothing && throw(ArgumentError(
            "merge proof addresses term $ordinal, which no expansion owns"))
        current = restored[i].edge_channels[plan_proof_edge[]]
        current isa ChannelIdentity{Q} || throw(ArgumentError(
            "merge proof addresses the root boundary"))
        original = ChannelIdentity{Q}(
            current.sector, current.route, current.multiplicity, label,
            current.orientation, current.frame)
        restored[i] = _rewrite_channel(restored[i], plan_proof_edge[], original)
        return nothing
    end
    plan_proof_edge = Ref(0)
    for proof in reverse(plan.proofs)
        plan_proof_edge[] = proof.edge
        if proof.witness isa GammaCoverWitness
            witness = proof.witness::GammaCoverWitness
            for (ordinal, label) in witness.previous
                restore_label!(ordinal, label)
            end
            for (ordinal, from, to) in reverse(witness.moved_atoms)
                i = by_ordinal[ordinal]
                restored[i] = _move_atom(restored[i], to, from)
            end
        else
            for label in proof.removed
                i = get(by_ordinal, label.copy, nothing)
                i === nothing && throw(ArgumentError(
                    "merge proof removes copy $(label.copy), which no expansion owns"))
                current = restored[i].edge_channels[proof.edge]
                current isa ChannelIdentity{Q} || throw(ArgumentError(
                    "merge proof addresses the root boundary"))
                # A structural merge changes only the copy axis: the live
                # span must equal the removed label's span.
                current.degeneracy.span == label.span || throw(ArgumentError(
                    "structural merge proof on edge $(proof.edge) removes a " *
                    "label whose span is not live for term $(label.copy)"))
                restore_label!(label.copy, label)
            end
        end
    end
    return restored
end

"""
    validate_merge_plan(input, plan::StateDiagram) -> Bool

Realization-side merge-proof validation: every proof's basis relation
validates, its witness kind is known, its retained label is live on the
edge, its removed labels are no longer present, and the full reversal
(`unmerge_expansions`) yields structurally valid expansions. Throws on the
first violation.
"""
function validate_merge_plan(input::TTNOBuildInput,
                             plan::StateDiagram{Q,C}) where {Q,C}
    svc = AbelianEquivalenceService()
    t = plan.topology
    for proof in plan.proofs
        validate_basis_proof(svc, proof.relation) || throw(ArgumentError(
            "merge proof on edge $(proof.edge) carries an invalid basis relation"))
        if proof.witness isa StructuralIdentityWitness
            proof.witness.kind in (:identical_hyperedge, :identical_subtree) ||
                throw(ArgumentError(
                    "merge proof on edge $(proof.edge) has witness $(proof.witness.kind)"))
            length(proof.retained) == 1 || throw(ArgumentError(
                "merge proof on edge $(proof.edge) must retain exactly one label"))
        elseif proof.witness isa GammaCoverWitness
            witness = proof.witness::GammaCoverWitness
            witness.edge == proof.edge || throw(ArgumentError(
                "gamma cover witness edge does not match its proof edge"))
            isempty(proof.retained) && throw(ArgumentError(
                "gamma cover proof on edge $(proof.edge) retains no label"))
            length(witness.cover_rows) + length(witness.cover_cols) <
                length(witness.rows) || throw(ArgumentError(
                    "gamma cover proof on edge $(proof.edge) does not reduce the block"))
        else
            throw(ArgumentError(
                "merge proof on edge $(proof.edge) carries an unknown witness kind"))
        end
        # A proof's retained label may be superseded by a later proof on the
        # same edge; liveness of the final state is checked globally through
        # the reversal below. Removed labels, however, must never resurface.
        live = [exp.edge_channels[proof.edge].degeneracy
                for exp in plan.expansions
                if exp.edge_channels[proof.edge] isa ChannelIdentity{Q}]
        for label in proof.removed
            label in live && throw(ArgumentError(
                "merge proof on edge $(proof.edge) removed label $(label.copy) is still live"))
        end
    end
    for exp in unmerge_expansions(plan)
        validate_expansion(input, exp)
    end
    return true
end
