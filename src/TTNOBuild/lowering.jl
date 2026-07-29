# SD1/CAT0 — Abelian scalar lowering into the canonical IR.
#
# `lower_terms` reuses the SD0 braided-term machinery of statediagram.jl
# verbatim (`_Euler`, `_insub`, `_subframe`, `_build_braided_term_plan`,
# `_local_morphism_signature`) and records its certificate as lowering
# provenance. Legacy `ttno_from_opsum` assembly is unchanged; the IR path is
# additive and internal.

"Fusion-trace-ordered charged-factor sectors below (and including) node `c`."
function _charged_fusion_leaves(t::TreeTopology, ops::Dict{Int,SiteOp},
                                c::Int, unit::Q) where {Q}
    leaves = Q[]
    function trace(node::Int)
        for child in t.children[node]
            trace(child)
        end
        if haskey(ops, node)
            q = charge(ops[node])
            q === nothing || q == unit || push!(leaves, q)
        end
        return nothing
    end
    trace(c)
    return leaves
end

function _route_from_leaves(leaves::Vector{Q}, unit::Q) where {Q}
    intermediates = Q[]
    acc = unit
    for leaf in leaves
        acc = _fuse_charge(acc, leaf)
        push!(intermediates, acc)
    end
    return FusionRoute(leaves, intermediates)
end

function _frame_content(pairs::Tuple)
    return Tuple((entry.first, entry.second) for entry in pairs)
end

"""
Classify the channel of one term on the oriented edge below `c` and build its
complete identity tuple. Copies are labelled by the originating term ordinal
so the direct-sum plan mechanically retains every term channel.
"""
function _term_edge_channel(t::TreeTopology, E::_Euler, ops::Dict{Int,SiteOp},
                            keys_by_node::Dict{Int,<:LocalOpKey},
                            opnodes::Vector{Int}, tm_nodes::Vector{Int},
                            plan, unit::Q, ordinal::Int, c::Int,
                            ::Type{C}) where {Q,C<:Number}
    frame_pairs = plan === nothing ? () : _subframe(plan.frame, E, c, unit).at
    subop = [x for x in opnodes if _insub(E, x, c)]
    all_inside = !isempty(tm_nodes) && all(x -> _insub(E, x, c), tm_nodes)

    if all_inside
        span = ChannelSpan(DONE_TRANSPORT, ())
        return ChannelIdentity{Q}(
            unit, FusionRoute{Q}(), MultiplicityLabels(),
            DegeneracyLabel(span, ordinal), ChannelOrientation(),
            AbelianFrameCertificate(),
        )
    elseif isempty(subop)
        if isempty(frame_pairs)
            span = ChannelSpan(PLAIN_IDLE, ())
            return ChannelIdentity{Q}(
                unit, FusionRoute{Q}(), MultiplicityLabels(),
                DegeneracyLabel(span, ordinal), ChannelOrientation(),
                AbelianFrameCertificate(),
            )
        end
        content = _frame_content(frame_pairs)
        span = ChannelSpan(FRAMED_IDLE, content)
        return ChannelIdentity{Q}(
            unit, FusionRoute{Q}(), MultiplicityLabels(),
            DegeneracyLabel(span, ordinal), ChannelOrientation(),
            AbelianFrameCertificate(frame_pairs),
        )
    end

    # ACTIVE: equal labelled factors are mergeable only when their actual
    # local lowering agrees, so the restriction identity carries the exact
    # per-factor morphism signature next to the fail-closed operator key.
    restriction = if plan === nothing
        Tuple(sort!([(x, keys_by_node[x]) for x in subop]; by=first))
    else
        Tuple(sort!(
            [begin
                 sig = _local_morphism_signature(plan, ops, x, unit)
                 (x, keys_by_node[x], sig.crossing_charge, C(sig.scalar))
             end for x in subop];
            by=first,
        ))
    end
    sector = plan === nothing ? unit : plan.restriction_charge[c]
    leaves = _charged_fusion_leaves(t, ops, c, unit)
    span = ChannelSpan(ACTIVE, (restriction, _frame_content(frame_pairs)))
    return ChannelIdentity{Q}(
        sector, _route_from_leaves(leaves, unit), MultiplicityLabels(),
        DegeneracyLabel(span, ordinal), ChannelOrientation(),
        AbelianFrameCertificate(frame_pairs),
    )
end

function _lower_one_term(input::TTNOBuildInput, t::TreeTopology, E::_Euler,
                         unit::Q, graded::Bool, ordinal::Int,
                         term::NormalizedTerm, ::Type{C}) where {Q,C<:Number}
    ops = Dict{Int,SiteOp}(f.node => f.op for f in term.factors)
    keys_by_node = Dict{Int,LocalOpKey}(f.node => f.key for f in term.factors)
    opnodes = sort!(collect(keys(ops)))
    coeff = coefficient_value(input.coefficients, term.atom)

    plan = graded && !isempty(opnodes) ?
        _build_braided_term_plan(t, E, input.phys_lookup, ops, opnodes, unit,
                                 coeff) : nothing

    frame_nodes = plan === nothing ? Int[] :
        Int[entry.first for entry in plan.frame.at]
    tm_nodes = sort!(unique!(vcat(copy(opnodes), frame_nodes)))

    completion = if isempty(tm_nodes)
        t.root
    else
        lca = tm_nodes[1]
        while !all(x -> _insub(E, x, lca), tm_nodes)
            lca = t.parent[lca]
        end
        lca
    end
    anchor = term.identity_only ? t.root : completion

    N = nnodes(t)
    edge_channels = Vector{Union{RootBoundary,ChannelIdentity{Q}}}(undef, N)
    for c in 1:N
        edge_channels[c] = c == t.root ? RootBoundary() :
            _term_edge_channel(t, E, ops, keys_by_node, opnodes, tm_nodes,
                               plan, unit, ordinal, c, C)
    end

    hyperedges = Vector{TermHyperedge{Q,C}}(undef, N)
    for n in 1:N
        transition = haskey(ops, n) ?
            ExplicitLocalTransition(keys_by_node[n]) :
            OmittedIdentityTransition()
        certificate = if plan === nothing
            LocalMorphismCertificate{Q,C}(unit, one(C))
        else
            sig = _local_morphism_signature(plan, ops, n, unit)
            LocalMorphismCertificate{Q,C}(sig.crossing_charge, C(sig.scalar))
        end
        slot = n == anchor ? CoeffAtomSlot(term.atom, one(C)) : ExactUnitSlot()
        hyperedges[n] = TermHyperedge{Q,C}(n, transition, certificate, slot)
    end

    provenance = plan === nothing ?
        LoweringProvenance{C}(ordinal, Int[], Int[], true, one(C)) :
        LoweringProvenance{C}(ordinal, copy(plan.canonical_word),
                              copy(plan.native_word), plan.uses_certificate,
                              C(plan.certificate_scale))

    return TermTTNOExpansion{Q,C}(ordinal, term.atom, anchor, edge_channels,
                                  hyperedges, provenance)
end

function lower_terms(input::TTNOBuildInput, ::AbelianScalarLowering)
    require_capability(input.category, :lowering, :lower_terms)
    t = input.topology
    E = _Euler(t)
    S = input_spacetype(input)
    C = input_coefftype(input)
    Q = sectortype(S)
    unit = one(Q)
    graded = S !== ComplexSpace
    expansions = Vector{TermTTNOExpansion{Q,C}}(undef, length(input.terms))
    for (ordinal, term) in enumerate(input.terms)
        expansions[ordinal] =
            _lower_one_term(input, t, E, unit, graded, ordinal, term, C)
    end
    return expansions
end

# ---------------------------------------------------------------------------
# Structural IR validation
# ---------------------------------------------------------------------------

"Local injected charge of a hyperedge transition (unit for omitted/neutral)."
function _transition_charge(tr::AbstractLocalTransition, unit::Q) where {Q}
    tr isa ExplicitLocalTransition || return unit
    q = tr.key.charge
    q === nothing && return unit
    return q
end

"""
    validate_expansion(input, expansion) -> Bool

Structural and semantic validation of one lowered term expansion: hyperedge
indexing, exactly one coefficient-owning anchor slot at the recorded anchor
node, per-node sector conservation (children ⊗ local charge == parent
channel sector), and canonical routes on every channel. Throws
`ArgumentError` on the first violation.
"""
function validate_expansion(input::TTNOBuildInput,
                            exp::TermTTNOExpansion{Q,C}) where {Q,C}
    t = input.topology
    N = nnodes(t)
    svc = AbelianEquivalenceService()
    length(exp.edge_channels) == N ||
        throw(ArgumentError("expansion has $(length(exp.edge_channels)) edge channels for $N nodes"))
    length(exp.hyperedges) == N ||
        throw(ArgumentError("expansion has $(length(exp.hyperedges)) hyperedges for $N nodes"))
    exp.edge_channels[t.root] isa RootBoundary ||
        throw(ArgumentError("root edge slot must be the RootBoundary"))

    anchors = [n for n in 1:N if is_anchor_slot(exp.hyperedges[n].coeff)]
    anchors == [exp.anchor_node] || throw(ArgumentError(
        "expansion of term $(exp.term_ordinal) has anchor slots at $anchors " *
        "but records anchor node $(exp.anchor_node); the coefficient atom " *
        "must occur exactly once"))
    anchor_slot = exp.hyperedges[exp.anchor_node].coeff::CoeffAtomSlot
    anchor_slot.atom == exp.atom ||
        throw(ArgumentError("anchor slot atom does not match the expansion atom"))

    unit = one(Q)
    for n in 1:N
        he = exp.hyperedges[n]
        he.node == n ||
            throw(ArgumentError("hyperedge at slot $n records node $(he.node)"))
        fused = unit
        for c in t.children[n]
            ch = exp.edge_channels[c]
            ch isa ChannelIdentity ||
                throw(ArgumentError("non-root edge $c carries no channel"))
            canonicalize_channel(svc, ch)
            fused = _fuse_charge(fused, ch.sector)
        end
        fused = _fuse_charge(fused, _transition_charge(he.transition, unit))
        parent_sector = if n == t.root
            unit
        else
            ch = exp.edge_channels[n]
            ch isa ChannelIdentity ||
                throw(ArgumentError("non-root edge $n carries no channel"))
            ch.sector
        end
        fused == parent_sector || throw(ArgumentError(
            "sector conservation fails at node $(nodeid(t, n)): children and " *
            "local charge fuse to $fused but the parent channel carries " *
            "$parent_sector"))
    end
    return true
end
