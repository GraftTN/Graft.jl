# SD4/SD5 — raw Gamma blocks, minimum vertex cover, restricted exact
# elimination, and proof-backed cover selection (state-diagram-compiler
# plan v2, SD4 and SD5 deliverables).
#
# Every child-to-parent cut is partitioned into categorical Gamma blocks
# keyed by the complete compatible cut hom-space (the equivalence service's
# mixing-block key: sector, route, multiplicity, orientation, frame; the
# exact scalar domain is the shared coefficient scalar type). Inside each
# block the raw exact Gamma relates lossless below-content rows to lossless
# above-context columns; a local deterministic Hopcroft-Karp maximum
# matching and the Konig construction give minimum vertex covers. With the
# SD5 optimizer the raw and restricted-elimination covers are computed
# independently and the eliminated result is selected only when strictly
# smaller — raw wins every tie. Coefficient atoms stay opaque throughout:
# reconstruction relocates atom slots along their own transition paths and
# introduces only ±1 exact scalar slots.

"""
    GammaCoverOptimizer()

SD4 optimizer for [`StateDiagramMerge`](@ref): the SD3 structural phases
followed by raw-Gamma minimum-cover merging on every cut, bottom-up in
postorder. No elimination runs; see [`SGEOptimizer`](@ref).
"""
struct GammaCoverOptimizer end

"""
    SGEOptimizer()

SD5 optimizer for [`StateDiagramMerge`](@ref): the SD3 structural phases,
then per cut and block the raw Gamma cover and the restricted-exact-
elimination cover computed independently, selecting the eliminated
reconstruction only when its cover is strictly smaller (raw wins ties).
"""
struct SGEOptimizer end

# ---------------------------------------------------------------------------
# Deterministic Hopcroft-Karp and Konig cover
# ---------------------------------------------------------------------------

"""
Deterministic local Hopcroft-Karp maximum matching on a bipartite support
graph with sorted adjacency. Returns (match_row, match_col) with 0 marking
unmatched vertices.
"""
function _hopcroft_karp(nrows::Int, ncols::Int, adj::Vector{Vector{Int}})
    INF = typemax(Int)
    match_row = zeros(Int, nrows)
    match_col = zeros(Int, ncols)
    dist = fill(INF, nrows)

    function bfs!()
        queue = Int[]
        for u in 1:nrows
            if match_row[u] == 0
                dist[u] = 0
                push!(queue, u)
            else
                dist[u] = INF
            end
        end
        found = false
        head = 1
        while head <= length(queue)
            u = queue[head]
            head += 1
            for v in adj[u]
                w = match_col[v]
                if w == 0
                    found = true
                elseif dist[w] == INF
                    dist[w] = dist[u] + 1
                    push!(queue, w)
                end
            end
        end
        return found
    end

    function dfs!(u::Int)
        for v in adj[u]
            w = match_col[v]
            if w == 0 || (dist[w] == dist[u] + 1 && dfs!(w))
                match_row[u] = v
                match_col[v] = u
                return true
            end
        end
        dist[u] = INF
        return false
    end

    while bfs!()
        for u in 1:nrows
            match_row[u] == 0 && dfs!(u)
        end
    end
    return match_row, match_col
end

"""
Konig construction: from a maximum matching, the minimum vertex cover is
(unreached rows) ∪ (reached columns) under alternating reachability from
unmatched rows.
"""
function _konig_cover(nrows::Int, ncols::Int, adj::Vector{Vector{Int}},
                      match_row::Vector{Int}, match_col::Vector{Int})
    visited_row = falses(nrows)
    visited_col = falses(ncols)
    stack = Int[]
    for u in 1:nrows
        if match_row[u] == 0
            visited_row[u] = true
            push!(stack, u)
        end
    end
    while !isempty(stack)
        u = pop!(stack)
        for v in adj[u]
            match_row[u] == v && continue
            if !visited_col[v]
                visited_col[v] = true
                w = match_col[v]
                if w != 0 && !visited_row[w]
                    visited_row[w] = true
                    push!(stack, w)
                end
            end
        end
    end
    cover_rows = [u for u in 1:nrows if !visited_row[u]]
    cover_cols = [v for v in 1:ncols if visited_col[v]]
    return cover_rows, cover_cols
end

function _support_cover(nrows::Int, ncols::Int,
                        support::Vector{Tuple{Int,Int}})
    adj = [Int[] for _ in 1:nrows]
    for (u, v) in support
        push!(adj[u], v)
    end
    foreach(sort!, adj)
    match_row, match_col = _hopcroft_karp(nrows, ncols, adj)
    cover_rows, cover_cols = _konig_cover(nrows, ncols, adj, match_row,
                                          match_col)
    matching = sort!([(u, match_row[u]) for u in 1:nrows if match_row[u] != 0])
    return matching, cover_rows, cover_cols
end

# ---------------------------------------------------------------------------
# Lossless cut content keys
# ---------------------------------------------------------------------------

_gamma_render(x) = sprint(_ir_ser, x)

"Lossless below-content key of one term at cut `c` (rows of raw Gamma)."
function _gamma_below_key(exp::TermTTNOExpansion, t::TreeTopology, E::_Euler,
                          c::Int)
    buf = IOBuffer()
    for m in 1:nnodes(t)
        _insub(E, m, c) || continue
        he = exp.hyperedges[m]
        print(buf, "(n", m, " ")
        _ir_ser(buf, he.transition)
        _ir_ser(buf, he.certificate)
        _ir_ser(buf, he.coeff)
        if m != c
            _ir_ser(buf, exp.edge_channels[m])
        end
        print(buf, ")")
    end
    return String(take!(buf))
end

"""
Lossless above-context key of one term at cut `c` (columns of raw Gamma):
the complete rest-of-tree structure with anchor slots rendered modulo their
coefficient atom, so terms differing only in the atom share a column.
"""
function _gamma_above_key(exp::TermTTNOExpansion, t::TreeTopology, E::_Euler,
                          c::Int)
    buf = IOBuffer()
    for m in 1:nnodes(t)
        _insub(E, m, c) && continue
        he = exp.hyperedges[m]
        print(buf, "(n", m, " ")
        _ir_ser(buf, he.transition)
        _ir_ser(buf, he.certificate)
        if he.coeff isa CoeffAtomSlot
            print(buf, "(anchor ")
            _ir_ser(buf, (he.coeff::CoeffAtomSlot).scale)
            print(buf, ")")
        else
            _ir_ser(buf, he.coeff)
        end
        m == t.root || _ir_ser(buf, exp.edge_channels[m])
        print(buf, ")")
    end
    return String(take!(buf))
end

# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------

"Relocate a term's coefficient atom along its own transition path."
function _move_atom(exp::TermTTNOExpansion{Q,C}, from::Int, to::Int) where {Q,C}
    from == to && return exp
    hyper = copy(exp.hyperedges)
    slot = hyper[from].coeff
    slot isa CoeffAtomSlot || throw(ArgumentError(
        "atom relocation source node $from carries no coefficient atom"))
    hyper[from] = TermHyperedge{Q,C}(from, hyper[from].transition,
                                     hyper[from].certificate, ExactUnitSlot())
    hyper[to].coeff isa ExactUnitSlot || throw(ArgumentError(
        "atom relocation target node $to already carries a coefficient"))
    hyper[to] = TermHyperedge{Q,C}(to, hyper[to].transition,
                                   hyper[to].certificate,
                                   CoeffAtomSlot(slot.atom, slot.scale))
    return TermTTNOExpansion{Q,C}(exp.term_ordinal, exp.atom, to,
                                  exp.edge_channels, hyper, exp.provenance)
end

"Replace one hyperedge's coefficient slot."
function _replace_slot(exp::TermTTNOExpansion{Q,C}, node::Int,
                       slot::AbstractCoeffSlot) where {Q,C}
    hyper = copy(exp.hyperedges)
    hyper[node] = TermHyperedge{Q,C}(node, hyper[node].transition,
                                     hyper[node].certificate, slot)
    anchor = is_anchor_slot(slot) ? node : exp.anchor_node
    return TermTTNOExpansion{Q,C}(exp.term_ordinal, exp.atom, anchor,
                                  exp.edge_channels, hyper, exp.provenance)
end

# ---------------------------------------------------------------------------
# Shared cover pass (raw for SD4, raw-versus-eliminated for SD5)
# ---------------------------------------------------------------------------

function _gamma_cover_pass!(input::TTNOBuildInput,
                            work::Vector{TermTTNOExpansion{Q,C}},
                            proofs::Vector{MergeProofStep},
                            log::Vector{OptimizerLogEntry},
                            step::Int, elimination::Bool) where {Q,C}
    t = input.topology
    E = _Euler(t)
    svc = AbelianEquivalenceService()

    for c in postorder(t)
        c == t.root && continue
        blocks = Dict{String,Vector{Int}}()
        for (i, exp) in enumerate(work)
            ch = exp.edge_channels[c]
            ch isa ChannelIdentity{Q} || continue
            push!(get!(blocks, _gamma_render(mixing_block_key(svc, ch)),
                       Int[]), i)
        end
        for blockkey in sort!(collect(keys(blocks)))
            members = blocks[blockkey]
            channels_now = unique(ChannelIdentity{Q}[
                work[i].edge_channels[c] for i in members])
            length(channels_now) > 1 || continue

            below_keys = String[_gamma_below_key(work[i], t, E, c)
                                for i in members]
            above_keys = String[_gamma_above_key(work[i], t, E, c)
                                for i in members]
            rows = sort!(unique(below_keys))
            cols = sort!(unique(above_keys))
            rowof = Dict(k => r for (r, k) in enumerate(rows))
            colof = Dict(k => v for (v, k) in enumerate(cols))
            member_rows = Int[rowof[k] for k in below_keys]
            member_cols = Int[colof[k] for k in above_keys]
            support = sort!(unique(Tuple{Int,Int}[
                (member_rows[j], member_cols[j]) for j in eachindex(members)]))

            matching, cover_rows, cover_cols =
                _support_cover(length(rows), length(cols), support)
            raw_size = length(cover_rows) + length(cover_cols)

            chosen = :raw
            elim_ops = AbstractSGEOperation[]
            elim_support = support
            elim_matching = matching
            elim_cover_rows = cover_rows
            elim_cover_cols = cover_cols
            row_remap = Dict{Int,Tuple{Int,C}}()  # eliminated row -> (source, factor)
            if elimination
                matrix = [GammaExpr{C}() for _ in rows, _ in cols]
                slot_of = Dict{Int,Tuple{Int,C}}()
                for (j, i) in enumerate(members)
                    slot = work[i].hyperedges[work[i].anchor_node].coeff::CoeffAtomSlot
                    slot_of[i] = (slot.atom.index, slot.scale)
                    matrix[member_rows[j], member_cols[j]] = _gexpr(vcat(
                        matrix[member_rows[j], member_cols[j]].terms,
                        [(slot.atom.index, slot.scale)]))
                end
                elim_matrix, ops = _restricted_sge(matrix)
                if !isempty(ops)
                    elim_support = sort!(Tuple{Int,Int}[
                        (u, v) for u in eachindex(rows), v in eachindex(cols)
                        if !_gexpr_iszero(elim_matrix[u, v])])
                    em, ecr, ecc = _support_cover(length(rows), length(cols),
                                                  elim_support)
                    elim_size = length(ecr) + length(ecc)
                    applicable = all(op isa SGERowElimination for op in ops)
                    if applicable
                        # Every affected term needs an above-cut anchor and an
                        # exact partner for the slot rewrite.
                        for op in ops
                            for j in eachindex(members)
                                member_rows[j] == op.target || continue
                                i = members[j]
                                if _insub(E, work[i].anchor_node, c)
                                    applicable = false
                                    break
                                end
                            end
                            applicable || break
                        end
                    end
                    if elim_size < raw_size && applicable
                        chosen = :eliminated
                        elim_ops = ops
                        elim_matching = em
                        elim_cover_rows = ecr
                        elim_cover_cols = ecc
                        for op in ops
                            op isa SGERowElimination &&
                                (row_remap[op.target] = (op.source, op.factor))
                        end
                    elseif !isempty(ops)
                        step += 1
                        push!(log, OptimizerLogEntry(step, :cover_selection,
                            "edge $(nodeid(t, c)): eliminated cover $elim_size " *
                            "vs raw $raw_size — raw retained " *
                            (elim_size < raw_size ?
                             "(elimination reconstruction inapplicable)" :
                             "(raw wins ties and smaller covers)")))
                    end
                end
            end

            use_cover_rows = chosen === :raw ? cover_rows : elim_cover_rows
            use_cover_cols = chosen === :raw ? cover_cols : elim_cover_cols
            cover_size = length(use_cover_rows) + length(use_cover_cols)
            cover_size < length(channels_now) || continue

            previous = Tuple{Int,DegeneracyLabel}[
                (work[i].term_ordinal,
                 (work[i].edge_channels[c]::ChannelIdentity{Q}).degeneracy)
                for i in members]
            slot_rewrites = Tuple{Int,Int,C,C}[]

            # SD5 elimination rewrites: fold the proportional-row factor into
            # a ±1 exact scalar slot at the cut node and align the anchor
            # scale with the partner so the shared above context deduplicates.
            if chosen === :eliminated
                for j in eachindex(members)
                    i = members[j]
                    haskey(row_remap, member_rows[j]) || continue
                    source, λ = row_remap[member_rows[j]]
                    exp = work[i]
                    aslot = exp.hyperedges[exp.anchor_node].coeff::CoeffAtomSlot
                    new_scale = aslot.scale / λ
                    exp = _replace_slot(exp, exp.anchor_node,
                                        CoeffAtomSlot(aslot.atom, new_scale))
                    exp.hyperedges[c].coeff isa ExactUnitSlot ||
                        throw(ArgumentError(
                            "elimination rewrite target slot is occupied"))
                    exp = _replace_slot(exp, c, ExactScalarSlot(λ))
                    push!(slot_rewrites,
                          (exp.term_ordinal, c, λ, aslot.scale))
                    work[i] = exp
                    member_rows[j] = source
                end
            end

            cover_row_set = Set(use_cover_rows)
            assignments = Tuple{Int,Int,Int}[]
            moved = Tuple{Int,Int,Int}[]
            removed_labels = DegeneracyLabel[]
            retained_labels = DegeneracyLabel[]

            for u in sort!(collect(cover_row_set))
                row_members = [members[j] for j in eachindex(members)
                               if member_rows[j] == u]
                isempty(row_members) && continue
                row_channels = sort!(unique(ChannelIdentity{Q}[
                    work[i].edge_channels[c] for i in row_members]);
                    by=channel_order_key)
                keep = first(row_channels)
                push!(retained_labels, keep.degeneracy)
                for (j, i) in enumerate(members)
                    member_rows[j] == u || continue
                    push!(assignments,
                          (work[i].term_ordinal, u, member_cols[j]))
                    current = work[i].edge_channels[c]::ChannelIdentity{Q}
                    current == keep && continue
                    push!(removed_labels, current.degeneracy)
                    work[i] = _rewrite_channel(work[i], c, keep)
                end
            end

            for v in sort!(use_cover_cols)
                col_members = [members[j] for j in eachindex(members)
                               if member_cols[j] == v &&
                                  !(member_rows[j] in cover_row_set)]
                isempty(col_members) && continue
                template = work[first(col_members)].edge_channels[c]::ChannelIdentity{Q}
                keys_of = sort!(unique(String[
                    below_keys[findfirst(==(i), members)] for i in col_members]))
                ordinal = minimum(work[i].term_ordinal for i in col_members)
                span = ChannelSpan(ACTIVE, (:gamma_combination, Tuple(keys_of)))
                merged = ChannelIdentity{Q}(
                    template.sector, template.route, template.multiplicity,
                    DegeneracyLabel(span, ordinal), template.orientation,
                    template.frame)
                push!(retained_labels, merged.degeneracy)
                for (j, i) in enumerate(members)
                    (member_cols[j] == v && !(member_rows[j] in cover_row_set)) ||
                        continue
                    push!(assignments,
                          (work[i].term_ordinal, member_rows[j], v))
                    exp = work[i]
                    if !_insub(E, exp.anchor_node, c)
                        from = exp.anchor_node
                        exp = _move_atom(exp, from, c)
                        push!(moved, (exp.term_ordinal, from, c))
                    end
                    current = exp.edge_channels[c]::ChannelIdentity{Q}
                    current == merged || push!(removed_labels, current.degeneracy)
                    work[i] = _rewrite_channel(exp, c, merged)
                end
            end

            isempty(removed_labels) && isempty(slot_rewrites) && continue
            raw_witness = GammaCoverWitness(
                c, blockkey, rows, cols, support, matching, cover_rows,
                cover_cols, sort!(assignments), moved, previous)
            span = ChannelSpan(ACTIVE, (:gamma_cover, Tuple(rows)))
            witness = chosen === :raw ? raw_witness :
                EliminationCoverWitness(
                    raw_witness, elim_ops, elim_support, elim_matching,
                    elim_cover_rows, elim_cover_cols, raw_size,
                    length(elim_cover_rows) + length(elim_cover_cols),
                    slot_rewrites)
            push!(proofs, MergeProofStep(
                chosen === :raw ? :gamma_raw_cover : :gamma_eliminated_cover,
                c, span, IdentitySpanRelation(span), witness,
                removed_labels, retained_labels))
            step += 1
            push!(log, OptimizerLogEntry(step,
                chosen === :raw ? :gamma_raw_cover : :gamma_eliminated_cover,
                "edge $(nodeid(t, c)) block $(length(rows))x$(length(cols)): " *
                "support $(length(support)), raw cover $raw_size" *
                (chosen === :eliminated ?
                 ", eliminated cover $(length(elim_cover_rows) + length(elim_cover_cols)) chosen" :
                 " chosen") *
                ", channels $(length(channels_now)) -> $cover_size"))
        end
    end
    return step
end

function _gamma_merge(input::TTNOBuildInput,
                      terms::Vector{TermTTNOExpansion{Q,C}},
                      elimination::Bool) where {Q,C}
    structural = merge_channels(input, terms,
                                StateDiagramMerge(StructuralOptimizer()))
    work = collect(structural.expansions)
    proofs = copy(structural.proofs)
    log = copy(structural.log.entries)
    step = length(log)

    step = _gamma_cover_pass!(input, work, proofs, log, step, elimination)

    for exp in work
        validate_expansion(input, exp)
    end
    step += 1
    push!(log, OptimizerLogEntry(step, :final_validate,
                                 "revalidated $(length(work)) expansions after cover pass"))
    return StateDiagram{Q,C}(input.topology, input.provenance, work, proofs,
                             OptimizerLog(log))
end

merge_channels(input::TTNOBuildInput, terms::Vector{TermTTNOExpansion{Q,C}},
               ::StateDiagramMerge{GammaCoverOptimizer}) where {Q,C} =
    _gamma_merge(input, terms, false)

merge_channels(input::TTNOBuildInput, terms::Vector{TermTTNOExpansion{Q,C}},
               ::StateDiagramMerge{SGEOptimizer}) where {Q,C} =
    _gamma_merge(input, terms, true)
