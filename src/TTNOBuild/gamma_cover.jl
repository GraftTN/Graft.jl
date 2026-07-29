# SD4 — raw Gamma blocks and minimum vertex cover (state-diagram-compiler
# plan v2, SD4 deliverables).
#
# Every child-to-parent cut is partitioned into categorical Gamma blocks
# keyed by the complete compatible cut hom-space (the equivalence service's
# mixing-block key: sector, route, multiplicity, orientation, frame; the
# exact scalar domain is the shared coefficient scalar type). Inside each
# block the raw exact Gamma support graph relates lossless below-content
# rows to lossless above-context columns; a local deterministic
# Hopcroft-Karp maximum matching and the Konig construction give the
# minimum vertex cover, and the raw-cover reconstruction rewires channels
# with an exact reconstruction log. Coefficient atoms stay opaque: the only
# relocation is moving one term's atom slot along its own transition path
# across the cut, which multiplies the same exact product in a different
# order.

"""
    GammaCoverOptimizer()

SD4 optimizer for [`StateDiagramMerge`](@ref): the SD3 structural phases
followed by raw-Gamma minimum-cover merging on every cut, bottom-up in
postorder. The eliminated-Gamma alternative and proof-backed raw-versus-
eliminated cover selection arrive with SD5.
"""
struct GammaCoverOptimizer end

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

# ---------------------------------------------------------------------------
# Raw-Gamma cover pass
# ---------------------------------------------------------------------------

function _gamma_raw_cover_pass!(input::TTNOBuildInput,
                                work::Vector{TermTTNOExpansion{Q,C}},
                                proofs::Vector{MergeProofStep},
                                log::Vector{OptimizerLogEntry},
                                step::Int) where {Q,C}
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
            adj = [Int[] for _ in rows]
            support = Tuple{Int,Int}[]
            for (j, i) in enumerate(members)
                u = rowof[below_keys[j]]
                v = colof[above_keys[j]]
                if !((u, v) in support)
                    push!(support, (u, v))
                    push!(adj[u], v)
                end
            end
            sort!(support)
            foreach(sort!, adj)

            match_row, match_col = _hopcroft_karp(length(rows), length(cols), adj)
            cover_rows, cover_cols = _konig_cover(length(rows), length(cols),
                                                  adj, match_row, match_col)
            cover_size = length(cover_rows) + length(cover_cols)
            cover_size < length(channels_now) || continue

            cover_row_set = Set(cover_rows)
            assignments = Tuple{Int,Int,Int}[]
            moved = Tuple{Int,Int,Int}[]
            removed_labels = DegeneracyLabel[]
            retained_labels = DegeneracyLabel[]
            previous = Tuple{Int,DegeneracyLabel}[
                (work[i].term_ordinal,
                 (work[i].edge_channels[c]::ChannelIdentity{Q}).degeneracy)
                for i in members]

            # Row-channel members: terms whose below-content row is covered
            # share one channel per row (their below contents are equal by
            # the lossless key).
            for u in sort!(collect(cover_row_set))
                row_members = [members[j] for (j, k) in enumerate(below_keys)
                               if rowof[k] == u]
                row_channels = sort!(unique(ChannelIdentity{Q}[
                    work[i].edge_channels[c] for i in row_members]);
                    by=channel_order_key)
                keep = first(row_channels)
                push!(retained_labels, keep.degeneracy)
                for i in row_members
                    push!(assignments,
                          (work[i].term_ordinal, u, colof[_gamma_above_key(work[i], t, E, c)]))
                    current = work[i].edge_channels[c]::ChannelIdentity{Q}
                    current == keep && continue
                    push!(removed_labels, current.degeneracy)
                    work[i] = _rewrite_channel(work[i], c, keep)
                end
            end

            # Column-channel members: terms whose row is uncovered attach to
            # the covered column wire; their coefficient atoms relocate below
            # the cut so the shared above context is atom-free.
            for v in sort!(cover_cols)
                col_members = [members[j] for (j, k) in enumerate(above_keys)
                               if colof[k] == v &&
                                  !(rowof[below_keys[j]] in cover_row_set)]
                isempty(col_members) && continue
                template = work[first(col_members)].edge_channels[c]::ChannelIdentity{Q}
                member_rows = sort!(unique(String[
                    below_keys[findfirst(==(i), members)] for i in col_members]))
                ordinal = minimum(work[i].term_ordinal for i in col_members)
                span = ChannelSpan(ACTIVE, (:gamma_combination, Tuple(member_rows)))
                merged = ChannelIdentity{Q}(
                    template.sector, template.route, template.multiplicity,
                    DegeneracyLabel(span, ordinal), template.orientation,
                    template.frame)
                push!(retained_labels, merged.degeneracy)
                for i in col_members
                    push!(assignments,
                          (work[i].term_ordinal,
                           rowof[_gamma_below_key(work[i], t, E, c)], v))
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

            isempty(removed_labels) && continue
            matching = sort!([(u, match_row[u]) for u in 1:length(rows)
                              if match_row[u] != 0])
            witness = GammaCoverWitness(
                c, blockkey, rows, cols, support, matching,
                cover_rows, cover_cols, sort!(assignments), moved, previous)
            span = ChannelSpan(ACTIVE, (:gamma_cover, Tuple(rows)))
            push!(proofs, MergeProofStep(
                :gamma_raw_cover, c, span, IdentitySpanRelation(span),
                witness, removed_labels, retained_labels))
            step += 1
            push!(log, OptimizerLogEntry(step, :gamma_raw_cover,
                "edge $(nodeid(t, c)) block $(length(rows))x$(length(cols)): " *
                "support $(length(support)), matching $(length(matching)), " *
                "cover $(cover_size), channels $(length(channels_now)) -> $(cover_size)"))
        end
    end
    return step
end

function merge_channels(input::TTNOBuildInput,
                        terms::Vector{TermTTNOExpansion{Q,C}},
                        kernel::StateDiagramMerge{GammaCoverOptimizer}) where {Q,C}
    structural = merge_channels(input, terms,
                                StateDiagramMerge(StructuralOptimizer()))
    work = collect(structural.expansions)
    proofs = copy(structural.proofs)
    log = copy(structural.log.entries)
    step = length(log)

    step = _gamma_raw_cover_pass!(input, work, proofs, log, step)

    for exp in work
        validate_expansion(input, exp)
    end
    step += 1
    push!(log, OptimizerLogEntry(step, :final_validate,
                                 "revalidated $(length(work)) expansions after raw-cover pass"))
    return StateDiagram{Q,C}(input.topology, input.provenance, work, proofs,
                             OptimizerLog(log))
end
