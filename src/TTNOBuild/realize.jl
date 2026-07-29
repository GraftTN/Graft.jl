# SD2 — direct-sum merge, producer-independent realization view, and
# `realize_ttno` (state-diagram-compiler plan v2, "Plan and realization
# boundary"; ADR-0003 decision 5).
#
# `realize_ttno` immediately consumes the canonical `RealizationPlanView`
# and never branches on the concrete producer layout. It reuses the
# tree-contiguous TensorKit-exact block coordinate utility
# (`_sector_tuple_groups`, `_add_block_entry!`) and the local twist
# materialization of statediagram.jl instead of duplicating any dense block
# enumeration.

# ---------------------------------------------------------------------------
# Realization view
# ---------------------------------------------------------------------------

"""
    RealizationEntry(node, child_channels, parent_channel, transition,
                     certificate, coeff)

One automaton transition of the realization view: ordered child channel slot
indices (into the per-edge channel lists), the parent channel slot (`0` at
the root boundary), the typed local transition, the exact morphism
certificate, and the coefficient slot. Identical transitions from different
term expansions deduplicate to one entry; distinct coefficient atoms never
deduplicate.
"""
struct RealizationEntry{Q,C<:Number}
    node::Int
    child_channels::Vector{Int}
    parent_channel::Int
    transition::AbstractLocalTransition
    certificate::LocalMorphismCertificate{Q,C}
    coeff::AbstractCoeffSlot
end

"""
    RealizationPlanView

The canonical producer-independent realization view: deterministic per-edge
channel lists (indexed by child node; empty at the root) and deterministic
transition entries. Views over a `DirectSumPlan` and over a zero-merge
`StateDiagram` of the same input are field-identical (ADR-0003 decision 5).
"""
struct RealizationPlanView{Q,C<:Number}
    provenance::BuildInputProvenance
    edge_channels::Vector{Vector{ChannelIdentity{Q}}}
    entries::Vector{RealizationEntry{Q,C}}
end

for T in (:RealizationEntry, :RealizationPlanView)
    @eval begin
        Base.:(==)(a::$T, b::$T) = _ir_isequal(a, b)
        Base.isequal(a::$T, b::$T) = _ir_isequal(a, b)
        Base.hash(x::$T, h::UInt) = _ir_hash(x, h)
    end
end

function _entry_order_key(e::RealizationEntry)
    return (e.node, e.parent_channel, Tuple(e.child_channels),
            string(e.transition), string(e.certificate), string(e.coeff))
end

function _view_from_expansions(t::TreeTopology, provenance::BuildInputProvenance,
                               expansions::Vector{TermTTNOExpansion{Q,C}}) where {Q,C}
    N = nnodes(t)
    per_edge = [ChannelIdentity{Q}[] for _ in 1:N]
    for exp in expansions
        length(exp.edge_channels) == N ||
            throw(ArgumentError("expansion does not match the plan topology"))
        for c in 1:N
            c == t.root && continue
            ch = exp.edge_channels[c]
            ch isa ChannelIdentity{Q} ||
                throw(ArgumentError("non-root edge $c carries no channel"))
            any(existing -> existing == ch, per_edge[c]) || push!(per_edge[c], ch)
        end
    end
    for c in 1:N
        sort!(per_edge[c]; by=channel_order_key)
    end
    slot = [Dict{ChannelIdentity{Q},Int}(ch => i for (i, ch) in enumerate(per_edge[c]))
            for c in 1:N]

    seen = Set{RealizationEntry{Q,C}}()
    entries = RealizationEntry{Q,C}[]
    for exp in expansions
        for n in 1:N
            he = exp.hyperedges[n]
            child_channels = Int[slot[c][exp.edge_channels[c]]
                                 for c in t.children[n]]
            parent_channel = n == t.root ? 0 : slot[n][exp.edge_channels[n]]
            entry = RealizationEntry{Q,C}(n, child_channels, parent_channel,
                                          he.transition, he.certificate,
                                          he.coeff)
            entry in seen && continue
            push!(seen, entry)
            push!(entries, entry)
        end
    end
    sort!(entries; by=_entry_order_key)
    return RealizationPlanView{Q,C}(provenance, per_edge, entries)
end

# ---------------------------------------------------------------------------
# Merge kernels (direct sum) and views
# ---------------------------------------------------------------------------

function merge_channels(input::TTNOBuildInput,
                        terms::Vector{TermTTNOExpansion{Q,C}},
                        ::DirectSumMerge) where {Q,C}
    require_capability(input.category, :merge, :merge_channels)
    length(terms) == length(input.terms) ||
        throw(ArgumentError("direct-sum merge requires one expansion per normalized term"))
    for exp in terms
        validate_expansion(input, exp)
    end
    return DirectSumPlan{Q,C}(input.topology, input.provenance, collect(terms))
end

realization_view(plan::DirectSumPlan) =
    _view_from_expansions(plan.topology, plan.provenance, plan.expansions)

realization_view(plan::StateDiagram) =
    _view_from_expansions(plan.topology, plan.provenance, plan.expansions)

_plan_kind(::DirectSumPlan) = :direct_sum
_plan_kind(::StateDiagram) = :state_diagram

# ---------------------------------------------------------------------------
# Materialization service methods (Abelian)
# ---------------------------------------------------------------------------

function materialize_virtual_space(::TensorKitMaterializationService,
                                   ::Type{S},
                                   channels::Vector{ChannelIdentity{Q}}) where {S<:ElementarySpace,Q}
    if S === ComplexSpace
        n = length(channels)
        return ComplexSpace(max(n, 1)), collect(1:n)
    end
    isempty(channels) && return trivialspace(S), Int[]
    groups = Dict{Q,Vector{Int}}()
    for (i, ch) in enumerate(channels)
        push!(get!(groups, ch.sector, Int[]), i)
    end
    pairs = [q => length(groups[q]) for q in sort!(collect(keys(groups)); by=string)]
    V = Vect[Q](pairs...)
    coords = Vector{Int}(undef, length(channels))
    offset = 0
    for q in sectors(V)
        for (j, i) in enumerate(groups[q])
            coords[i] = offset + j
        end
        offset += length(groups[q])
    end
    return V, coords
end

"""
    LocalMorphismSpec(node, physical, op, crossing)

Typed specification of one local invariant morphism: the physical space, the
explicit factor (`nothing` for omitted identity), and the fused crossing
charge braided past the physical input. Materialized behind
[`materialize_morphism`](@ref) so the twist convention lives in one place.
"""
struct LocalMorphismSpec{Q,P<:ElementarySpace}
    node::Int
    physical::P
    op::Union{Nothing,SiteOp}
    crossing::Q
end

function materialize_morphism(::TensorKitMaterializationService,
                              spec::LocalMorphismSpec{Q},
                              ::Type{elt}) where {Q,elt<:Number}
    localentry = spec.op === nothing ? _OMITTED_IDENTITY :
        _ExplicitLocal(spec.op.name)
    opmats = Dict{Tuple{Int,Symbol},AbstractTensorMap}()
    spec.op === nothing || (opmats[(spec.node, spec.op.name)] = spec.op.op)
    if Q === Trivial
        return _siteop_matrix(opmats, spec.node, localentry, elt,
                              dim(spec.physical))
    end
    return _graded_siteop_matrix(opmats, spec.node, localentry, elt,
                                 spec.physical, spec.crossing, one(Q))
end

function validate_morphism(svc::TensorKitMaterializationService,
                           morphism::AbstractMatrix,
                           spec::LocalMorphismSpec, ::Type{elt}) where {elt<:Number}
    d = dim(spec.physical)
    size(morphism) == (d, d) || return false
    return morphism == materialize_morphism(svc, spec, elt)
end

# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

"""
    TTNOBuildEdgeReport(child, parent, dimension, sector_dimensions)

Realized bond layout of one oriented edge: total dimension and per-sector
dimensions in deterministic sector order.
"""
struct TTNOBuildEdgeReport
    child::Symbol
    parent::Symbol
    dimension::Int
    sector_dimensions::Vector{Pair{String,Int}}
end

"""
    TTNOBuildReport

Complete build and capability report of one `realize_ttno` call: input
provenance, producer kind, the evaluated category capability matrix, the
Hermiticity contract, term/entry counts, per-edge bond layout, and applied
merge-proof/optimizer-log counts. Deterministic across repeated runs on
equal input.
"""
struct TTNOBuildReport{CS<:CategorySemantics}
    provenance::BuildInputProvenance
    plan_kind::Symbol
    capability::CategoryCapabilityReport{CS}
    hermitian_asserted::Bool
    nterms::Int
    nentries::Int
    edges::Vector{TTNOBuildEdgeReport}
    proofs_applied::Int
    optimizer_log_entries::Int
end

for T in (:TTNOBuildEdgeReport, :TTNOBuildReport)
    @eval begin
        Base.:(==)(a::$T, b::$T) = _ir_isequal(a, b)
        Base.isequal(a::$T, b::$T) = _ir_isequal(a, b)
        Base.hash(x::$T, h::UInt) = _ir_hash(x, h)
    end
end

function _edge_sector_dimensions(V::ElementarySpace)
    spacetype(V) === ComplexSpace && return Pair{String,Int}["trivial" => dim(V)]
    entries = Pair{String,Int}[string(q) => dim(V, q) for q in sectors(V)]
    return sort!(entries; by=first)
end

# ---------------------------------------------------------------------------
# realize_ttno
# ---------------------------------------------------------------------------

_slot_value(::CoeffTable, ::ExactUnitSlot) = 1
_slot_value(table::CoeffTable, slot::CoeffAtomSlot) =
    coefficient_value(table, slot.atom) * slot.scale

function realize_ttno(input::TTNOBuildInput{S,C}, plan::AbstractTTNOMergePlan;
                      elt::Type{<:Number}=ComplexF64) where {S,C}
    require_capability(input.category, :materialization, :materialize_morphism)
    plan.provenance == input.provenance || throw(ArgumentError(
        "merge plan provenance $(provenance_hex(plan.provenance)) does not " *
        "match the build input $(provenance_hex(input.provenance))"))
    if plan isa StateDiagram && !isempty(plan.proofs)
        throw(ArgumentError(
            "merge-proof replay validation lands with SD3; only zero-merge " *
            "StateDiagram plans are realizable at SD2"))
    end
    plan_view = realization_view(plan)
    t = input.topology
    N = nnodes(t)
    Q = sectortype(S)
    unit_sector = one(Q)
    graded = S !== ComplexSpace
    svc = TensorKitMaterializationService()

    vspaces = Vector{S}(undef, N)
    coords = Vector{Vector{Int}}(undef, N)
    for c in 1:N
        if c == t.root
            vspaces[c] = tensor_unit(svc, S)
            coords[c] = Int[]
        else
            vspaces[c], coords[c] =
                materialize_virtual_space(svc, S, plan_view.edge_channels[c])
        end
    end
    vspaces[t.root] == trivialspace(S) || throw(ArgumentError(
        "realized root boundary must equal the category tensor unit"))

    opdict = Dict{LocalOpKey,SiteOp}(input.operator_table)
    entries_by_node = [RealizationEntry{Q,C}[] for _ in 1:N]
    for entry in plan_view.entries
        push!(entries_by_node[entry.node], entry)
    end

    unit_space = oneunit(S)
    tensors = map(1:N) do n
        K = nchildren(t, n)
        site = nodeid(t, n)
        hp = haskey(input.phys_lookup, site)
        P = hp ? input.phys_lookup[site] : nothing
        d = hp ? dim(P) : 1
        χp = n == t.root ? 1 : dim(vspaces[n])
        cods = S[]
        for c in t.children[n]
            push!(cods, vspaces[c])
        end
        hp && push!(cods, P)
        cod = isempty(cods) ? one(unit_space) : reduce(⊗, cods)
        Vp = n == t.root ? unit_space : vspaces[n]
        dom = hp ? P ⊗ Vp : ProductSpace(Vp)

        local_matrix(entry) = begin
            op = entry.transition isa ExplicitLocalTransition ?
                opdict[entry.transition.key] : nothing
            materialize_morphism(
                svc, LocalMorphismSpec(n, P, op, entry.certificate.crossing),
                elt)
        end

        if graded
            W = zeros(elt, cod ← dom)
            blockmap = Dict{Q,Any}(q => b for (q, b) in blocks(W))
            _, codcoord = _sector_tuple_groups(cods, unit_sector)
            doms = hp ? S[P, Vp] : S[Vp]
            _, domcoord = _sector_tuple_groups(doms, unit_sector)
            for entry in entries_by_node[n]
                value = elt(_slot_value(input.coefficients, entry.coeff)) *
                    elt(entry.certificate.scalar)
                αidx = ntuple(k -> coords[t.children[n][k]][entry.child_channels[k]], K)
                βidx = entry.parent_channel == 0 ? 1 : coords[n][entry.parent_channel]
                if hp
                    mat = local_matrix(entry)
                    for pout in 1:d, pin in 1:d
                        val = value * mat[pout, pin]
                        iszero(val) && continue
                        _add_block_entry!(blockmap, codcoord, domcoord,
                                          (αidx..., pout), (pin, βidx), val)
                    end
                else
                    entry.transition isa OmittedIdentityTransition ||
                        throw(ArgumentError(
                            "operator factor on branching node $(nodeid(t, n))"))
                    _add_block_entry!(blockmap, codcoord, domcoord,
                                      αidx, (βidx,), value)
                end
            end
            W
        else
            dims = (ntuple(k -> dim(vspaces[t.children[n][k]]), K)...,
                    (hp ? (d, d) : ())..., χp)
            W = zeros(elt, dims)
            for entry in entries_by_node[n]
                value = elt(_slot_value(input.coefficients, entry.coeff)) *
                    elt(entry.certificate.scalar)
                αidx = ntuple(k -> coords[t.children[n][k]][entry.child_channels[k]], K)
                βidx = entry.parent_channel == 0 ? 1 : coords[n][entry.parent_channel]
                if hp
                    mat = local_matrix(entry)
                    Base.view(W, αidx..., :, :, βidx) .+= value .* mat
                else
                    entry.transition isa OmittedIdentityTransition ||
                        throw(ArgumentError(
                            "operator factor on branching node $(nodeid(t, n))"))
                    W[αidx..., βidx] += value
                end
            end
            TensorMap(W, cod ← dom)
        end
    end

    operator = TTNO(t, tensors;
                    ishermitian=input.hermiticity isa AssertedHermitian)
    edge_reports = TTNOBuildEdgeReport[
        TTNOBuildEdgeReport(nodeid(t, c), nodeid(t, t.parent[c]),
                            dim(vspaces[c]),
                            _edge_sector_dimensions(vspaces[c]))
        for c in 1:N if c != t.root
    ]
    report = TTNOBuildReport(
        input.provenance, _plan_kind(plan), capability_report(input.category),
        input.hermiticity isa AssertedHermitian, length(input.terms),
        length(plan_view.entries), edge_reports,
        plan isa StateDiagram ? length(plan.proofs) : 0,
        plan isa StateDiagram ? length(plan.log.entries) : 0,
    )
    return operator, report
end
