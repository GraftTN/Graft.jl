# TTNO assembly from an OpSum — tree finite-state-machine construction.
#
# Channel structure per edge e = (child c, parent p) — cutting the tree at e,
# every term t is classified by its restriction to the subtree below e:
#   * IDLE:  t has no site below e            (identity flows below)
#   * DONE:  t lies entirely below e          (identity flows above; all such
#            terms share one channel, coefficients applied at completion)
#   * ACTIVE(r): proper nonempty restriction r (terms with identical
#            restriction share the channel — the "state diagram" merge)
# A term's coefficient is applied exactly once, at its completion node (the
# lowest node whose subtree contains all its sites). This reproduces the
# channels of PyTreeNet's state-diagram pipeline (single-term diagrams +
# hyperedge/vertex merging) for product-term Hamiltonians.
#
# TODO(port, §4b): PyTreeNet's edge-cut compression pass can beat the channel
#   construction when term combinations are linearly dependent (not just
#   identical). Port order (per state_diagram.py::from_hamiltonian_modified):
#   per edge cut, build the exact-rational Γ bond matrix (rows/cols = child-/
#   parent-side hyperedges), symbolic Gaussian elimination over Rational
#   (symbolic_gaussian_elimination_fraction.py), then Kőnig minimum vertex
#   cover via Hopcroft–Karp (bipartite_graph.py, generic and portable
#   verbatim); keep whichever of raw-Γ / eliminated-Γ gives the smaller cover
#   (TTNOFinder.SGE semantics). Run bottom-up level by level.
# Sector-graded virtual legs: each ACTIVE channel carries the fused charge flux
# of its restriction, while IDLE/DONE channels are trivial. The trivial-sector
# path below still uses dense `ComplexSpace(χ)` channels.

# subtree membership via Euler (DFS in/out) intervals
struct _Euler
    tin::Vector{Int}
    tout::Vector{Int}
end
function _Euler(t::TreeTopology)
    tin = zeros(Int, nnodes(t))
    tout = zeros(Int, nnodes(t))
    clock = Ref(0)
    function dfs(n)
        tin[n] = (clock[] += 1)
        for c in t.children[n]
            dfs(c)
        end
        tout[n] = (clock[] += 1)
    end
    dfs(t.root)
    return _Euler(tin, tout)
end
_insub(E::_Euler, x::Int, c::Int) = E.tin[c] <= E.tin[x] <= E.tout[c]

abstract type _ChannelKey end

"Plain identity transport with no categorical physical-leg frame."
struct _PlainIdle <: _ChannelKey end

"Completed restriction transport; it always carries the unit virtual sector."
struct _Done <: _ChannelKey end

# Non-abelian migration contract
# ------------------------------
# `_CrossingFrame` is intentionally the typed replacement seam for this
# abelian scalar specialization.  A future non-abelian lowerer must preserve
# the public site-labelled `Term` contract and upgrade this internal payload,
# rather than making `Term.ops` order semantic or changing the tree topology:
#
# 1. Replace each `site => fused_charge` entry by an oriented braid word or
#    fusion route carrying the ordered legs, intermediate sectors, duality,
#    and fusion-multiplicity indices.
# 2. Include that route in `_Active`/`_FramedIdle` channel identity, so equal
#    labelled restrictions with different fusion trees or multiplicity paths
#    can never be merged by the state diagram.
# 3. Lower neutral-local and charged-local crossings as explicit TensorKit
#    morphisms: perform the required F moves, R moves, and bends/twists along
#    the stored route.  Do not collapse them to one total charge or scalar.
# 4. Keep the resulting route data internal to TTNO construction; dense
#    fallback and topology-dependent public ordering conventions remain
#    forbidden.
#
# Until that route-aware lowering exists, `_fuse_charge`, scalar `Rsymbol`,
# and scalar local-twist guards below deliberately fail closed on non-unique
# fusion or non-scalar categorical data.  SU(2) is therefore rejected here
# even though its common fusion rules are multiplicity-free: the intermediate
# fusion channel is still not determined by one abelian total-charge label.

"Immutable per-physical-site abelian braid plan for one term restriction."
struct _CrossingFrame{Q,A<:Tuple}
    at::A
end

"Identity restriction with a unit virtual sector and a nonempty braid plan."
struct _FramedIdle{F<:_CrossingFrame} <: _ChannelKey
    frame::F
end

"Non-identity restriction plus any physical-leg braid plan below the edge."
struct _Active{R<:Tuple,F<:_CrossingFrame} <: _ChannelKey
    restriction::R
    frame::F
end

const _PLAIN_IDLE = _PlainIdle()
const _DONE = _Done()

_is_plain_idle(key::_ChannelKey) = key isa _PlainIdle
_is_framed_idle(key::_ChannelKey) = key isa _FramedIdle
_is_done(key::_ChannelKey) = key isa _Done
_is_active(key::_ChannelKey) = key isa _Active

abstract type _EntryLocal end

"Identity padding introduced by the state diagram, not an explicit Term factor."
struct _OmittedIdentity <: _EntryLocal end

"One explicit labelled local factor, including an explicit SiteOp named :I."
struct _ExplicitLocal <: _EntryLocal
    name::Symbol
end

const _OMITTED_IDENTITY = _OmittedIdentity()

Base.isempty(frame::_CrossingFrame) = isempty(frame.at)

function _crossing_frame(entries, unit)
    pairs = Pair{Int,typeof(unit)}[
        Int(entry.first) => entry.second for entry in entries if entry.second != unit
    ]
    sort!(pairs; by=first)
    allunique(first.(pairs)) || throw(ArgumentError(
        "canonical TTNO crossing frame repeats one physical node",
    ))
    at = Tuple(pairs)
    return _CrossingFrame{typeof(unit),typeof(at)}(at)
end

_empty_crossing_frame(unit) = _CrossingFrame{typeof(unit),Tuple{}}(())

function _frame_at(frame::_CrossingFrame, node::Int, unit)
    for entry in frame.at
        entry.first == node && return entry.second
    end
    return unit
end

function _subframe(frame::_CrossingFrame, E::_Euler, child::Int, unit)
    return _crossing_frame((entry for entry in frame.at if _insub(E, entry.first, child)),
                           unit)
end

function _channel_sort_key(key::_ChannelKey)
    _is_plain_idle(key) && return (0, "")
    _is_framed_idle(key) && return (1, string(key.frame.at))
    _is_done(key) && return (2, "")
    _is_active(key) && return (3, string((key.restriction, key.frame.at)))
    throw(ArgumentError("unknown state-diagram channel key $(typeof(key))"))
end

_rkey(pairs::AbstractVector) = Tuple(sort!(collect(pairs); by=string))

"Canonical internal-node order and the TTNO physical-input traversal order."
function _physical_node_orders(t::TreeTopology, phys)
    canonical = [node for node in 1:nnodes(t) if haskey(phys, nodeid(t, node))]
    native = Int[]
    function visit(node::Int)
        haskey(phys, nodeid(t, node)) && push!(native, node)
        # TTNO node maps consume the local physical input before their child
        # branch maps; the codomain child-leg orientation reverses this walk.
        for child in reverse(t.children[node])
            visit(child)
        end
        return nothing
    end
    visit(t.root)
    length(native) == length(canonical) && Set(native) == Set(canonical) ||
        throw(ArgumentError("TTNO physical traversal does not cover the declared spaces"))
    return canonical, native
end

function _prefix_charges(order::AbstractVector{Int}, ops::Dict{Int,SiteOp}, unit)
    prefixes = Dict{Int,Any}()
    total = unit
    for node in order
        prefixes[node] = total
        haskey(ops, node) && (total = _fuse_charge(total, charge(ops[node])))
    end
    return prefixes
end

"Per-site physical-input frames needed to convert native TTNO traversal to canonical order."
function _canonical_crossing_frame(t::TreeTopology, phys,
                                   ops::Dict{Int,SiteOp}, unit)
    canonical, native = _physical_node_orders(t, phys)
    canonical_prefix = _prefix_charges(canonical, ops, unit)
    native_prefix = _prefix_charges(native, ops, unit)
    entries = Pair{Int,typeof(unit)}[]
    for node in canonical
        local_charge = haskey(ops, node) ? charge(ops[node]) : unit
        # Existing charged-wire R/bend paths own the charged local endpoint.
        # This complementary frame records only neutral physical inputs,
        # including omitted identities and explicit neutral site operators.
        local_charge == unit || continue
        crossing = _fuse_charge(canonical_prefix[node], dual(native_prefix[node]))
        _has_fermionic_crossing(crossing, unit) && push!(entries, node => crossing)
    end
    return _crossing_frame(entries, unit)
end

# SD0 braid certificate
# ---------------------
# These events keep the native tree morphisms and the global word correction
# as separate, uniquely-owned pieces of one term plan.  They are scalar/
# abelian today, but their ownership and ordering survive a later route-aware
# non-abelian lowering.
abstract type _BraidedTermEvent end

"One adjacent native-to-canonical wire swap, owned by the pair's tree LCA."
struct _WireCrossingEvent{Q,S<:Number} <: _BraidedTermEvent
    lhs::Int
    rhs::Int
    owner::Int
    lhs_charge::Q
    rhs_charge::Q
    fused_charge::Q
    scalar::S
end

"Native bend of a charged local factor past earlier sibling restrictions."
struct _LocalFactorBendEvent{Q} <: _BraidedTermEvent
    node::Int
    crossing_charge::Q
end

"Pivotal bend where a charged local factor closes a multi-branch term."
struct _LocalCompletionBendEvent{Q,S<:Number} <: _BraidedTermEvent
    node::Int
    charge::Q
    scalar::S
end

"Native bend of fused child charge past a neutral local physical input."
struct _PhysicalInputBendEvent{Q} <: _BraidedTermEvent
    node::Int
    crossing_charge::Q
end

"Canonical/native physical-input frame crossing at one physical node."
struct _PhysicalInputFrameEvent{Q} <: _BraidedTermEvent
    node::Int
    crossing_charge::Q
end

"CG-009 charge-leg exit braid and its orientation normalization."
struct _ChargeLegExitEvent{Q,S<:Number} <: _BraidedTermEvent
    node::Int
    crossing_charge::Q
    orientation_scalar::S
end

"All native morphism payloads and certificate swaps consumed at one node."
struct _LocalMorphismPlan{Q,S<:Number,
                          B<:_LocalCompletionBendEvent{Q},
                          E<:_ChargeLegExitEvent{Q}}
    node::Int
    factor_bend::_LocalFactorBendEvent{Q}
    completion_bend::B
    physical_bend::_PhysicalInputBendEvent{Q}
    input_frame::_PhysicalInputFrameEvent{Q}
    charge_exit::E
    wire_crossings::Vector{_WireCrossingEvent{Q}}
    word_scale::S
end

"Internal per-term SD0 certificate; factor identities are canonical node ids."
struct _BraidedTermPlan{Q,F<:_CrossingFrame,C<:Number,L<:Number}
    canonical_word::Vector{Int}
    native_word::Vector{Int}
    factor_class::Dict{Int,Int}
    crossings::Vector{_WireCrossingEvent{Q}}
    local_plans::Vector{_LocalMorphismPlan{Q}}
    restriction_charge::Vector{Q}
    frame::F
    uses_certificate::Bool
    certificate_scale::C
    legacy_scale::L
end

function _term_lca(t::TreeTopology, lhs::Int, rhs::Int)
    ancestors = Set{Int}()
    node = lhs
    while node != 0
        push!(ancestors, node)
        node = t.parent[node]
    end
    node = rhs
    while !(node in ancestors)
        node = t.parent[node]
    end
    return node
end

"Legacy multi-child scalar, retained only for parity-only fallback/shadowing."
function _legacy_canonical_junction_braid(t::TreeTopology,
                                           ops::Dict{Int,SiteOp},
                                           opnodes::Vector{Int}, unit_sector,
                                           coefficient, n::Int)
    phase = one(coefficient)
    child_wires = [
        [x for x in opnodes if begin
            child_node = x
            while child_node != 0 && child_node != child
                child_node = t.parent[child_node]
            end
            child_node == child
        end]
        for child in t.children[n]
    ]
    count(!isempty, child_wires) >= 2 || return phase
    if haskey(ops, n)
        localq = charge(ops[n])
        if localq != unit_sector
            local_twist = twist(localq)
            local_twist isa Number || throw(ArgumentError(
                "graded TTNO local junction twists require scalar abelian data; got $(typeof(local_twist))",
            ))
            phase *= local_twist
        end
    end
    wires = reduce(vcat, child_wires; init=Int[])
    for i in 1:(length(wires) - 1), j in (i + 1):length(wires)
        x, y = wires[i], wires[j]
        x > y || continue
        qx, qy = charge(ops[x]), charge(ops[y])
        (qx == unit_sector || qy == unit_sector) && continue
        qxy = _fuse_charge(qx, qy)
        r = Rsymbol(qx, qy, qxy)
        r isa Number || throw(ArgumentError(
            "graded TTNO canonical junction braids require scalar abelian R-symbols; got $(typeof(r))",
        ))
        phase *= r
    end
    return phase
end

function _braid_certificate_debug_enabled()
    value = lowercase(strip(get(ENV, "GRAFT_DEBUG_BRAID_CERTIFICATE", "false")))
    return value in ("1", "true", "yes", "on")
end

function _build_braided_term_plan(t::TreeTopology, E::_Euler, phys,
                                  ops::Dict{Int,SiteOp}, opnodes::Vector{Int},
                                  unit_sector, coefficient)
    canonical_nodes, native_nodes = _physical_node_orders(t, phys)
    canonical_rank = Dict(node => rank for (rank, node) in enumerate(canonical_nodes))
    native_rank = Dict(node => rank for (rank, node) in enumerate(native_nodes))

    factor_class = Dict{Int,Int}()
    odd_count = 0
    unclassifiable_odd = Int[]
    certificate_ready = true
    for node in opnodes
        q = charge(ops[node])
        net = _net_u1_charge(q)
        odd = _has_fermionic_crossing(q, unit_sector)
        odd_count += odd

        # U(1) orientation is authoritative when present.  A parity-only odd
        # factor is still structurally classifiable when all of its nonzero
        # tensor blocks act from one input-twist eigensector: under the same
        # standard-orientation convention as the CG-009 exit scalar, λ=+1 is
        # creation and λ=-1 is annihilation.  Even factors need no orientation
        # because they live in the neutral word class.
        class = if net !== nothing && !iszero(net)
            sign(net)
        elseif odd
            _input_twist_parity(ops[node].op, q)
        else
            0
        end
        if class === nothing
            certificate_ready = false
            push!(unclassifiable_odd, node)
        else
            factor_class[node] = class
        end
    end
    if odd_count >= 4 && !isempty(unclassifiable_odd)
        sites = join((string(nodeid(t, node)) for node in unclassifiable_odd), ", ")
        throw(ArgumentError(
            "graded TTNO terms with four or more fermionic-odd factors require " *
            "every odd factor to have either a nonzero net U(1) charge or one " *
            "structural input-twist parity; factors on $sites have neither, " *
            "and the class-normal braid word cannot be inferred from labels",
        ))
    end

    canonical_word = Int[]
    native_word = Int[]
    if certificate_ready
        canonical_word = sort(copy(opnodes); by=node -> begin
            class = factor_class[node]
            (class > 0 ? 0 : class < 0 ? 1 : 2, canonical_rank[node])
        end)

        # Charged fusion trace: planar child traces, then the charged local
        # factor.  This is the tree's actual abelian fusion order.
        fusion_trace = Int[]
        function trace!(node::Int)
            for child in t.children[node]
                trace!(child)
            end
            haskey(ops, node) && charge(ops[node]) != unit_sector &&
                push!(fusion_trace, node)
            return nothing
        end
        trace!(t.root)
        positives = [node for node in fusion_trace if factor_class[node] > 0]
        negatives = [node for node in fusion_trace if factor_class[node] < 0]
        neutrals = sort(
            [node for node in opnodes if iszero(factor_class[node])];
            by=node -> native_rank[node],
        )
        native_word = [reverse(positives); negatives; neutrals]
    end

    Q = typeof(unit_sector)
    restriction_charge = Vector{Q}(undef, nnodes(t))
    for node in 1:nnodes(t)
        restriction_charge[node] = _fuse_charges(
            (charge(ops[x]) for x in opnodes if _insub(E, x, node)),
            unit_sector,
        )
    end
    frame = _canonical_crossing_frame(t, phys, ops, unit_sector)

    # Stable adjacent swaps lower the native word to the canonical word.  The
    # R-symbol orientation follows the current adjacent pair before it swaps.
    crossings = _WireCrossingEvent{Q}[]
    by_owner = [_WireCrossingEvent{Q}[] for _ in 1:nnodes(t)]
    certificate_scale = one(coefficient)
    if certificate_ready
        work = copy(native_word)
        for (target_position, target) in enumerate(canonical_word)
            current_position = findfirst(==(target), work)
            current_position === nothing && throw(ArgumentError(
                "native TTNO braid word omits physical node $(nodeid(t, target))",
            ))
            while current_position > target_position
                lhs = work[current_position - 1]
                rhs = work[current_position]
                qlhs, qrhs = charge(ops[lhs]), charge(ops[rhs])
                fused = _fuse_charge(qlhs, qrhs)
                scalar = Rsymbol(qlhs, qrhs, fused)
                scalar isa Number || throw(ArgumentError(
                    "graded TTNO braid certificates require scalar abelian R-symbols; got $(typeof(scalar))",
                ))
                owner = _term_lca(t, lhs, rhs)
                event = _WireCrossingEvent(
                    lhs, rhs, owner, qlhs, qrhs, fused, scalar,
                )
                push!(crossings, event)
                push!(by_owner[owner], event)
                certificate_scale *= scalar
                work[current_position - 1], work[current_position] =
                    work[current_position], work[current_position - 1]
                current_position -= 1
            end
        end
        work == canonical_word || throw(ArgumentError(
            "native-to-canonical TTNO braid decomposition did not close",
        ))
        length(Set((event.lhs, event.rhs) for event in crossings)) ==
            length(crossings) || throw(ArgumentError(
                "TTNO braid certificate assigns one wire pair more than once",
            ))
        all(event.owner == _term_lca(t, event.lhs, event.rhs) for event in crossings) ||
            throw(ArgumentError("TTNO braid certificate has a non-LCA crossing owner"))
    end

    legacy_by_node = [
        _legacy_canonical_junction_braid(
            t, ops, opnodes, unit_sector, coefficient, node,
        ) for node in 1:nnodes(t)
    ]
    legacy_scale = prod(legacy_by_node; init=one(coefficient))
    support_lca = foldl((lhs, rhs) -> _term_lca(t, lhs, rhs), opnodes)

    local_plans = _LocalMorphismPlan{Q}[]
    for node in 1:nnodes(t)
        factor_cross = unit_sector
        if haskey(ops, node)
            child = node
            while t.parent[child] != 0
                parent = t.parent[child]
                for sibling in t.children[parent]
                    sibling == child && break
                    factor_cross = _fuse_charge(
                        factor_cross, restriction_charge[sibling],
                    )
                end
                child = parent
            end
        end

        localq = haskey(ops, node) ? charge(ops[node]) : unit_sector
        completion_bend_scale = one(coefficient)
        if certificate_ready && node == support_lca && localq != unit_sector
            occupied_children = count(t.children[node]) do child
                any(factor -> _insub(E, factor, child), opnodes)
            end
            if occupied_children >= 2
                completion_bend_scale = twist(localq)
                completion_bend_scale isa Number || throw(ArgumentError(
                    "graded TTNO local completion bends require scalar " *
                    "abelian twists; got $(typeof(completion_bend_scale))",
                ))
                certificate_scale *= completion_bend_scale
            end
        end
        physical_cross = localq == unit_sector ? _fuse_charges(
            (restriction_charge[child] for child in t.children[node]),
            unit_sector,
        ) : unit_sector
        frame_cross = _frame_at(frame, node, unit_sector)

        exit_cross = unit_sector
        exit_scale = one(coefficient)
        if haskey(ops, node) && localq != unit_sector
            later = _fuse_charges(
                (charge(ops[x]) for x in opnodes
                 if canonical_rank[x] > canonical_rank[node]),
                unit_sector,
            )
            if _has_fermionic_crossing(later, unit_sector)
                exit_cross = later
                exit_scale = _exit_orientation_scalar(
                    ops[node].op, localq, nodeid(t, node),
                )
            end
        end

        word_scale = certificate_ready ?
            prod((event.scalar for event in by_owner[node]);
                 init=completion_bend_scale) : legacy_by_node[node]
        completion_bend = _LocalCompletionBendEvent(
            node, localq, completion_bend_scale,
        )
        charge_exit = _ChargeLegExitEvent(node, exit_cross, exit_scale)
        push!(local_plans, _LocalMorphismPlan(
            node,
            _LocalFactorBendEvent{Q}(node, factor_cross),
            completion_bend,
            _PhysicalInputBendEvent{Q}(node, physical_cross),
            _PhysicalInputFrameEvent{Q}(node, frame_cross),
            charge_exit,
            by_owner[node], word_scale,
        ))
    end
    all(plan.node == node for (node, plan) in enumerate(local_plans)) ||
        throw(ArgumentError("TTNO local morphism plans lost canonical node indexing"))

    if certificate_ready
        local_scale = prod((plan.word_scale for plan in local_plans);
                           init=one(coefficient))
        local_scale == certificate_scale || throw(ArgumentError(
            "TTNO braid certificate local scales do not reproduce its word scale",
        ))
        if _braid_certificate_debug_enabled()
            crossing_summary = [
                (event.lhs, event.rhs, event.owner, event.scalar)
                for event in crossings
            ]
            completion_bend_summary = [
                (plan.node, plan.completion_bend.charge,
                 plan.completion_bend.scalar) for plan in local_plans
                if plan.completion_bend.scalar != one(coefficient)
            ]
            @info "TTNO braid certificate shadow comparison" canonical_word native_word certificate_scale legacy_scale crossing_summary completion_bend_summary
        end
    end

    return _BraidedTermPlan(
        canonical_word, native_word, factor_class, crossings, local_plans,
        restriction_charge, frame, certificate_ready, certificate_scale,
        legacy_scale,
    )
end

"""
    ttno_from_opsum(H::OpSum, topo, phys; elt=ComplexF64, hermitian=false) -> TTNO

Assemble a TTNO from a sum of product terms. `phys :: Dict{Symbol,<:ElementarySpace}`
gives the physical space of every site-carrying node (others become branching
tensors). For graded abelian spaces, charged [`SiteOp`](@ref) factors thread
their fused restriction charge through TTNO virtual channels. Physical spaces
with sector degeneracy (several fermionic modes on one site) are supported for
number-oriented gradings such as fZ2 ⊠ U(1): the charge-leg exit braid is the
per-sector input twist normalized by the ±1 particle-number orientation
(CG-009); unoriented degenerate charged factors fail closed. `hermitian` sets
the `ishermitian` trait on the result — a wrong `true` is a caller bug (§9.8).
"""
function ttno_from_opsum(H::OpSum, topo::TreeTopology, phys::Dict{Symbol,<:ElementarySpace};
                         elt::Type{<:Number}=ComplexF64,
                         hermitian::Bool=false)
    t = topo
    E = _Euler(t)
    N = nnodes(t)
    isempty(phys) && throw(ArgumentError("phys must contain at least one physical space"))
    S = spacetype(first(values(phys)))
    all(P -> spacetype(P) === S, values(phys)) ||
        throw(ArgumentError("TTNO builder requires all physical spaces to share one concrete spacetype"))
    graded = S !== ComplexSpace
    unit_sector = graded ? one(sectortype(first(values(phys)))) : nothing

    # Preprocess terms into node-indexed factors plus a canonical physical-leg
    # braid plan. The public Term vector is never used as an ordering signal.
    terms = map(H.terms) do term
        ops = Dict{Int,SiteOp}()
        for so in term.ops
            n = nodeindex(t, so.site)
            haskey(phys, so.site) || throw(ArgumentError("term factor on $(so.site), which has no physical space"))
            spacetype(codomain(so.op)[1]) == S ||
                throw(ArgumentError("term factor on $(so.site) uses a physical-space symmetry incompatible with `phys`"))
            ops[n] = so
        end
        opnodes = sort!(collect(keys(ops)))
        plan = graded ? _build_braided_term_plan(
            t, E, phys, ops, opnodes, unit_sector, term.coeff,
        ) : nothing
        frame = graded ? plan.frame : _empty_crossing_frame(nothing)
        nodes = sort!(unique!(vcat(
            copy(opnodes), Int[entry.first for entry in frame.at],
        )))
        (; coeff=term.coeff, ops, opnodes, plan, frame, nodes)
    end
    isempty(terms) && throw(ArgumentError("empty OpSum"))

    opmats = Dict{Tuple{Int,Symbol},AbstractTensorMap}()   # (node, opname) -> P ← P
    used = [Set{_ChannelKey}() for _ in 1:N]               # channel usage, edge keyed by child
    chsector = [Dict{_ChannelKey,Any}() for _ in 1:N]       # channel sector, edge keyed by child
    # entry accumulator: (node, αkeys, βkey, local factor, crossing) => coefficient.
    # Keep omitted padding distinct from an explicitly labelled `:I` factor:
    # the latter owns a real SiteOp tensor and must not merge with transport.
    # The native fZ2×U1 crossing charge is part of the entry identity: two
    # terms sharing every channel but crossing the local physical input with
    # different parities write different local matrices into the same block
    # coordinates, which is a legal additive merge. The input `Term.ops`
    # vector never contributes to this metadata: it is derived solely from
    # labelled sites, canonical node indices, and the topology's child order.
    entries = Dict{Tuple{Int,Tuple,_ChannelKey,_EntryLocal,Any},Any}()

    # Canonical scale attached to a state-diagram entry: the uniquely-owned
    # native-to-canonical word crossings, times the charge-leg exit
    # orientation (CG-009). Terms merged into one entry must agree on it.
    entry_scale = Dict{Tuple{Int,Tuple,_ChannelKey,_EntryLocal,Any},Any}()

    restriction_charge(tm, c) = tm.plan.restriction_charge[c]

    # Every edge restriction has one explicit semantic kind. A framed idle is
    # not an active channel with an empty tuple: it carries a unit virtual
    # sector plus the physical-leg braid plan that must be transported below
    # that edge. This tag is internal state-diagram data, never Term.ops order.
    function restriction_key(tm, c::Int)
        frame = graded ? _subframe(tm.frame, E, c, unit_sector) : tm.frame
        opnodes = [x for x in tm.opnodes if _insub(E, x, c)]
        isempty(opnodes) && return isempty(frame) ? _PLAIN_IDLE : _FramedIdle(frame)
        restriction = if graded
            _rkey([
                (x, tm.ops[x].name,
                 tm.plan.local_plans[x].factor_bend.crossing_charge) for x in opnodes
            ])
        else
            _rkey([(x, tm.ops[x].name) for x in opnodes])
        end
        return _Active(restriction, frame)
    end

    function register!(edge::Int, key::_ChannelKey, q)
        push!(used[edge], key)
        if graded
            old = get(chsector[edge], key, nothing)
            if old === nothing
                chsector[edge][key] = q
            elseif old != q
                throw(ArgumentError("state-diagram channel $key on edge $(nodeid(t, edge)) has inconsistent charges $old and $q"))
            end
        end
        return key
    end

    for tm in terms
        for so in values(tm.ops)
            get!(opmats, (nodeindex(t, so.site), so.name), so.op)
        end
        lca = tm.nodes[1]
        while !all(x -> _insub(E, x, lca), tm.nodes)
            lca = t.parent[lca]
        end
        # per-term entries exist at nodes n in subtree(lca) where the term is
        # not confined to a single child branch: exactly lca itself and the
        # "spine" nodes below it where the term is partially present.
        active = Set{Int}(tm.nodes)
        # climb from every site towards lca, collecting pass-through nodes
        for s in tm.nodes
            x = s
            while x != lca
                x = t.parent[x]
                push!(active, x)
            end
        end
        for n in sort!(collect(active))
            # skip nodes where the whole term sits inside one child branch
            any(c -> all(x -> _insub(E, x, c), tm.nodes), t.children[n]) && continue
            αkeys = _ChannelKey[]
            αcharges = Any[]
            for c in t.children[n]
                child_key = restriction_key(tm, c)
                push!(αkeys, child_key)
                if graded
                    child_charge = _is_active(child_key) ?
                        restriction_charge(tm, c) : unit_sector
                    push!(αcharges, child_charge)
                end
            end
            complete = all(x -> _insub(E, x, n), tm.nodes)
            βkey::_ChannelKey = complete ? _DONE : restriction_key(tm, n)
            βcharge = if graded
                complete || !_is_active(βkey) ? unit_sector : restriction_charge(tm, n)
            else
                nothing
            end
            localentry = haskey(tm.ops, n) ? _ExplicitLocal(tm.ops[n].name) :
                _OMITTED_IDENTITY
            localq = graded ? (haskey(tm.ops, n) ? charge(tm.ops[n]) : unit_sector) : nothing
            if graded
                lhs = _fuse_charges(αcharges, unit_sector)
                lhs = _fuse_charge(lhs, localq)
                lhs == βcharge ||
                    throw(ArgumentError("state-diagram charge mismatch at $(nodeid(t, n)): children/local fuse to $lhs but parent channel has $βcharge"))
            end
            for (c, k, q) in zip(t.children[n], αkeys, graded ? αcharges : fill(nothing, length(αkeys)))
                register!(c, k, q)
            end
            t.parent[n] == 0 ? (@assert complete) : register!(n, βkey, βcharge)
            crossing = nothing
            scale = 1
            if graded
                morphism = tm.plan.local_plans[n]
                local_cross = haskey(tm.ops, n) && localq != unit_sector ?
                    morphism.factor_bend.crossing_charge : unit_sector
                node_cross = morphism.physical_bend.crossing_charge
                frame_cross = morphism.input_frame.crossing_charge
                exit_cross = morphism.charge_exit.crossing_charge
                crossing = _fuse_charge(
                    _fuse_charge(_fuse_charge(local_cross, node_cross), frame_cross),
                    exit_cross,
                )
                scale = morphism.charge_exit.orientation_scalar *
                    morphism.word_scale
            end
            key = (n, Tuple(αkeys), βkey, localentry, crossing)
            previous_scale = get(entry_scale, key, nothing)
            if previous_scale === nothing
                entry_scale[key] = scale
            elseif previous_scale != scale
                throw(ArgumentError(
                    "state-diagram entry at $(nodeid(t, n)) has inconsistent canonical scales",
                ))
            end
            if complete
                entries[key] = get(entries, key, zero(tm.coeff)) + tm.coeff * scale
            else
                # A non-completion merge is still a physical fusion junction;
                # retain its canonical scale rather than postponing it to the
                # term's completion node.
                entries[key] = scale
            end
        end
    end

    # DONE propagation upward: a completed term must ride the done channel on
    # every edge up to the root; done-pass entries reference idle on siblings.
    for n in postorder(t)
        t.parent[n] == 0 && continue
        if _DONE in used[n] && t.parent[t.parent[n]] != 0
            register!(t.parent[n], _DONE, unit_sector)
        end
    end
    for p in 1:N, c in t.children[p]
        if _DONE in used[c]
            for c2 in t.children[p]
                c2 == c || register!(c2, _PLAIN_IDLE, unit_sector)
            end
        end
    end
    # IDLE propagation downward: an idle edge needs identity flowing through
    # every node below it.
    for n in preorder(t)
        t.parent[n] == 0 && continue
        if _PLAIN_IDLE in used[n]
            for c in t.children[n]
                register!(c, _PLAIN_IDLE, unit_sector)
            end
        end
    end
    # transport entries
    transport_cross = graded ? unit_sector : nothing
    for p in 1:N
        for c in t.children[p]
            if _DONE in used[c]
                αkeys = _ChannelKey[c2 == c ? _DONE : _PLAIN_IDLE for c2 in t.children[p]]
                entries[(p, Tuple(αkeys), _DONE, _OMITTED_IDENTITY, transport_cross)] = 1
            end
        end
    end
    for n in 1:N
        t.parent[n] == 0 && continue
        if _PLAIN_IDLE in used[n]
            αkeys = ntuple(_ -> _PLAIN_IDLE, nchildren(t, n))
            entries[(n, αkeys, _PLAIN_IDLE, _OMITTED_IDENTITY, transport_cross)] = 1
        end
    end

    # channel index assignment (deterministic: idle, done, then sorted actives)
    chindex = [Dict{_ChannelKey,Int}() for _ in 1:N]
    chcoord = [Dict{_ChannelKey,Int}() for _ in 1:N]
    vspaces = Vector{S}(undef, N)
    for c in 1:N
        if t.parent[c] == 0
            vspaces[c] = oneunit(S)
            continue
        end
        ordered = sort!(collect(used[c]); by=_channel_sort_key)
        for (i, k) in enumerate(ordered)
            chindex[c][k] = i
        end
        if graded
            vspaces[c], chcoord[c] = _channel_layout(S, ordered, chsector[c], unit_sector)
        else
            vspaces[c] = ComplexSpace(max(length(ordered), 1))
            for (i, k) in enumerate(ordered)
                chcoord[c][k] = i
            end
        end
    end

    # assemble dense per-node tensors
    unit = oneunit(S)
    tensors = map(1:N) do n
        K = nchildren(t, n)
        hp = haskey(phys, nodeid(t, n))
        P = hp ? phys[nodeid(t, n)] : nothing
        d = hp ? dim(P) : 1
        χp = t.parent[n] == 0 ? 1 : dim(vspaces[n])
        cods = S[]
        for c in t.children[n]
            push!(cods, vspaces[c])
        end
        hp && push!(cods, P)
        cod = isempty(cods) ? one(unit) : reduce(⊗, cods)
        Vp = t.parent[n] == 0 ? oneunit(S) : vspaces[n]
        doms = hp ? S[P, Vp] : S[Vp]
        dom = hp ? P ⊗ Vp : ProductSpace(Vp)

        if graded
            W = zeros(elt, cod ← dom)
            blockmap = Dict{typeof(unit_sector),Any}(q => b for (q, b) in blocks(W))
            _, codcoord = _sector_tuple_groups(cods, unit_sector)
            _, domcoord = _sector_tuple_groups(doms, unit_sector)
            for ((m, αkeys, βkey, localentry, crossing), coeff) in entries
                m == n || continue
                αidx = ntuple(k -> chcoord[t.children[n][k]][αkeys[k]], K)
                βidx = t.parent[n] == 0 ? 1 : chcoord[n][βkey]
                if hp
                    mat = _graded_siteop_matrix(
                        opmats, n, localentry, elt, P, crossing, unit_sector,
                    )
                    for pout in 1:d, pin in 1:d
                        val = elt(coeff) * mat[pout, pin]
                        iszero(val) && continue
                        _add_block_entry!(blockmap, codcoord, domcoord,
                                          (αidx..., pout), (pin, βidx), val)
                    end
                else
                    localentry === _OMITTED_IDENTITY || throw(ArgumentError(
                        "operator factor on branching node $(nodeid(t, n))",
                    ))
                    _add_block_entry!(blockmap, codcoord, domcoord,
                                      αidx, (βidx,), elt(coeff))
                end
            end
            W
        else
            dims = (ntuple(k -> dim(vspaces[t.children[n][k]]), K)..., (hp ? (d, d) : ())..., χp)
            W = zeros(elt, dims)
            for ((m, αkeys, βkey, localentry, _), coeff) in entries
                m == n || continue
                αidx = ntuple(k -> chcoord[t.children[n][k]][αkeys[k]], K)
                βidx = t.parent[n] == 0 ? 1 : chcoord[n][βkey]
                if hp
                    mat = _siteop_matrix(opmats, n, localentry, elt, d)
                    view(W, αidx..., :, :, βidx) .+= elt(coeff) .* mat
                else
                    localentry === _OMITTED_IDENTITY || throw(ArgumentError(
                        "operator factor on branching node $(nodeid(t, n))",
                    ))
                    W[αidx..., βidx] += elt(coeff)
                end
            end
            TensorMap(W, cod ← dom)
        end
    end

    return TTNO(t, tensors; ishermitian=hermitian)
end

_eye(::Type{T}, d::Int) where {T} = T[i == j ? one(T) : zero(T) for i in 1:d, j in 1:d]

function _siteop_matrix(opmats, n::Int, localentry::_EntryLocal,
                        ::Type{T}, d::Int) where {T}
    localentry === _OMITTED_IDENTITY && return _eye(T, d)
    opname = (localentry::_ExplicitLocal).name
    op = opmats[(n, opname)]
    if numout(op) == 1 && numin(op) == 1
        arr = _tensor_dense_from_blocks(op)
        return T.(reshape(arr, d, d))
    elseif numout(op) == 1 && numin(op) == 2 && dim(domain(op)[2]) == 1
        arr = reshape(_tensor_dense_from_blocks(op), d, d, :)
        size(arr, 3) == 1 || throw(ArgumentError("charged SiteOp charge leg must be one-dimensional"))
        return T.(arr[:, :, 1])
    else
        throw(ArgumentError("SiteOp tensor for `$opname` must be `P ← P` or charged `P ← P ⊗ C`"))
    end
end

function _has_fermionic_crossing(q, unit_sector)
    q == unit_sector && return false
    θ = twist(q)
    θ == one(θ) && return false
    θ == -one(θ) || throw(ArgumentError(
        "graded TTNO canonical crossings currently support only fZ2×abelian twists; got twist $θ for charge $q",
    ))
    return true
end

"All U1Irrep components of an abelian (product) sector."
function _u1_components(q)
    q isa U1Irrep && return (q,)
    hasproperty(q, :sectors) || return ()
    components = ()
    for part in q.sectors
        components = (components..., _u1_components(part)...)
    end
    return components
end

"Summed U(1) charge over every U1Irrep component, or `nothing` without one."
function _net_u1_charge(q)
    components = _u1_components(q)
    isempty(components) && return nothing
    total = sum(float(component.charge) for component in components)
    return isinteger(total) ? Int(total) : nothing
end

"Structural physical-input twist parity of a charged SiteOp tensor: ±1 when
every nonzero block shares one input-sector parity (every one-mode carrier
does), `nothing` otherwise."
function _input_twist_parity(op::AbstractTensorMap, q)
    parity = nothing
    for (coupled, b) in blocks(op)
        all(iszero, b) && continue
        θ = twist(_fuse_charge(coupled, dual(q)))
        sign = θ == one(θ) ? 1 : θ == -one(θ) ? -1 : return nothing
        parity === nothing || parity == sign || return nothing
        parity = sign
    end
    return parity === nothing ? 1 : parity
end

# The scalar `s(q)` normalizing the charge-leg exit braid `s(q)·twist` to the
# identity on one-mode carriers: the input twist eigenvalue a one-mode factor
# of charge `q` would have. A net particle creation acts out of the even
# sector (+1), a net annihilation out of the odd sector (-1). Without a U(1)
# orientation the sector label cannot distinguish the two (fZ2 alone is
# self-dual), so a sector-degenerate carrier fails closed instead of guessing;
# a twist-eigenvector factor is exactly the one-mode-like case where the
# normalized braid is the identity regardless of orientation.
function _exit_orientation_scalar(op::AbstractTensorMap, q, site)
    net = _net_u1_charge(q)
    net == 1 && return 1
    net == -1 && return -1
    parity = _input_twist_parity(op, q)
    parity === nothing && throw(ArgumentError(
        "charged factor on $site braids out of a sector-degenerate physical " *
        "space, but its charge $q carries no ±1 particle-number orientation; " *
        "use an fZ2 ⊠ U(1) number-graded carrier (CG-009)",
    ))
    return parity
end

"Native fZ2×U1 braid specialization on a local physical input leg."
function _graded_siteop_matrix(opmats, n::Int, localentry::_EntryLocal, ::Type{T},
                                P::ElementarySpace, crossing, unit_sector) where {T}
    op = localentry === _OMITTED_IDENTITY ? one(T) * id(P) :
        opmats[(n, (localentry::_ExplicitLocal).name)]
    # For an odd carried charge, braiding it past the local physical input is
    # exactly TensorKit's pivotal `twist` on that input in fZ2×U1. This is a
    # tensor operation, not an empirical scalar: it gives -C, +Cd, and the
    # required spectator-parity map for a neutral local entry.
    _has_fermionic_crossing(crossing, unit_sector) &&
        (op = twist(op, numout(op) + 1))
    d = dim(P)
    if numout(op) == 1 && numin(op) == 1
        arr = _tensor_dense_from_blocks(op)
        return T.(reshape(arr, d, d))
    elseif numout(op) == 1 && numin(op) == 2 && dim(domain(op)[2]) == 1
        arr = reshape(_tensor_dense_from_blocks(op), d, d, :)
        size(arr, 3) == 1 || throw(ArgumentError(
            "charged SiteOp charge leg must be one-dimensional",
        ))
        return T.(arr[:, :, 1])
    end
    throw(ArgumentError(
        "SiteOp tensor for `$localentry` must be `P ← P` or charged `P ← P ⊗ C`",
    ))
end

function _fuse_charge(a, b)
    a === nothing && return nothing
    fused = a ⊗ b
    length(fused) == 1 ||
        throw(ArgumentError("non-abelian TTNO charge bookkeeping needs SU2Reduce/graded fusion-tree support (TODO(M3))"))
    return only(fused)
end

function _fuse_charges(qs, unit_sector)
    unit_sector === nothing && return nothing
    qtot = unit_sector
    for q in qs
        qtot = _fuse_charge(qtot, q)
    end
    return qtot
end

function _channel_layout(::Type{S}, ordered::Vector{_ChannelKey},
                         sector_of::Dict{_ChannelKey,Any}, unit_sector) where {S<:ElementarySpace}
    if isempty(ordered)
        return oneunit(S), Dict{_ChannelKey,Int}()
    end
    Q = typeof(unit_sector)
    groups = Dict{Q,Vector{_ChannelKey}}()
    for key in ordered
        q = sector_of[key]
        dim(Vect[Q](q => 1)) == 1 ||
            throw(ArgumentError("TTNO virtual channels currently require abelian one-dimensional sectors (TODO(M3))"))
        push!(get!(groups, q, _ChannelKey[]), key)
    end
    pairs = [q => length(groups[q]) for q in sort!(collect(keys(groups)); by=string)]
    V = Vect[Q](pairs...)
    coord = Dict{_ChannelKey,Int}()
    offset = 0
    for q in sectors(V)
        for (j, key) in enumerate(groups[q])
            coord[key] = offset + j
        end
        offset += length(groups[q])
    end
    return V, coord
end

function _basis_sectors(V::ElementarySpace, unit_sector)
    qs = typeof(unit_sector)[]
    for q in sectors(V)
        for _ in 1:dim(V, q)
            push!(qs, q)
        end
    end
    return qs
end

function _sector_tuple_groups(legs::Vector{S}, unit_sector) where {S<:ElementarySpace}
    Q = typeof(unit_sector)
    groups = Dict{Q,Vector{Tuple}}()
    coord = Dict{Tuple,Tuple{Q,Int}}()
    if isempty(legs)
        groups[unit_sector] = [()]
        coord[()] = (unit_sector, 1)
        return groups, coord
    end
    # TensorKit block-row layout: abelian fusion trees (uncoupled sector
    # tuples) iterate first-leg-fastest, and each tree owns one contiguous
    # row range with degeneracy indices column-major inside it. A plain
    # column-major sweep over basis positions coincides with this only while
    # every sector is one-dimensional; with sector degeneracy it interleaves
    # rows of different trees and silently permutes degenerate states.
    K = length(legs)
    legsectors = [collect(sectors(V)) for V in legs]
    legoffsets = map(legs) do V
        offsets = Dict{Q,Int}()
        offset = 0
        for s in sectors(V)
            offsets[s] = offset
            offset += dim(V, s)
        end
        offsets
    end
    for T in CartesianIndices(Tuple(length.(legsectors)))
        secs = ntuple(j -> legsectors[j][T[j]], K)
        q = unit_sector
        for s in secs
            q = _fuse_charge(q, s)
        end
        rows = get!(groups, q, Tuple[])
        for D in CartesianIndices(ntuple(j -> dim(legs[j], secs[j]), K))
            idx = ntuple(j -> legoffsets[j][secs[j]] + D[j], K)
            push!(rows, idx)
            coord[idx] = (q, length(rows))
        end
    end
    return groups, coord
end

function _add_block_entry!(blockmap, codcoord, domcoord,
                           codidx::Tuple, domidx::Tuple, val)
    cq, row = codcoord[codidx]
    dq, col = domcoord[domidx]
    cq == dq || return nothing
    blockmap[cq][row, col] += val
    return nothing
end

function _tensor_dense_from_blocks(t::AbstractTensorMap)
    T = scalartype(t)
    dims = ntuple(i -> dim(space(t, i)), numind(t))
    arr = zeros(T, dims)
    if spacetype(t) === ComplexSpace
        return convert(Array, t)
    end
    unit_sector = one(sectortype(space(t, 1)))
    codlegs = [codomain(t)[i] for i in 1:numout(t)]
    domlegs = [domain(t)[i] for i in 1:numin(t)]
    codgroups, _ = _sector_tuple_groups(codlegs, unit_sector)
    domgroups, _ = _sector_tuple_groups(domlegs, unit_sector)
    for (q, b) in blocks(t)
        for row in axes(b, 1), col in axes(b, 2)
            codidx = codgroups[q][row]
            domidx = domgroups[q][col]
            arr[codidx..., domidx...] = b[row, col]
        end
    end
    return arr
end
