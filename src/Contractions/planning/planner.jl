function _heuristic_tree(spec::ContractionSpec)
    slots = spec.preferred_slots
    isempty(slots) && throw(ArgumentError("contraction spec has no input slots"))
    tree = spec.dynamic_slot === nothing ? first(slots) : spec.dynamic_slot
    first_slot = spec.dynamic_slot === nothing ? 2 : 1
    for slot in Iterators.drop(slots, first_slot - 1)
        tree = Any[tree, slot]
    end
    return tree
end

function _optimaltree(spec::ContractionSpec, label_dims::Dict{Int,Int})
    # TensorOperations optimizes dense FLOPs. We later score its candidate
    # ourselves and retain the Phase-1 memory-safe plan if its peak is worse.
    optdata = Dict{Int,Float64}(label => Float64(d) for (label, d) in label_dims)
    return TensorOperations.optimaltree(spec.labels, optdata)[1]
end

"""
Deterministic dense fallback for unusually large local networks. It is not a
replacement for TensorOperations' exact optimizer: it only supplies a bounded
candidate, and the Phase-1 env-first plan remains the memory-safe floor during
selection below.
"""
function _greedy_tree(spec::ContractionSpec, dims::Vector{Vector{Int}})
    nodes = [(tree=i, labels=copy(spec.labels[i]), dims=copy(dims[i]))
             for i in eachindex(spec.labels)]
    while length(nodes) > 1
        best_i, best_j = 0, 0
        best_metrics = nothing
        best_score = (Inf, Inf)
        # Prefer genuine contractions. Only form an outer product when the
        # remaining network is disconnected, matching ncon's last-resort rule.
        connected = false
        for i in 1:(length(nodes) - 1), j in (i + 1):length(nodes)
            shared = !isempty(Set(nodes[i].labels) ∩ Set(nodes[j].labels))
            connected |= shared
            !shared && continue
            metrics = dense_cost(nodes[i].labels, nodes[i].dims,
                                 nodes[j].labels, nodes[j].dims)
            score = (metrics.peak_elements, metrics.flops)
            if score < best_score
                best_i, best_j, best_metrics, best_score = i, j, metrics, score
            end
        end
        if !connected
            best_i, best_j = length(nodes) - 1, length(nodes)
            best_metrics = dense_cost(nodes[best_i].labels, nodes[best_i].dims,
                                      nodes[best_j].labels, nodes[best_j].dims)
        end
        merged = (tree=Any[nodes[best_i].tree, nodes[best_j].tree],
                  labels=best_metrics.labels, dims=best_metrics.dims)
        nodes[best_i] = merged
        deleteat!(nodes, best_j)
    end
    return only(nodes).tree
end

"""
Small, bounded dense forest state for the hard-cap fallback.  It models only
reclaimable intermediate payloads: every leaf operand is an order-independent
persistent baseline in the full live-byte model, so including it here would
not improve the beam ordering.
"""
struct _MemoryBeamNode
    tree::Any
    labels::Vector{Int}
    dims::Vector{Int}
    payload::Float64
    leaf::Bool
end

struct _MemoryBeamState
    nodes::Vector{_MemoryBeamNode}
    live::Float64
    peak::Float64
    flops::Float64
end

"""
    _memory_beam_trees(spec, dims; width=16) -> Vector{Any}

Deterministic bounded fallback used only when a caller supplied a hard memory
cap above the exact-DP limit.  It explores forest merges by simulated live
intermediate payload before FLOPs, retaining at most `width` states at each
depth.  Final candidates are still compiled and checked against the exact
byte model and the env-first floors, so this heuristic can never permit an
over-cap plan merely because its coarse ordering estimate was optimistic.
"""
function _memory_beam_trees(spec::ContractionSpec, dims::Vector{Vector{Int}};
                            width::Integer=16)
    width > 0 || throw(ArgumentError("memory beam width must be positive"))
    leaves = _MemoryBeamNode[
        _MemoryBeamNode(i, copy(spec.labels[i]), copy(dims[i]),
                        _prod_dims(dims[i]), true)
        for i in eachindex(spec.labels)
    ]
    states = _MemoryBeamState[_MemoryBeamState(leaves, 0.0, 0.0, 0.0)]
    while any(length(state.nodes) > 1 for state in states)
        candidates = _MemoryBeamState[]
        for state in states
            length(state.nodes) > 1 || begin
                push!(candidates, state)
                continue
            end
            pairs = Tuple{Int,Int}[]
            for i in 1:(length(state.nodes) - 1), j in (i + 1):length(state.nodes)
                _has_shared_positive_label(state.nodes[i].labels,
                                           state.nodes[j].labels) &&
                    push!(pairs, (i, j))
            end
            if isempty(pairs)
                for i in 1:(length(state.nodes) - 1), j in (i + 1):length(state.nodes)
                    push!(pairs, (i, j))
                end
            end
            for (i, j) in pairs
                left, right = state.nodes[i], state.nodes[j]
                metrics = dense_cost(left.labels, left.dims, right.labels, right.dims)
                released = (left.leaf ? 0.0 : left.payload) +
                           (right.leaf ? 0.0 : right.payload)
                # One result payload is always live during the contraction.
                # The complete compiler subsequently adds known permutation
                # buffers, which is why this beam remains a candidate source
                # rather than an authority on cap compliance.
                peak = max(state.peak, state.live + metrics.peak_elements)
                merged = _MemoryBeamNode(Any[left.tree, right.tree],
                                         metrics.labels, metrics.dims,
                                         metrics.peak_elements, false)
                nodes = _MemoryBeamNode[]
                for k in eachindex(state.nodes)
                    k == i || k == j || push!(nodes, state.nodes[k])
                end
                push!(nodes, merged)
                push!(candidates,
                      _MemoryBeamState(nodes,
                                       state.live - released + merged.payload,
                                       peak, state.flops + metrics.flops))
            end
        end
        sort!(candidates; by=state -> (state.peak, state.flops))
        keep = min(Int(width), length(candidates))
        states = candidates[1:keep]
        all(length(state.nodes) == 1 for state in states) && break
    end
    return Any[only(state.nodes).tree for state in states]
end

"""
Structural DP state used only by the Phase-3 unique-fusion planner.
`space` is the exact TensorKit HomSpace of the all-codomain intermediate, so
states with equal label sets but different fusion/leg layouts cannot be
incorrectly merged.  It is intentionally local to Planning: the long-lived
EnvCache stores only the final executable `ContractionPlan`.
"""
struct _SectorDPState
    tree::Any
    labels::Vector{Int}
    dims::Vector{Int}
    space::Any
    conj::Bool
    flops::Float64
    peak::Float64
    sector_flops::Float64
    sector_peak::Float64
    sector_block_peak::Float64
end

# The dense TensorOperations candidate and the exact sector DP deliberately
# share this local-network limit.  Above it, the Phase-1 env-first plan and
# the Phase-2 dense candidates remain the bounded, memory-safe fallback.
const _EXACT_TENSOR_LIMIT = 10
const _SECTOR_EXACT_TENSOR_LIMIT = _EXACT_TENSOR_LIMIT

@inline function _has_shared_positive_label(a::Vector{Int}, b::Vector{Int})
    return any(label -> label > 0 && label in b, a)
end

@inline _canonical_label_key(label::Int) = label > 0 ? (0, label) : (1, -label)

"""
    _canonical_intermediate_partition(labels) -> pAB

Materialize every non-root intermediate in a deterministic flat-leg order:
remaining contracted labels first in ascending order, then open labels in
`-1, -2, …` order.  Phase 3 is restricted to unique-fusion symmetric braiding
and currently fixes the permutation coefficient to zero, so TensorKit's
braiding makes this canonicalization an exact representative of either input
operand order.  It also prevents a high-degree star's harmless leaf-order
permutations from creating factorially many DP states.
"""
function _canonical_intermediate_partition(labels::Vector{Int})
    order = sortperm(eachindex(labels); by=i -> _canonical_label_key(labels[i]))
    return (Tuple(order), ())
end

function _reordered_metrics(metrics, out::Tuple)
    order = Int[out[1]...]
    return metrics.labels[order], metrics.dims[order]
end

@inline _sector_structure_key(s::_SectorDPState) =
    (Tuple(s.labels), Tuple(s.dims), s.space, s.conj)

@inline _same_sector_structure(a::_SectorDPState, b::_SectorDPState) =
    _sector_structure_key(a) == _sector_structure_key(b)

@inline function _sector_no_worse(a::_SectorDPState, b::_SectorDPState)
    # `sector_block_peak` is a reported allocator/workspace diagnostic, not a
    # configured objective or hard guard.  It must not retain otherwise
    # dominated states and turn the exact λ_mem frontier into a larger,
    # unselected multi-objective search.
    no_worse = a.peak <= b.peak &&
               a.sector_peak <= b.sector_peak &&
               a.sector_flops <= b.sector_flops
    return no_worse
end

"""Insert a state into its exact-layout Pareto frontier without approximation."""
function _insert_sector_state!(frontier::Vector{_SectorDPState}, candidate::_SectorDPState)
    same = Int[]
    for (i, current) in enumerate(frontier)
        _same_sector_structure(current, candidate) || continue
        # Equal objective tuples are interchangeable for every later merge,
        # because the structural state is equal too.  Keep the first tree to
        # avoid duplicate derivations without dropping a Pareto alternative.
        _sector_no_worse(current, candidate) && return frontier
        push!(same, i)
    end
    for i in reverse(same)
        _sector_no_worse(candidate, frontier[i]) && deleteat!(frontier, i)
    end
    push!(frontier, candidate)
    return frontier
end

function _sector_merge_state(left::_SectorDPState, right::_SectorDPState,
                             profiles::Dict{Any,Any})
    metrics = dense_cost(left.labels, left.dims, right.labels, right.dims)
    out = _canonical_intermediate_partition(metrics.labels)
    labels, dims = _reordered_metrics(metrics, out)
    key = (_sector_structure_key(left), _sector_structure_key(right))
    profile = get!(profiles, key) do
        _sector_pair_profile(metrics, left.space, left.conj,
                             right.space, right.conj, out)
    end
    profile.supported || return nothing
    return _SectorDPState(Any[left.tree, right.tree], labels, dims,
                          profile.output, false,
                          left.flops + right.flops + metrics.flops,
                          max(left.peak, right.peak, metrics.peak_elements),
                          left.sector_flops + right.sector_flops + profile.sector_flops,
                          max(left.sector_peak, right.sector_peak,
                              profile.output_elements),
                          max(left.sector_block_peak, right.sector_block_peak,
                              profile.peak_block_elements))
end

"""
    _sector_dp_trees(spec, dims, protos) -> Vector{Any}

Generate every Pareto-relevant *connected* binary contraction tree for an
eligible local network of at most `_SECTOR_EXACT_TENSOR_LIMIT` tensors.  Each
non-root result is materialized in a canonical leg order.  In the supported
unique-fusion, symmetric-braiding, `λ_perm = 0` model, this makes reversed
operand layouts structurally equivalent instead of factorially distinct DP
states.  Only mathematically dominated canonical states are pruned.  An early
outer product in a connected effective-map graph is Pareto-dominated by
delaying that product until one component reaches its bridge tensor, so shared
positive-label merges span the exact optimum under this model.  The Phase-1
env-first plan stays the bounded memory-safe fallback outside this deliberately
small local scope.
"""
Base.@noinline function _sector_dp_trees(spec::ContractionSpec,
                                         dims::Vector{Vector{Int}}, protos)
    n = length(spec.labels)
    n <= _SECTOR_EXACT_TENSOR_LIMIT || return Any[]
    spaces = [_prototype_space(proto) for proto in protos]
    all(Backend.sector_cost_supported, spaces) || return Any[]
    any(Backend.sector_cost_nontrivial, spaces) || return Any[]

    fullmask = (1 << n) - 1
    frontiers = [_SectorDPState[] for _ in 1:fullmask]
    profiles = Dict{Any,Any}()
    for i in 1:n
        space_i = spaces[i]
        frontiers[1 << (i - 1)] = [_SectorDPState(
            i, copy(spec.labels[i]), copy(dims[i]), space_i, spec.conjs[i],
            0.0, _prod_dims(dims[i]), 0.0, Float64(dim(space_i)),
            Backend.sector_block_peak(space_i),
        )]
    end

    for mask in 1:fullmask
        count_ones(mask) < 2 && continue
        leftmask = (mask - 1) & mask
        while leftmask != 0
            rightmask = mask ⊻ leftmask
            # `_canonical_intermediate_partition` materializes an exact
            # canonical representative of either operand order under the
            # supported symmetric-braiding cost model, so each unordered
            # bipartition is considered once.
            if rightmask != 0 && leftmask < rightmask &&
               !isempty(frontiers[leftmask]) && !isempty(frontiers[rightmask])
                for left in frontiers[leftmask], right in frontiers[rightmask]
                    # An early outer product in a connected effective-map
                    # graph can always be delayed until one component reaches
                    # its bridge tensor. In this λ_perm=0 symmetric model it
                    # cannot improve work or either peak metric.
                    _has_shared_positive_label(left.labels, right.labels) || continue
                    merged = _sector_merge_state(left, right, profiles)
                    merged === nothing || _insert_sector_state!(frontiers[mask], merged)
                end
            end
            leftmask = (leftmask - 1) & mask
        end
    end
    return Any[state.tree for state in frontiers[fullmask]]
end

mutable struct _CompileState
    nextslot::Int
    steps::Vector{PairStep}
    flops::Float64
    peak::Float64
    sector_flops::Float64
    sector_peak::Float64
    sector_block_peak::Float64
    output_dense_payloads::Vector{Float64}
    output_sector_payloads::Vector{Float64}
    temporary_dense_payloads::Vector{Float64}
    permutation_dense_payloads::Vector{Float64}
    temporary_sector_payloads::Vector{Float64}
    permutation_sector_payloads::Vector{Float64}
end

function _output_partition(source_labels::Vector{Int}, spec::ContractionSpec)
    wanted = Int[-i for i in 1:spec.nopen]
    length(source_labels) == length(wanted) ||
        throw(ArgumentError("planned root has $(length(source_labels)) open legs; expected $(length(wanted))"))
    positions = Int[]
    for label in wanted
        pos = findfirst(==(label), source_labels)
        pos === nothing && throw(ArgumentError("planned root is missing output label $label"))
        push!(positions, pos)
    end
    nout, nin = spec.out_partition
    return (Tuple(positions[1:nout]), Tuple(positions[(nout + 1):(nout + nin)]))
end

function _compile_tree!(state::_CompileState, tree, spec::ContractionSpec,
                        dims::Vector{Vector{Int}}, protos,
                        structural_metrics::Bool,
                        canonical_intermediates::Bool,
                        root::Bool=false)
    if tree isa Integer
        slot = Int(tree)
        space_ = _prototype_space(protos[slot])
        dense_payload = _prod_dims(dims[slot])
        return slot, copy(spec.labels[slot]), copy(dims[slot]), space_,
               spec.conjs[slot], dense_payload,
               _stored_payload_elements(space_, dense_payload)
    end
    length(tree) == 2 || throw(ArgumentError("contraction tree nodes must be binary"))
    a, labels_a, dims_a, space_a, conj_a, dense_a, sector_a =
        _compile_tree!(state, tree[1], spec, dims, protos,
                       structural_metrics, canonical_intermediates)
    b, labels_b, dims_b, space_b, conj_b, dense_b, sector_b =
        _compile_tree!(state, tree[2], spec, dims, protos,
                       structural_metrics, canonical_intermediates)
    metrics = dense_cost(labels_a, dims_a, labels_b, dims_b)
    state.nextslot += 1
    dst = state.nextslot
    out = root ? _output_partition(metrics.labels, spec) :
                 (canonical_intermediates ?
                  _canonical_intermediate_partition(metrics.labels) :
                  (Tuple(1:length(metrics.labels)), ()))
    out_labels, out_dims = root || !canonical_intermediates ?
                           (metrics.labels, metrics.dims) :
                           _reordered_metrics(metrics, out)
    profile = structural_metrics ?
              _sector_pair_profile(metrics, space_a, conj_a, space_b, conj_b, out) :
              nothing
    dense_out = metrics.peak_elements
    sector_out = profile === nothing ? dense_out : profile.output_elements
    transforms = _known_transform_payloads(metrics, out,
                                           dense_a, dense_b, dense_out,
                                           sector_a, sector_b, sector_out,
                                           length(dims_a), length(dims_b),
                                           conj_a, conj_b; profile=profile)
    push!(state.steps,
          PairStep(a, b, dst,
                   Tuple(metrics.oindA), Tuple(metrics.cindA),
                   Tuple(metrics.oindB), Tuple(metrics.cindB),
                   conj_a, conj_b, out))
    push!(state.output_dense_payloads, dense_out)
    push!(state.output_sector_payloads, sector_out)
    push!(state.temporary_dense_payloads, transforms.temporary_dense)
    push!(state.permutation_dense_payloads, transforms.permutation_dense)
    push!(state.temporary_sector_payloads, transforms.temporary_sector)
    push!(state.permutation_sector_payloads, transforms.permutation_sector)
    state.flops += metrics.flops
    state.peak = max(state.peak, metrics.peak_elements)
    if profile === nothing
        # The trivial/no-sector case is exactly one dense block.  Retain
        # finite, comparable diagnostics without trying to compose synthetic
        # HomSpaces whose arrow metadata is intentionally absent from a
        # dimensions-only fixture.
        state.sector_flops += 2 * metrics.flops
        state.sector_peak = max(state.sector_peak, metrics.peak_elements)
        state.sector_block_peak = max(state.sector_block_peak, metrics.peak_elements)
        return dst, out_labels, out_dims, nothing, false, dense_out, sector_out
    elseif profile.supported && isfinite(state.sector_flops)
        state.sector_flops += profile.sector_flops
    else
        state.sector_flops = NaN
    end
    state.sector_peak = max(state.sector_peak, profile.output_elements)
    state.sector_block_peak = max(state.sector_block_peak,
                                  profile.peak_block_elements)
    return dst, out_labels, out_dims, profile.output, false, dense_out, sector_out
end

"""
    _compiled_live_memory_metrics(state, dims, spaces, scalar_bytes)

Model the executor's allocation high-water mark rather than only the largest
result.  Leaf operands are a persistent baseline: the caller owns the dynamic
input and `EffectiveMap.statics` owns every static input, so clearing executor
slots cannot release them.  Internal outputs follow the plan's postorder
liveness.  At a binary step both source intermediates, the new output, and
the explicit known transformation buffers are live together.
"""
function _compiled_live_memory_metrics(state::_CompileState,
                                       dims::Vector{Vector{Int}}, spaces,
                                       scalar_bytes::Int)
    dense_operand_elements = sum(_prod_dims(d) for d in dims; init=0.0)
    sector_operand_elements = sum(
        _stored_payload_elements(space_, _prod_dims(dims[i]))
        for (i, space_) in enumerate(spaces);
        init=0.0,
    )
    dense_live = Dict{Int,Float64}()
    sector_live = Dict{Int,Float64}()
    dense_peak = dense_operand_elements
    sector_peak = sector_operand_elements
    temporary_peak = 0.0
    permutation_peak = 0.0
    sector_temporary_peak = 0.0
    sector_permutation_peak = 0.0

    for i in eachindex(state.steps)
        step = state.steps[i]
        dense_output = state.output_dense_payloads[i]
        sector_output = state.output_sector_payloads[i]
        dense_temporary = state.temporary_dense_payloads[i]
        sector_temporary = state.temporary_sector_payloads[i]
        dense_peak = max(dense_peak,
                         dense_operand_elements + sum(values(dense_live)) +
                         dense_output + dense_temporary)
        sector_peak = max(sector_peak,
                          sector_operand_elements + sum(values(sector_live)) +
                          sector_output + sector_temporary)
        temporary_peak = max(temporary_peak, dense_temporary)
        permutation_peak = max(permutation_peak,
                               state.permutation_dense_payloads[i])
        sector_temporary_peak = max(sector_temporary_peak, sector_temporary)
        sector_permutation_peak = max(sector_permutation_peak,
                                      state.permutation_sector_payloads[i])
        delete!(dense_live, step.a)
        delete!(dense_live, step.b)
        delete!(sector_live, step.a)
        delete!(sector_live, step.b)
        dense_live[step.dst] = dense_output
        sector_live[step.dst] = sector_output
    end

    scale = Float64(scalar_bytes)
    return (operand_bytes=dense_operand_elements * scale,
            live_peak_bytes=dense_peak * scale,
            known_temporary_peak_bytes=temporary_peak * scale,
            known_permutation_peak_bytes=permutation_peak * scale,
            sector_operand_bytes=sector_operand_elements * scale,
            sector_live_peak_bytes=sector_peak * scale,
            sector_known_temporary_peak_bytes=sector_temporary_peak * scale,
            sector_known_permutation_peak_bytes=sector_permutation_peak * scale)
end

function _compile_plan(tree, spec::ContractionSpec, dims::Vector{Vector{Int}}, protos;
                       strategy::Symbol, structural_metrics::Bool,
                       canonical_intermediates::Bool=false,
                       scalar_type::DataType=_planning_scalar_type(protos))
    spaces = [_prototype_space(proto) for proto in protos]
    initial_peak = maximum(_prod_dims(d) for d in dims; init=0.0)
    initial_sector_peak = structural_metrics ?
                          maximum(Float64(dim(space_)) for space_ in spaces; init=0.0) :
                          initial_peak
    initial_sector_block_peak = structural_metrics ?
                                maximum(Backend.sector_block_peak(space_) for space_ in spaces;
                                        init=0.0) :
                                initial_peak
    initial_sector_flops = 0.0
    state = _CompileState(length(dims), PairStep[], 0.0, initial_peak,
                          initial_sector_flops, initial_sector_peak,
                          initial_sector_block_peak,
                          Float64[], Float64[], Float64[], Float64[],
                          Float64[], Float64[])
    output, _, _, _, _, _, _ = _compile_tree!(state, tree, spec, dims, protos,
                                              structural_metrics,
                                              canonical_intermediates, true)
    scalar_bytes = _scalar_byte_width(scalar_type)
    live = _compiled_live_memory_metrics(state, dims, spaces, scalar_bytes)
    return ContractionPlan(state.nextslot, output, state.steps;
                           strategy, flops=state.flops, peak_elements=state.peak,
                           sector_flops=state.sector_flops,
                           sector_peak_elements=state.sector_peak,
                           sector_peak_block_elements=state.sector_block_peak,
                           scalar_bytes=scalar_bytes,
                           operand_bytes=live.operand_bytes,
                           live_peak_bytes=live.live_peak_bytes,
                           known_temporary_peak_bytes=live.known_temporary_peak_bytes,
                           known_permutation_peak_bytes=live.known_permutation_peak_bytes,
                           sector_operand_bytes=live.sector_operand_bytes,
                           sector_live_peak_bytes=live.sector_live_peak_bytes,
                           sector_known_temporary_peak_bytes=live.sector_known_temporary_peak_bytes,
                           sector_known_permutation_peak_bytes=live.sector_known_permutation_peak_bytes,
                           scalar_output=spec.nopen == 0)
end

function _canonical_memory_cap(memory_cap_bytes::Union{Nothing,Real})
    memory_cap_bytes === nothing && return Inf
    cap = Float64(memory_cap_bytes)
    isfinite(cap) && cap >= 0 ||
        throw(ArgumentError("memory_cap_bytes must be finite and nonnegative"))
    return cap
end

"""
Hard-cap guard.  Dense live bytes are an intentionally conservative upper
bound; the sector-stored estimate is retained too so a malformed structural
profile cannot make the cap less strict.  A plan must fit both estimates.
"""
@inline function _within_memory_cap(plan::ContractionPlan, cap::Float64)
    return plan.live_peak_bytes <= cap && plan.sector_live_peak_bytes <= cap
end

@inline function _within_envfirst_memory_floors(candidate::ContractionPlan,
                                                 envfirst::ContractionPlan)
    return candidate.peak_elements <= envfirst.peak_elements &&
           candidate.sector_peak_elements <= envfirst.sector_peak_elements &&
           candidate.live_peak_bytes <= envfirst.live_peak_bytes &&
           candidate.sector_live_peak_bytes <= envfirst.sector_live_peak_bytes
end

"""
    plan_contraction(spec, protos;
                     optimize=true, memory_weight=1, sector_aware=true,
                     scalar_type=..., memory_cap_bytes=nothing) -> ContractionPlan

Compile the Phase-1 env-first plan plus Phase-2 dense candidates: TensorOperations'
FLOP-optimal tree and a separate memory-greedy tree.  On eligible unique-fusion
spaces, `sector_aware=true` additionally supplies an exact local DP (up to ten
tensors) using TensorKit's per-block GEMM model.  Dense peak and
symmetry-reduced stored peak are both hard-constrained by the env-first
candidate; the new conservative live-byte metrics obey the same floors.  A
finite `memory_cap_bytes` is hard: no over-cap candidate is returned.  Above
the exact-DP limit a bounded dense live-memory beam replaces the single greedy
candidate only when such a cap is requested.  Larger local networks,
non-unique fusion spaces, and non-symmetric braiding otherwise cleanly retain
Phase-2 dense selection until their corresponding cost/execution models are
calibrated.
"""
Base.@noinline function plan_contraction(
        spec::ContractionSpec, protos;
        optimize::Bool=true, memory_weight::Real=1,
        sector_aware::Bool=true,
        scalar_type::DataType=_planning_scalar_type(protos),
        memory_cap_bytes::Union{Nothing,Real}=nothing)
    isfinite(memory_weight) && memory_weight >= 0 ||
        throw(ArgumentError("memory_weight must be finite and nonnegative"))
    cap = _canonical_memory_cap(memory_cap_bytes)
    isfinite(cap) && !isbitstype(scalar_type) &&
        throw(ArgumentError("memory_cap_bytes requires an isbits scalar type; " *
                            "$scalar_type has heap-owned payload outside the " *
                            "conservative fixed-width model"))
    dims, label_dims = _label_dimensions(spec, protos)
    spaces = [_prototype_space(proto) for proto in protos]
    # `optimize=false` is the low-latency env-first path. Sector metrics cannot
    # affect its fixed tree; retaining dense fallback metrics is conservative
    # for a hard memory cap and avoids compiling the exact block-cost machinery.
    structural_metrics = optimize &&
                         all(Backend.sector_cost_supported, spaces) &&
                         any(Backend.sector_cost_nontrivial, spaces)
    heuristic = _compile_plan(_heuristic_tree(spec), spec, dims, protos;
                              strategy=:env_first, structural_metrics=structural_metrics,
                              scalar_type=scalar_type)
    if !optimize
        _within_memory_cap(heuristic, cap) ||
            throw(ArgumentError("env-first contraction plan requires at least " *
                                "$(max(heuristic.live_peak_bytes, heuristic.sector_live_peak_bytes)) bytes, " *
                                "above memory_cap_bytes=$cap"))
        return heuristic
    end
    candidates = ContractionPlan[heuristic]

    use_sector_model = sector_aware && structural_metrics
    if use_sector_model
        for tree in _sector_dp_trees(spec, dims, protos)
            try
                push!(candidates,
                      _compile_plan(tree, spec, dims, protos;
                                    strategy=:sector_exact,
                                    structural_metrics=structural_metrics,
                                    canonical_intermediates=true,
                                    scalar_type=scalar_type))
            catch err
                err isa InterruptException && rethrow()
            end
        end
    end

    # A FLOP-optimal tree is the ecosystem candidate required by Phase 2. For
    # bigger maps the bounded greedy tree is the only search candidate.
    if length(spec.labels) <= _EXACT_TENSOR_LIMIT
        try
            push!(candidates,
                  _compile_plan(_optimaltree(spec, label_dims), spec, dims, protos;
                                strategy=:dense_optimal,
                                structural_metrics=structural_metrics,
                                scalar_type=scalar_type))
        catch err
            err isa InterruptException && rethrow()
        end
        # This independently minimises each next intermediate's peak before
        # FLOPs, supplying a usable memory-sensitive candidate rather than
        # treating the single FLOP tree as the entire Phase-2 search space.
        try
            push!(candidates,
                  _compile_plan(_greedy_tree(spec, dims), spec, dims, protos;
                                strategy=:memory_greedy,
                                structural_metrics=structural_metrics,
                                scalar_type=scalar_type))
        catch err
            err isa InterruptException && rethrow()
        end
    else
        if isfinite(cap)
            for tree in _memory_beam_trees(spec, dims)
                try
                    push!(candidates,
                          _compile_plan(tree, spec, dims, protos;
                                        strategy=:memory_beam,
                                        structural_metrics=structural_metrics,
                                        scalar_type=scalar_type))
                catch err
                    err isa InterruptException && rethrow()
                end
            end
        else
            try
                push!(candidates,
                      _compile_plan(_greedy_tree(spec, dims), spec, dims, protos;
                                    strategy=:dense_greedy,
                                    structural_metrics=structural_metrics,
                                    scalar_type=scalar_type))
            catch err
                err isa InterruptException && rethrow()
            end
        end
    end

    best = nothing
    best_score = Inf
    for candidate in candidates
        _within_envfirst_memory_floors(candidate, heuristic) || continue
        _within_memory_cap(candidate, cap) || continue
        score = use_sector_model ? _sector_score(candidate, memory_weight) :
                                   _score(candidate, memory_weight)
        if score < best_score
            best, best_score = candidate, score
        end
    end
    best === nothing &&
        throw(ArgumentError("no contraction plan fits memory_cap_bytes=$cap; " *
                            "env-first requires at least " *
                            "$(max(heuristic.live_peak_bytes, heuristic.sector_live_peak_bytes)) bytes"))
    return best
end
