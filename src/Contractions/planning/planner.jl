function _heuristic_tree(spec::ContractionSpec)
    tree = spec.dynamic_slot
    for slot in spec.preferred_slots
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
Bounded structural DP state used only by the Phase-3 unique-fusion planner.
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

const _SECTOR_DP_TENSOR_LIMIT = 7
const _SECTOR_DP_FRONTIER_LIMIT = 8

@inline function _has_shared_positive_label(a::Vector{Int}, b::Vector{Int})
    return any(label -> label > 0 && label in b, a)
end

@inline _same_sector_structure(a::_SectorDPState, b::_SectorDPState) =
    Tuple(a.labels) == Tuple(b.labels) && a.space == b.space

@inline function _sector_dominates(a::_SectorDPState, b::_SectorDPState)
    no_worse = a.peak <= b.peak &&
               a.sector_peak <= b.sector_peak &&
               a.sector_flops <= b.sector_flops
    strict = a.peak < b.peak ||
             a.sector_peak < b.sector_peak ||
             a.sector_flops < b.sector_flops
    return no_worse && strict
end

"""Insert a state into its exact-layout Pareto frontier, with a hard size cap."""
function _insert_sector_state!(frontier::Vector{_SectorDPState}, candidate::_SectorDPState)
    same = Int[]
    for (i, current) in enumerate(frontier)
        _same_sector_structure(current, candidate) || continue
        _sector_dominates(current, candidate) && return frontier
        push!(same, i)
    end
    for i in reverse(same)
        _sector_dominates(candidate, frontier[i]) && deleteat!(frontier, i)
    end
    push!(frontier, candidate)

    # A pathological graph can generate many non-dominated trees with the
    # same externally visible intermediate.  Keeping the smallest sector
    # objective representatives is bounded planning overhead, not a semantic
    # fallback: env-first remains a separate candidate below.
    same = Int[i for (i, current) in enumerate(frontier)
               if _same_sector_structure(current, candidate)]
    if length(same) > _SECTOR_DP_FRONTIER_LIMIT
        sort!(same; by=i -> (frontier[i].sector_flops,
                             frontier[i].sector_peak,
                             frontier[i].peak))
        for i in reverse(same[(_SECTOR_DP_FRONTIER_LIMIT + 1):end])
            deleteat!(frontier, i)
        end
    end
    return frontier
end

function _sector_merge_state(left::_SectorDPState, right::_SectorDPState)
    metrics = dense_cost(left.labels, left.dims, right.labels, right.dims)
    out = (Tuple(1:length(metrics.labels)), ())
    profile = _sector_pair_profile(metrics, left.space, left.conj,
                                   right.space, right.conj, out)
    profile.supported || return nothing
    return _SectorDPState(Any[left.tree, right.tree], metrics.labels, metrics.dims,
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

Generate a small Pareto frontier of binary trees using exact HomSpace block
profiles.  The canonical subset orientation deliberately avoids exploring both
braid/layout orientations of the same merge; execution remains general, while
the Phase-1 env-first candidate provides the memory-safe fallback.
"""
function _sector_dp_trees(spec::ContractionSpec, dims::Vector{Vector{Int}}, protos)
    n = length(spec.labels)
    n <= _SECTOR_DP_TENSOR_LIMIT || return Any[]
    spaces = [_prototype_space(proto) for proto in protos]
    all(Backend.sector_cost_supported, spaces) || return Any[]
    any(Backend.sector_cost_nontrivial, spaces) || return Any[]

    fullmask = (1 << n) - 1
    frontiers = [_SectorDPState[] for _ in 1:fullmask]
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
            # Canonical operand orientation: every unordered bipartition is
            # considered exactly once, avoiding unpriced permutation variants.
            if rightmask != 0 && leftmask < rightmask &&
               !isempty(frontiers[leftmask]) && !isempty(frontiers[rightmask])
                for left in frontiers[leftmask], right in frontiers[rightmask]
                    _has_shared_positive_label(left.labels, right.labels) || continue
                    merged = _sector_merge_state(left, right)
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
                        structural_metrics::Bool, root::Bool=false)
    if tree isa Integer
        slot = Int(tree)
        return slot, copy(spec.labels[slot]), copy(dims[slot]),
               _prototype_space(protos[slot]), spec.conjs[slot]
    end
    length(tree) == 2 || throw(ArgumentError("contraction tree nodes must be binary"))
    a, labels_a, dims_a, space_a, conj_a =
        _compile_tree!(state, tree[1], spec, dims, protos, structural_metrics)
    b, labels_b, dims_b, space_b, conj_b =
        _compile_tree!(state, tree[2], spec, dims, protos, structural_metrics)
    metrics = dense_cost(labels_a, dims_a, labels_b, dims_b)
    state.nextslot += 1
    dst = state.nextslot
    out = root ? _output_partition(metrics.labels, spec) :
                 (Tuple(1:length(metrics.labels)), ())
    profile = structural_metrics ?
              _sector_pair_profile(metrics, space_a, conj_a, space_b, conj_b, out) :
              nothing
    push!(state.steps,
          PairStep(a, b, dst,
                   Tuple(metrics.oindA), Tuple(metrics.cindA),
                   Tuple(metrics.oindB), Tuple(metrics.cindB),
                   conj_a, conj_b, out))
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
        return dst, metrics.labels, metrics.dims, nothing, false
    elseif profile.supported && isfinite(state.sector_flops)
        state.sector_flops += profile.sector_flops
    else
        state.sector_flops = NaN
    end
    state.sector_peak = max(state.sector_peak, profile.output_elements)
    state.sector_block_peak = max(state.sector_block_peak,
                                  profile.peak_block_elements)
    return dst, metrics.labels, metrics.dims, profile.output, false
end

function _compile_plan(tree, spec::ContractionSpec, dims::Vector{Vector{Int}}, protos;
                       strategy::Symbol, structural_metrics::Bool)
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
                          initial_sector_block_peak)
    output, _, _, _, _ = _compile_tree!(state, tree, spec, dims, protos,
                                         structural_metrics, true)
    return ContractionPlan(state.nextslot, output, state.steps;
                           strategy, flops=state.flops, peak_elements=state.peak,
                           sector_flops=state.sector_flops,
                           sector_peak_elements=state.sector_peak,
                           sector_peak_block_elements=state.sector_block_peak)
end

"""
    plan_contraction(spec, protos;
                     optimize=true, memory_weight=1, sector_aware=true) -> ContractionPlan

Compile the Phase-1 env-first plan plus Phase-2 dense candidates: TensorOperations'
FLOP-optimal tree and a separate memory-greedy tree.  On unique-fusion spaces,
`sector_aware=true` additionally supplies a bounded own DP using TensorKit's
per-block GEMM model.  Dense peak and symmetry-reduced stored peak are both
hard-constrained by the env-first candidate; only then does selection optimize
the relevant cost model.  Its `:sector_bounded` result is a bounded candidate,
not a global optimality claim: the frontier and operand orientation are capped
deliberately.  Non-unique fusion spaces cleanly retain Phase-2 dense selection
until recoupling/permutation cost is calibrated.
"""
const _EXACT_TENSOR_LIMIT = 10

function plan_contraction(spec::ContractionSpec, protos;
                          optimize::Bool=true, memory_weight::Real=1,
                          sector_aware::Bool=true)
    isfinite(memory_weight) && memory_weight >= 0 ||
        throw(ArgumentError("memory_weight must be finite and nonnegative"))
    dims, label_dims = _label_dimensions(spec, protos)
    spaces = [_prototype_space(proto) for proto in protos]
    structural_metrics = all(Backend.sector_cost_supported, spaces) &&
                         any(Backend.sector_cost_nontrivial, spaces)
    heuristic = _compile_plan(_heuristic_tree(spec), spec, dims, protos;
                              strategy=:env_first, structural_metrics=structural_metrics)
    optimize || return heuristic
    candidates = ContractionPlan[heuristic]

    use_sector_model = sector_aware && structural_metrics
    if use_sector_model
        for tree in _sector_dp_trees(spec, dims, protos)
            try
                push!(candidates,
                      _compile_plan(tree, spec, dims, protos;
                                    strategy=:sector_bounded,
                                    structural_metrics=structural_metrics))
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
                                structural_metrics=structural_metrics))
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
                                structural_metrics=structural_metrics))
        catch err
            err isa InterruptException && rethrow()
        end
    else
        try
            push!(candidates,
                  _compile_plan(_greedy_tree(spec, dims), spec, dims, protos;
                                strategy=:dense_greedy,
                                structural_metrics=structural_metrics))
        catch err
            err isa InterruptException && rethrow()
        end
    end

    best = heuristic
    best_score = use_sector_model ? _sector_score(best, memory_weight) :
                                   _score(best, memory_weight)
    for candidate in candidates
        candidate.peak_elements <= heuristic.peak_elements || continue
        candidate.sector_peak_elements <= heuristic.sector_peak_elements || continue
        score = use_sector_model ? _sector_score(candidate, memory_weight) :
                                   _score(candidate, memory_weight)
        if score < best_score
            best, best_score = candidate, score
        end
    end
    return best
end
