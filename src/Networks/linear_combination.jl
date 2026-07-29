"""
    exact_linear_combination(states, coefficients;
                             max_bond=typemax(Int),
                             max_payload=typemax(Int)) -> TTNS

Construct `sum(coefficients[i] * states[i])` as an exact TTNS block sum.
Every non-root edge is replaced by the direct sum of the corresponding source
spaces.  A source-specific injection is applied to every incident virtual leg
of a node, so a branching tensor is nonzero only when all of its virtual legs
carry the same source label.  Coefficients are inserted exactly once, at the
root.

This operation does not truncate or variationally fit.  The constructed direct
sum is canonicalized to the topology root, which may reduce its bond spaces;
callers must not rely on source-block identities or zero-coefficient blocks
being retained.
"""
function exact_linear_combination(
        states::AbstractVector{<:TTNS}, coefficients;
        max_bond::Int=typemax(Int),
        max_payload::Int=typemax(Int))
    isempty(states) &&
        throw(ArgumentError("exact_linear_combination requires at least one state"))
    max_bond >= 1 ||
        throw(ArgumentError("exact_linear_combination: max_bond must be positive"))
    max_payload >= 1 ||
        throw(ArgumentError("exact_linear_combination: max_payload must be positive"))
    inputs = collect(states)
    reference = first(inputs)
    t = topology(reference)
    S = spacetype(reference)
    T = eltype(reference)
    coeffs = _exact_combination_coefficients(T, coefficients, length(inputs))

    for source in inputs
        topology(source) == t ||
            throw(ArgumentError("exact_linear_combination: state topologies differ"))
        source.hasphys == reference.hasphys ||
            throw(ArgumentError(
                "exact_linear_combination: physical-leg layouts differ"))
        spacetype(source) == S ||
            throw(ArgumentError("exact_linear_combination: state spacetypes differ"))
        eltype(source) == T ||
            throw(ArgumentError("exact_linear_combination: state eltypes differ"))
        _check_exact_combination_spaces(reference, source)
    end
    sources = map(inputs) do source
        aligned = copy(source)
        _canonicalize_apply!(aligned, t.root)
        return aligned
    end
    reference = first(sources)
    _check_exact_combination_guards(
        reference, sources; max_bond, max_payload)

    edge_embeddings = Dict{Int,Vector{AbstractTensorMap{T,S}}}()
    for child in 1:nnodes(t)
        t.parent[child] == 0 && continue
        edge_spaces = S[virtualspace(source, child) for source in sources]
        edge_embeddings[child] = _direct_sum_embeddings(T, edge_spaces)
    end

    tensors = Vector{AbstractTensorMap{T,S}}(undef, nnodes(t))
    for n in 1:nnodes(t)
        combined = nothing
        for (source_index, source) in pairs(sources)
            mapped = source.tensors[n]
            for (slot, child) in enumerate(t.children[n])
                mapped = transform_leg(
                    mapped, edge_embeddings[child][source_index], slot)
            end
            if t.parent[n] != 0
                mapped = mapped * adjoint(edge_embeddings[n][source_index])
            end
            n == t.root && (mapped = coeffs[source_index] * mapped)
            combined = combined === nothing ? mapped : combined + mapped
        end
        tensors[n] = combined::AbstractTensorMap{T,S}
    end

    result = TTNS(t, tensors, t.root)
    return _canonicalize_apply!(result, t.root)
end

function _check_exact_combination_guards(
        reference::TTNS, sources::Vector{<:TTNS};
        max_bond::Int, max_payload::Int)
    t = topology(reference)
    summed_bonds = Dict{Int,Int}()
    for child in 1:nnodes(t)
        t.parent[child] == 0 && continue
        dimension = sum(dim(virtualspace(source, child)) for source in sources)
        dimension <= max_bond ||
            throw(ArgumentError(
                "exact_linear_combination: edge " *
                string(nodeid(t, child)) * " requires bond dimension " *
                string(dimension) * ", exceeding max_bond=" *
                string(max_bond)))
        summed_bonds[child] = dimension
    end
    for n in 1:nnodes(t)
        payload = BigInt(1)
        for child in t.children[n]
            payload *= summed_bonds[child]
        end
        hasphys(reference, n) &&
            (payload *= dim(physspace(reference, n)))
        payload *= t.parent[n] == 0 ?
            dim(domain(reference.tensors[n])[1]) : summed_bonds[n]
        payload <= max_payload ||
            throw(ArgumentError(
                "exact_linear_combination: node " * string(nodeid(t, n)) *
                " requires a conservative payload of " * string(payload) *
                " scalars, exceeding max_payload=" * string(max_payload)))
    end
    return nothing
end

function _exact_combination_coefficients(
        ::Type{T}, coefficients, count::Int) where {T<:Number}
    raw = collect(coefficients)
    length(raw) == count ||
        throw(ArgumentError(
            "exact_linear_combination: coefficient and state counts differ"))
    if T <: Real && any(c -> c isa Complex && !isreal(c), raw)
        throw(ArgumentError(
            "exact_linear_combination: real states require real coefficients"))
    end
    try
        return convert(Vector{T}, raw)
    catch error
        error isa InexactError || rethrow()
        throw(ArgumentError(
            "exact_linear_combination: coefficients are not representable by $T"))
    end
end

function _check_exact_combination_spaces(reference::TTNS, source::TTNS)
    t = topology(reference)
    for n in 1:nnodes(t)
        hasphys(reference, n) || continue
        physspace(reference, n) == physspace(source, n) ||
            throw(ArgumentError(
                "exact_linear_combination: physical spaces differ at " *
                string(nodeid(t, n))))
    end
    domain(reference.tensors[t.root])[1] == domain(source.tensors[t.root])[1] ||
        throw(ArgumentError(
            "exact_linear_combination: global charge spaces differ"))
    return nothing
end

function _direct_sum_embeddings(
        ::Type{T}, spaces::Vector{S}) where
        {T<:Number,S<:ElementarySpace}
    isempty(spaces) && throw(ArgumentError("cannot sum an empty space list"))
    orientation = isdual(first(spaces))
    all(V -> isdual(V) == orientation, spaces) ||
        throw(ArgumentError("direct-sum edge spaces have mixed dual orientation"))
    combined = foldl(⊕, spaces)
    embeddings = Vector{AbstractTensorMap{T,S}}(undef, length(spaces))
    for (source_index, source) in pairs(spaces)
        embedding = zeros(T, combined ← source)
        for (sector, block_) in blocks(embedding)
            source_dim = dim(source, sector)
            source_dim == 0 && continue
            offset = sum(
                dim(spaces[j], sector) for j in 1:(source_index - 1);
                init=0)
            for k in 1:source_dim
                block_[offset + k, k] = one(T)
            end
        end
        embeddings[source_index] = embedding
    end
    return embeddings
end
