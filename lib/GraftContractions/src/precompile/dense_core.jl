# Deterministic dense TTNS/TTNO fixtures built solely from the owner package's
# declared runtime dependencies.
function _precompile_dense_product_state(topo, local_states)
    unit = Backend.ℂ^1
    tensors = map(1:Trees.nnodes(topo)) do node
        children = Trees.nchildren(topo, node)
        virtual_codomain = children == 0 ? one(unit) :
            reduce(Backend.:⊗, ntuple(_ -> unit, children))
        state = get(local_states, Trees.nodeid(topo, node), nothing)
        if state === nothing
            data = ones(ComplexF64, ntuple(_ -> 1, children + 1))
            Backend.TensorMap(data, Backend.:←(virtual_codomain, unit))
        else
            physical = Backend.ℂ^length(state)
            codomain = Backend.:⊗(virtual_codomain, physical)
            dims = (ntuple(_ -> 1, children)..., length(state), 1)
            Backend.TensorMap(
                reshape(ComplexF64.(state), dims),
                Backend.:←(codomain, unit),
            )
        end
    end
    return Networks.TTNS(topo, tensors, topo.root)
end

function _precompile_dense_product_operator(topo, local_operators)
    unit = Backend.ℂ^1
    tensors = map(1:Trees.nnodes(topo)) do node
        children = Trees.nchildren(topo, node)
        virtual_codomain = children == 0 ? one(unit) :
            reduce(Backend.:⊗, ntuple(_ -> unit, children))
        operator = get(local_operators, Trees.nodeid(topo, node), nothing)
        if operator === nothing
            data = ones(ComplexF64, ntuple(_ -> 1, children + 1))
            Backend.TensorMap(data, Backend.:←(virtual_codomain, unit))
        else
            dimension = size(operator, 1)
            physical = Backend.ℂ^dimension
            codomain = Backend.:⊗(virtual_codomain, physical)
            domain = Backend.:⊗(physical, unit)
            dims = (ntuple(_ -> 1, children)..., dimension, dimension, 1)
            Backend.TensorMap(
                reshape(ComplexF64.(operator), dims),
                Backend.:←(codomain, domain),
            )
        end
    end
    return Networks.TTNO(topo, tensors; ishermitian=true)
end

function _precompile_dense_contractions(topo, local_states, local_operators)
    state = _precompile_dense_product_state(topo, local_states)
    operator = _precompile_dense_product_operator(topo, local_operators)

    Contractions.inner(state, state)
    cache = Contractions.EnvCache(topo)
    Contractions.expect(state, operator; cache)
    root = topo.root
    Contractions.eff_h1(cache, state, operator, root)(state.tensors[root])

    child = first(topo.children[root])
    edge_state = Networks.move_center!(copy(state), child)
    edge_cache = Contractions.EnvCache(topo)
    theta = Contractions.two_site_tensor(edge_state, child, root)
    Contractions.eff_h2(
        edge_cache, edge_state, operator, child, root)(theta)
    link = Backend.id(Networks.virtualspace(edge_state, child))
    Contractions.eff_h0(
        edge_cache, edge_state, operator, child, root)(link)

    target = Networks.apply(operator, state)
    Networks.fit!(copy(state), target; nsweeps=1, tol=0.0, verbose=false)
    Networks.fit!(
        copy(state), (state,); Hs=(operator,), nsweeps=1,
        tol=0.0, verbose=false,
    )
    return nothing
end

# Environments, effective maps, and the variational fit extension belong to
# Contractions. Cover both a chain and a physical-leg-free branch point.
PrecompileTools.@compile_workload begin
    let
        plus = ComplexF64[1, 1] / sqrt(2)
        topo = Trees.mps_topology(2)
        _precompile_dense_contractions(
            topo,
            Dict(:site1 => plus, :site2 => plus),
            Dict(
                :site1 => ComplexF64[0 1; 1 0],
                :site2 => ComplexF64[1 0; 0 -1],
            ),
        )
    end
end

PrecompileTools.@compile_workload begin
    let
        plus = ComplexF64[1, 1] / sqrt(2)
        topo = Trees.star_topology(3, 1)
        _precompile_dense_contractions(
            topo,
            Dict(:b1_1 => plus, :b2_1 => plus, :b3_1 => plus),
            Dict(
                :b1_1 => ComplexF64[1 0; 0 -1],
                :b2_1 => ComplexF64[0 1; 1 0],
                :b3_1 => ComplexF64[1 0; 0 -1],
            ),
        )
    end
end
