# Deterministic, owner-local dense fixtures. They intentionally use only the
# GraftFoundation and GraftNetworks runtime APIs, so precompiling Networks does
# not pull a test-support or upper-layer package into its dependency graph.
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

# Dense construction, gauge movement, exact application, and state arithmetic
# stay with the package that owns those methods.
PrecompileTools.@compile_workload begin
    let
        topo = Trees.mps_topology(2)
        plus = ComplexF64[1, 1] / sqrt(2)
        state = _precompile_dense_product_state(
            topo, Dict(:site1 => plus, :site2 => plus))
        operator = _precompile_dense_product_operator(
            topo,
            Dict(
                :site1 => ComplexF64[0 1; 1 0],
                :site2 => ComplexF64[1 0; 0 -1],
            ),
        )

        Networks.check_arrows(state)
        Networks.move_center!(copy(state), :site1)
        applied = Networks.apply(operator, state)
        combined = Networks.exact_linear_combination(
            [state, applied], ComplexF64[1, -0.25])
        Networks.truncated_linear_combination(
            [state, combined], ComplexF64[1, 0.1];
            trunc=Backend.TruncationScheme(maxdim=2),
        )
    end
end
