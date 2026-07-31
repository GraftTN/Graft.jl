# Ground-state workloads build their deterministic fixture from Foundation and
# Networks instead of depending on Symbolic, TTNOBuild, or test support.
function _precompile_ground_state_fixture()
    topo = Trees.mps_topology(2)
    unit = Backend.ℂ^1
    physical = Backend.ℂ^2
    plus = ComplexF64[1, 1] / sqrt(2)
    state_tensors = map(1:Trees.nnodes(topo)) do node
        children = Trees.nchildren(topo, node)
        virtual_codomain = children == 0 ? one(unit) :
            reduce(Backend.:⊗, ntuple(_ -> unit, children))
        codomain = Backend.:⊗(virtual_codomain, physical)
        dims = (ntuple(_ -> 1, children)..., 2, 1)
        Backend.TensorMap(
            reshape(plus, dims), Backend.:←(codomain, unit))
    end
    state = Networks.TTNS(topo, state_tensors, topo.root)

    local_operators = Dict(
        :site1 => ComplexF64[0 1; 1 0],
        :site2 => ComplexF64[1 0; 0 -1],
    )
    operator_tensors = map(1:Trees.nnodes(topo)) do node
        children = Trees.nchildren(topo, node)
        virtual_codomain = children == 0 ? one(unit) :
            reduce(Backend.:⊗, ntuple(_ -> unit, children))
        codomain = Backend.:⊗(virtual_codomain, physical)
        domain = Backend.:⊗(physical, unit)
        dims = (ntuple(_ -> 1, children)..., 2, 2, 1)
        Backend.TensorMap(
            reshape(local_operators[Trees.nodeid(topo, node)], dims),
            Backend.:←(codomain, domain),
        )
    end
    operator = Networks.TTNO(topo, operator_tensors; ishermitian=true)
    return state, operator
end

PrecompileTools.@compile_workload begin
    let
        state, operator = _precompile_ground_state_fixture()
        truncation = Backend.TruncationScheme(maxdim=2)

        GroundState.dmrg1!(
            copy(state), operator; nsweeps=1, krylovdim=4, verbose=false)
        GroundState.dmrg2!(
            copy(state), operator;
            trunc=truncation, nsweeps=1, krylovdim=4, verbose=false,
        )
        GroundState.dmrg1_3s!(
            copy(state), operator;
            trunc=truncation, nsweeps=1, max_add=1,
            krylovdim=4, verbose=false,
        )
    end
end
