# Evolution owns all propagation and implicit-linear-solve specialization. The
# small product fixture uses only its declared lower-layer runtime APIs.
function _precompile_evolution_fixture()
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
        state, operator = _precompile_evolution_fixture()
        truncation = Backend.TruncationScheme(maxdim=2)

        Evolution.evolve!(
            Evolution.TDVP1(
                order=1, krylovdim=4, tol=1e-8, verbose=false),
            copy(state), operator, -0.01im, 1,
        )
        Evolution.step!(
            Evolution.TDVP2(
                order=1, trunc=truncation, krylovdim=4,
                tol=1e-8, verbose=false,
            ),
            copy(state), operator, -0.01im,
        )
        Evolution.step!(
            Evolution.TDVP1_CBE(
                cbe=Evolution.PredictorCBE(max_add=1),
                order=1, trunc=truncation,
                krylovdim=4, tol=1e-8, verbose=false,
            ),
            copy(state), operator, -0.01im,
        )
        Evolution.step!(
            Evolution.TDVP1_GSE(
                ancillary_shift=0.01, order=1, trunc=truncation, max_add=1,
                krylovdim=4, tol=1e-8, verbose=false,
            ),
            copy(state), operator, -0.01im,
        )
        Evolution.step!(
            Evolution.TDVP1_LSE(
                order=1, trunc=truncation, max_add=1,
                krylovdim=4, tol=1e-8, verbose=false,
            ),
            copy(state), operator, -0.01im,
        )
        Evolution.step!(
            Evolution.GlobalKrylov(
                krylovdim=4, maxiter=2, tol=1e-8,
                fit_nsweeps=1, fit_tol=1e-8,
            ),
            copy(state), operator, -0.01im,
        )

        Evolution.linsolve!(
            copy(state), operator, state;
            a0=1.0, a1=0.01, krylovdim=4, maxiter=2,
            tol=1e-8, fit_nsweeps=1, fit_tol=1e-8,
        )
        for scheme in (
                Evolution.LogTrapezoid(), Evolution.LogBackwardEuler(),
                Evolution.LogGaussLegendre(1))
            Evolution.step!(
                Evolution.ImplicitLogTime(
                    scheme=scheme, krylovdim=4, maxiter=2,
                    tol=1e-8, fit_nsweeps=1, fit_tol=1e-8,
                ),
                copy(state), operator, -0.01,
            )
        end
    end
end
