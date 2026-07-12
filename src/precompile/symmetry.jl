# Abelian boson and fermion-parity construction plus block-sparse contraction
# paths. Solver families are compiled by dense_solvers.jl; this layer targets
# the sector-specialized TensorKit dispatch.
PrecompileTools.@compile_workload begin
    let
        topo = mps_topology(2)
        bosons = boson_ops_u1(2)
        phys = Dict(:site1 => bosons.P, :site2 => bosons.P)
        hamiltonian = OpSum() +
            Term(0.5, SiteOp(:site1, :Bd, bosons.Bd),
                      SiteOp(:site2, :B, bosons.B)) +
            Term(0.5, SiteOp(:site1, :B, bosons.B),
                      SiteOp(:site2, :Bd, bosons.Bd))
        operator = ttno_from_opsum(hamiltonian, topo, phys; hermitian=true)
        state = TestUtils.product_ttns(
            ComplexF64, topo, phys,
            Dict(:site1 => Backend.U1Irrep(1),
                 :site2 => Backend.U1Irrep(0)))

        inner(state, state)
        cache = EnvCache(topo)
        expect(state, operator; cache)
        root = topo.root
        eff_h1(cache, state, operator, root)(state.tensors[root])
        applied = apply(operator, state)
        inner(applied, applied)
    end
end

PrecompileTools.@compile_workload begin
    let
        topo = mps_topology(2)
        fermions = fermion_ops_z2()
        phys = Dict(:site1 => fermions.P, :site2 => fermions.P)
        hamiltonian = OpSum() +
            Term(-1.0, SiteOp(:site1, :Cd, fermions.Cd),
                       SiteOp(:site2, :C, fermions.C)) +
            Term(-1.0, SiteOp(:site1, :C, fermions.C),
                       SiteOp(:site2, :Cd, fermions.Cd))
        operator = ttno_from_opsum(hamiltonian, topo, phys; hermitian=true)
        state = TestUtils.product_ttns(
            ComplexF64, topo, phys,
            Dict(:site1 => Backend.FermionParity(1),
                 :site2 => Backend.FermionParity(0)))

        inner(state, state)
        cache = EnvCache(topo)
        expect(state, operator; cache)
        root = topo.root
        eff_h1(cache, state, operator, root)(state.tensors[root])
        applied = apply(operator, state)
        inner(applied, applied)
    end
end
