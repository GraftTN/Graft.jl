# Dense TTNS/TTNO construction, contractions, gauge moves, application, and
# variational fitting on both a chain and a physical-leg-free branch point.
PrecompileTools.@compile_workload begin
    let
        topo = mps_topology(2)
        spins = spin_ops()
        phys = Dict(:site1 => spins.P, :site2 => spins.P)
        hamiltonian = OpSum() +
            Term(-1.0, SiteOp(:site1, :Z, spins.Z),
                       SiteOp(:site2, :Z, spins.Z)) +
            Term(-0.3, SiteOp(:site1, :X, spins.X)) +
            Term(-0.3, SiteOp(:site2, :X, spins.X))
        operator = ttno_from_opsum(hamiltonian, topo, phys; hermitian=true)
        state = TestUtils.random_ttns(
            Random.Xoshiro(0x6a09e667), ComplexF64, topo, phys, Backend.ℂ^2)

        inner(state, state)
        cache = EnvCache(topo)
        expect(state, operator; cache)
        site = nodeindex(topo, :site1)
        eff_h1(cache, state, operator, site)(state.tensors[site])

        edge_state = move_center!(copy(state), site)
        parent = topo.parent[site]
        theta = Contractions.two_site_tensor(edge_state, site, parent)
        eff_h2(EnvCache(topo), edge_state, operator, site, parent)(theta)
        link = Backend.id(virtualspace(edge_state, site))
        eff_h0(EnvCache(topo), edge_state, operator, site, parent)(link)

        target = apply(operator, state)
        fit!(copy(state), target; nsweeps=1, tol=0.0, verbose=false)
        fit!(copy(state), (state,); Hs=(operator,), nsweeps=1,
             tol=0.0, verbose=false)
        move_center!(copy(state), :site1)
    end
end

PrecompileTools.@compile_workload begin
    let
        topo = star_topology(3, 1)
        spins = spin_ops()
        phys = Dict(Symbol(:b, branch, :_1) => spins.P for branch in 1:3)
        hamiltonian = OpSum() +
            Term(0.8, SiteOp(:b1_1, :Z, spins.Z),
                      SiteOp(:b2_1, :Z, spins.Z)) +
            Term(-0.5, SiteOp(:b3_1, :X, spins.X))
        operator = ttno_from_opsum(hamiltonian, topo, phys; hermitian=true)
        state = TestUtils.random_ttns(
            Random.Xoshiro(0x510e527f), ComplexF64, topo, phys, Backend.ℂ^2)

        inner(state, state)
        expect(state, operator; cache=EnvCache(topo))
        root = topo.root
        eff_h1(EnvCache(topo), state, operator, root)(state.tensors[root])

        child = nodeindex(topo, :b1_1)
        edge_state = move_center!(copy(state), child)
        theta = Contractions.two_site_tensor(edge_state, child, root)
        eff_h2(EnvCache(topo), edge_state, operator, child, root)(theta)
        link = Backend.id(virtualspace(edge_state, child))
        eff_h0(EnvCache(topo), edge_state, operator, child, root)(link)

        target = apply(operator, state)
        fit!(copy(state), target; nsweeps=1, tol=0.0, verbose=false)
    end
end
