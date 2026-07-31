# Symbolic-to-TTNO lowering is owned by GraftTTNOBuild. The fixtures exercise
# both a chain and a physical-leg-free branch point without importing an
# upper-layer state or solver package.
PrecompileTools.@compile_workload begin
    let
        topo = Trees.mps_topology(2)
        spins = Symbolic.spin_ops()
        phys = Dict(:site1 => spins.P, :site2 => spins.P)
        hamiltonian = Symbolic.OpSum() +
            Symbolic.Term(
                -1.0,
                Symbolic.SiteOp(:site1, :Z, spins.Z),
                Symbolic.SiteOp(:site2, :Z, spins.Z),
            ) +
            Symbolic.Term(-0.3, Symbolic.SiteOp(:site1, :X, spins.X)) +
            Symbolic.Term(-0.3, Symbolic.SiteOp(:site2, :X, spins.X))
        LegacyTTNOBuild.ttno_from_opsum(
            hamiltonian, topo, phys; hermitian=true)
    end
end

PrecompileTools.@compile_workload begin
    let
        topo = Trees.star_topology(3, 1)
        spins = Symbolic.spin_ops()
        phys = Dict(Symbol(:b, branch, :_1) => spins.P for branch in 1:3)
        hamiltonian = Symbolic.OpSum() +
            Symbolic.Term(
                0.8,
                Symbolic.SiteOp(:b1_1, :Z, spins.Z),
                Symbolic.SiteOp(:b2_1, :Z, spins.Z),
            ) +
            Symbolic.Term(-0.5, Symbolic.SiteOp(:b3_1, :X, spins.X))
        LegacyTTNOBuild.ttno_from_opsum(
            hamiltonian, topo, phys; hermitian=true)
    end
end
