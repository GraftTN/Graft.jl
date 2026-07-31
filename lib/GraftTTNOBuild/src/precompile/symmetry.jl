# Abelian and fermion-parity symbolic lowering stays with the TTNO builder;
# solver and contraction dispatch are compiled by their respective owners.
PrecompileTools.@compile_workload begin
    let
        topo = Trees.mps_topology(2)
        bosons = Symbolic.boson_ops_u1(2)
        phys = Dict(:site1 => bosons.P, :site2 => bosons.P)
        hamiltonian = Symbolic.OpSum() +
            Symbolic.Term(
                0.5,
                Symbolic.SiteOp(:site1, :Bd, bosons.Bd),
                Symbolic.SiteOp(:site2, :B, bosons.B),
            ) +
            Symbolic.Term(
                0.5,
                Symbolic.SiteOp(:site1, :B, bosons.B),
                Symbolic.SiteOp(:site2, :Bd, bosons.Bd),
            )
        LegacyTTNOBuild.ttno_from_opsum(
            hamiltonian, topo, phys; hermitian=true)
    end
end

PrecompileTools.@compile_workload begin
    let
        topo = Trees.mps_topology(2)
        fermions = Symbolic.fermion_ops_z2()
        phys = Dict(:site1 => fermions.P, :site2 => fermions.P)
        hamiltonian = Symbolic.OpSum() +
            Symbolic.Term(
                -1.0,
                Symbolic.SiteOp(:site1, :Cd, fermions.Cd),
                Symbolic.SiteOp(:site2, :C, fermions.C),
            ) +
            Symbolic.Term(
                -1.0,
                Symbolic.SiteOp(:site1, :C, fermions.C),
                Symbolic.SiteOp(:site2, :Cd, fermions.Cd),
            )
        LegacyTTNOBuild.ttno_from_opsum(
            hamiltonian, topo, phys; hermitian=true)
    end
end
