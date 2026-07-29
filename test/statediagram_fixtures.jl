# Shared SD1+ fixture inputs for the typed StateDiagram compiler tests and
# the golden-fixture generator (scripts regenerate the committed .sexp files
# from exactly these builders after a reviewed IR schema bump).

function _sd1_dense_input(; hermiticity=Graft.TTNOBuild.NoHermiticityAssertion())
    S = spin_ops()
    topo = mps_topology(3)
    phys = Dict(:site1 => S.P, :site2 => S.P, :site3 => S.P)
    H = OpSum()
    H += Term(-1.0, SiteOp(:site1, :Z, S.Z), SiteOp(:site2, :Z, S.Z))
    H += Term(-1.0, SiteOp(:site2, :Z, S.Z), SiteOp(:site3, :Z, S.Z))
    H += Term(-0.7, SiteOp(:site2, :X, S.X))
    return Graft.TTNOBuild.TTNOBuildInput(H, topo, phys; hermiticity), topo
end

function _sd1_fz2_input()
    F = fermion_ops_z2()
    topo = fork_topology(2, 1)
    phys = Dict(nodeid(topo, i) => F.P for i in 1:nnodes(topo))
    sites = [nodeid(topo, i) for i in 1:nnodes(topo)]
    a, b = sites[1], sites[end]
    H = OpSum()
    H += Term(1.0, SiteOp(a, :Cd, F.Cd), SiteOp(b, :C, F.C))
    H += Term(1.0, SiteOp(b, :Cd, F.Cd), SiteOp(a, :C, F.C))
    H += Term(0.5, SiteOp(sites[2], :N, F.N))
    return Graft.TTNOBuild.TTNOBuildInput(H, topo, phys), topo
end
