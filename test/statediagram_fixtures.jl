# Shared SD1+ fixture inputs for the typed StateDiagram compiler tests and
# the golden-fixture generator (scripts regenerate the committed .sexp files
# from exactly these builders after a reviewed IR schema bump).

using Graft.Backend: ⊠, ⊗, ←

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

const _SD2_MM_Q = typeof(Graft.Backend.FermionParity(0) ⊠ Graft.Backend.U1Irrep(0))
_sd2_mm_sector(n::Int) =
    Graft.Backend.FermionParity(n % 2) ⊠ Graft.Backend.U1Irrep(n)

"Two-mode fZ2 ⊠ U(1) fermionic carrier (CG-009 contract, intra-site JW)."
function _sd2_multimode_carrier(mode_count::Int)
    states = sort!(collect(0:((1 << mode_count) - 1)); by=s -> (count_ones(s), s))
    d = length(states)
    pos = Dict(s => i for (i, s) in enumerate(states))
    P = Graft.Backend.Vect[_SD2_MM_Q](
        (_sd2_mm_sector(n) => binomial(mode_count, n) for n in 0:mode_count)...)
    annihilate = Graft.Backend.Vect[_SD2_MM_Q](
        (Graft.Backend.FermionParity(1) ⊠ Graft.Backend.U1Irrep(-1)) => 1)
    create = Graft.Backend.Vect[_SD2_MM_Q](
        (Graft.Backend.FermionParity(1) ⊠ Graft.Backend.U1Irrep(1)) => 1)
    C = Vector{Any}(undef, mode_count)
    Cd = Vector{Any}(undef, mode_count)
    N = Vector{Any}(undef, mode_count)
    for j in 1:mode_count
        mask = 1 << (j - 1)
        a = zeros(ComplexF64, d, d)
        c = zeros(ComplexF64, d, d)
        nn = zeros(ComplexF64, d, d)
        for s in states
            sgn = isodd(count_ones(s & (mask - 1))) ? -1.0 : 1.0
            if (s & mask) != 0
                a[pos[s & ~mask], pos[s]] = sgn
                nn[pos[s], pos[s]] = 1.0
            else
                c[pos[s | mask], pos[s]] = sgn
            end
        end
        C[j] = Graft.Backend.TensorMap(reshape(a, d, d, 1),
                                       P ← P ⊗ annihilate)
        Cd[j] = Graft.Backend.TensorMap(reshape(c, d, d, 1), P ← P ⊗ create)
        N[j] = Graft.Backend.TensorMap(nn, P ← P)
    end
    return (; P, C=Tuple(C), Cd=Tuple(Cd), N=Tuple(N))
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
