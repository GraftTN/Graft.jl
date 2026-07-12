### Graft.jl

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/GraftTN/Graft.jl@main/assets/graftjl-logo.png" alt="Graft.jl logo" width="240">
</p>

A general-purpose tree tensor network core library. DMFT/EDMFT impurity-solver workflows are provided by the companion `GraftImpurity.jl` package, which depends on Graft rather than being embedded in it.

The architecture is largely inspired by [PyTreeNet](https://github.com/Drachier/PyTreeNet), and its tensor network foundation is built on [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl), which lets us exploit the abelian and non-abelian symmetries. It is designed to flexibly adopt new algorithms from papers for experimentation and verification — i.e., tree *graft*ing.


#### Quick example

```julia
using Graft, Graft.TestUtils, Random
using Graft.Backend: ℂ

topo = star_topology(3, 2)                       # generic star geometry
S = spin_ops()
phys = Dict(nodeid(topo, i) => S.P for i in 1:nnodes(topo))

H = OpSum()
for (c, p) in Graft.Trees.edges(topo)
    H += Term(-1.0, SiteOp(nodeid(topo, c), :Z, S.Z), SiteOp(nodeid(topo, p), :Z, S.Z))
end
for i in 1:nnodes(topo)
    H += Term(-0.9, SiteOp(nodeid(topo, i), :X, S.X))
end
O = ttno_from_opsum(H, topo, phys; hermitian=true)

ψ = random_ttns(Xoshiro(1), ComplexF64, topo, phys, ℂ^2)
ψ, energies = dmrg2!(ψ, O; trunc=TruncationScheme(maxdim=32))

ev = TDVP1_CBE(trunc=TruncationScheme(maxdim=64), d_tilde_max=16)
evolve!(ev, ψ, O, -0.05im, 100)                  # real-time evolution, bond-adaptive
```

#### License

Graft.jl is licensed under the [Apache License 2.0](LICENSE).
