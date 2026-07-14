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

#### References

Selected algorithmic references for Graft.jl, in implementation order:

1. **TTNO state diagrams:** R. M. Milbradt, Q. Huang, and C. B. Mendl, “State Diagrams to determine Tree Tensor Network Operators,” *SciPost Physics Core* **7**, 036 (2024). [doi:10.21468/SciPostPhysCore.7.2.036](https://doi.org/10.21468/SciPostPhysCore.7.2.036); [arXiv:2311.13433](https://arxiv.org/abs/2311.13433).
2. **PyTreeNet:** R. M. Milbradt, Q. Huang, and C. B. Mendl, “PyTreeNet: A Python Library for easy Utilisation of Tree Tensor Networks,” arXiv:2407.13249 (2024). [arXiv:2407.13249](https://arxiv.org/abs/2407.13249).
3. **Tree TDVP / ForkTPS:** D. Bauernfeind and M. Aichhorn, “Time Dependent Variational Principle for Tree Tensor Networks,” *SciPost Physics* **8**, 024 (2020). [doi:10.21468/SciPostPhys.8.2.024](https://doi.org/10.21468/SciPostPhys.8.2.024); [arXiv:1908.03090](https://arxiv.org/abs/1908.03090).
4. **CBE-TDVP:** J.-W. Li, A. Gleis, and J. von Delft, “Time-dependent variational principle with controlled bond expansion for matrix product states,” *Physical Review Letters* **133**, 026401 (2024). [doi:10.1103/PhysRevLett.133.026401](https://doi.org/10.1103/PhysRevLett.133.026401); [arXiv:2208.10972](https://arxiv.org/abs/2208.10972).
5. **DMRG3S:** C. Hubig, I. P. McCulloch, U. Schollwöck, and F. A. Wolf, “A Strictly Single-Site DMRG Algorithm with Subspace Expansion,” *Physical Review B* **91**, 155115 (2015). [doi:10.1103/PhysRevB.91.155115](https://doi.org/10.1103/PhysRevB.91.155115); [arXiv:1501.05504](https://arxiv.org/abs/1501.05504).
6. **RSVD post-expansion:** I. P. McCulloch and J. J. Osborne, “Comment on ‘Controlled Bond Expansion for Density Matrix Renormalization Group Ground State Search at Single-Site Costs’ (Extended Version),” arXiv:2403.00562 (2024). [arXiv:2403.00562](https://arxiv.org/abs/2403.00562).
7. **Global Krylov:** S. Paeckel, T. Köhler, A. Swoboda, S. R. Manmana, U. Schollwöck, and C. Hubig, “Time-evolution methods for matrix-product states,” *Annals of Physics* **411**, 167998 (2019). [doi:10.1016/j.aop.2019.167998](https://doi.org/10.1016/j.aop.2019.167998); [arXiv:1901.05824](https://arxiv.org/abs/1901.05824).
8. **GSE/LSE TDVP:** M. Yang and S. R. White, “Time Dependent Variational Principle with Ancillary Krylov Subspace,” *Physical Review B* **102**, 094315 (2020). This is the global ancillary-Krylov foundation used by Graft's GSE/LSE expansion family. [doi:10.1103/PhysRevB.102.094315](https://doi.org/10.1103/PhysRevB.102.094315); [arXiv:2005.06104](https://arxiv.org/abs/2005.06104).
9. **Implicit logarithmic-time evolution:** J. P. Zima, E. M. Stoudenmire, S. R. White, O. Parcollet, and J. Kaye, “Fast Tensor Network Imaginary Time Evolution by Implicit Stepping on Logarithmic Grids,” arXiv:2606.02930 (2026). [arXiv:2606.02930](https://arxiv.org/abs/2606.02930).
10. **Projected purification for bosons:** T. Köhler, J. Stolpp, and S. Paeckel, “Efficient and Flexible Approach to Simulate Low-Dimensional Quantum Lattice Models with Large Local Hilbert Spaces,” *SciPost Physics* **10**, 058 (2021). [doi:10.21468/SciPostPhys.10.3.058](https://doi.org/10.21468/SciPostPhys.10.3.058); [arXiv:2008.08466](https://arxiv.org/abs/2008.08466).

#### License

Graft.jl is licensed under the [Apache License 2.0](LICENSE).
