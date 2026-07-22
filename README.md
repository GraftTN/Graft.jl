# Graft.jl

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/GraftTN/Graft.jl@main/assets/graftjl-logo.png" alt="Graft.jl logo" width="240">
</p>

A general-purpose tree tensor network core library. DMFT/EDMFT impurity-solver workflows are provided by the companion `GraftImpurity.jl` package, which depends on Graft rather than being embedded in it.

The architecture is largely inspired by [PyTreeNet](https://github.com/Drachier/PyTreeNet), and its tensor network foundation is built on [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl), which lets us exploit the abelian and non-abelian symmetries. It is designed to flexibly adopt new algorithms from papers for experimentation and verification — i.e., tree *graft*ing.


## Quick example

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

## Algorithmic References and Provenance

References are grouped by the Graft functionality they inform. Each entry states
whether it is an implementation basis or a design reference; citing a method
does not imply that every variant in the paper is implemented.

### Tree Operators and Software Architecture

1. **TTNO state diagrams** — *implemented; algorithmic basis*

   R. M. Milbradt, Q. Huang, and C. B. Mendl, “State Diagrams to determine Tree Tensor Network Operators,” *SciPost Physics Core* **7**, 036 (2024).
   [DOI](https://doi.org/10.21468/SciPostPhysCore.7.2.036) ·
   [arXiv](https://arxiv.org/abs/2311.13433)

   **Provenance:** Basis for constructing TTNOs from operator sums through state diagrams.

2. **Tree-network architecture** — *implemented; PyTreeNet design reference*

   R. M. Milbradt, Q. Huang, and C. B. Mendl, “PyTreeNet: A Python Library for easy Utilisation of Tree Tensor Networks,” arXiv:2407.13249 (2024).
   [arXiv](https://arxiv.org/abs/2407.13249)

   **Provenance:** Informs the package's tree-network organization, terminology, and parts of its TDVP implementation lineage.

### Ground-State and Time-Evolution Algorithms

1. **Tree TDVP / ForkTPS** — *implemented; algorithmic basis*

   D. Bauernfeind and M. Aichhorn, “Time Dependent Variational Principle for Tree Tensor Networks,” *SciPost Physics* **8**, 024 (2020).
   [DOI](https://doi.org/10.21468/SciPostPhys.8.2.024) ·
   [arXiv](https://arxiv.org/abs/1908.03090)

   **Provenance:** Basis for TDVP sweeps and local time evolution on tree tensor networks.

2. **CBE-TDVP** — *implemented; adapted algorithmic basis*

   J.-W. Li, A. Gleis, and J. von Delft, “Time-dependent variational principle with controlled bond expansion for matrix product states,” *Physical Review Letters* **133**, 026401 (2024).
   [DOI](https://doi.org/10.1103/PhysRevLett.133.026401) ·
   [arXiv](https://arxiv.org/abs/2208.10972)

   **Provenance:** Basis for controlled bond expansion, adapted from chains to tree tensor networks.

3. **DMRG3S** — *planned; design reference*

   C. Hubig, I. P. McCulloch, U. Schollwöck, and F. A. Wolf, “A Strictly Single-Site DMRG Algorithm with Subspace Expansion,” *Physical Review B*
   **91**, 155115 (2015).
   [DOI](https://doi.org/10.1103/PhysRevB.91.155115) ·
   [arXiv](https://arxiv.org/abs/1501.05504)

   **Provenance:** Design reference for single-site DMRG with subspace expansion.

4. **RSVD post-expansion** — *planned; design reference*

   I. P. McCulloch and J. J. Osborne, “Comment on ‘Controlled Bond Expansion for Density Matrix Renormalization Group Ground State Search at Single-Site Costs’ (Extended Version),” arXiv:2403.00562 (2024).
   [arXiv](https://arxiv.org/abs/2403.00562)

   **Provenance:** Design reference for randomized-SVD post-expansion choices.

5. **Global Krylov** — *design reference*

   S. Paeckel, T. Köhler, A. Swoboda, S. R. Manmana, U. Schollwöck, and C. Hubig, “Time-evolution methods for matrix-product states,” *Annals of
   Physics* **411**, 167998 (2019).
   [DOI](https://doi.org/10.1016/j.aop.2019.167998) ·
   [arXiv](https://arxiv.org/abs/1901.05824)

   **Provenance:** Design reference for global Krylov time evolution.

6. **GSE/LSE TDVP** — *algorithmic basis*

   M. Yang and S. R. White, “Time Dependent Variational Principle with Ancillary Krylov Subspace,” *Physical Review B* **102**, 094315 (2020).
   [DOI](https://doi.org/10.1103/PhysRevB.102.094315) ·
   [arXiv](https://arxiv.org/abs/2005.06104)

   **Provenance:** Global ancillary-Krylov foundation for the planned GSE/LSE expansion family.

7. **Implicit logarithmic-time evolution** — *design reference*

   J. P. Zima, E. M. Stoudenmire, S. R. White, O. Parcollet, and J. Kaye, “Fast Tensor Network Imaginary Time Evolution by Implicit Stepping on Logarithmic Grids,” arXiv:2606.02930 (2026).
   [arXiv](https://arxiv.org/abs/2606.02930)

   **Provenance:** Design reference for implicit imaginary-time stepping on logarithmic grids.

### Thermal-State Algorithms

1. **Projected purification for bosons** — *planned; algorithmic basis*

   T. Köhler, J. Stolpp, and S. Paeckel, “Efficient and Flexible Approach to Simulate Low-Dimensional Quantum Lattice Models with Large Local Hilbert Spaces,” *SciPost Physics* **10**, 058 (2021).
   [DOI](https://doi.org/10.21468/SciPostPhys.10.3.058) ·
   [arXiv](https://arxiv.org/abs/2008.08466)

   **Provenance:** Algorithmic basis for the planned projected-purification treatment of large bosonic local spaces.

## License

Graft.jl is licensed under the [Apache License 2.0](LICENSE).
