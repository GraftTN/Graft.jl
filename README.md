### GRAFT.jl

A tree tensor network library featuring DMFT impurity solver, with the eventual goal of applying it to real material simulations.

The architecture is largely inspired by [PyTreeNet](https://github.com/Drachier/PyTreeNet), and its tensor network foundation is built on [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl), which lets us exploit the abelian and non-abelian symmetries. It is designed to flexibly adopt new algorithms from papers for experimentation and verification — i.e., tree *graft*ing.

#### Layering

```
GRAFT.jl
├── Backend        # L0  TensorKit adapter: spaces, sectors, arrow convention, TruncationScheme
├── Trees          # L1  immutable TreeTopology, traversal/paths, geometries (mps/star/binary/fork)
├── Networks       # L3  TTNS / TTNO / TTNDO, canonical form, move_center!, update_tensor!
├── Contractions   # L2  EnvCache (directed-edge environments, explicit invalidation),
│                  #     effective Hamiltonians (0/1/2-site), inner/expect
├── Symbolic       # L4a OpSum / Term / SiteOp; PPDress / SU2Reduce / ModeReorder passes (TODO)
├── TTNOBuild      # L4b symbolic Hamiltonian → TTNO (tree FSM / state-diagram channels)
├── GroundState    # L5a dmrg1!, dmrg2!; dmrg1_3s! + expand! primitive (TODO)
├── Evolution      # L5b complex-step Evolver family: TDVP1, TDVP2, TDVP1_CBE;
│                  #     GK / GSE / LSE / TEBD / BUG / ImplicitLogTime (TODO)
├── Thermal        # L5c Purified / METTS / HybridMETTS representations (TODO, M2)
├── FreqDomain     # L5d TaSK tangent-space resolvent kernel (TODO, M6)
├── Impurity       # L6  Partition (user-declared), fit_bath / audit_partition / solve (TODO)
├── Checkpoints    # ✕   atomic JLD2 checkpoint!/resume with rotation
├── Parallel       # ✕   threading/MPI roll-out plan (TODO)
└── TestUtils      # ✕   random/product TTNS, dense references, exact diagonalization
```

The design contract lives in `GRAFT_architecture_v0` (design principles §0, global
constraints §9): geometry is data, kernels are orthogonal families, every `Evolver`
takes a complex step `dz` (`ψ ← exp(dz·H)ψ`; real time `dz = -im*δt`, imaginary time
`dz = -δτ`), hermiticity is a trait, truncation has a single entry point, one
orthogonality center guarded by `move_center!`/`update_tensor!` events.

#### Implemented (validated against ED / exact propagation on small trees)

- TTNS canonical form and gauge moves on arbitrary trees, abelian sectors included
  (`FermionParity ⊠ U(1) ⊠ U(1)` type aliases ready; no Jordan–Wigner strings by
  construction — trees are planar, TensorKit's graded braiding does the rest).
- TTNO assembly from `OpSum` product terms (idle/done/active channel construction —
  reproduces PyTreeNet's state-diagram channels; matches dense Hamiltonians exactly).
  Graded/charged operator assembly is the next TODO (fermionic hopping needs it).
- `dmrg1!`, `dmrg2!` — converge to ED ground energies at 1e-10 tolerances.
- `TDVP1` (first/second order tree projector splitting, sweep-equivalent to
  PyTreeNet's `FirstOrder`/`SecondOrderOneSiteTDVP`), `TDVP2` (two-site, bond-adaptive),
  and **`TDVP1_CBE`** — the controlled-bond-expansion 1TDVP ported from the local
  PyTreeNet fork (two-site predictor + shrewd selection + enriched split;
  `enabled=false` reproduces `TDVP1` exactly, same contract as the fork's tests).
- `EnvCache` sandwich environments with explicit invalidation events; effective
  0/1/2-site Hamiltonians as matrix-free maps for KrylovKit.
- Atomic, rotating JLD2 checkpoints (`checkpoint!`/`resume`).

#### Quick example

```julia
using GRAFT, GRAFT.TestUtils, Random
using GRAFT.Backend: ℂ

topo = star_topology(3, 2)                       # impurity-style star geometry
S = spin_ops()
phys = Dict(nodeid(topo, i) => S.P for i in 1:nnodes(topo))

H = OpSum()
for (c, p) in GRAFT.Trees.edges(topo)
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

#### Tests

```julia
julia --project -e 'using Pkg; Pkg.test()'
```

Every kernel ships with a ≲16-site tree test against exact diagonalization and a
gauge-invariance property test (architecture constraint §9.11).
