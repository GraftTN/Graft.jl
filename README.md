### GRAFT.jl

A tree tensor network library featuring DMFT impurity solver, with the eventual goal of applying it to real material simulations.

The architecture is largely inspired by [PyTreeNet](https://github.com/Drachier/PyTreeNet), and its tensor network foundation is built on [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl), which lets us exploit the abelian and non-abelian symmetries. It is designed to flexibly adopt new algorithms from papers for experimentation and verification — i.e., tree *graft*ing.

