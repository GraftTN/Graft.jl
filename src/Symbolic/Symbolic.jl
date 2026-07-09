"""
L4a — Symbolic operator layer (architecture §4a).

`OpSum`-style input language. PyTreeNet counterparts: `operators/tensorproduct.py`
(`TensorProduct` = one product term as a site→operator map) and
`operators/hamiltonian.py` (list of terms + label→matrix conversion).

Design rules (§4a):
* Coefficients may be *structured tensors* (Hopping matrices, Coulomb V_ijkl) —
  structured generators expand into plain product terms before TTNO assembly.
* The symbolic layer never knows about temperature or the word "retarded":
  finite-T baths and retarded interactions arrive here only as explicit modes
  (`Impurity.Bath` does the fitting and emits plain `OpSum` terms).
* Rewrite passes (PPDress, SU2Reduce, ModeReorder) are compiler-style passes on
  the term list — all TODO at this milestone.
"""
module Symbolic

using ..Backend

export SiteOp, Term, OpSum, sites, coefficient, nterms, spin_ops, boson_ops,
    Lindbladian, ppdress, su2reduce, modereorder

"""
    SiteOp(site, name, op)

One local operator factor: `op :: P ← P` acting on the physical space of node
`site`. `name` is the symbolic identity used for term merging in TTNO
construction (PyTreeNet merges hyperedges by comparing label strings — floats
are never compared).

TODO(M0 fermion path): charged operators (c†, c) carrying a nontrivial sector
on an auxiliary leg (`op :: P ← P ⊗ C`); the state diagram then threads the
charge flux through every virtual space on the tree path between paired
factors. Until then every factor must be charge-diagonal.
"""
struct SiteOp
    site::Symbol
    name::Symbol
    op::AbstractTensorMap
    function SiteOp(site::Symbol, name::Symbol, op::AbstractTensorMap)
        numout(op) == numin(op) == 1 || throw(ArgumentError("SiteOp must be a single-site endomorphism P ← P (charged ops TODO)"))
        codomain(op)[1] == domain(op)[1] || throw(ArgumentError("SiteOp must map P ← P"))
        return new(site, name, op)
    end
end

Base.:(==)(a::SiteOp, b::SiteOp) = a.site == b.site && a.name == b.name
Base.hash(a::SiteOp, h::UInt) = hash(a.name, hash(a.site, hash(:SiteOp, h)))

"""
    Term(coeff, ops)

`coeff * ⊗_i ops[i]`, identity-padded on all unlisted sites. Factors must act
on distinct sites (products on the same site should be pre-multiplied).
"""
struct Term{C<:Number}
    coeff::C
    ops::Vector{SiteOp}
    function Term(coeff::C, ops::Vector{SiteOp}) where {C<:Number}
        allunique(op.site for op in ops) || throw(ArgumentError("Term factors must act on distinct sites"))
        return new{C}(coeff, ops)
    end
end

Term(coeff::Number, ops::SiteOp...) = Term(coeff, collect(SiteOp, ops))
sites(t::Term) = [op.site for op in t.ops]
coefficient(t::Term) = t.coeff
Base.:*(λ::Number, t::Term) = Term(λ * t.coeff, t.ops)

"""
    OpSum()

A sum of product terms. Build with `+=`:

    H = OpSum()
    H += Term(-1.0, SiteOp(:s1, :Z, Z), SiteOp(:s2, :Z, Z))

TODO(M0/M5, §4a): structured generators `Hopping(t)`, `Coulomb(V)` (via
ISDF-THC/SVD pre-factorization, §4b), `Hybridization(Vk, εk)`,
`BosonBath(ImU)`, `BosonCoupling(g, :density|:hopping)` — each expands into
plain `Term`s here.
"""
struct OpSum
    terms::Vector{Term}
end
OpSum() = OpSum(Term[])

nterms(H::OpSum) = length(H.terms)
Base.:+(H::OpSum, t::Term) = OpSum(vcat(H.terms, t))
Base.:+(H::OpSum, G::OpSum) = OpSum(vcat(H.terms, G.terms))
Base.:*(λ::Number, H::OpSum) = OpSum([λ * t for t in H.terms])
Base.iterate(H::OpSum, s...) = iterate(H.terms, s...)
Base.length(H::OpSum) = length(H.terms)
Base.eltype(::Type{OpSum}) = Term

"""
    Lindbladian(H, jumps)

Symbolic-layer citizen (§3, §11.3): kept so the TTNDO / quasi-Lindblad route
needs no new symbolic machinery. TODO(post-M2): vectorization pass emitting the
Liouvillian TTNO.
"""
struct Lindbladian
    H::OpSum
    jumps::Vector{Term}
end

# ---------------------------------------------------------------------------
# local operator libraries (trivial-sector; graded versions TODO)
# ---------------------------------------------------------------------------

"""
    spin_ops(; elt=ComplexF64) -> (; X, Y, Z, Sp, Sm, N, I, P)

Spin-1/2 operators on `ℂ²` (trivial sector). `P = ℂ²` is the physical space.
TODO(M0): graded U(1)_Sz versions; TODO(M3): SU(2) reduced tensor operators
(the `SU2Reduce` pass owns those).
"""
function spin_ops(; elt::Type{<:Number}=ComplexF64)
    P = ℂ^2
    mk(m) = TensorMap(elt.(m), P ← P)
    X = mk([0 1; 1 0]); Z = mk([1 0; 0 -1])
    Y = elt <: Complex ? mk([0 -im; im 0]) : nothing
    Sp = mk([0 1; 0 0]); Sm = mk([0 0; 1 0])
    N = mk([0 0; 0 1]); I = mk([1 0; 0 1])
    return (; X, Y, Z, Sp, Sm, N, I, P)
end

"""
    boson_ops(nmax; elt=Float64) -> (; B, Bd, N, I, P)

Truncated boson (d = nmax+1) on trivial-sector space. TODO(M5): U(1)_PP graded
version for projected purification (PPDress pass).
"""
function boson_ops(nmax::Int; elt::Type{<:Number}=Float64)
    d = nmax + 1
    P = ℂ^d
    b = zeros(elt, d, d)
    for n in 1:nmax
        b[n, n + 1] = sqrt(elt(n))
    end
    B = TensorMap(b, P ← P)
    Bd = TensorMap(collect(b'), P ← P)
    N = TensorMap(collect((b' * b)), P ← P)
    I = TensorMap(Matrix{elt}(LinearAlgebra_I(d)), P ← P)
    return (; B, Bd, N, I, P)
end

# small identity helper without pulling LinearAlgebra into the public surface
LinearAlgebra_I(d::Int) = [i == j ? 1.0 : 0.0 for i in 1:d, j in 1:d]

# ---------------------------------------------------------------------------
# rewrite passes — compiler-style, applied in order to the term list (§4a)
# ---------------------------------------------------------------------------

# TODO(M5): PPDress pass — rewrite every `b†` into `b†_P · b_B` pairs
# (Köhler–Stolpp–Paeckel, SciPost Phys. 10, 058 (2021)); adds the artificial
# U(1)_PP conservation (n_P + n_B = d − 1). Truncating the P–B bond through
# `TruncationScheme` is then equivalent to LBO for free (§2).
"""
    ppdress(H::OpSum) -> OpSum

Projected-purification rewrite pass. TODO(M5) — no methods yet.
"""
function ppdress end

# TODO(M3): SU2Reduce pass — c† as spin-1/2 tensor operator, hoppings and
# Coulomb terms as scalar contractions; emits reduced matrix elements + fusion
# tree labels. Hardest layer of the whole stack (§11.1); the abelian path
# bypasses it entirely, so it blocks nothing before M3.
"""
    su2reduce(H::OpSum) -> OpSum

SU(2) reduction rewrite pass. TODO(M3) — no methods yet.
"""
function su2reduce end

# TODO(lattice problems only, §4a): ModeReorder — mutual-information / Fiedler
# ordering hooks (Legeza); relabels term sites, never touches the core.
# Impurity problems do NOT take this path — partitioning there is a user
# declaration (§6.2).
"""
    modereorder(H::OpSum; method=:fiedler) -> OpSum

Site-relabelling reordering pass. TODO — no methods yet.
"""
function modereorder end

end # module Symbolic
