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
    boson_modes, BosonCoupling, Lindbladian, ppdress, su2reduce, modereorder

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

Implemented discrete boson helpers: [`boson_modes`](@ref) and
[`BosonCoupling`](@ref). TODO(M0/M5, §4a): structured fermion generators
`Hopping(t)`, `Coulomb(V)` (via ISDF-THC/SVD pre-factorization, §4b),
`Hybridization(Vk, εk)`. Continuous `BosonBath(ImU)` lives in `Impurity` and
lowers to these discrete helpers after fitting.
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
    boson_ops(nmax; elt=Float64) -> (; B, Bd, X, N, I, P)

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
    bd = collect(transpose(b))
    B = TensorMap(b, P ← P)
    Bd = TensorMap(bd, P ← P)
    X = TensorMap(b .+ bd, P ← P)
    N = TensorMap(collect((b' * b)), P ← P)
    I = TensorMap(Matrix{elt}(LinearAlgebra_I(d)), P ← P)
    return (; B, Bd, X, N, I, P)
end

# small identity helper without pulling LinearAlgebra into the public surface
LinearAlgebra_I(d::Int) = [i == j ? 1.0 : 0.0 for i in 1:d, j in 1:d]

"""
    boson_modes(modes; ops, site_prefix=:b) -> OpSum

Expand already-discretized boson modes into `sum_k omega_k n_k`. `ops` is the
operator tuple from [`boson_ops`](@ref). Each mode may be `site => omega`,
`(omega, site)`, `(site, omega)`, a bare `omega` (site `Symbol(site_prefix, k)`),
or a named tuple with `site` and `omega`/`ω`.
"""
function boson_modes(modes; ops, site_prefix::Symbol=:b)
    H = OpSum()
    for (k, mode) in enumerate(modes)
        site, omega = _mode_site_omega(mode, k, site_prefix)
        H += Term(omega, SiteOp(site, :N, ops.N))
    end
    return H
end

"""
    BosonCoupling(couplings, kind; matter_ops, boson_ops, density=:N) -> OpSum

Expand discrete matter-boson couplings into plain product terms. `kind=:density`
accepts couplings as `(g, matter_site, boson_site)` or
`(matter_site, boson_site) => g` and emits `g n_m X_b`.

`kind=:hopping` accepts `(g, left_site, right_site, boson_site)` or
`(left_site, right_site, boson_site) => g` and emits the SSH/Peierls form
`g Sp_left Sm_right X_b + conj(g) Sm_left Sp_right X_b`. The default
`Sp`/`Sm` names match `spin_ops`; callers with fermionic charged operators can
pass compatible operator tuples once B2 lands.
"""
function BosonCoupling(couplings, kind::Symbol; matter_ops, boson_ops,
                       density::Symbol=:N, create::Symbol=:Sp,
                       destroy::Symbol=:Sm)
    H = OpSum()
    X = _opfield(boson_ops, :X)
    if kind == :density
        n_op = _opfield(matter_ops, density)
        for c in couplings
            g, matter_site, boson_site = _density_coupling(c)
            H += Term(g, SiteOp(matter_site, density, n_op),
                      SiteOp(boson_site, :X, X))
        end
    elseif kind == :hopping
        c_op = _opfield(matter_ops, create)
        a_op = _opfield(matter_ops, destroy)
        for c in couplings
            g, left_site, right_site, boson_site = _hopping_coupling(c)
            H += Term(g, SiteOp(left_site, create, c_op),
                      SiteOp(right_site, destroy, a_op),
                      SiteOp(boson_site, :X, X))
            H += Term(conj(g), SiteOp(left_site, destroy, a_op),
                      SiteOp(right_site, create, c_op),
                      SiteOp(boson_site, :X, X))
        end
    else
        throw(ArgumentError("unknown BosonCoupling kind $kind; expected :density or :hopping"))
    end
    return H
end

function _mode_site_omega(mode::Pair, ::Int, ::Symbol)
    mode.first isa Symbol || throw(ArgumentError("mode pair must be site => omega"))
    return mode.first, mode.second
end
function _mode_site_omega(mode::Tuple, ::Int, ::Symbol)
    length(mode) == 2 || throw(ArgumentError("mode tuple must have two entries"))
    a, b = mode
    if a isa Number && b isa Symbol
        return b, a
    elseif a isa Symbol && b isa Number
        return a, b
    else
        throw(ArgumentError("mode tuple must be (omega, site) or (site, omega)"))
    end
end
_mode_site_omega(omega::Number, k::Int, site_prefix::Symbol) = Symbol(site_prefix, k), omega
function _mode_site_omega(mode::NamedTuple, ::Int, ::Symbol)
    haskey(mode, :site) || throw(ArgumentError("mode named tuple needs a `site` field"))
    omega = haskey(mode, :omega) ? mode.omega :
        (haskey(mode, :ω) ? getproperty(mode, :ω) :
         throw(ArgumentError("mode named tuple needs `omega` or `ω`")))
    return mode.site, omega
end

function _density_coupling(c::Pair)
    sites = c.first
    sites isa Tuple && length(sites) == 2 ||
        throw(ArgumentError("density coupling pair must be (matter_site, boson_site) => g"))
    return c.second, sites[1], sites[2]
end
function _density_coupling(c::Tuple)
    length(c) == 3 ||
        throw(ArgumentError("density coupling tuple must be (g, matter_site, boson_site)"))
    return c[1], c[2], c[3]
end

function _hopping_coupling(c::Pair)
    sites = c.first
    sites isa Tuple && length(sites) == 3 ||
        throw(ArgumentError("hopping coupling pair must be (left_site, right_site, boson_site) => g"))
    return c.second, sites[1], sites[2], sites[3]
end
function _hopping_coupling(c::Tuple)
    length(c) == 4 ||
        throw(ArgumentError("hopping coupling tuple must be (g, left_site, right_site, boson_site)"))
    return c[1], c[2], c[3], c[4]
end

function _opfield(ops, name::Symbol)
    hasproperty(ops, name) ||
        throw(ArgumentError("operator tuple has no field `$name`"))
    return getproperty(ops, name)
end

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
