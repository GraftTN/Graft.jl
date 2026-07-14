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
  (the companion `GraftImpurity.jl` package does the fitting and emits plain
  `OpSum` terms).
* Rewrite passes (PPDress, SU2Reduce, ModeReorder) are compiler-style passes on
  the term list — all TODO at this milestone.
"""
module Symbolic

using ..Backend
using ..Trees: TreeTopology, mount_chain, nodeindex

export SiteOp, Term, OpSum, charge, sites, coefficient, nterms, spin_ops,
    spin_ops_u1, boson_ops, boson_ops_u1, boson_ops_pp, fermion_ops_z2,
    boson_modes, BosonCoupling, Lindbladian, ppdress, su2reduce, modereorder

"""
    SiteOp(site, name, op)

One local operator factor acting on node `site`. The fast path is a neutral
endomorphism `op :: P ← P`. Charged abelian factors use
`op :: P ← P ⊗ C`, where `C` is a one-dimensional graded charge space carrying
the sector injected by the operator and flowing toward the root in TTNO
assembly. `name` is the symbolic identity used for term merging in TTNO
construction (PyTreeNet merges hyperedges by comparing label strings — floats
are never compared).
"""
struct SiteOp
    site::Symbol
    name::Symbol
    op::AbstractTensorMap
    charge
    function SiteOp(site::Symbol, name::Symbol, op::AbstractTensorMap)
        q = _siteop_charge(op)
        return new(site, name, op, q)
    end
end

charge(op::SiteOp) = op.charge

Base.:(==)(a::SiteOp, b::SiteOp) =
    a.site == b.site && a.name == b.name && charge(a) == charge(b)
Base.hash(a::SiteOp, h::UInt) =
    hash(charge(a), hash(a.name, hash(a.site, hash(:SiteOp, h))))

"""
    Term(coeff, ops)

`coeff * ⊗_i ops[i]`, identity-padded on all unlisted sites. This is a
site-labelled tensor product: the vector is construction storage, not a
sequential product, so permuting factors with the same distinct
site-to-operator assignment leaves the term unchanged. Factors must act on
distinct sites (products on the same site should be pre-multiplied). If any
factor is charged, the fused total charge must be trivial; TTNO root caps stay
neutral.
"""
struct Term{C<:Number}
    coeff::C
    ops::Vector{SiteOp}
    function Term(coeff::C, ops::Vector{SiteOp}) where {C<:Number}
        allunique(op.site for op in ops) || throw(ArgumentError("Term factors must act on distinct sites"))
        _assert_neutral_term(ops)
        return new{C}(coeff, ops)
    end
end

Term(coeff::Number, ops::SiteOp...) = Term(coeff, collect(SiteOp, ops))
sites(t::Term) = [op.site for op in t.ops]
coefficient(t::Term) = t.coeff
Base.:*(λ::Number, t::Term) = Term(λ * t.coeff, t.ops)

function _siteop_charge(op::AbstractTensorMap)
    numout(op) == 1 || throw(ArgumentError("SiteOp must have one physical output leg"))
    if numin(op) == 1
        codomain(op)[1] == domain(op)[1] ||
            throw(ArgumentError("neutral SiteOp must map P ← P"))
        P = codomain(op)[1]
        return spacetype(P) === ComplexSpace ? nothing : one(sectortype(P))
    elseif numin(op) == 2
        P = codomain(op)[1]
        domain(op)[1] == P || throw(ArgumentError("charged SiteOp must map P ← P ⊗ C"))
        spacetype(P) === ComplexSpace &&
            throw(ArgumentError("charged SiteOp requires a graded physical space"))
        C = domain(op)[2]
        spacetype(C) == spacetype(P) ||
            throw(ArgumentError("charged SiteOp charge leg must use the physical-space symmetry"))
        dim(C) == 1 || throw(ArgumentError("charged SiteOp charge leg must be one-dimensional"))
        qs = collect(sectors(C))
        length(qs) == 1 ||
            throw(ArgumentError("charged SiteOp charge leg must carry exactly one sector"))
        return only(qs)
    else
        throw(ArgumentError("SiteOp must be `P ← P` or charged `P ← P ⊗ C`"))
    end
end

function _assert_neutral_term(ops::Vector{SiteOp})
    qs = [charge(op) for op in ops if charge(op) !== nothing]
    isempty(qs) && return true
    Q = typeof(qs[1])
    qtot = one(Q)
    for q in qs
        q isa Q || throw(ArgumentError("all charged factors in a Term must use the same sector type"))
        fused = qtot ⊗ q
        length(fused) == 1 ||
            throw(ArgumentError("non-abelian charged Terms need SU2Reduce/graded fusion-tree support (TODO(M3))"))
        qtot = only(fused)
    end
    qtot == one(Q) ||
        throw(ArgumentError("charged Term is not neutral: fused charge $qtot"))
    return true
end

"""
    OpSum()

A sum of product terms. Build with `+=`:

    H = OpSum()
    H += Term(-1.0, SiteOp(:s1, :Z, Z), SiteOp(:s2, :Z, Z))

Implemented discrete boson helpers: [`boson_modes`](@ref) and
[`BosonCoupling`](@ref). Local abelian charged libraries are provided by
[`spin_ops_u1`](@ref), [`boson_ops_u1`](@ref), and
[`fermion_ops_z2`](@ref). TODO(future structured-generator milestone, §4a):
structured matrix generators
`Hopping(t)`, `Coulomb(V)` (via ISDF-THC/SVD pre-factorization, §4b),
`Hybridization(Vk, εk)`. Continuous `BosonBath(ImU)` lives in the companion
`GraftImpurity.jl` package and lowers to these discrete helpers after fitting.
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
# local operator libraries
# ---------------------------------------------------------------------------

"""
    spin_ops(; elt=ComplexF64) -> (; X, Y, Z, Sp, Sm, N, I, P)

Spin-1/2 operators on `ℂ²` (trivial sector). `P = ℂ²` is the physical space.
Use [`spin_ops_u1`](@ref) for the abelian charged `U(1)_Sz` library. TODO(M3):
SU(2) reduced tensor operators (the `SU2Reduce` pass owns those).
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
    spin_ops_u1(; elt=ComplexF64) -> (; Sp, Sm, Z, N, I, P, Cp, Cm)

Spin-1/2 operators over `U(1)` charge sectors `0,1`. `Sp` carries charge `+1`
and `Sm` carries charge `-1`, both as `P <- P ⊗ C` charged local operators.
The diagonal convention is `N = diag(0,1)` and `Z = 2N - I`, matching the
sector ordering used by charged TTNO tests.
"""
function spin_ops_u1(; elt::Type{<:Number}=ComplexF64)
    P = U1Space(0 => 1, 1 => 1)
    Cp = U1Space(1 => 1)
    Cm = U1Space(-1 => 1)
    charged(mat, C) = TensorMap(reshape(elt.(mat), 2, 2, 1), P ← P ⊗ C)
    neutral(mat) = TensorMap(elt.(mat), P ← P)
    Sp = charged([0 0; 1 0], Cp)
    Sm = charged([0 1; 0 0], Cm)
    N = neutral([0 0; 0 1])
    Z = neutral([-1 0; 0 1])
    I = neutral([1 0; 0 1])
    return (; Sp, Sm, Z, N, I, P, Cp, Cm)
end

"""
    boson_ops(nmax; elt=Float64) -> (; B, Bd, X, N, I, P)

Truncated boson (d = nmax+1) on trivial-sector space. Use
[`boson_ops_u1`](@ref) for number-conserving charged boson hopping and
[`boson_ops_pp`](@ref) for projected purification.
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
    boson_ops_u1(nmax; elt=ComplexF64) -> (; B, Bd, N, I, P, Cp, Cm)

Truncated boson over `U(1)` occupation sectors `0:nmax`. `B` carries charge
`-1` and `Bd` carries charge `+1`; neutral number and identity operators remain
`P <- P`. This is the public B2/M0 helper for number-conserving hopping terms
such as `b†_i b_j`.
"""
function boson_ops_u1(nmax::Int; elt::Type{<:Number}=ComplexF64)
    nmax >= 0 || throw(ArgumentError("nmax must be nonnegative"))
    d = nmax + 1
    P = U1Space([n => 1 for n in 0:nmax]...)
    Cp = U1Space(1 => 1)
    Cm = U1Space(-1 => 1)
    lower = zeros(elt, d, d, 1)
    raise = zeros(elt, d, d, 1)
    nmat = zeros(elt, d, d)
    for n in 1:nmax
        lower[n, n + 1, 1] = sqrt(elt(n))
        raise[n + 1, n, 1] = sqrt(elt(n))
    end
    for n in 0:nmax
        nmat[n + 1, n + 1] = elt(n)
    end
    B = TensorMap(lower, P ← P ⊗ Cm)
    Bd = TensorMap(raise, P ← P ⊗ Cp)
    N = TensorMap(nmat, P ← P)
    I = TensorMap(Matrix{elt}(LinearAlgebra_I(d)), P ← P)
    return (; B, Bd, N, I, P, Cp, Cm)
end

"""
    fermion_ops_z2(; elt=ComplexF64) -> (; C, Cd, N, I, F, P, Q)

Spinless-fermion operators over TensorKit's fermion-parity `fℤ₂` grading. Both
annihilation `C` and creation `Cd` carry the odd charge leg `Q`, while `N`,
identity `I`, and parity `F = (-1)^N` are neutral. TensorKit's graded braiding
handles fermionic exchange; no Jordan-Wigner string is generated here.
"""
function fermion_ops_z2(; elt::Type{<:Number}=ComplexF64)
    even = FermionParity(0)
    odd = FermionParity(1)
    P = Vect[FermionParity](even => 1, odd => 1)
    Q = Vect[FermionParity](odd => 1)
    Cd = zeros(elt, P ← P ⊗ Q)
    C = zeros(elt, P ← P ⊗ Q)
    N = zeros(elt, P ← P)
    I = zeros(elt, P ← P)
    F = zeros(elt, P ← P)
    for (sector, block_) in blocks(Cd)
        sector == odd && (block_[1, 1] = one(elt))
    end
    for (sector, block_) in blocks(C)
        sector == even && (block_[1, 1] = one(elt))
    end
    for (sector, block_) in blocks(N)
        block_[1, 1] = sector == odd ? one(elt) : zero(elt)
    end
    for (_, block_) in blocks(I)
        block_[1, 1] = one(elt)
    end
    for (sector, block_) in blocks(F)
        block_[1, 1] = sector == odd ? -one(elt) : one(elt)
    end
    return (; C, Cd, N, I, F, P, Q)
end

"""
    boson_ops_pp(nmax; elt=Float64) -> (; Bp, B, Bpd, Bd, Bb, Bbd, N, I, P, Bspace)

Projected-purification boson operators over a shared U(1) charge. `P` carries
occupation sectors `0:nmax`; `Bspace` is the dual ancilla representation. The
charged pair rewrite is `b† -> Bpd_P * Bbd_B` and `b -> Bp_P * Bb_B`, while
`N`/`I` stay neutral on the physical P site. `Bb`/`Bbd` are balancing
operators: they shift the ancilla basis with unit matrix elements, not
canonical bosonic `sqrt(n)` amplitudes.
"""
function boson_ops_pp(nmax::Int; elt::Type{<:Number}=Float64)
    nmax >= 0 || throw(ArgumentError("nmax must be nonnegative"))
    d = nmax + 1
    P = U1Space([n => 1 for n in 0:nmax]...)
    Bspace = dual(P)
    Cp = U1Space(1 => 1)
    Cm = U1Space(-1 => 1)
    lower = zeros(elt, d, d, 1)
    raise = zeros(elt, d, d, 1)
    # Projected-purification balancing operators are unit shifts, not bosons.
    bal_lower = zeros(elt, d, d, 1)
    bal_raise = zeros(elt, d, d, 1)
    for n in 1:nmax
        lower[n, n + 1, 1] = sqrt(elt(n))
        raise[n + 1, n, 1] = sqrt(elt(n))
        bal_lower[n, n + 1, 1] = one(elt)
        bal_raise[n + 1, n, 1] = one(elt)
    end
    Bp = TensorMap(lower, P ← P ⊗ Cm)
    Bpd = TensorMap(raise, P ← P ⊗ Cp)
    # Ancilla operators act in the dual representation; the charge legs are
    # opposite so each P-B pair is neutral.
    Bb = TensorMap(bal_lower, Bspace ← Bspace ⊗ Cp)
    Bbd = TensorMap(bal_raise, Bspace ← Bspace ⊗ Cm)
    nmat = zeros(elt, d, d)
    for n in 0:nmax
        nmat[n + 1, n + 1] = elt(n)
    end
    N = TensorMap(nmat, P ← P)
    I = TensorMap(Matrix{elt}(LinearAlgebra_I(d)), P ← P)
    return (; Bp, B=Bp, Bpd, Bd=Bpd, Bb, Bbd, N, I, P, Bspace)
end

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

"""
    ppdress(H, topo, phys; nmax, boson_sites=keys(phys), prefix=:ppB)
        -> (Hprime, topoprime, physprime)

Projected-purification rewrite pass. Each boson site `s` gains an ancilla leaf
`Symbol(s, "_B")` mounted directly on `s`; topology and physical spaces change,
so the pass returns all three. The rewrite follows the pinned B3 design:
`Bd_s -> Bpd_s * Bbd_s_B`, `B_s -> Bp_s * Bb_s_B`, while neutral `N`/`I`
factors remain on `s`. All dressed modes share one U(1)_PP. Neutral
trivial-sector matter sites are lifted into the shared zero-charge sector so
mixed spin-boson PP Hamiltonians still satisfy TTNO's one-spacetype invariant.
"""
function ppdress(H::OpSum, topo::TreeTopology, phys::Dict{Symbol,<:ElementarySpace};
                 nmax::Int, boson_sites=keys(phys), prefix::Symbol=:ppB)
    pp = boson_ops_pp(nmax)
    bsites = Set(Symbol.(collect(boson_sites)))
    topo′ = topo
    anc = Dict{Symbol,Symbol}()
    for s in sort!(collect(bsites); by=string)
        haskey(phys, s) || throw(ArgumentError("ppdress boson site $s is absent from phys"))
        nodeindex(topo′, s) # validate before mounting
        a = Symbol(s, :_B, 1)
        anc[s] = a
        topo′ = mount_chain(topo′, s, 1; prefix=Symbol(s, :_B))
    end
    phys′ = Dict{Symbol,ElementarySpace}()
    liftspaces = Dict{Symbol,ElementarySpace}()
    for (site, P) in phys
        if site in bsites
            phys′[site] = pp.P
        else
            P′ = _pp_lift_space(P, pp)
            phys′[site] = P′
            spacetype(P) === ComplexSpace && (liftspaces[site] = P′)
        end
    end
    for s in bsites
        phys′[anc[s]] = pp.Bspace
    end

    H′ = OpSum()
    for term in H
        partial = [SiteOp[]]
        for so in term.ops
            expanded = _pp_factor(so, bsites, anc, pp, liftspaces)
            partial = [vcat(base, add) for base in partial for add in expanded]
        end
        for ops in partial
            H′ += Term(term.coeff, ops)
        end
    end
    return H′, topo′, phys′
end

function _pp_lift_space(P::ElementarySpace, pp)
    if spacetype(P) === ComplexSpace
        return U1Space(0 => dim(P))
    elseif spacetype(P) === spacetype(pp.P)
        return P
    else
        throw(ArgumentError("ppdress can lift ComplexSpace matter or reuse U(1)_PP-compatible spaces; got $P"))
    end
end

function _pp_lift_neutral_op(op::AbstractTensorMap, P::ElementarySpace)
    numout(op) == 1 && numin(op) == 1 ||
        throw(ArgumentError("ppdress can lift only neutral trivial-sector matter operators"))
    dim(codomain(op)[1]) == dim(P) && dim(domain(op)[1]) == dim(P) ||
        throw(ArgumentError("ppdress neutral lift dimension mismatch"))
    return TensorMap(reshape(convert(Array, op), dim(P), dim(P)), P ← P)
end

function _pp_factor(so::SiteOp, bsites, anc, pp, liftspaces)
    if !(so.site in bsites)
        haskey(liftspaces, so.site) || return [[so]]
        return [[SiteOp(so.site, so.name, _pp_lift_neutral_op(so.op, liftspaces[so.site]))]]
    end
    if so.name == :Bd
        return [[SiteOp(so.site, :Bpd, pp.Bpd), SiteOp(anc[so.site], :Bbd, pp.Bbd)]]
    elseif so.name == :B
        return [[SiteOp(so.site, :Bp, pp.Bp), SiteOp(anc[so.site], :Bb, pp.Bb)]]
    elseif so.name == :X
        return [[SiteOp(so.site, :Bpd, pp.Bpd), SiteOp(anc[so.site], :Bbd, pp.Bbd)],
                [SiteOp(so.site, :Bp, pp.Bp), SiteOp(anc[so.site], :Bb, pp.Bb)]]
    elseif so.name in (:N, :I)
        return [[SiteOp(so.site, so.name, so.name == :N ? pp.N : pp.I)]]
    elseif _pp_is_diagonal_neutral(so.op)
        return [[SiteOp(so.site, so.name, _pp_lift_neutral_op(so.op, pp.P))]]
    else
        throw(ArgumentError("ppdress does not know how to rewrite boson operator `$(so.name)` at $(so.site)"))
    end
end

function _pp_is_diagonal_neutral(op::AbstractTensorMap)
    numout(op) == 1 && numin(op) == 1 || return false
    codomain(op)[1] == domain(op)[1] || return false
    A = convert(Array, op)
    return all(i == j || iszero(A[i, j]) for i in axes(A, 1), j in axes(A, 2))
end

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
