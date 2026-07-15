using Test
using Graft
using Graft.TestUtils
using Graft.Backend: FermionParity, U1Irrep, Vect, ⊠, ⊗, ←, blocks, domain,
    TensorMap
using LinearAlgebra: I, norm
using Random

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

# CG-009 gate: a physical site carrying several fermionic modes is a graded
# space with sector degeneracy. The carrier contract below is GraftImpurity's
# `ParticleNumberSector` convention: Fock states sorted by (particle number,
# state integer), mode j on bit (j-1), the intra-site Jordan-Wigner string on
# earlier modes (`c_j |n> = (-1)^(sum_{i<j} n_i) n_j |n-e_j>`), and charged
# tensors written as dense `P ← P ⊗ C` reshapes. The independent oracle is the
# global Jordan-Wigner action in canonical site order with the declared
# intra-site mode order.

const _MM_Q = typeof(FermionParity(0) ⊠ U1Irrep(0))
_mm_sector(n::Int) = FermionParity(n % 2) ⊠ U1Irrep(n)

function _multimode_fermion_carrier(mode_count::Int)
    states = sort!(collect(0:((1 << mode_count) - 1)); by=s -> (count_ones(s), s))
    d = length(states)
    pos = Dict(s => i for (i, s) in enumerate(states))
    P = Vect[_MM_Q]((_mm_sector(n) => binomial(mode_count, n) for n in 0:mode_count)...)
    annihilate = Vect[_MM_Q]((FermionParity(1) ⊠ U1Irrep(-1)) => 1)
    create = Vect[_MM_Q]((FermionParity(1) ⊠ U1Irrep(1)) => 1)
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
        C[j] = TensorMap(reshape(a, d, d, 1), P ← P ⊗ annihilate)
        Cd[j] = TensorMap(reshape(c, d, d, 1), P ← P ⊗ create)
        N[j] = TensorMap(nn, P ← P)
    end
    Iop = TensorMap(Matrix{ComplexF64}(I, d, d), P ← P)
    return (; P, C=Tuple(C), Cd=Tuple(Cd), N=Tuple(N), I=Iop,
            states, pos, mode_count)
end

function _fz2_multimode_carrier(mode_count::Int)
    states = sort!(collect(0:((1 << mode_count) - 1));
                   by=s -> (isodd(count_ones(s)), s))
    d = length(states)
    pos = Dict(s => i for (i, s) in enumerate(states))
    P = Vect[FermionParity](FermionParity(0) => 1 << (mode_count - 1),
                            FermionParity(1) => 1 << (mode_count - 1))
    odd = Vect[FermionParity](FermionParity(1) => 1)
    a = zeros(ComplexF64, d, d)
    c = zeros(ComplexF64, d, d)
    mask = 1 << (mode_count - 1)   # last declared mode: intra-site string is live
    for s in states
        sgn = isodd(count_ones(s & (mask - 1))) ? -1.0 : 1.0
        if (s & mask) != 0
            a[pos[s & ~mask], pos[s]] = sgn
        else
            c[pos[s | mask], pos[s]] = sgn
        end
    end
    return (; P,
            C=TensorMap(reshape(a, d, d, 1), P ← P ⊗ odd),
            Cd=TensorMap(reshape(c, d, d, 1), P ← P ⊗ odd),
            states, pos, mode_count)
end

"Independent JW annihilator over the flattened global mode list."
function _mm_jw_annihilator(total_modes::Int, mode::Int)
    dimension = 1 << total_modes
    matrix = zeros(ComplexF64, dimension, dimension)
    bit = 1 << (mode - 1)
    for s in 0:(dimension - 1)
        s & bit == 0 && continue
        matrix[(s & ~bit) + 1, s + 1] = isodd(count_ones(s & (bit - 1))) ? -1.0 : 1.0
    end
    return matrix
end

_mm_sites(topo, phys) = [nodeid(topo, n) for n in 1:nnodes(topo)
                         if haskey(phys, nodeid(topo, n))]

"(sector => degeneracy index, local Fock state) labels in carrier basis order."
_mm_local_basis(car) = [(_mm_sector(count_ones(s)) =>
                             count(x -> count_ones(x) == count_ones(s) && x <= s,
                                   car.states),
                         s) for s in car.states]

"Degeneracy-resolved product basis plus each state's global JW basis index."
function _mm_product_basis(topo, phys, carriers)
    sites = _mm_sites(topo, phys)
    locals_ = [_mm_local_basis(carriers[site]) for site in sites]
    offsets = cumsum([0; [carriers[site].mode_count for site in sites]])
    basis = Vector{Any}()
    jwidx = Int[]
    for combo in Iterators.product((eachindex(l) for l in locals_)...)
        sectors_ = Dict{Symbol,Any}()
        bits = 0
        for (k, site) in enumerate(sites)
            label, s = locals_[k][combo[k]]
            sectors_[site] = label
            bits |= s << offsets[k]
        end
        push!(basis, product_ttns(ComplexF64, topo, phys, sectors_))
        push!(jwidx, bits + 1)
    end
    return basis, jwidx
end

"Exact action matrix on a product basis; plans are cached and reused."
function _mm_action_matrix(O, basis)
    dimension = length(basis)
    values = zeros(ComplexF64, dimension, dimension)
    root = topology(O).root
    plan_cache = EnvCache(topology(O))
    for (column, input) in enumerate(basis)
        output = apply(O, input; optimize=false)
        output_root = domain(output.tensors[root])[1]
        for (row, bra) in enumerate(basis)
            domain(bra.tensors[root])[1] == output_root || continue
            values[row, column] = inner(bra, output; plan_cache, optimize=false)
        end
    end
    return values
end

"Global embedding of one labelled factor: JW strings over canonically-earlier
sites, the supplied (left-string convention) matrix at its own site."
function _mm_global_factor(topo, phys, carriers, site::Symbol, op, odd::Bool)
    sites = _mm_sites(topo, phys)
    position = Dict(s => i for (i, s) in enumerate(sites))
    mats = Matrix{ComplexF64}[]
    for s in sites
        car = carriers[s]
        d = 1 << car.mode_count
        if s == site
            perm = [car.pos[b] for b in 0:(d - 1)]
            dense = convert(Array, op)
            m = ndims(dense) == 3 ? dense[:, :, 1] : dense
            push!(mats, ComplexF64.(m[perm, perm]))
        elseif position[s] < position[site] && odd
            push!(mats, ComplexF64[(i == j ? (-1.0)^count_ones(i - 1) : 0.0)
                                   for i in 1:d, j in 1:d])
        else
            push!(mats, Matrix{ComplexF64}(I, d, d))
        end
    end
    out = mats[1]
    for k in 2:length(mats)
        out = kron(mats[k], out)   # site-1 modes are the fastest bits
    end
    return out
end

# `Term` is a labelled tensor product; embed in the canonical Fock convention
# (creation-class factors first, then annihilation, then neutral, canonical
# site order within a class) exactly as `_jw_normal_term` does for one-mode
# carriers. `class` is the factor's net particle number (+1, -1, or 0).
function _mm_required(topo, phys, carriers, factors)
    sites = _mm_sites(topo, phys)
    position = Dict(s => i for (i, s) in enumerate(sites))
    ordered = sort(collect(factors); by=f -> (
        f.class > 0 ? 0 : f.class < 0 ? 1 : 2, position[f.site],
    ))
    total = sum(carriers[s].mode_count for s in sites)
    out = Matrix{ComplexF64}(I, 1 << total, 1 << total)
    for f in reverse(ordered)
        out = _mm_global_factor(topo, phys, carriers, f.site, f.op, f.class != 0) * out
    end
    return out
end

function _mm_assert_action(topo, phys, carriers, coefficient, terms;
                           hermitian=true)
    H = OpSum()
    expected = zeros(ComplexF64,
                     1 << sum(carriers[s].mode_count for s in _mm_sites(topo, phys)),
                     1 << sum(carriers[s].mode_count for s in _mm_sites(topo, phys)))
    for (coeff, factors) in terms
        H += Term(coeff, SiteOp[SiteOp(f.site, f.name, f.op) for f in factors])
        expected .+= coeff .* _mm_required(topo, phys, carriers, factors)
    end
    O = ttno_from_opsum(H, topo, phys; hermitian)
    basis, jwidx = _mm_product_basis(topo, phys, carriers)
    actual = _mm_action_matrix(O, basis)
    @test actual ≈ expected[jwidx, jwidx] atol=1e-12 rtol=1e-12
    hermitian && @test norm(actual - actual') < 1e-12
    return O, basis, actual
end

_mm_pair(car2, car1, mode, xsite, ysite) = [
    (1.0, [(; site=xsite, name=Symbol(:Cd, mode), op=car2.Cd[mode], class=1),
           (; site=ysite, name=:C, op=car1.C[1], class=-1)]),
    (1.0, [(; site=xsite, name=Symbol(:C, mode), op=car2.C[mode], class=-1),
           (; site=ysite, name=:Cd, op=car1.Cd[1], class=1)]),
]

@graft_testset "graded multi-mode physical carriers (CG-009)" begin
    car2 = _multimode_fermion_carrier(2)
    car1 = _multimode_fermion_carrier(1)
    hopping = 0.7

    # Cross-site coupling through each local mode of a two-mode site, with the
    # multi-mode site canonically before and after its one-mode partner. The
    # before-partner cases are the CG-009 failure class: the charge leg exits
    # through the site's own physical input (per-sector twist × orientation).
    for (desc, topo, xsite, ysite) in [
        ("head", TreeTopology(:X, [:X => :Y]), :X, :Y),
        ("tail", TreeTopology(:Y, [:Y => :X]), :X, :Y),
    ]
        phys = Dict(:X => car2.P, :Y => car1.P)
        carriers = Dict(:X => car2, :Y => car1)
        for mode in 1:2
            terms = [(t[1] * hopping, t[2]) for t in _mm_pair(car2, car1, mode, xsite, ysite)]
            _mm_assert_action(topo, phys, carriers, hopping, terms)
        end
    end

    # Same-site two-mode block hopping is one neutral premultiplied factor.
    xa = _mm_jw_annihilator(2, 1)
    xb = _mm_jw_annihilator(2, 2)
    block_hop = TensorMap(ComplexF64.(xa' * xb .+ xb' * xa), car2.P ← car2.P)
    _mm_assert_action(
        TreeTopology(:X, [:X => :Y]), Dict(:X => car2.P, :Y => car1.P),
        Dict(:X => car2, :Y => car1), hopping,
        [(hopping, [(; site=:X, name=:hopab, op=block_hop, class=0)])],
    )

    # An idle two-mode site is a spectator for a hopping across it, both on an
    # ancestor chain and as a sibling under a physless junction.
    spectator_terms = [
        (hopping, [(; site=:A, name=:Cd, op=car1.Cd[1], class=1),
                   (; site=:B, name=:C, op=car1.C[1], class=-1)]),
        (hopping, [(; site=:A, name=:C, op=car1.C[1], class=-1),
                   (; site=:B, name=:Cd, op=car1.Cd[1], class=1)]),
    ]
    spectator_phys = Dict(:A => car1.P, :X => car2.P, :B => car1.P)
    spectator_carriers = Dict(:A => car1, :X => car2, :B => car1)
    _mm_assert_action(TreeTopology(:A, [:A => :X, :X => :B]),
                      spectator_phys, spectator_carriers, hopping, spectator_terms)
    _mm_assert_action(TreeTopology(:hub, [:hub => :A, :hub => :X, :hub => :B]),
                      spectator_phys, spectator_carriers, hopping, spectator_terms)

    # Physical one-mode junction root with the two-mode site as a sibling, in
    # both sibling orders (head/tail canonical positions).
    for edges in [[:hub => :a, :hub => :X], [:hub => :X, :hub => :a]]
        phys = Dict(:a => car1.P, :X => car2.P)
        carriers = Dict(:a => car1, :X => car2)
        for mode in 1:2
            terms = [
                (hopping, [(; site=:a, name=:Cd, op=car1.Cd[1], class=1),
                           (; site=:X, name=Symbol(:C, mode), op=car2.C[mode], class=-1)]),
                (hopping, [(; site=:a, name=:C, op=car1.C[1], class=-1),
                           (; site=:X, name=Symbol(:Cd, mode), op=car2.Cd[mode], class=1)]),
            ]
            _mm_assert_action(TreeTopology(:hub, edges), phys, carriers,
                              hopping, terms)
        end
    end

    # Multi-mode charged local factor at a completion junction, coupling to
    # either child; a neutral local mode-number factor with two odd children.
    junction = TreeTopology(:X, [:X => :u, :X => :v])
    junction_phys = Dict(:X => car2.P, :u => car1.P, :v => car1.P)
    junction_carriers = Dict(:X => car2, :u => car1, :v => car1)
    for mode in 1:2, partner in (:u, :v)
        terms = [
            (hopping, [(; site=:X, name=Symbol(:Cd, mode), op=car2.Cd[mode], class=1),
                       (; site=partner, name=:C, op=car1.C[1], class=-1)]),
            (hopping, [(; site=:X, name=Symbol(:C, mode), op=car2.C[mode], class=-1),
                       (; site=partner, name=:Cd, op=car1.Cd[1], class=1)]),
        ]
        _mm_assert_action(junction, junction_phys, junction_carriers,
                          hopping, terms)
    end
    _mm_assert_action(junction, junction_phys, junction_carriers, hopping, [
        (hopping, [(; site=:X, name=:Na, op=car2.N[1], class=0),
                   (; site=:u, name=:Cd, op=car1.Cd[1], class=1),
                   (; site=:v, name=:C, op=car1.C[1], class=-1)]),
        (hopping, [(; site=:X, name=:Na, op=car2.N[1], class=0),
                   (; site=:u, name=:C, op=car1.C[1], class=-1),
                   (; site=:v, name=:Cd, op=car1.Cd[1], class=1)]),
    ])

    # The labelled contract is permutation invariant: reversing the factor
    # vectors is the same term.
    perm_topo = TreeTopology(:X, [:X => :Y])
    perm_phys = Dict(:X => car2.P, :Y => car1.P)
    H_forward = OpSum()
    H_reversed = OpSum()
    for t in _mm_pair(car2, car1, 2, :X, :Y)
        ops = SiteOp[SiteOp(f.site, f.name, f.op) for f in t[2]]
        H_forward += Term(hopping, ops)
        H_reversed += Term(hopping, reverse(ops))
    end
    perm_basis, _ = _mm_product_basis(perm_topo, perm_phys,
                                      Dict(:X => car2, :Y => car1))
    @test _mm_action_matrix(ttno_from_opsum(H_reversed, perm_topo, perm_phys;
                                            hermitian=true), perm_basis) ≈
          _mm_action_matrix(ttno_from_opsum(H_forward, perm_topo, perm_phys;
                                            hermitian=true), perm_basis) atol=1e-12

    # The exit orientation must survive the exact-rank compression pipeline.
    O_compress, compress_basis, action_before = _mm_assert_action(
        perm_topo, perm_phys, Dict(:X => car2, :Y => car1), hopping,
        [(t[1] * hopping, t[2]) for t in _mm_pair(car2, car1, 2, :X, :Y)],
    )
    compress!(O_compress; compression_atol=1e-12)
    @test check_arrows(O_compress)
    @test _mm_action_matrix(O_compress, compress_basis) ≈ action_before atol=1e-12

    # A parity-only graded space cannot orient a degenerate charged exit: the
    # sector label of C and Cd is identical, so the builder fails closed
    # instead of guessing an intra-site string convention (CG-009).
    fz2_two = _fz2_multimode_carrier(2)
    fz2_one = _fz2_multimode_carrier(1)
    fz2_head = TreeTopology(:X, [:X => :Y])
    fz2_phys = Dict(:X => fz2_two.P, :Y => fz2_one.P)
    H_fz2 = OpSum()
    H_fz2 += Term(hopping, SiteOp(:X, :Cd, fz2_two.Cd), SiteOp(:Y, :C, fz2_one.C))
    @test_throws ArgumentError ttno_from_opsum(H_fz2, fz2_head, fz2_phys)
    # The tail canonical position never braids the charge leg out through the
    # degenerate input, so it still assembles.
    fz2_tail = TreeTopology(:Y, [:Y => :X])
    O_fz2 = ttno_from_opsum(H_fz2, fz2_tail, fz2_phys)
    @test O_fz2 isa TTNO
end

@graft_extended_testset "graded multi-mode four-factor junctions (CG-009)" begin
    car2 = _multimode_fermion_carrier(2)
    car1 = _multimode_fermion_carrier(1)
    # local_partial topology: interleaved creation/annihilation classes with a
    # charged factor at a non-completion junction; the two-mode carrier takes
    # each canonical position, exercising odd and even charge-leg exits.
    topo = TreeTopology(:root, [:root => :x, :root => :d, :x => :b, :x => :c])
    sites = [:x, :b, :c, :d]
    kinds = Dict(:x => 1, :b => -1, :c => 1, :d => -1)
    for xsite in sites
        phys = Dict(s => (s == xsite ? car2.P : car1.P) for s in sites)
        carriers = Dict(s => (s == xsite ? car2 : car1) for s in sites)
        adjoint_kinds = Dict(s => -kinds[s] for s in sites)
        terms = Any[]
        for signs in (kinds, adjoint_kinds)
            factors = [(; site=s,
                        name=Symbol(signs[s] > 0 ? :Cd : :C, s == xsite ? 2 : 1),
                        op=(signs[s] > 0 ?
                            (s == xsite ? car2.Cd[2] : car1.Cd[1]) :
                            (s == xsite ? car2.C[2] : car1.C[1])),
                        class=signs[s]) for s in sites]
            push!(terms, (0.7, factors))
        end
        _mm_assert_action(topo, phys, carriers, 0.7, terms)
    end
end

include("braided_sign_regressions.jl")
