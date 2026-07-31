using Test
using Graft
using GraftTestUtils
using Graft.Backend: FermionParity, U1Irrep, Vect, ⊠, ⊗, ←, TensorMap, domain,
    isdual, dual, oneunit
using LinearAlgebra: I, norm, kron
using Random: Xoshiro

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

# Charged `apply_local` is a categorical transport, not a sector relabelling:
# one charge wire runs from the physical leg of the insertion node to the
# root's domain leg. Its phase is fixed by the stored leg order, the incoming
# and outgoing wires, and their primal/dual orientation — never by the depth or
# the parity of the path index. These gates pin it against an independent
# Jordan-Wigner Fock oracle in the canonical internal-node order (the same
# convention `fermionic_operator_algebra.jl` pins for TTNO actions) under every
# gauge, root, branch ordering and bond orientation the evolvers can produce.

const _CLO_TOL = 2e-12

_clo_sites(topo, phys) = [nodeid(topo, n) for n in 1:nnodes(topo)
                          if haskey(phys, nodeid(topo, n))]

"Chain, fork, branch-node, physless-root and non-preorder storage geometries."
function _clo_topologies()
    return (
        ("chain4", mps_topology(4), [:site1, :site2, :site3, :site4]),
        ("branch3", TreeTopology(:hub, [:hub => :a, :hub => :sp, :hub => :b]),
         [:hub, :a, :sp, :b]),
        ("deep", TreeTopology(:r, [:r => :m, :m => :x, :m => :y]),
         [:r, :m, :x, :y]),
        ("physless-root", TreeTopology(:r, [:r => :a, :r => :b]), [:a, :b]),
        # Stores a child subtree after a later sibling, so the tree's native
        # physical-wire order and the canonical node order genuinely disagree.
        ("nonpreorder", TreeTopology(:hub, [:hub => :a, :hub => :b, :a => :x]),
         [:hub, :a, :b, :x]),
        ("fork", TreeTopology(:r, [:r => :u1, :u1 => :u2, :r => :d1, :d1 => :d2]),
         [:r, :u1, :u2, :d1, :d2]),
    )
end

_clo_bond() = Vect[FermionParity](FermionParity(0) => 2, FermionParity(1) => 2)

"Independent canonical Jordan-Wigner annihilator, first site fastest."
function _clo_jw_annihilator(site_count::Int, position::Int)
    dimension = 1 << site_count
    matrix = zeros(ComplexF64, dimension, dimension)
    bit = 1 << (position - 1)
    for state in 0:(dimension - 1)
        state & bit == 0 && continue
        matrix[(state & ~bit) + 1, state + 1] =
            isodd(count_ones(state & (bit - 1))) ? -1.0 : 1.0
    end
    return matrix
end

_clo_jw_creation(count, position) =
    Matrix(adjoint(_clo_jw_annihilator(count, position)))

"All product-sector basis states of a one-mode fℤ₂ carrier, in bit order."
function _clo_product_basis(topo, phys)
    sites = _clo_sites(topo, phys)
    return [product_ttns(ComplexF64, topo, phys,
                         Dict(site => FermionParity((bits >> (position - 1)) & 1)
                              for (position, site) in enumerate(sites)))
            for bits in 0:((1 << length(sites)) - 1)]
end

"Coordinates in a prebuilt product-sector basis (a cached `categorical_coordinates`)."
function _clo_coordinates(state, basis, plan_cache)
    out = zeros(ComplexF64, length(basis))
    root = topology(state).root
    rootspace = domain(state.tensors[root])[1]
    for (row, bra) in enumerate(basis)
        domain(bra.tensors[root])[1] == rootspace || continue
        out[row] = inner(bra, state; plan_cache, optimize=false)
    end
    return out
end

"""
Compare every single-site creation/annihilation insertion on `state` against
the Fock oracle. `state` may carry any gauge, center, and bond orientation.
"""
function _clo_assert_insertions(topo, phys, operators, state, basis, plan_cache)
    sites = _clo_sites(topo, phys)
    count = length(sites)
    coordinates = _clo_coordinates(state, basis, plan_cache)
    for (position, site) in enumerate(sites)
        for (op, reference) in ((operators.C, _clo_jw_annihilator(count, position)),
                                (operators.Cd, _clo_jw_creation(count, position)))
            inserted = apply_local(state, op, site)
            @test check_arrows(inserted)
            got = _clo_coordinates(inserted, basis, plan_cache)
            expected = reference * coordinates
            @test norm(got - expected) < _CLO_TOL
            @test abs(norm(got) - norm(expected)) < _CLO_TOL
        end
    end
    return nothing
end

"Bond orientation signature of every edge, child-node ordered."
_clo_orientations(state) = [isdual(virtualspace(state, n))
                            for n in 1:nnodes(topology(state))
                            if topology(state).parent[n] != 0]

# Gates C3/C4/C5. The product-sector basis spans every charge sector, so the
# odd-parity representations a fixed-parity random state cannot reach are
# covered too. Each center move re-orients the bonds it crosses, which is the
# only sanctioned way to change the gauge (§9.1); the same unrooted geometry
# stored with a different root or a different child-slot order is a separate
# entry in `_clo_topologies`.
@graft_testset "charged apply_local gauge/root/branch-order covariance" begin
    operators = fermion_ops_z2()
    for (label, topo, sites) in _clo_topologies()
        @testset "$label" begin
            phys = Dict(site => operators.P for site in sites)
            basis = _clo_product_basis(topo, phys)
            plan_cache = EnvCache(topo)
            centers = GRAFT_EXTENDED_TESTS ? (1:nnodes(topo)) :
                unique((topo.root, argmax(topo.depth)))
            seen = Set{Vector{Bool}}()
            for center in centers, reference in basis
                state = move_center!(copy(reference), center)
                push!(seen, _clo_orientations(state))
                _clo_assert_insertions(topo, phys, operators, state, basis, plan_cache)
            end
            # The sweep must really have exercised more than one orientation
            # pattern wherever the tree has an edge to re-orient.
            @test length(seen) == length(centers)
        end
    end
end

# Gate C2: a bond-dimension-2 (sector-degenerate) state whose virtual bonds
# were left dual by a downward gauge move — the shape the experimental
# snapshot path produced, kept as an explicit non-regression.
@graft_testset "charged apply_local dual-oriented degenerate representation" begin
    operators = fermion_ops_z2()
    for (label, topo, sites) in _clo_topologies()
        @testset "$label" begin
            phys = Dict(site => operators.P for site in sites)
            basis = _clo_product_basis(topo, phys)
            plan_cache = EnvCache(topo)
            state = move_center!(
                random_ttns(Xoshiro(20260727), ComplexF64, topo, phys, _clo_bond()),
                argmax(topo.depth))
            @test any(_clo_orientations(state))
            _clo_assert_insertions(topo, phys, operators, state, basis, plan_cache)
        end
    end
end

"""
Uncanonical random TTNS with a prescribed primal/dual orientation per edge and
a declared center. `move_center!` only ever hands a downward path dual bonds,
so a leaf center above a primal bond is unreachable through gauge moves alone —
but a two-site split (TDVP2, CBE, a snapshot reload) produces exactly that.
"""
function _clo_oriented_ttns(rng, topo, phys, bond, duals, center::Int)
    S = typeof(bond)
    edge(n) = get(duals, n, false) ? dual(bond) : bond
    tensors = map(1:nnodes(topo)) do n
        cod = S[edge(c) for c in topo.children[n]]
        haskey(phys, nodeid(topo, n)) && push!(cod, phys[nodeid(topo, n)])
        dom = topo.parent[n] == 0 ? oneunit(S) : edge(n)
        randn(rng, ComplexF64, reduce(⊗, cod) ← dom)
    end
    return TTNS(topo, tensors, center)
end

# Gate C1: every primal/dual orientation pattern of the traversed edges, with
# the center pinned at the insertion node so `apply_local` cannot re-gauge the
# path first. This is the representation the failing uniform-TDVP2 checkpoint
# carried, and the one the earlier snapshot-derived regression never reached.
@graft_testset "charged apply_local primal/dual edge-pattern covariance" begin
    operators = fermion_ops_z2()
    bond = _clo_bond()
    for (label, topo, sites) in _clo_topologies()
        @testset "$label" begin
            phys = Dict(site => operators.P for site in sites)
            ordered = _clo_sites(topo, phys)
            count = length(ordered)
            basis = _clo_product_basis(topo, phys)
            plan_cache = EnvCache(topo)
            edges = [n for n in 1:nnodes(topo) if topo.parent[n] != 0]
            patterns = GRAFT_EXTENDED_TESTS ? (0:(1 << length(edges)) - 1) :
                (0, (1 << length(edges)) - 1, 1)
            for mask in unique(patterns)
                duals = Dict(e => ((mask >> (k - 1)) & 1) == 1
                             for (k, e) in enumerate(edges))
                for (position, site) in enumerate(ordered)
                    state = _clo_oriented_ttns(Xoshiro(20260727), topo, phys, bond,
                                               duals, nodeindex(topo, site))
                    @test check_arrows(state)
                    coordinates = _clo_coordinates(state, basis, plan_cache)
                    for (op, reference) in
                            ((operators.C, _clo_jw_annihilator(count, position)),
                             (operators.Cd, _clo_jw_creation(count, position)))
                        got = _clo_coordinates(apply_local(state, op, site),
                                               basis, plan_cache)
                        expected = reference * coordinates
                        @test norm(got - expected) <
                            _CLO_TOL * max(1.0, norm(coordinates))
                    end
                end
            end
        end
    end
end

# Gate C6: the canonical anticommutation relations must hold in every one of
# those representations, including through the odd intermediate state (charged
# root sector, shifted bonds) that the first insertion produces.
@graft_testset "charged apply_local creation/annihilation algebra" begin
    operators = fermion_ops_z2()
    for (label, topo, sites) in _clo_topologies()
        @testset "$label" begin
            phys = Dict(site => operators.P for site in sites)
            basis = _clo_product_basis(topo, phys)
            plan_cache = EnvCache(topo)
            base = random_ttns(Xoshiro(20260727), ComplexF64, topo, phys, _clo_bond())
            for center in unique((topo.root, argmax(topo.depth)))
                state = move_center!(copy(base), center)
                coordinates = _clo_coordinates(state, basis, plan_cache)
                for site in sites
                    lower = apply_local(state, operators.C, site)
                    raise = apply_local(state, operators.Cd, site)
                    @test norm(_clo_coordinates(
                        apply_local(lower, operators.C, site), basis, plan_cache)) < _CLO_TOL
                    @test norm(_clo_coordinates(
                        apply_local(raise, operators.Cd, site), basis, plan_cache)) < _CLO_TOL
                    anticommutator =
                        _clo_coordinates(apply_local(lower, operators.Cd, site),
                                         basis, plan_cache) .+
                        _clo_coordinates(apply_local(raise, operators.C, site),
                                         basis, plan_cache)
                    @test norm(anticommutator - coordinates) < _CLO_TOL
                end
                for site in sites, other in sites
                    site == other && continue
                    forward = _clo_coordinates(apply_local(
                        apply_local(state, operators.C, other), operators.Cd, site),
                        basis, plan_cache)
                    reversed = _clo_coordinates(apply_local(
                        apply_local(state, operators.Cd, site), operators.C, other),
                        basis, plan_cache)
                    @test norm(forward + reversed) < _CLO_TOL
                end
            end
        end
    end
end

# Two charged insertions must reproduce the state-diagram TTNO for the same
# labelled word. That path is validated independently in
# `fermionic_operator_algebra.jl`, so agreement pins the insertion convention
# instead of restating it.
@graft_testset "charged apply_local words agree with the TTNO lowerer" begin
    operators = fermion_ops_z2()
    for (label, topo, sites) in _clo_topologies()
        @testset "$label" begin
            phys = Dict(site => operators.P for site in sites)
            basis = _clo_product_basis(topo, phys)
            plan_cache = EnvCache(topo)
            state = random_ttns(Xoshiro(20260727), ComplexF64, topo, phys, _clo_bond())
            ordered = _clo_sites(topo, phys)      # canonical internal-node order
            count = length(ordered)
            coordinates = _clo_coordinates(state, basis, plan_cache)
            positions = Dict(site => index for (index, site) in enumerate(ordered))
            for site in sites, other in sites
                site == other && continue
                operator = ttno_from_opsum(
                    OpSum() + Term(1.0, SiteOp(site, :Cd, operators.Cd),
                                   SiteOp(other, :C, operators.C)),
                    topo, phys; hermitian=false)
                word = _clo_coordinates(apply_local(
                    apply_local(state, operators.C, other), operators.Cd, site),
                    basis, plan_cache)
                @test norm(word - _clo_coordinates(
                    apply(operator, state; optimize=false), basis, plan_cache)) < _CLO_TOL
                @test norm(word -
                    _clo_jw_creation(count, positions[site]) *
                    _clo_jw_annihilator(count, positions[other]) *
                    coordinates) < _CLO_TOL
            end
        end
    end
end

# ---------------------------------------------------------------------------
# multi-mode carriers (CG-009 contract)
# ---------------------------------------------------------------------------

const _CLO_MMQ = typeof(FermionParity(0) ⊠ U1Irrep(0))
_clo_mm_sector(n::Int) = FermionParity(n % 2) ⊠ U1Irrep(n)

"GraftImpurity `ParticleNumberSector` carrier: (n, state) order, intra-site JW."
function _clo_multimode_carrier(mode_count::Int)
    states = sort!(collect(0:((1 << mode_count) - 1)); by=s -> (count_ones(s), s))
    d = length(states)
    pos = Dict(s => i for (i, s) in enumerate(states))
    P = Vect[_CLO_MMQ]((_clo_mm_sector(n) => binomial(mode_count, n)
                        for n in 0:mode_count)...)
    annihilate = Vect[_CLO_MMQ]((FermionParity(1) ⊠ U1Irrep(-1)) => 1)
    create = Vect[_CLO_MMQ]((FermionParity(1) ⊠ U1Irrep(1)) => 1)
    C = Vector{Any}(undef, mode_count)
    Cd = Vector{Any}(undef, mode_count)
    for j in 1:mode_count
        mask = 1 << (j - 1)
        lower = zeros(ComplexF64, d, d)
        raise = zeros(ComplexF64, d, d)
        for s in states
            sign_ = isodd(count_ones(s & (mask - 1))) ? -1.0 : 1.0
            if (s & mask) != 0
                lower[pos[s & ~mask], pos[s]] = sign_
            else
                raise[pos[s | mask], pos[s]] = sign_
            end
        end
        C[j] = TensorMap(reshape(lower, d, d, 1), P ← P ⊗ annihilate)
        Cd[j] = TensorMap(reshape(raise, d, d, 1), P ← P ⊗ create)
    end
    return (; P, C=Tuple(C), Cd=Tuple(Cd), states, pos, mode_count)
end

_clo_mm_local_basis(car) =
    [(_clo_mm_sector(count_ones(s)) =>
      count(x -> count_ones(x) == count_ones(s) && x <= s, car.states), s)
     for s in car.states]

"Degeneracy-resolved product basis plus each state's global JW basis index."
function _clo_mm_product_basis(topo, phys, carriers)
    sites = _clo_sites(topo, phys)
    locals_ = [_clo_mm_local_basis(carriers[site]) for site in sites]
    offsets = cumsum([0; [carriers[site].mode_count for site in sites]])
    basis = Any[]
    indices = Int[]
    for combo in Iterators.product((eachindex(l) for l in locals_)...)
        labels = Dict{Symbol,Any}()
        bits = 0
        for (k, site) in enumerate(sites)
            label, state = locals_[k][combo[k]]
            labels[site] = label
            bits |= state << offsets[k]
        end
        push!(basis, product_ttns(ComplexF64, topo, phys, labels))
        push!(indices, bits + 1)
    end
    return basis, indices
end

"Global Fock matrix of one charged factor: JW string over canonically earlier sites."
function _clo_mm_global_factor(topo, phys, carriers, site::Symbol, op)
    sites = _clo_sites(topo, phys)
    position = Dict(s => i for (i, s) in enumerate(sites))
    mats = Matrix{ComplexF64}[]
    for s in sites
        car = carriers[s]
        d = 1 << car.mode_count
        if s == site
            permutation = [car.pos[b] for b in 0:(d - 1)]
            dense = convert(Array, op)
            matrix = ndims(dense) == 3 ? dense[:, :, 1] : dense
            push!(mats, ComplexF64.(matrix[permutation, permutation]))
        elseif position[s] < position[site]
            push!(mats, ComplexF64[(i == j ? (-1.0)^count_ones(i - 1) : 0.0)
                                   for i in 1:d, j in 1:d])
        else
            push!(mats, Matrix{ComplexF64}(I, d, d))
        end
    end
    out = mats[1]
    for k in 2:length(mats)
        out = kron(mats[k], out)         # site-1 modes are the fastest bits
    end
    return out
end

function _clo_mm_coordinates(state, basis, indices, dimension, plan_cache)
    out = zeros(ComplexF64, dimension)
    root = topology(state).root
    rootspace = domain(state.tensors[root])[1]
    for (k, bra) in enumerate(basis)
        domain(bra.tensors[root])[1] == rootspace || continue
        out[indices[k]] = inner(bra, state; plan_cache, optimize=false)
    end
    return out
end

@graft_testset "charged apply_local multi-mode carrier covariance" begin
    cases = (
        ("chain3 two-mode", mps_topology(3), [:site1, :site2, :site3],
         Dict(:site1 => 2, :site2 => 2, :site3 => 2)),
        ("branch mixed-mode", TreeTopology(:r, [:r => :m, :m => :x, :m => :y]),
         [:r, :m, :x, :y], Dict(:r => 1, :m => 2, :x => 1, :y => 2)),
    )
    for (label, topo, sites, modes) in cases
        @testset "$label" begin
            carriers = Dict(site => _clo_multimode_carrier(modes[site]) for site in sites)
            phys = Dict(site => carriers[site].P for site in sites)
            basis, indices = _clo_mm_product_basis(topo, phys, carriers)
            plan_cache = EnvCache(topo)
            dimension = 1 << sum(modes[site] for site in sites)
            centers = GRAFT_EXTENDED_TESTS ? (1:nnodes(topo)) :
                unique((topo.root, argmax(topo.depth)))
            for center in centers, reference in basis
                state = move_center!(copy(reference), center)
                coordinates = _clo_mm_coordinates(state, basis, indices, dimension,
                                                  plan_cache)
                for site in sites, mode in 1:carriers[site].mode_count
                    for op in (carriers[site].C[mode], carriers[site].Cd[mode])
                        inserted = apply_local(state, op, site)
                        @test check_arrows(inserted)
                        @test norm(_clo_mm_coordinates(inserted, basis, indices,
                                                       dimension, plan_cache) -
                            _clo_mm_global_factor(topo, phys, carriers, site, op) *
                            coordinates) < _CLO_TOL
                    end
                    for (op, expected) in (
                            (adjoint(carriers[site].C[mode]),
                             carriers[site].Cd[mode]),
                            (adjoint(carriers[site].Cd[mode]),
                             carriers[site].C[mode]))
                        inserted = apply_local(state, op, site)
                        @test check_arrows(inserted)
                        @test norm(_clo_mm_coordinates(
                            inserted, basis, indices, dimension, plan_cache) -
                            _clo_mm_global_factor(
                                topo, phys, carriers, site, expected) *
                            coordinates) < _CLO_TOL
                    end
                end
            end
        end
    end
end

# Gates C1/C7: the representation an evolver actually hands to a correlator.
# A uniform TDVP2 sweep re-splits every bond, so the insertion runs on
# evolver-produced virtual spaces rather than on a freshly canonicalized state.
# Each propagated charged channel is then checked against its own dense
# trajectory, never against the other channel.
@graft_testset "charged apply_local on an evolver-produced gauge" begin
    operators = fermion_ops_z2()
    topo = TreeTopology(:imp, [:imp => :bath1, :imp => :bath2])
    sites = [:imp, :bath1, :bath2]
    phys = Dict(site => operators.P for site in sites)

    hamiltonian = OpSum()
    hamiltonian += Term(-0.35, SiteOp(:imp, :N, operators.N))
    for bath in (:bath1, :bath2)
        hamiltonian += Term(-0.8, SiteOp(:imp, :Cd, operators.Cd),
                            SiteOp(bath, :C, operators.C))
        hamiltonian += Term(-0.8, SiteOp(bath, :Cd, operators.Cd),
                            SiteOp(:imp, :C, operators.C))
    end
    H = ttno_from_opsum(hamiltonian, topo, phys; hermitian=true)

    state = random_ttns(Xoshiro(20260727), ComplexF64, topo, phys, _clo_bond())
    evolver = TDVP2(order=2, trunc=TruncationScheme(maxdim=16, atol=1e-12),
                    verbose=false)
    for _ in 1:4
        step!(evolver, state, H, -0.05)
    end
    normalize!(state)

    basis = _clo_product_basis(topo, phys)
    plan_cache = EnvCache(topo)
    _clo_assert_insertions(topo, phys, operators, state, basis, plan_cache)

    dense = to_dense(H)
    dz = -1e-3im
    for site in sites, op in (operators.C, operators.Cd)
        inserted = apply_local(state, op, site)
        exact = exact_evolve(dense, _clo_coordinates(inserted, basis, plan_cache), dz)
        propagated = copy(inserted)
        step!(TDVP2(order=2, trunc=TruncationScheme(maxdim=16, atol=1e-12),
                    verbose=false), propagated, H, dz)
        @test check_arrows(propagated)
        @test norm(_clo_coordinates(propagated, basis, plan_cache) - exact) < 1e-7
    end
end
