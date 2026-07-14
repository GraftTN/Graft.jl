using Test
using Graft
using Graft.TestUtils
using Graft.Backend: FermionParity, U1Irrep, SU2Irrep, Vect, ⊠, ⊗, ←,
    blocks, domain
using LinearAlgebra: I, norm

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

"Independent canonical Jordan-Wigner annihilator, first physical site fastest."
function _jw_annihilator(mode_count::Int, mode::Int)
    dimension = 1 << mode_count
    matrix = zeros(ComplexF64, dimension, dimension)
    bit = 1 << (mode - 1)
    for state in 0:(dimension - 1)
        state & bit == 0 && continue
        matrix[(state & ~bit) + 1, state + 1] =
            isodd(count_ones(state & (bit - 1))) ? -1.0 : 1.0
    end
    return matrix
end

function _graded_physical_sites(topo, physical)
    return [nodeid(topo, node) for node in 1:nnodes(topo)
            if haskey(physical, nodeid(topo, node))]
end

function _graded_product_basis(topo, physical, vacuum, occupied)
    sites = _graded_physical_sites(topo, physical)
    return [
        product_ttns(
            ComplexF64,
            topo,
            physical,
            Dict(site => (((bits >> (index - 1)) & 1) == 0 ? vacuum : occupied)
                 for (index, site) in enumerate(sites)),
        ) for bits in 0:((1 << length(sites)) - 1)
    ]
end

"Exact graded action matrix from product-basis TTNO application and overlaps."
function _graded_action_matrix(O, basis)
    dimension = length(basis)
    values = zeros(ComplexF64, dimension, dimension)
    root = topology(O).root
    for (column, input) in enumerate(basis)
        output = apply(O, input)
        output_root = domain(output.tensors[root])[1]
        for (row, bra) in enumerate(basis)
            domain(bra.tensors[root])[1] == output_root || continue
            values[row, column] = inner(bra, output)
        end
    end
    return values
end

function _jw_normal_term(topo, physical, factors)
    sites = _graded_physical_sites(topo, physical)
    index = Dict(site => position for (position, site) in enumerate(sites))
    annihilators = [_jw_annihilator(length(sites), mode) for mode in eachindex(sites)]
    # `Term` is a labelled tensor product, not a sequential product. Embed its
    # local morphisms in the canonical Fock convention: creation legs first,
    # annihilation legs second, then neutral local factors, with canonical
    # physical-site order within each family. This makes the labelled h.c. assignment
    # `(:a => C, :b => Cd)` the Fock adjoint `c†_b c_a`, without consulting
    # the input vector order.
    ordered = sort!(collect(factors); by=factor -> (
        factor[2] === :Cd ? 0 : factor[2] === :C ? 1 : 2,
        index[factor[1]],
    ))
    operator = Matrix{ComplexF64}(I, 1 << length(sites), 1 << length(sites))
    for (site, kind) in reverse(ordered)
        local_operator = kind === :Cd ? adjoint(annihilators[index[site]]) :
            kind === :C ? annihilators[index[site]] :
            kind === :N ? adjoint(annihilators[index[site]]) * annihilators[index[site]] :
            kind === :I ? Matrix{ComplexF64}(I, size(operator, 1), size(operator, 2)) :
            throw(ArgumentError("expected :C, :Cd, :N, or :I factors"))
        operator = local_operator * operator
    end
    return operator
end

function _graded_term(topology, physical, operators, coefficient, factors)
    siteops = SiteOp[
        SiteOp(site, kind, getproperty(operators, kind))
        for (site, kind) in factors
    ]
    return ttno_from_opsum(OpSum() + Term(coefficient, siteops), topology, physical)
end

function _graded_two_term_hamiltonian(topology, physical, operators,
                                      coefficient, factors, adjoint_factors)
    term = Term(coefficient, SiteOp[
        SiteOp(site, kind, getproperty(operators, kind))
        for (site, kind) in factors
    ])
    adjoint_term = Term(conj(coefficient), SiteOp[
        SiteOp(site, kind, getproperty(operators, kind))
        for (site, kind) in adjoint_factors
    ])
    return ttno_from_opsum(OpSum() + term + adjoint_term, topology, physical;
                           hermitian=true)
end

function _assert_graded_action_matches_jw(topology, physical, operators,
                                           coefficient, factors, adjoint_factors;
                                           vacuum=FermionParity(0),
                                           occupied=FermionParity(1))
    O = _graded_two_term_hamiltonian(
        topology, physical, operators, coefficient, factors, adjoint_factors,
    )
    basis = _graded_product_basis(topology, physical, vacuum, occupied)
    expected = coefficient * _jw_normal_term(topology, physical, factors) +
        conj(coefficient) * _jw_normal_term(topology, physical, adjoint_factors)
    actual = _graded_action_matrix(O, basis)
    @test actual ≈ expected atol=1e-12 rtol=1e-12
    @test norm(actual - actual') < 1e-12
    return O, basis, expected
end

"Small fZ2 x U1 local fermion carrier used to exercise abelian product braids."
function _fermion_ops_z2_u1()
    Q = FermionParity ⊠ U1Irrep
    vacuum = FermionParity(0) ⊠ U1Irrep(0)
    occupied = FermionParity(1) ⊠ U1Irrep(1)
    create_charge = FermionParity(1) ⊠ U1Irrep(1)
    annihilate_charge = FermionParity(1) ⊠ U1Irrep(-1)
    P = Vect[Q](vacuum => 1, occupied => 1)
    Cp = Vect[Q](create_charge => 1)
    Cm = Vect[Q](annihilate_charge => 1)
    C = zeros(ComplexF64, P ← P ⊗ Cm)
    Cd = zeros(ComplexF64, P ← P ⊗ Cp)
    N = zeros(ComplexF64, P ← P)
    Iop = zeros(ComplexF64, P ← P)
    for (sector, block_) in blocks(C)
        sector == vacuum && (block_[1, 1] = 1)
    end
    for (sector, block_) in blocks(Cd)
        sector == occupied && (block_[1, 1] = 1)
    end
    for (sector, block_) in blocks(N)
        block_[1, 1] = sector == occupied ? 1 : 0
    end
    for (_, block_) in blocks(Iop)
        block_[1, 1] = 1
    end
    return (; C, Cd, N, I=Iop, P, vacuum, occupied)
end

@graft_testset "graded labelled TTNO embedding" begin
    operators = fermion_ops_z2()
    hopping = 0.7 + 0.2im

    # Ancestor/descendant chain control: canonical physical-site order is
    # :a, :b, :c regardless of the tree's child-to-parent tensor direction.
    chain = TreeTopology(:a, [:a => :b, :b => :c])
    chain_physical = Dict(:a => operators.P, :b => operators.P, :c => operators.P)
    _assert_graded_action_matches_jw(
        chain, chain_physical, operators, hopping,
        [(:a, :Cd), (:c, :C)], [(:a, :C), (:c, :Cd)],
    )

    # A physless sibling junction must not depend on the input vector order.
    hub = TreeTopology(:hub, [:hub => :a, :hub => :b])
    hub_physical = Dict(:a => operators.P, :b => operators.P)
    factors = [(:a, :Cd), (:b, :C)]
    adjoint_factors = [(:a, :C), (:b, :Cd)]
    O_hub, hub_basis, _ = _assert_graded_action_matches_jw(
        hub, hub_physical, operators, hopping, factors, adjoint_factors,
    )
    O_hub_permuted = _graded_two_term_hamiltonian(
        hub, hub_physical, operators, hopping,
        reverse(factors), reverse(adjoint_factors),
    )
    @test _graded_action_matrix(O_hub_permuted, hub_basis) ≈
          _graded_action_matrix(O_hub, hub_basis) atol=1e-12 rtol=1e-12

    # Two terms can share x's labelled local restriction while their charged
    # partners lie on different ancestor siblings.  The state diagram must
    # retain their distinct abelian framing rather than rejecting or merging
    # the local x channel.
    framed = TreeTopology(:root, [
        :root => :y,
        :root => :x,
        :root => :z,
    ])
    framed_physical = Dict(:x => operators.P, :y => operators.P, :z => operators.P)
    framed_terms = [
        (0.31 + 0.11im, [(:x, :Cd), (:y, :C)]),
        (-0.23 + 0.17im, [(:x, :Cd), (:z, :C)]),
    ]
    H_framed = OpSum()
    expected_framed = zeros(ComplexF64, 8, 8)
    for (coefficient, term_factors) in framed_terms
        adjoint_term = [(site, kind === :Cd ? :C : :Cd) for (site, kind) in term_factors]
        H_framed += Term(coefficient, SiteOp[
            SiteOp(site, kind, kind === :Cd ? operators.Cd : operators.C)
            for (site, kind) in term_factors
        ])
        H_framed += Term(conj(coefficient), SiteOp[
            SiteOp(site, kind, kind === :Cd ? operators.Cd : operators.C)
            for (site, kind) in adjoint_term
        ])
        expected_framed .+= coefficient .* _jw_normal_term(framed, framed_physical, term_factors)
        expected_framed .+= conj(coefficient) .* _jw_normal_term(
            framed, framed_physical, adjoint_term,
        )
    end
    O_framed = ttno_from_opsum(H_framed, framed, framed_physical; hermitian=true)
    framed_basis = _graded_product_basis(
        framed, framed_physical, FermionParity(0), FermionParity(1),
    )
    framed_action = _graded_action_matrix(O_framed, framed_basis)
    @test framed_action ≈ expected_framed atol=1e-12 rtol=1e-12
    @test norm(framed_action - framed_action') < 1e-12

    # A physical sibling junction has a spectator local fermion. The oracle
    # catches a missing braid as a spectator-parity dependent sign.
    physical_siblings = TreeTopology(:a, [:a => :b, :a => :c])
    physical_siblings_physical = Dict(
        :a => operators.P, :b => operators.P, :c => operators.P,
    )
    _assert_graded_action_matches_jw(
        physical_siblings, physical_siblings_physical, operators, hopping,
        [(:b, :Cd), (:c, :C)], [(:b, :C), (:c, :Cd)],
    )

    # The same physical junction must use TensorKit's fZ2 x U1 product
    # braiding, not a parity-only scalar specialization.
    operators_z2u1 = _fermion_ops_z2_u1()
    _assert_graded_action_matches_jw(
        physical_siblings,
        Dict(:a => operators_z2u1.P, :b => operators_z2u1.P, :c => operators_z2u1.P),
        operators_z2u1,
        hopping,
        [(:b, :Cd), (:c, :C)], [(:b, :C), (:c, :Cd)];
        vacuum=operators_z2u1.vacuum,
        occupied=operators_z2u1.occupied,
    )

    # The four odd factors merge at x/y before the whole term completes at
    # root. Physical indices deliberately interleave x/y subtrees:
    # canonical order is :a, :c, :b, :d, not planar subtree order.
    nested = TreeTopology(:root, [
        :root => :x,
        :root => :y,
        :x => :a,
        :y => :c,
        :x => :b,
        :y => :d,
    ])
    nested_physical = Dict(
        :a => operators.P, :b => operators.P,
        :c => operators.P, :d => operators.P,
    )
    _, nested_basis, expected_nested = _assert_graded_action_matches_jw(
        nested, nested_physical, operators, hopping,
        [(:a, :Cd), (:b, :C), (:c, :Cd), (:d, :C)],
        [(:a, :C), (:b, :Cd), (:c, :C), (:d, :Cd)],
    )
    # Deliberately interleaved construction order is the same labelled tensor
    # product and must give the same action.
    nested_interleaved = _graded_two_term_hamiltonian(
        nested, nested_physical, operators, hopping,
        [(:d, :C), (:a, :Cd), (:c, :Cd), (:b, :C)],
        [(:c, :C), (:d, :Cd), (:b, :Cd), (:a, :C)],
    )
    @test _graded_action_matrix(nested_interleaved, nested_basis) ≈
          expected_nested atol=1e-12 rtol=1e-12

    # Local physical charged factor at the completion junction.
    local_completion = TreeTopology(:a, [
        :a => :x,
        :a => :y,
        :x => :b,
        :x => :c,
        :y => :d,
    ])
    local_completion_physical = Dict(
        :a => operators.P, :b => operators.P,
        :c => operators.P, :d => operators.P,
    )
    _assert_graded_action_matches_jw(
        local_completion, local_completion_physical, operators, hopping,
        [(:a, :Cd), (:b, :C), (:c, :Cd), (:d, :C)],
        [(:a, :C), (:b, :Cd), (:c, :C), (:d, :Cd)],
    )

    # Local physical charged factor at a non-completion junction.
    local_partial = TreeTopology(:root, [
        :root => :x,
        :root => :d,
        :x => :b,
        :x => :c,
    ])
    local_partial_physical = Dict(
        :x => operators.P, :b => operators.P,
        :c => operators.P, :d => operators.P,
    )
    _assert_graded_action_matches_jw(
        local_partial, local_partial_physical, operators, hopping,
        [(:x, :Cd), (:b, :C), (:c, :Cd), (:d, :C)],
        [(:x, :C), (:b, :Cd), (:c, :C), (:d, :Cd)],
    )

    # Q-M5-009: an odd wire crossing a physically present but operator-idle
    # sibling is a framed unit-sector channel, not plain identity transport.
    idle_hub = TreeTopology(:hub, [
        :hub => :a,
        :hub => :spectator,
        :hub => :c,
    ])
    idle_hub_physical = Dict(
        :a => operators.P, :spectator => operators.P, :c => operators.P,
    )
    _assert_graded_action_matches_jw(
        idle_hub, idle_hub_physical, operators, hopping,
        [(:a, :Cd), (:c, :C)], [(:a, :C), (:c, :Cd)],
    )

    if GRAFT_EXTENDED_TESTS
        # Reversing the planar child order changes the native fusion route but
        # not the site-labelled contract in that topology's canonical order.
        reversed_idle_hub = TreeTopology(:hub, [
            :hub => :c,
            :hub => :spectator,
            :hub => :a,
        ])
        reversed_idle_hub_physical = Dict(
            :a => operators.P, :spectator => operators.P, :c => operators.P,
        )
        _assert_graded_action_matches_jw(
            reversed_idle_hub, reversed_idle_hub_physical, operators, hopping,
            [(:a, :Cd), (:c, :C)], [(:a, :C), (:c, :Cd)],
        )
    end

    # The mapped-Cayley failure was fZ2 x U1 on an ancestor chain with an idle
    # physical site between the two charged endpoints. Keep that exact carrier
    # class as a permanent core gate.
    chain_z2u1_physical = Dict(
        :a => operators_z2u1.P,
        :b => operators_z2u1.P,
        :c => operators_z2u1.P,
    )
    O_idle_z2u1, idle_z2u1_basis, expected_idle_z2u1 =
        _assert_graded_action_matches_jw(
            chain, chain_z2u1_physical, operators_z2u1, hopping,
            [(:a, :Cd), (:c, :C)], [(:a, :C), (:c, :Cd)];
            vacuum=operators_z2u1.vacuum,
            occupied=operators_z2u1.occupied,
        )

    # Omitted identity, an explicitly labelled identity, and a neutral number
    # operator all receive the same physical-input frame. Explicit `:I` stays a
    # real factor, so a scaled identity cannot collide with padding transport.
    scaled_identity = 2 * operators_z2u1.I
    H_explicit_identity = OpSum()
    H_explicit_identity += Term(hopping, SiteOp[
        SiteOp(:a, :Cd, operators_z2u1.Cd),
        SiteOp(:b, :I, scaled_identity),
        SiteOp(:c, :C, operators_z2u1.C),
    ])
    H_explicit_identity += Term(conj(hopping), SiteOp[
        SiteOp(:a, :C, operators_z2u1.C),
        SiteOp(:b, :I, scaled_identity),
        SiteOp(:c, :Cd, operators_z2u1.Cd),
    ])
    O_explicit_identity = ttno_from_opsum(
        H_explicit_identity, chain, chain_z2u1_physical; hermitian=true,
    )
    explicit_identity_action = _graded_action_matrix(O_explicit_identity, idle_z2u1_basis)
    @test explicit_identity_action ≈ 2 .* expected_idle_z2u1 atol=1e-12 rtol=1e-12
    @test norm(explicit_identity_action - explicit_identity_action') < 1e-12

    _assert_graded_action_matches_jw(
        chain, chain_z2u1_physical, operators_z2u1, hopping,
        [(:a, :Cd), (:b, :N), (:c, :C)],
        [(:a, :C), (:b, :N), (:c, :Cd)];
        vacuum=operators_z2u1.vacuum,
        occupied=operators_z2u1.occupied,
    )

    # A framed idle physical leaf nested under one child subtree must propagate
    # through branch nodes without becoming an active charged restriction.
    nested_idle = TreeTopology(:root, [
        :root => :x,
        :root => :y,
        :x => :a,
        :x => :spectator,
        :y => :b,
    ])
    nested_idle_physical = Dict(
        :a => operators_z2u1.P,
        :spectator => operators_z2u1.P,
        :b => operators_z2u1.P,
    )
    _assert_graded_action_matches_jw(
        nested_idle, nested_idle_physical, operators_z2u1, hopping,
        [(:a, :Cd), (:b, :C)], [(:a, :C), (:b, :Cd)];
        vacuum=operators_z2u1.vacuum,
        occupied=operators_z2u1.occupied,
    )

    # The corrected idle frame must survive the exact-rank core compression
    # pipeline; the action oracle, not a raw graded TTNO contraction, is final.
    idle_action_before = _graded_action_matrix(O_idle_z2u1, idle_z2u1_basis)
    compress!(O_idle_z2u1; compression_atol=1e-12)
    @test check_arrows(O_idle_z2u1)
    @test _graded_action_matrix(O_idle_z2u1, idle_z2u1_basis) ≈
          idle_action_before atol=1e-12 rtol=1e-12

    # The typed frame is an abelian scalar specialization, not an implicit
    # SU(2) approximation. The current first fail-closed boundary is SiteOp's
    # one-dimensional charge carrier; after that boundary is generalized, two
    # spinor legs will additionally require fusion-route metadata and native
    # F/R-move lowering instead of the scalar charge payload.
    Psu2 = Vect[SU2Irrep](SU2Irrep(0) => 1, SU2Irrep(1 // 2) => 1)
    Qsu2 = Vect[SU2Irrep](SU2Irrep(1 // 2) => 1)
    charged_su2 = zeros(ComplexF64, Psu2 ← Psu2 ⊗ Qsu2)
    @test_throws ArgumentError SiteOp(:site1, :spinor, charged_su2)
end
