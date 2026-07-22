using Test
using Graft
using Graft.TestUtils
using Graft.Backend: FermionParity, Vect, domain
using LinearAlgebra: I, norm
using Random: Xoshiro, randn, randperm

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

# These gates use a fixed Jordan-Wigner action matrix before and after
# exact-rank compression.  CAR words are lowered separately so an accidental
# cancellation cannot hide two wrong signs; seeded random TTNS probes ensure
# the same identities hold away from the product-state action basis.

function _foa_physical_sites(topo, physical)
    return [
        nodeid(topo, node) for node in 1:nnodes(topo)
        if haskey(physical, nodeid(topo, node))
    ]
end

function _foa_product_basis(topo, physical)
    sites = _foa_physical_sites(topo, physical)
    return [
        product_ttns(
            ComplexF64,
            topo,
            physical,
            Dict(
                site => FermionParity((bits >> (position - 1)) & 1)
                for (position, site) in enumerate(sites)
            ),
        ) for bits in 0:((1 << length(sites)) - 1)
    ]
end

"Exact categorical action on the canonical parity product basis."
function _foa_action_matrix(operator, basis)
    dimension = length(basis)
    action = zeros(ComplexF64, dimension, dimension)
    root = topology(operator).root
    plan_cache = EnvCache(topology(operator))
    for (column, ket) in enumerate(basis)
        output = apply(operator, ket; optimize=false)
        output_root = domain(output.tensors[root])[1]
        for (row, bra) in enumerate(basis)
            domain(bra.tensors[root])[1] == output_root || continue
            action[row, column] = inner(
                bra, output; plan_cache, optimize=false,
            )
        end
    end
    return action
end

"Independent canonical Jordan-Wigner annihilator, first site fastest."
function _foa_jw_annihilator(site_count::Int, position::Int)
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

function _foa_pairing_operator(topo, physical, operators, delta,
                               left::Symbol, right::Symbol;
                               reverse_factors::Bool=false)
    annihilation = SiteOp[
        SiteOp(left, :opaque_pair_destroy_left, operators.C),
        SiteOp(right, :opaque_pair_destroy_right, operators.C),
    ]
    creation = SiteOp[
        SiteOp(left, :opaque_pair_create_left, operators.Cd),
        SiteOp(right, :opaque_pair_create_right, operators.Cd),
    ]
    reverse_factors && (reverse!(annihilation); reverse!(creation))

    # Term factors are a labelled tensor product and are class-normalized.
    # The creation word lowers as c_left^dagger c_right^dagger, whereas the
    # adjoint of c_left c_right is c_right^dagger c_left^dagger; hence the
    # explicit fermionic minus sign on the creation coefficient.
    hamiltonian = OpSum()
    hamiltonian += Term(delta, annihilation)
    hamiltonian += Term(-conj(delta), creation)
    return ttno_from_opsum(hamiltonian, topo, physical; hermitian=true)
end

"Lower one ordered two-odd-factor word through class-normal Term storage."
function _foa_ordered_word_operator(topo, physical, operators,
                                    left_site::Symbol, left_kind::Symbol,
                                    right_site::Symbol, right_kind::Symbol;
                                    reverse_storage::Bool=false)
    sites = _foa_physical_sites(topo, physical)
    positions = Dict(site => position for (position, site) in enumerate(sites))
    factors = [(left_site, left_kind), (right_site, right_kind)]
    canonical = sort(copy(factors); by=factor -> (
        factor[2] === :Cd ? 0 : factor[2] === :C ? 1 : 2,
        positions[factor[1]],
    ))
    coefficient = factors == canonical ? 1.0 : -1.0
    storage = reverse_storage ? reverse(factors) : factors
    siteops = SiteOp[
        SiteOp(site, Symbol(:opaque_word_, position), getproperty(operators, kind))
        for (position, (site, kind)) in enumerate(storage)
    ]
    return ttno_from_opsum(
        OpSum() + Term(coefficient, siteops), topo, physical; hermitian=false,
    )
end

function _foa_assert_ordered_word(topo, physical, operators,
                                  left_site::Symbol, left_kind::Symbol,
                                  right_site::Symbol, right_kind::Symbol,
                                  expected;
                                  reverse_storage::Bool=false,
                                  probe_states=())
    operator = _foa_ordered_word_operator(
        topo, physical, operators,
        left_site, left_kind, right_site, right_kind; reverse_storage,
    )
    basis = _foa_product_basis(topo, physical)
    raw = _foa_action_matrix(operator, basis)
    @test raw ≈ expected atol=1e-12 rtol=1e-12
    for state in probe_states
        coordinates = categorical_coordinates(state)
        @test categorical_coordinates(apply(operator, state; optimize=false)) ≈
            expected * coordinates atol=1e-12 rtol=1e-12
    end
    compress!(operator; compression_atol=1e-12)
    @test check_arrows(operator)
    compressed = _foa_action_matrix(operator, basis)
    @test compressed ≈ expected atol=1e-12 rtol=1e-12
    @test compressed ≈ raw atol=1e-12 rtol=1e-12
    for state in probe_states
        coordinates = categorical_coordinates(state)
        @test categorical_coordinates(apply(operator, state; optimize=false)) ≈
            expected * coordinates atol=1e-12 rtol=1e-12
    end
    return raw, compressed
end

function _foa_assert_local_neutral(topo, physical, site, local_operator,
                                   expected; probe_states=())
    hamiltonian = OpSum() + Term(
        1.0, SiteOp(site, :opaque_local_car, local_operator),
    )
    operator = ttno_from_opsum(
        hamiltonian, topo, physical; hermitian=true,
    )
    basis = _foa_product_basis(topo, physical)
    raw = _foa_action_matrix(operator, basis)
    @test raw ≈ expected atol=1e-12 rtol=1e-12
    for state in probe_states
        coordinates = categorical_coordinates(state)
        @test categorical_coordinates(apply(operator, state; optimize=false)) ≈
            expected * coordinates atol=1e-12 rtol=1e-12
    end
    compress!(operator; compression_atol=1e-12)
    @test check_arrows(operator)
    compressed = _foa_action_matrix(operator, basis)
    @test compressed ≈ expected atol=1e-12 rtol=1e-12
    @test compressed ≈ raw atol=1e-12 rtol=1e-12
    return raw, compressed
end

function _foa_assert_pairing_case(label, topo, physical, operators, delta;
                                  reverse_factors::Bool=false)
    @testset "$label" begin
        sites = _foa_physical_sites(topo, physical)
        positions = Dict(site => position for (position, site) in enumerate(sites))
        left, right = sort([:a, :b]; by=site -> positions[site])
        annihilators = [
            _foa_jw_annihilator(length(sites), position)
            for position in eachindex(sites)
        ]
        pair = annihilators[positions[left]] * annihilators[positions[right]]
        expected = delta .* pair .+ conj(delta) .* adjoint(pair)

        operator = _foa_pairing_operator(
            topo, physical, operators, delta, left, right; reverse_factors,
        )
        basis = _foa_product_basis(topo, physical)
        raw = _foa_action_matrix(operator, basis)
        @test raw ≈ expected atol=1e-12 rtol=1e-12
        @test norm(raw - raw') < 1e-12

        compress!(operator; compression_atol=1e-12)
        @test check_arrows(operator)
        compressed = _foa_action_matrix(operator, basis)
        @test compressed ≈ expected atol=1e-12 rtol=1e-12
        @test compressed ≈ raw atol=1e-12 rtol=1e-12
    end
    return nothing
end

function _foa_dense_quadratic_terms(labels, h)
    terms = Vector{Tuple{ComplexF64,Vector{Tuple{Symbol,Symbol}}}}()
    for i in eachindex(labels)
        push!(terms, (ComplexF64(real(h[i, i])), [(labels[i], :N)]))
        for j in (i + 1):length(labels)
            push!(terms, (h[i, j], [(labels[i], :Cd), (labels[j], :C)]))
            push!(terms, (conj(h[i, j]), [(labels[j], :Cd), (labels[i], :C)]))
        end
    end
    return terms
end

function _foa_dense_quadratic_operator(topo, physical, operators, terms,
                                       order, reverse_factors::Bool)
    hamiltonian = OpSum()
    for term_index in order
        coefficient, factors = terms[term_index]
        siteops = SiteOp[
            SiteOp(site, kind, getproperty(operators, kind))
            for (site, kind) in factors
        ]
        reverse_factors && reverse!(siteops)
        hamiltonian += Term(coefficient, siteops)
    end
    return ttno_from_opsum(hamiltonian, topo, physical; hermitian=true)
end

function _foa_dense_jw_matrix(topo, physical, labels, h)
    sites = _foa_physical_sites(topo, physical)
    positions = Dict(site => position for (position, site) in enumerate(sites))
    annihilators = [
        _foa_jw_annihilator(length(sites), position) for position in eachindex(sites)
    ]
    expected = zeros(ComplexF64, 1 << length(sites), 1 << length(sites))
    for i in eachindex(labels), j in eachindex(labels)
        ci = annihilators[positions[labels[i]]]
        cj = annihilators[positions[labels[j]]]
        expected .+= h[i, j] .* (ci' * cj)
    end
    return expected
end

"Canonical indices and fermionic reorder signs for a fixed label-bit basis."
function _foa_label_order_basis_map(topo, physical, labels)
    sites = _foa_physical_sites(topo, physical)
    label_position = Dict(site => position for (position, site) in enumerate(labels))
    canonical_rank = Dict(site => position for (position, site) in enumerate(sites))
    indices = Vector{Int}(undef, 1 << length(labels))
    signs = Vector{Float64}(undef, length(indices))
    for label_bits in 0:((1 << length(labels)) - 1)
        canonical_bits = 0
        for (position, site) in enumerate(sites)
            occupied = (label_bits >> (label_position[site] - 1)) & 1
            canonical_bits |= occupied << (position - 1)
        end
        indices[label_bits + 1] = canonical_bits + 1
        occupied_labels = [
            site for (position, site) in enumerate(labels)
            if ((label_bits >> (position - 1)) & 1) == 1
        ]
        inversions = count(
            canonical_rank[occupied_labels[i]] >
                canonical_rank[occupied_labels[j]]
            for i in 1:(length(occupied_labels) - 1)
            for j in (i + 1):length(occupied_labels)
        )
        signs[label_bits + 1] = isodd(inversions) ? -1.0 : 1.0
    end
    return indices, signs
end

@graft_testset "fermionic canonical anticommutation relations" begin
    operators = fermion_ops_z2()
    topo = TreeTopology(:hub, [
        :hub => :a,
        :hub => :spectator,
        :hub => :b,
    ])
    physical = Dict(
        :a => operators.P,
        :spectator => operators.P,
        :b => operators.P,
    )
    bond = Vect[FermionParity](
        FermionParity(0) => 2,
        FermionParity(1) => 2,
    )
    even_state = random_ttns(
        Xoshiro(20260722), ComplexF64, topo, physical, bond,
    )
    odd_state = normalize!(apply_local(even_state, operators.Cd, :a))
    sites = _foa_physical_sites(topo, physical)
    annihilators = [
        _foa_jw_annihilator(length(sites), position) for position in eachindex(sites)
    ]

    probes = (even_state, odd_state)

    # Same-site CAR uses the premultiplied neutral factors required by Term's
    # distinct-site contract: c_i^dagger c_i = N_i and c_i c_i^dagger = I-N_i.
    identity_matrix = Matrix{ComplexF64}(I, 1 << length(sites), 1 << length(sites))
    for (position, site) in enumerate(sites)
        @testset "same site $site" begin
            c_site = annihilators[position]
            number_raw, number_compressed = _foa_assert_local_neutral(
                topo, physical, site, operators.N, c_site' * c_site;
                probe_states=probes,
            )
            hole_raw, hole_compressed = _foa_assert_local_neutral(
                topo, physical, site, operators.I - operators.N,
                c_site * c_site'; probe_states=probes,
            )
            @test number_raw + hole_raw ≈
                identity_matrix atol=1e-12 rtol=1e-12
            @test number_compressed + hole_compressed ≈
                identity_matrix atol=1e-12 rtol=1e-12
        end
    end

    # For distinct sites, check each ordered word separately before checking
    # the cancellation.  This is the public neutral-TTNO bridge for CAR: the
    # package does not expose a single charged global c_i as a TTNO.
    zero_matrix = zeros(ComplexF64, size(identity_matrix))
    for left_position in 1:(length(sites) - 1)
        for right_position in (left_position + 1):length(sites)
            left_site, right_site = sites[left_position], sites[right_position]
            c_left, c_right = (
                annihilators[left_position], annihilators[right_position],
            )
            @testset "distinct sites $left_site / $right_site" begin
                word_specs = [
                    (left_site, :C, right_site, :Cd, c_left * c_right'),
                    (right_site, :Cd, left_site, :C, c_right' * c_left),
                    (left_site, :Cd, right_site, :C, c_left' * c_right),
                    (right_site, :C, left_site, :Cd, c_right * c_left'),
                    (left_site, :C, right_site, :C, c_left * c_right),
                    (right_site, :C, left_site, :C, c_right * c_left),
                ]
                actions = [
                    _foa_assert_ordered_word(
                        topo, physical, operators,
                        first_site, first_kind, second_site, second_kind, expected;
                        reverse_storage=isodd(word_index), probe_states=probes,
                    ) for (word_index,
                           (first_site, first_kind, second_site, second_kind,
                            expected)) in enumerate(word_specs)
                ]
                @test actions[1][1] + actions[2][1] ≈
                    zero_matrix atol=1e-12 rtol=1e-12
                @test actions[3][1] + actions[4][1] ≈
                    zero_matrix atol=1e-12 rtol=1e-12
                @test actions[5][1] + actions[6][1] ≈
                    zero_matrix atol=1e-12 rtol=1e-12
                @test actions[1][2] + actions[2][2] ≈
                    zero_matrix atol=1e-12 rtol=1e-12
                @test actions[3][2] + actions[4][2] ≈
                    zero_matrix atol=1e-12 rtol=1e-12
                @test actions[5][2] + actions[6][2] ≈
                    zero_matrix atol=1e-12 rtol=1e-12
                for state in probes
                    coordinates = categorical_coordinates(state)
                    @test norm(
                        (actions[1][1] + actions[2][1]) * coordinates,
                    ) < 2e-11
                    @test norm(
                        (actions[3][1] + actions[4][1]) * coordinates,
                    ) < 2e-11
                    @test norm(
                        (actions[5][1] + actions[6][1]) * coordinates,
                    ) < 2e-11
                end
            end
        end
    end
end

@graft_testset "fermionic anomalous pairing action" begin
    operators = fermion_ops_z2()
    delta = 0.37 + 0.29im
    cases = [
        (
            "chain root-head",
            TreeTopology(:a, [:a => :spectator, :spectator => :b]),
            false,
        ),
        (
            "chain root-tail",
            TreeTopology(:b, [:b => :spectator, :spectator => :a]),
            true,
        ),
        (
            "branch spectator",
            TreeTopology(:hub, [
                :hub => :a, :hub => :spectator, :hub => :b,
            ]),
            false,
        ),
        (
            "branch spectator reversed",
            TreeTopology(:hub, [
                :hub => :b, :hub => :spectator, :hub => :a,
            ]),
            true,
        ),
    ]
    for (label, topo, reverse_factors) in cases
        physical = Dict(
            :a => operators.P,
            :spectator => operators.P,
            :b => operators.P,
        )
        _foa_assert_pairing_case(
            label, topo, physical, operators, delta; reverse_factors,
        )
    end
end

@graft_extended_testset "dense complex quadratic fermion action" begin
    operators = fermion_ops_z2()
    labels = [:a, :b, :c, :d, :e]
    rng = Xoshiro(2026072201)
    random_matrix = randn(rng, ComplexF64, length(labels), length(labels))
    h = ComplexF64.((random_matrix + random_matrix') ./ 2)

    # Pin a gauge-invariant complex triangle: h_ab h_bc h_ca = 0.14im.
    h[1, 2] = 0.7
    h[2, 1] = conj(h[1, 2])
    h[2, 3] = 0.5
    h[3, 2] = conj(h[2, 3])
    h[3, 1] = 0.4im
    h[1, 3] = conj(h[3, 1])
    @test h ≈ h'
    @test abs(imag(h[1, 2] * h[2, 3] * h[3, 1])) > 0.1

    terms = _foa_dense_quadratic_terms(labels, h)
    shuffled = randperm(rng, length(terms))
    topologies = [
        (
            "forward children",
            TreeTopology(:root, [
                :root => :x, :root => :y, :root => :e,
                :x => :a, :x => :b, :y => :c, :y => :d,
            ]),
        ),
        (
            "reversed children",
            TreeTopology(:root, [
                :root => :e, :root => :y, :root => :x,
                :y => :d, :y => :c, :x => :b, :x => :a,
            ]),
        ),
    ]
    fixed_reference = nothing
    for (topology_label, topo) in topologies
        physical = Dict(site => operators.P for site in labels)
        basis = _foa_product_basis(topo, physical)
        expected = _foa_dense_jw_matrix(topo, physical, labels, h)
        fixed_indices, fixed_signs = _foa_label_order_basis_map(
            topo, physical, labels,
        )
        topology_reference = nothing
        for (order_label, order, reverse_factors) in [
            ("declared terms", collect(eachindex(terms)), false),
            ("shuffled terms and factors", shuffled, true),
        ]
            @testset "$topology_label / $order_label" begin
                operator = _foa_dense_quadratic_operator(
                    topo, physical, operators, terms, order, reverse_factors,
                )
                raw = _foa_action_matrix(operator, basis)
                @test raw ≈ expected atol=2e-12 rtol=2e-12
                @test norm(raw - raw') < 2e-12

                compress!(operator; compression_atol=1e-12)
                @test check_arrows(operator)
                compressed = _foa_action_matrix(operator, basis)
                @test compressed ≈ expected atol=2e-12 rtol=2e-12
                @test compressed ≈ raw atol=2e-12 rtol=2e-12

                fixed_action = (fixed_signs * fixed_signs') .*
                    raw[fixed_indices, fixed_indices]
                if topology_reference === nothing
                    topology_reference = fixed_action
                else
                    @test fixed_action ≈ topology_reference atol=2e-12 rtol=2e-12
                end
                if fixed_reference === nothing
                    fixed_reference = fixed_action
                else
                    @test fixed_action ≈ fixed_reference atol=2e-12 rtol=2e-12
                end
            end
        end
    end
end
