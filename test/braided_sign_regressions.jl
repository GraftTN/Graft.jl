# Permanent SD0 gates for the state-diagram braid certificate.  This file is
# included after graded_multimode_carrier.jl so the categorical action and
# multi-mode carrier helpers are already available.

"Independent canonical physical order: physical nodes by internal node index."
function _sd0_canonical_sites(topo, phys)
    return [
        nodeid(topo, node) for node in 1:nnodes(topo)
        if haskey(phys, nodeid(topo, node))
    ]
end

"Parity-only product basis, independent of graded_ttno_embedding.jl helpers."
function _sd0_fz2_product_basis(topo, phys)
    sites = _sd0_canonical_sites(topo, phys)
    return [
        product_ttns(
            ComplexF64,
            topo,
            phys,
            Dict(
                site => FermionParity((bits >> (index - 1)) & 1)
                for (index, site) in enumerate(sites)
            ),
        ) for bits in 0:((1 << length(sites)) - 1)
    ]
end

"Independent canonical Jordan-Wigner annihilator for one-mode sites."
function _sd0_jw_annihilator(mode_count::Int, mode::Int)
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

"Creation/annihilation class from charge orientation, never from SiteOp.name."
function _sd0_factor_class(factor)
    q = charge(SiteOp(factor.site, factor.name, factor.op))
    class = Graft.TTNOBuild._net_u1_charge(q)
    class === nothing && throw(ArgumentError(
        "SD0 class-normal oracle requires a U(1)-oriented factor",
    ))
    return class
end

function _sd0_global_factor(canonical_sites, carriers, factor)
    position = Dict(site => index for (index, site) in enumerate(canonical_sites))
    class = _sd0_factor_class(factor)
    matrices = Matrix{ComplexF64}[]
    for site in canonical_sites
        carrier = carriers[site]
        dimension = 1 << carrier.mode_count
        if site == factor.site
            permutation = [carrier.pos[bits] for bits in 0:(dimension - 1)]
            dense = convert(Array, factor.op)
            local_matrix = ndims(dense) == 3 ? dense[:, :, 1] : dense
            push!(matrices, ComplexF64.(local_matrix[permutation, permutation]))
        elseif position[site] < position[factor.site] && isodd(class)
            push!(matrices, ComplexF64[
                row == column ? (-1.0)^count_ones(row - 1) : 0.0
                for row in 1:dimension, column in 1:dimension
            ])
        else
            push!(matrices, Matrix{ComplexF64}(I, dimension, dimension))
        end
    end
    result = matrices[1]
    for index in 2:length(matrices)
        result = kron(matrices[index], result)
    end
    return result
end

"Independent class-normal JW action using the two pinned SD0 metadata sources."
function _sd0_required(topo, phys, carriers, factors)
    canonical_sites = _sd0_canonical_sites(topo, phys)
    position = Dict(site => index for (index, site) in enumerate(canonical_sites))
    ordered = sort(collect(factors); by=factor -> begin
        class = _sd0_factor_class(factor)
        (class > 0 ? 0 : class < 0 ? 1 : 2, position[factor.site])
    end)
    total_modes = sum(carriers[site].mode_count for site in canonical_sites)
    result = Matrix{ComplexF64}(I, 1 << total_modes, 1 << total_modes)
    for factor in reverse(ordered)
        result = _sd0_global_factor(canonical_sites, carriers, factor) * result
    end
    return result
end

"Build the internal certificate directly for event-ownership invariants."
function _sd0_certificate_plan(topo, phys, factors)
    ops = Dict(
        nodeindex(topo, factor.site) => SiteOp(factor.site, factor.name, factor.op)
        for factor in factors
    )
    opnodes = sort!(collect(keys(ops)))
    unit_sector = one(charge(first(values(ops))))
    return Graft.TTNOBuild._build_braided_term_plan(
        topo, Graft.TTNOBuild._Euler(topo), phys, ops, opnodes,
        unit_sector, 1.0,
    )
end

"Full categorical action for small gates; sparse columns plus zero controls otherwise."
function _sd0_action_probe(operator, basis, expected)
    nonzero_columns = [
        column for column in axes(expected, 2)
        if any(abs(value) > 1e-14 for value in view(expected, :, column))
    ]
    isempty(nonzero_columns) && throw(ArgumentError(
        "SD0 probe oracle has no nonzero columns",
    ))
    zero_columns = [
        column for column in axes(expected, 2)
        if all(abs(value) <= 1e-14 for value in view(expected, :, column))
    ]
    zero_controls = isempty(zero_columns) ? Int[] :
        unique([first(zero_columns), last(zero_columns)])
    columns = length(basis) <= 16 ? collect(axes(expected, 2)) :
        sort!(unique([nonzero_columns; zero_controls]))
    actual = zeros(ComplexF64, length(basis), length(columns))
    root = topology(operator).root
    plan_cache = EnvCache(topology(operator))
    for (probe_column, basis_column) in enumerate(columns)
        output = apply(operator, basis[basis_column]; optimize=false)
        output_root = domain(output.tensors[root])[1]
        for (row, bra) in enumerate(basis)
            domain(bra.tensors[root])[1] == output_root || continue
            actual[row, probe_column] = inner(
                bra, output; plan_cache, optimize=false,
            )
        end
    end
    return actual, expected[:, columns]
end

function _sd0_assert_raw_and_compressed(label, topo, phys, carriers, factors;
                                         coefficient=0.37 + 0.11im)
    @testset "$label" begin
        siteops = SiteOp[
            SiteOp(factor.site, factor.name, factor.op) for factor in factors
        ]
        hamiltonian = OpSum() + Term(coefficient, siteops)
        operator = ttno_from_opsum(
            hamiltonian, topo, phys; hermitian=false,
        )
        basis, jw_indices = _mm_product_basis(topo, phys, carriers)
        expected = coefficient .* _sd0_required(topo, phys, carriers, factors)
        expected = expected[jw_indices, jw_indices]

        raw_action, probed_expected = _sd0_action_probe(operator, basis, expected)
        @test raw_action ≈ probed_expected atol=1e-12 rtol=1e-12

        compress!(operator; compression_atol=1e-12)
        @test check_arrows(operator)
        compressed_action, compressed_expected = _sd0_action_probe(
            operator, basis, expected,
        )
        @test compressed_action ≈ compressed_expected atol=1e-12 rtol=1e-12
        @test compressed_action ≈ raw_action atol=1e-12 rtol=1e-12
    end
    return nothing
end

"Check raw and compressed action for several Term.ops storage orders."
function _sd0_assert_permutations(label, topo, phys, carriers, factors,
                                  permutations;
                                  coefficient=0.37 + 0.11im)
    @testset "$label permutations" begin
        basis, jw_indices = _mm_product_basis(topo, phys, carriers)
        expected = coefficient .* _sd0_required(topo, phys, carriers, factors)
        expected = expected[jw_indices, jw_indices]
        raw_reference = nothing
        compressed_reference = nothing
        for permutation in permutations
            siteops = SiteOp[
                SiteOp(factor.site, factor.name, factor.op)
                for factor in factors[permutation]
            ]
            operator = ttno_from_opsum(
                OpSum() + Term(coefficient, siteops), topo, phys; hermitian=false,
            )
            raw_action, raw_expected = _sd0_action_probe(operator, basis, expected)
            @test raw_action ≈ raw_expected atol=1e-12 rtol=1e-12

            compress!(operator; compression_atol=1e-12)
            @test check_arrows(operator)
            compressed_action, compressed_expected = _sd0_action_probe(
                operator, basis, expected,
            )
            @test compressed_action ≈ compressed_expected atol=1e-12 rtol=1e-12
            @test compressed_action ≈ raw_action atol=1e-12 rtol=1e-12

            if raw_reference === nothing
                raw_reference = raw_action
                compressed_reference = compressed_action
            else
                @test raw_action ≈ raw_reference atol=1e-12 rtol=1e-12
                @test compressed_action ≈ compressed_reference atol=1e-12 rtol=1e-12
            end
        end
    end
    return nothing
end

function _sd0_chain(site_count::Int, reverse_root::Bool)
    sites = [Symbol(:s, index) for index in 1:site_count]
    if reverse_root
        edges = Pair{Symbol,Symbol}[
            sites[index] => sites[index - 1] for index in site_count:-1:2
        ]
        return TreeTopology(sites[end], edges), sites
    end
    edges = Pair{Symbol,Symbol}[
        sites[index] => sites[index + 1] for index in 1:(site_count - 1)
    ]
    return TreeTopology(sites[1], edges), sites
end

function _sd0_permutations(values::Vector{Int})
    length(values) <= 1 && return [copy(values)]
    permutations = Vector{Vector{Int}}()
    for index in eachindex(values)
        head = values[index]
        tail = [values[j] for j in eachindex(values) if j != index]
        for suffix in _sd0_permutations(tail)
            push!(permutations, [head; suffix])
        end
    end
    return permutations
end

function _sd0_oriented_factor(carrier, site, class::Int, ordinal::Int)
    class == 1 && return (
        ; site, name=Symbol(:opaque_, ordinal), op=carrier.Cd[1],
    )
    class == -1 && return (
        ; site, name=Symbol(:opaque_, ordinal), op=carrier.C[1],
    )
    class == 0 && return (
        ; site, name=Symbol(:opaque_, ordinal), op=carrier.N[1],
    )
    throw(ArgumentError("unsupported SD0 test class $class"))
end

@graft_testset "SD0 Gap A ancestor-chain braid regressions" begin
    carrier = _multimode_fermion_carrier(1)

    # Each class pattern is exercised in both root orientations.  The
    # deliberately non-canonical factor vectors pin Term permutation
    # invariance without allowing SiteOp labels to encode a class.
    chain_cases = [
        (4, false, [1, -1, 1, -1], [1, 2, 3, 4], "H0 chain4 interleaved root-head"),
        (4, true,  [1, -1, 1, -1], [3, 1, 4, 2], "H0 chain4 interleaved root-tail"),
        (4, false, [1, 1, -1, -1], [4, 2, 1, 3], "H0b chain4 class-aligned root-head"),
        (4, true,  [1, 1, -1, -1], [2, 4, 3, 1], "H0b chain4 class-aligned root-tail"),
        (6, false, [1, -1, 1, -1, 1, -1], [1, 2, 3, 4, 5, 6],
         "chain6 interleaved root-head"),
        (6, true,  [1, -1, 1, -1, 1, -1], [5, 2, 6, 1, 4, 3],
         "chain6 interleaved root-tail"),
        (6, false, [1, 1, 1, -1, -1, -1], [6, 2, 4, 1, 5, 3],
         "chain6 class-aligned root-head"),
        (6, true,  [1, 1, 1, -1, -1, -1], [3, 5, 1, 6, 2, 4],
         "chain6 class-aligned root-tail"),
    ]
    for (site_count, reverse_root, classes, permutation, label) in chain_cases
        topo, declared_sites = _sd0_chain(site_count, reverse_root)
        phys = Dict(site => carrier.P for site in declared_sites)
        carriers = Dict(site => carrier for site in declared_sites)
        canonical_sites = _sd0_canonical_sites(topo, phys)
        factors = [
            _sd0_oriented_factor(carrier, site, class, ordinal)
            for (ordinal, (site, class)) in enumerate(zip(canonical_sites, classes))
        ]
        if site_count == 4
            reference_plan = _sd0_certificate_plan(topo, phys, factors)
            permutation_plans = [
                _sd0_certificate_plan(topo, phys, factors[p])
                for p in _sd0_permutations(collect(eachindex(factors)))
            ]
            @test all(plan -> plan.canonical_word == reference_plan.canonical_word &&
                              plan.native_word == reference_plan.native_word &&
                              plan.certificate_scale == reference_plan.certificate_scale &&
                              [(event.lhs, event.rhs, event.owner) for event in plan.crossings] ==
                              [(event.lhs, event.rhs, event.owner)
                               for event in reference_plan.crossings],
                      permutation_plans)
        end
        if site_count == 4 && !reverse_root && classes == [1, -1, 1, -1]
            plan = _sd0_certificate_plan(topo, phys, factors)
            @test nodeid.(Ref(topo), plan.canonical_word) == [:s1, :s3, :s2, :s4]
            @test nodeid.(Ref(topo), plan.native_word) == [:s1, :s3, :s4, :s2]
            @test length(plan.crossings) == 1
            @test nodeid(topo, only(plan.crossings).owner) == :s2
            @test plan.certificate_scale == -1
            @test plan.legacy_scale == 1
            @test prod(local_plan.word_scale for local_plan in plan.local_plans) ==
                plan.certificate_scale
        end
        _sd0_assert_raw_and_compressed(
            label, topo, phys, carriers, factors[permutation],
        )
        _sd0_assert_permutations(
            label, topo, phys, carriers, factors,
            [collect(eachindex(factors)), reverse(collect(eachindex(factors)))],
        )
    end

    # Neutral N is a real factor between odd factors, not padding transport.
    for (reverse_root, permutation, label) in [
        (false, [1, 2, 3, 4, 5], "chain5 neutral N root-head"),
        (true, [5, 2, 1, 4, 3], "chain5 neutral N root-tail permuted"),
    ]
        topo, declared_sites = _sd0_chain(5, reverse_root)
        phys = Dict(site => carrier.P for site in declared_sites)
        carriers = Dict(site => carrier for site in declared_sites)
        canonical_sites = _sd0_canonical_sites(topo, phys)
        factors = [
            _sd0_oriented_factor(carrier, site, class, ordinal)
            for (ordinal, (site, class)) in enumerate(
                zip(canonical_sites, [1, -1, 0, 1, -1]),
            )
        ]
        _sd0_assert_raw_and_compressed(
            label, topo, phys, carriers, factors[permutation],
        )
        _sd0_assert_permutations(
            label, topo, phys, carriers, factors,
            [collect(eachindex(factors)), reverse(collect(eachindex(factors)))],
        )
    end


    # Gap A composed with the CG-009 sector-degenerate charge-leg exit.  The
    # two-mode carrier occupies each canonical position and uses its second
    # mode, so the certificate scalar and per-sector exit morphism are both live.
    two_mode = _multimode_fermion_carrier(2)
    topo, declared_sites = _sd0_chain(4, false)
    canonical_sites = _sd0_canonical_sites(
        topo, Dict(site => carrier.P for site in declared_sites),
    )
    for multimode_position in eachindex(canonical_sites)
        multimode_site = canonical_sites[multimode_position]
        phys = Dict(
            site => (site == multimode_site ? two_mode.P : carrier.P)
            for site in declared_sites
        )
        carriers = Dict(
            site => (site == multimode_site ? two_mode : carrier)
            for site in declared_sites
        )
        classes = [1, -1, 1, -1]
        factors = [begin
            local_carrier = carriers[site]
            mode = site == multimode_site ? 2 : 1
            op = classes[index] > 0 ? local_carrier.Cd[mode] : local_carrier.C[mode]
            (; site, name=Symbol(:opaque_h_, index), op)
        end for (index, site) in enumerate(canonical_sites)]
        _sd0_assert_raw_and_compressed(
            "H multimode position $multimode_position",
            topo, phys, carriers, factors[[4, 2, 1, 3]],
        )
    end

    # Two terms share the complete s2:s4 ACTIVE restriction and therefore the
    # same non-completion entry at s2.  Its certificate scale must be attached
    # exactly once to the merged entry, while the two root completions remain
    # distinct through their local factors.
    one_mode = carrier
    topo, sites = _sd0_chain(4, false)
    carriers = Dict(
        :s1 => two_mode,
        :s2 => one_mode,
        :s3 => one_mode,
        :s4 => one_mode,
    )
    phys = Dict(site => carriers[site].P for site in sites)
    shared = [
        (; site=:s2, name=:shared_s2_C, op=one_mode.C[1]),
        (; site=:s3, name=:shared_s3_Cd, op=one_mode.Cd[1]),
        (; site=:s4, name=:shared_s4_C, op=one_mode.C[1]),
    ]
    factors_a = [
        (; site=:s1, name=:root_Cd1, op=two_mode.Cd[1]),
        shared...,
    ]
    factors_b = [
        (; site=:s1, name=:root_Cd2, op=two_mode.Cd[2]),
        shared...,
    ]
    owner = nodeindex(topo, :s2)
    for factors in (factors_a, factors_b)
        plan = _sd0_certificate_plan(topo, phys, factors)
        @test nodeid.(Ref(topo), plan.canonical_word) == [:s1, :s3, :s2, :s4]
        @test nodeid.(Ref(topo), plan.native_word) == [:s1, :s3, :s4, :s2]
        @test only(plan.crossings).owner == owner
        @test plan.local_plans[owner].word_scale == -1
    end

    coefficient_a = 0.31 + 0.07im
    coefficient_b = -0.19 + 0.13im
    hamiltonian = OpSum() +
        Term(coefficient_a, SiteOp[
            SiteOp(f.site, f.name, f.op) for f in factors_a
        ]) +
        Term(coefficient_b, SiteOp[
            SiteOp(f.site, f.name, f.op) for f in factors_b
        ])
    operator = ttno_from_opsum(hamiltonian, topo, phys; hermitian=false)
    basis, jw_indices = _mm_product_basis(topo, phys, carriers)
    expected = coefficient_a .* _sd0_required(
        topo, phys, carriers, factors_a,
    ) .+ coefficient_b .* _sd0_required(
        topo, phys, carriers, factors_b,
    )
    expected = expected[jw_indices, jw_indices]
    raw_action, raw_expected = _sd0_action_probe(operator, basis, expected)
    @test raw_action ≈ raw_expected atol=1e-12 rtol=1e-12
    compress!(operator; compression_atol=1e-12)
    @test check_arrows(operator)
    compressed_action, compressed_expected = _sd0_action_probe(
        operator, basis, expected,
    )
    @test compressed_action ≈ compressed_expected atol=1e-12 rtol=1e-12
    @test compressed_action ≈ raw_action atol=1e-12 rtol=1e-12
end

@graft_testset "SD0 Gap B insertion orders and green junction controls" begin
    one_mode = _multimode_fermion_carrier(1)
    two_mode = _multimode_fermion_carrier(2)

    # All four nested insertion variants: root (x,y)/(y,x) crossed with x's
    # (a,SPEC)/(SPEC,a), for both one- and two-mode idle spectators.  F1 is
    # the forward/forward positive control; the other routes include F2.
    for root_reversed in (false, true), x_reversed in (false, true),
        spectator_modes in (1, 2)
        root_edges = root_reversed ? [:root => :y, :root => :x] :
            [:root => :x, :root => :y]
        x_edges = x_reversed ? [:x => :SPEC, :x => :a] :
            [:x => :a, :x => :SPEC]
        topo = TreeTopology(:root, [root_edges; x_edges; [:y => :b]])
        spectator = spectator_modes == 1 ? one_mode : two_mode
        phys = Dict(:a => one_mode.P, :SPEC => spectator.P, :b => one_mode.P)
        carriers = Dict(:a => one_mode, :SPEC => spectator, :b => one_mode)
        factors = [
            (; site=:a, name=:opaque_left, op=one_mode.Cd[1]),
            (; site=:b, name=:opaque_right, op=one_mode.C[1]),
        ]
        isodd(Int(root_reversed) + Int(x_reversed) + spectator_modes) &&
            reverse!(factors)
        if root_reversed && x_reversed && spectator_modes == 1
            plan = _sd0_certificate_plan(topo, phys, factors)
            @test nodeid.(Ref(topo), plan.canonical_word) == [:a, :b]
            @test plan.native_word == plan.canonical_word
            @test isempty(plan.crossings)
            @test plan.certificate_scale == 1
            @test plan.legacy_scale == -1
        end
        label = "nested idle root=$(root_reversed ? "yx" : "xy") " *
            "x=$(x_reversed ? "SPEC-a" : "a-SPEC") spectator=$(spectator_modes)m"
        _sd0_assert_raw_and_compressed(label, topo, phys, carriers, factors)
        _sd0_assert_permutations(
            label, topo, phys, carriers, factors,
            [[1, 2], [2, 1]],
        )
    end

    # H0c: nested junction stays exact while the spine correction changes.
    nested = TreeTopology(:root, [
        :root => :x, :root => :y, :x => :a,
        :y => :c, :x => :b, :y => :d,
    ])
    nested_sites = [:a, :b, :c, :d]
    nested_phys = Dict(site => one_mode.P for site in nested_sites)
    nested_carriers = Dict(site => one_mode for site in nested_sites)
    nested_classes = Dict(:a => 1, :b => -1, :c => 1, :d => -1)
    nested_factors = [
        _sd0_oriented_factor(one_mode, site, nested_classes[site], ordinal)
        for (ordinal, site) in enumerate(nested_sites)
    ]
    _sd0_assert_raw_and_compressed(
        "H0c nested junction", nested, nested_phys, nested_carriers,
        nested_factors[[4, 1, 3, 2]],
    )

    # CG-009 J controls become permanent: the two-mode carrier occupies each
    # canonical position of the validated local_partial junction.
    local_partial = TreeTopology(:root, [
        :root => :x, :root => :d, :x => :b, :x => :c,
    ])
    local_sites = [:x, :b, :c, :d]
    local_classes = Dict(:x => 1, :b => -1, :c => 1, :d => -1)
    for multimode_site in local_sites
        phys = Dict(
            site => (site == multimode_site ? two_mode.P : one_mode.P)
            for site in local_sites
        )
        carriers = Dict(
            site => (site == multimode_site ? two_mode : one_mode)
            for site in local_sites
        )
        factors = [begin
            carrier = carriers[site]
            mode = site == multimode_site ? 2 : 1
            class = local_classes[site]
            op = class > 0 ? carrier.Cd[mode] : carrier.C[mode]
            (; site, name=Symbol(:opaque_j_, ordinal), op)
        end for (ordinal, site) in enumerate(local_sites)]
        _sd0_assert_raw_and_compressed(
            "J local_partial two-mode@$multimode_site",
            local_partial, phys, carriers, factors[[3, 1, 4, 2]],
        )
    end

    # Flat reversed hub remains a green framed-idle control.
    reversed_hub = TreeTopology(:hub, [
        :hub => :b, :hub => :SPEC, :hub => :a,
    ])
    hub_phys = Dict(:a => one_mode.P, :SPEC => two_mode.P, :b => one_mode.P)
    hub_carriers = Dict(:a => one_mode, :SPEC => two_mode, :b => one_mode)
    hub_factors = [
        (; site=:b, name=:opaque_hub_right, op=one_mode.C[1]),
        (; site=:a, name=:opaque_hub_left, op=one_mode.Cd[1]),
    ]
    _sd0_assert_raw_and_compressed(
        "flat reversed idle hub", reversed_hub, hub_phys, hub_carriers, hub_factors,
    )
end

@graft_testset "SD0 parity-only orientation boundary" begin
    operators = fermion_ops_z2()

    # Two odd factors remain supported.  Opaque labels make this a direct gate
    # against recovering creation/annihilation class from SiteOp.name.
    pair_topo = TreeTopology(:a, [:a => :b])
    pair_phys = Dict(:a => operators.P, :b => operators.P)
    pair_hamiltonian = OpSum() + Term(
        0.41,
        SiteOp(:b, :opaque_right, operators.C),
        SiteOp(:a, :opaque_left, operators.Cd),
    )
    pair_operator = ttno_from_opsum(
        pair_hamiltonian, pair_topo, pair_phys; hermitian=false,
    )
    pair_basis = _sd0_fz2_product_basis(pair_topo, pair_phys)
    annihilators = [_sd0_jw_annihilator(2, mode) for mode in 1:2]
    pair_expected = 0.41 .* (annihilators[1]' * annihilators[2])
    pair_raw = _mm_action_matrix(pair_operator, pair_basis)
    @test pair_raw ≈ pair_expected atol=1e-12 rtol=1e-12
    compress!(pair_operator; compression_atol=1e-12)
    @test check_arrows(pair_operator)
    @test _mm_action_matrix(pair_operator, pair_basis) ≈
          pair_expected atol=1e-12 rtol=1e-12

    # Promote the exact flat reversed-hub control that was previously
    # extended-only.  The spectator is idle and all labels remain opaque.
    hub_topo = TreeTopology(:hub, [
        :hub => :c, :hub => :spectator, :hub => :a,
    ])
    hub_phys = Dict(
        :a => operators.P, :spectator => operators.P, :c => operators.P,
    )
    hub_sites = _sd0_canonical_sites(hub_topo, hub_phys)
    hub_position = Dict(site => index for (index, site) in enumerate(hub_sites))
    hub_annihilators = [
        _sd0_jw_annihilator(length(hub_sites), mode) for mode in eachindex(hub_sites)
    ]
    coefficient = 0.29 + 0.13im
    hub_hamiltonian = OpSum()
    hub_hamiltonian += Term(
        coefficient,
        SiteOp(:a, :opaque_hub_a_create, operators.Cd),
        SiteOp(:c, :opaque_hub_c_destroy, operators.C),
    )
    hub_hamiltonian += Term(
        conj(coefficient),
        SiteOp(:a, :opaque_hub_a_destroy, operators.C),
        SiteOp(:c, :opaque_hub_c_create, operators.Cd),
    )
    hub_operator = ttno_from_opsum(
        hub_hamiltonian, hub_topo, hub_phys; hermitian=true,
    )
    hub_basis = _sd0_fz2_product_basis(hub_topo, hub_phys)
    c_a = hub_annihilators[hub_position[:a]]
    c_c = hub_annihilators[hub_position[:c]]
    hub_expected = coefficient .* (c_a' * c_c) .+
        conj(coefficient) .* (c_c' * c_a)
    hub_raw = _mm_action_matrix(hub_operator, hub_basis)
    @test hub_raw ≈ hub_expected atol=1e-12 rtol=1e-12
    compress!(hub_operator; compression_atol=1e-12)
    @test check_arrows(hub_operator)
    @test _mm_action_matrix(hub_operator, hub_basis) ≈
          hub_expected atol=1e-12 rtol=1e-12

    # Parity-only one-mode factors derive their creation/annihilation class
    # structurally from the physical-input twist eigensector.  This promotes
    # both four-odd Gap A patterns without consulting SiteOp.name.
    chain_topo, chain_sites = _sd0_chain(4, false)
    chain_phys = Dict(site => operators.P for site in chain_sites)
    chain_basis = _sd0_fz2_product_basis(chain_topo, chain_phys)
    chain_annihilators = [_sd0_jw_annihilator(4, mode) for mode in 1:4]
    coefficient = 0.37 + 0.11im
    chain_patterns = [
        (
            "H0 interleaved",
            [operators.Cd, operators.C, operators.Cd, operators.C],
            [1, -1, 1, -1],
            chain_annihilators[1]' * chain_annihilators[3]' *
                chain_annihilators[2] * chain_annihilators[4],
        ),
        (
            "H0b class-aligned",
            [operators.Cd, operators.Cd, operators.C, operators.C],
            [1, 1, -1, -1],
            chain_annihilators[1]' * chain_annihilators[2]' *
                chain_annihilators[3] * chain_annihilators[4],
        ),
    ]
    permutations = [[1, 2, 3, 4], [4, 2, 1, 3], [3, 1, 4, 2]]
    for (label, local_operators, classes, expected_word) in chain_patterns
        factors = [
            (
                ; site,
                name=Symbol("opaque_parity_", ordinal),
                op=local_operators[ordinal],
            ) for (ordinal, site) in enumerate(chain_sites)
        ]
        plan = _sd0_certificate_plan(chain_topo, chain_phys, factors)
        @test plan.uses_certificate
        @test [plan.factor_class[nodeindex(chain_topo, site)]
               for site in chain_sites] == classes
        @test plan.certificate_scale == -1
        @test plan.legacy_scale == 1

        expected = coefficient .* expected_word
        for permutation in permutations
            siteops = SiteOp[
                SiteOp(f.site, f.name, f.op) for f in factors[permutation]
            ]
            operator = ttno_from_opsum(
                OpSum() + Term(coefficient, siteops),
                chain_topo,
                chain_phys;
                hermitian=false,
            )
            raw_action = _mm_action_matrix(operator, chain_basis)
            @test raw_action ≈ expected atol=1e-12 rtol=1e-12
            compress!(operator; compression_atol=1e-12)
            @test check_arrows(operator)
            compressed_action = _mm_action_matrix(operator, chain_basis)
            @test compressed_action ≈ expected atol=1e-12 rtol=1e-12
            @test compressed_action ≈ raw_action atol=1e-12 rtol=1e-12
        end
    end

    # A charged factor at the support LCA closes two occupied child branches.
    # That pivotal completion bend is distinct from both the class-normal word
    # crossings and the CG-009 charge-leg exit orientation.
    completion_topo = TreeTopology(:a, [
        :a => :x, :a => :y, :x => :b, :x => :c, :y => :d,
    ])
    completion_sites = [:a, :b, :c, :d]
    completion_phys = Dict(site => operators.P for site in completion_sites)
    completion_factors = [
        (; site=:a, name=:opaque_completion_a, op=operators.Cd),
        (; site=:b, name=:opaque_completion_b, op=operators.C),
        (; site=:c, name=:opaque_completion_c, op=operators.Cd),
        (; site=:d, name=:opaque_completion_d, op=operators.C),
    ]
    completion_plan = _sd0_certificate_plan(
        completion_topo, completion_phys, completion_factors,
    )
    completion_lca = nodeindex(completion_topo, :a)
    @test completion_plan.uses_certificate
    @test completion_plan.local_plans[completion_lca].completion_bend.scalar == -1
    @test completion_plan.certificate_scale == -1

    completion_basis = _sd0_fz2_product_basis(
        completion_topo, completion_phys,
    )
    completion_annihilators = [
        _sd0_jw_annihilator(4, mode) for mode in 1:4
    ]
    completion_expected = coefficient .* (
        completion_annihilators[1]' * completion_annihilators[3]' *
        completion_annihilators[2] * completion_annihilators[4]
    )
    completion_term = Term(
        coefficient,
        SiteOp[
            SiteOp(factor.site, factor.name, factor.op)
            for factor in completion_factors
        ],
    )
    completion_operator = ttno_from_opsum(
        OpSum() + completion_term,
        completion_topo,
        completion_phys;
        hermitian=false,
    )
    completion_raw = _mm_action_matrix(completion_operator, completion_basis)
    @test completion_raw ≈ completion_expected atol=1e-12 rtol=1e-12
    compress!(completion_operator; compression_atol=1e-12)
    @test check_arrows(completion_operator)
    completion_compressed = _mm_action_matrix(
        completion_operator, completion_basis,
    )
    @test completion_compressed ≈ completion_expected atol=1e-12 rtol=1e-12
    @test completion_compressed ≈ completion_raw atol=1e-12 rtol=1e-12

    # A sector-degenerate parity-only charged factor has mixed nonzero input
    # twist sectors, so neither class source exists and a class-sensitive term
    # still fails closed.
    degenerate = _fz2_multimode_carrier(2)
    degenerate_phys = Dict(site => degenerate.P for site in chain_sites)
    degenerate_ops = [
        SiteOp(:s1, :opaque_mixed_1, degenerate.Cd),
        SiteOp(:s2, :opaque_mixed_2, degenerate.C),
        SiteOp(:s3, :opaque_mixed_3, degenerate.Cd),
        SiteOp(:s4, :opaque_mixed_4, degenerate.C),
    ]
    @test Graft.TTNOBuild._input_twist_parity(
        degenerate.Cd, charge(first(degenerate_ops)),
    ) === nothing
    @test_throws ArgumentError ttno_from_opsum(
        OpSum() + Term(1.0, degenerate_ops),
        chain_topo,
        degenerate_phys;
        hermitian=false,
    )
end
