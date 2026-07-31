using Test
using LinearAlgebra: dot, norm
using Random: Xoshiro, randn
using Graft
using Graft.Backend
using GraftTestUtils

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

@graft_testset "graded effective maps and TDVP1 sweep order" begin
    F = fermion_ops_z2()
    even = FermionParity(0)
    odd = FermionParity(1)

    topo = mps_topology(4)
    site(i) = nodeid(topo, i)
    dup, bup, ddn, bdn = site(1), site(2), site(3), site(4)
    phys = Dict(site(i) => F.P for i in 1:4)

    H = OpSum()
    H += Term(-0.4, SiteOp(dup, :N, F.N))
    H += Term(-0.4, SiteOp(ddn, :N, F.N))
    H += Term(0.8, SiteOp(dup, :N, F.N), SiteOp(ddn, :N, F.N))
    for (impurity, bath) in ((dup, bup), (ddn, bdn))
        H += Term(
            -0.2,
            SiteOp(impurity, :Cd, F.Cd),
            SiteOp(bath, :C, F.C),
        )
        H += Term(
            -0.2,
            SiteOp(impurity, :C, F.C),
            SiteOp(bath, :Cd, F.Cd),
        )
    end
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    initial_basis = Dict(dup => odd, bup => even, ddn => odd, bdn => even)
    initial = product_ttns(ComplexF64, topo, phys, initial_basis)

    # Two untruncated TDVP2 steps open every dynamically accessible Schmidt
    # sector.  Its split_two_site! output is the historically failing gauge:
    # the state is canonical, but parent environments are consumed across
    # non-dual virtual bonds.  Keep this object untouched below.
    tdvp2_gauge = copy(initial)
    tdvp2 = TDVP2(
        trunc=TruncationScheme(maxdim=16, atol=1e-12),
        verbose=false,
    )
    seed_steps = 2
    dz = -0.025
    for _ in 1:seed_steps
        step!(tdvp2, tdvp2_gauge, O, dz)
        Graft.normalize!(tdvp2_gauge)
    end
    bond_nodes = [n for n in 1:nnodes(topo) if topo.parent[n] != 0]
    @test [(dim(virtualspace(tdvp2_gauge, n), even),
            dim(virtualspace(tdvp2_gauge, n), odd)) for n in bond_nodes] ==
          [(1, 1), (0, 2), (1, 1)]

    canonical_gauge = canonicalize!(copy(tdvp2_gauge), tdvp2_gauge.center)
    @test categorical_coordinates(canonical_gauge) ≈
          categorical_coordinates(tdvp2_gauge) atol = 2e-12 rtol = 0

    @testset "effective-map Rayleigh closure" begin
        for (label, center_state) in (
                ("left-orthogonal gauge", n -> move_center!(copy(tdvp2_gauge), n)),
                ("recanonicalized gauge", n -> canonicalize!(copy(tdvp2_gauge), n)))
            @testset "$label" begin
                for n in 1:nnodes(topo)
                    state = center_state(n)
                    expected = expect(state, O)
                    A = state.tensors[n]
                    h1 = eff_h1(
                        EnvCache(topo), state, O, n;
                        optimize=false, sector_aware=false,
                    )
                    @test dot(A, h1(A)) ≈ expected atol = 2e-11 rtol = 0
                end

                for n in bond_nodes
                    m = topo.parent[n]
                    state = center_state(n)
                    expected = expect(state, O)
                    theta = Graft.Contractions.two_site_tensor(state, n, m)
                    h2 = eff_h2(
                        EnvCache(topo), state, O, n, m;
                        optimize=false, sector_aware=false,
                    )
                    @test dot(theta, h2(theta)) ≈ expected atol = 2e-11 rtol = 0
                end
            end
        end
    end

    @testset "zero-site link coordinates" begin
        dense_operator = to_dense(O)
        link_rng = Xoshiro(0x810)

        function check_link(base, u, v; down::Bool)
            state = copy(base)
            move_center!(state, u)
            C = down ?
                Graft.Evolution._split_link_down(TDVP1(), state, O, u, v, dz) :
                Graft.Evolution._split_link_up(TDVP1(), state, O, u, v, dz)
            child, parent = down ? (v, u) : (u, v)
            child_tensor = state.tensors[child]
            parent_tensor = state.tensors[parent]
            h0 = eff_h0(
                EnvCache(topo), state, O, child, parent;
                optimize=false, sector_aware=false,
            )

            for x in (C, randn(link_rng, ComplexF64, space(C)))
                x_before = copy(x)
                y = h0(x)
                y_workspace = Graft.workspace_map(h0)(x)
                @test x ≈ x_before atol = 0 rtol = 0
                @test y_workspace ≈ y atol = 2e-12 rtol = 0

                tensors = copy(state.tensors)
                pivotal_x = Graft.Networks.pivotal_link(x)
                if down
                    tensors[child] = child_tensor * pivotal_x
                    embedded = TTNS(topo, tensors, child)
                else
                    tensors[parent] = Graft.Backend.absorb_on_leg(
                        parent_tensor, pivotal_x,
                        Graft.Trees.childslot(topo, parent, child),
                    )
                    embedded = TTNS(topo, tensors, parent)
                end
                coordinates = categorical_coordinates(embedded)
                exact_rayleigh = dot(
                    coordinates, dense_operator * coordinates)
                @test dot(x, y) ≈ exact_rayleigh atol = 2e-10 rtol = 0
            end
        end

        for child in bond_nodes
            parent = topo.parent[child]
            check_link(tdvp2_gauge, child, parent; down=false)
            check_link(tdvp2_gauge, parent, child; down=true)
        end
        # A dualized starting bond makes an up split rectangular in arrow
        # orientation even when its dimensions are square.  Its raw h0
        # contraction already carries the pivotal bend and must not receive
        # the down-split similarity correction a second time.
        first_child = first(bond_nodes)
        check_link(canonical_gauge, first_child, topo.parent[first_child];
                   down=false)
    end

    @testset "full-rank TDVP1 propagation" begin
        total_steps = 8
        exact = exact_evolve(
            to_dense(O),
            categorical_coordinates(initial),
            total_steps * dz,
        )
        infidelity(state) = begin
            evolved = categorical_coordinates(state)
            1 - abs(dot(evolved, exact)) / (norm(evolved) * norm(exact))
        end

        order2_state = nothing
        for (order, tolerance) in ((1, 2e-5), (2, 1e-10))
            @testset "order=$order" begin
                state = copy(tdvp2_gauge)
                evolver = TDVP1(order=order, verbose=false)
                for _ in (seed_steps + 1):total_steps
                    step!(evolver, state, O, dz)
                    Graft.normalize!(state)
                end
                @test infidelity(state) < tolerance
                order == 2 && (order2_state = state)
            end
        end

        @testset "topology reversal" begin
            swap_dup, swap_bup, swap_ddn, swap_bdn =
                site(3), site(4), site(1), site(2)
            Hswap = OpSum()
            Hswap += Term(-0.4, SiteOp(swap_dup, :N, F.N))
            Hswap += Term(-0.4, SiteOp(swap_ddn, :N, F.N))
            Hswap += Term(
                0.8,
                SiteOp(swap_dup, :N, F.N),
                SiteOp(swap_ddn, :N, F.N),
            )
            for (impurity, bath) in (
                    (swap_dup, swap_bup), (swap_ddn, swap_bdn))
                Hswap += Term(
                    -0.2,
                    SiteOp(impurity, :Cd, F.Cd),
                    SiteOp(bath, :C, F.C),
                )
                Hswap += Term(
                    -0.2,
                    SiteOp(impurity, :C, F.C),
                    SiteOp(bath, :Cd, F.Cd),
                )
            end
            Oswap = ttno_from_opsum(Hswap, topo, phys; hermitian=true)
            swapped_initial = product_ttns(
                ComplexF64, topo, phys,
                Dict(
                    swap_dup => odd, swap_bup => even,
                    swap_ddn => odd, swap_bdn => even,
                ),
            )
            swapped = copy(swapped_initial)
            swapped_seed = TDVP2(
                trunc=TruncationScheme(maxdim=16, atol=1e-12),
                verbose=false,
            )
            for _ in 1:seed_steps
                step!(swapped_seed, swapped, Oswap, dz)
                Graft.normalize!(swapped)
            end
            swapped_evolver = TDVP1(order=2, verbose=false)
            for _ in (seed_steps + 1):total_steps
                step!(swapped_evolver, swapped, Oswap, dz)
                Graft.normalize!(swapped)
            end
            swapped_exact = exact_evolve(
                to_dense(Oswap),
                categorical_coordinates(swapped_initial),
                total_steps * dz,
            )
            swapped_coordinates = categorical_coordinates(swapped)
            swapped_infidelity =
                1 - abs(dot(swapped_coordinates, swapped_exact)) /
                    (norm(swapped_coordinates) * norm(swapped_exact))
            @test swapped_infidelity < 1e-10

            number_operator(site_) = ttno_from_opsum(
                OpSum() + Term(1.0, SiteOp(site_, :N, F.N)),
                topo, phys; hermitian=true,
            )
            original_occupations = [
                real(expect(order2_state, number_operator(site_)))
                for site_ in (dup, bup, ddn, bdn)
            ]
            swapped_occupations = [
                real(expect(swapped, number_operator(site_)))
                for site_ in (swap_dup, swap_bup, swap_ddn, swap_bdn)
            ]
            @test swapped_occupations ≈ original_occupations atol = 2e-10 rtol = 0
        end
    end
end
