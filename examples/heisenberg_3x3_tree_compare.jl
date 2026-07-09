using GRAFT
using GRAFT.TestUtils
using GRAFT.Backend: ComplexSpace, dim, domain
using Printf
using Random

function balanced_binary_leaf_topology(leaves::AbstractVector{Symbol}; prefix::Symbol=:bin)
    isempty(leaves) && throw(ArgumentError("need at least one leaf"))
    allunique(leaves) || throw(ArgumentError("leaf ids must be unique"))
    length(leaves) == 1 && return TreeTopology(only(leaves), Pair{Symbol,Symbol}[])

    root = Symbol(prefix, :_root)
    root in leaves && throw(ArgumentError("internal root id $root collides with a leaf"))
    tree_edges = Pair{Symbol,Symbol}[]
    counter = Ref(0)

    function attach_range!(parent::Symbol, lo::Int, hi::Int)
        if lo == hi
            push!(tree_edges, parent => leaves[lo])
            return nothing
        end
        counter[] += 1
        node = Symbol(prefix, :_, counter[])
        push!(tree_edges, parent => node)
        split_at = (lo + hi) ÷ 2
        attach_range!(node, lo, split_at)
        attach_range!(node, split_at + 1, hi)
        return nothing
    end

    root_split = div(length(leaves), 2)
    attach_range!(root, firstindex(leaves), root_split)
    attach_range!(root, root_split + 1, lastindex(leaves))
    return TreeTopology(root, tree_edges)
end

function comb_t3ns_topology(sites::AbstractVector{Symbol}; prefix::Symbol=:t3)
    isempty(sites) && throw(ArgumentError("need at least one physical site"))
    allunique(sites) || throw(ArgumentError("site ids must be unique"))
    length(sites) == 1 && return TreeTopology(only(sites), Pair{Symbol,Symbol}[])

    spine(i) = Symbol(prefix, :_spine, i)
    root = spine(1)
    tree_edges = Pair{Symbol,Symbol}[]
    for i in 2:length(sites)
        push!(tree_edges, spine(i - 1) => spine(i))
    end
    for i in eachindex(sites)
        push!(tree_edges, spine(i) => sites[i])
    end
    return TreeTopology(root, tree_edges)
end

site_id(x::Int, y::Int, Lx::Int) = Symbol(:s, x + Lx * (y - 1))
site_id(i::Int) = Symbol(:s, i)

function open_boundary_bonds_square(Lx::Int, Ly::Int)
    bonds = Tuple{Symbol,Symbol}[]
    for y in 1:Ly, x in 1:Lx
        x < Lx && push!(bonds, (site_id(x, y, Lx), site_id(x + 1, y, Lx)))
        y < Ly && push!(bonds, (site_id(x, y, Lx), site_id(x, y + 1, Lx)))
    end
    return bonds
end

function heisenberg_opsum(bonds; J::Real=1.0)
    S = spin_ops()
    H = OpSum()
    for (a, b) in bonds
        # spin_ops uses Pauli Z and sigma ladder operators, so
        # S_i dot S_j = 1/4 Z_i Z_j + 1/2(S+_i S-_j + S-_i S+_j).
        H += Term(J / 4, SiteOp(a, :Z, S.Z), SiteOp(b, :Z, S.Z))
        H += Term(J / 2, SiteOp(a, :Sp, S.Sp), SiteOp(b, :Sm, S.Sm))
        H += Term(J / 2, SiteOp(a, :Sm, S.Sm), SiteOp(b, :Sp, S.Sp))
    end
    return H
end

function max_bond_dim(psi::TTNS)
    t = topology(psi)
    dims = [dim(domain(psi.tensors[n])[1]) for n in 1:nnodes(t) if t.parent[n] != 0]
    return isempty(dims) ? 1 : maximum(dims)
end

function run_tree_case(label::String, topo::TreeTopology, phys, H;
                       seed::Int, bond_dim::Int=16, maxdim::Int=32,
                       nsweeps::Int=6, krylovdim::Int=48)
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    psi = random_ttns(Xoshiro(seed), ComplexF64, topo, phys, ComplexSpace(bond_dim))
    Hd = dense_hamiltonian(H, topo, phys)
    E_ed, _ = exact_groundstate(Hd)

    _, energies = dmrg2!(psi, O;
                         trunc=TruncationScheme(maxdim=maxdim, atol=1e-12),
                         nsweeps, krylovdim)
    E_dmrg = real(expect(psi, O))

    return (; label, topo, E_ed, E_dmrg, err=abs(E_dmrg - E_ed),
            energies, maxbond=max_bond_dim(psi),
            t3ns=is_t3ns(topo; physical=collect(keys(phys))))
end

function main()
    Lx = 3
    Ly = 3
    J = 1.0
    sites = [site_id(i) for i in 1:(Lx * Ly)]
    bonds = open_boundary_bonds_square(Lx, Ly)
    S = spin_ops()
    phys = Dict(site => S.P for site in sites)
    H = heisenberg_opsum(bonds; J)

    binary = balanced_binary_leaf_topology(sites; prefix=:heis_bin)
    comb = comb_t3ns_topology(sites; prefix=:heis_t3)
    cases = [
        run_tree_case("balanced binary leaves", binary, phys, H; seed=20260710),
        run_tree_case("comb T3NS", comb, phys, H; seed=20260711),
    ]

    println("3x3 spin-1/2 Heisenberg model: binary-tree vs T3NS tree")
    println("boundary = open")
    println("open nearest-neighbor bonds = ", bonds)
    @printf("parameters: J = %.6g, H = J * sum_<ij> S_i dot S_j\n", J)
    println("physical sites = ", sites)
    println("dense Hilbert dimension = ", 2^(Lx * Ly))

    for result in cases
        println()
        println("tree = ", result.label)
        println("tree nodes = ", nnodes(result.topo),
                ", physical leaves = ", length(sites),
                ", is_t3ns = ", result.t3ns)
        @printf("ED ground energy      = %.12f\n", result.E_ed)
        @printf("DMRG ground energy    = %.12f\n", result.E_dmrg)
        @printf("abs(DMRG - ED)        = %.3e\n", result.err)
        println("DMRG sweep energies   = ", result.energies)
        println("max DMRG bond dim     = ", result.maxbond)
    end

    println()
    @printf("cross-tree |E_DMRG(binary) - E_DMRG(T3NS)| = %.3e\n",
            abs(cases[1].E_dmrg - cases[2].E_dmrg))
    @printf("cross-tree |E_ED(binary) - E_ED(T3NS)|     = %.3e\n",
            abs(cases[1].E_ed - cases[2].E_ed))
    return nothing
end

main()
