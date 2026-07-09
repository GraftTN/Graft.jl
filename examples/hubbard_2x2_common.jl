module Hubbard2x2Common

using GRAFT
using GRAFT.Backend: U1Irrep, dim, domain

export balanced_binary_leaf_topology, electron_ops, electron_orbitals,
    phonon_sites, pp_ancilla_site, open_boundary_bonds_2x2,
    hubbard_2x2_opsum, hubbard_holstein_2x2_opsum,
    hubbard_2x2_binary, hubbard_holstein_2x2_binary,
    half_filled_product_states, half_filled_pp_basis,
    electron_number_opsum, double_occupancy_opsum, phonon_number_opsum,
    dress_like_holstein, max_bond_dim

const SPINS = (:up, :dn)
const LATTICE_SITES = 1:4

"""
    balanced_binary_leaf_topology(leaves; prefix=:bin)

Build a balanced binary tree whose leaves, in the supplied order, are the
physical sites. Internal nodes are branching tensors with no physical leg.
"""
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
        node in leaves && throw(ArgumentError("internal node id $node collides with a leaf"))
        push!(tree_edges, parent => node)
        split_at = (lo + hi) ÷ 2
        attach_range!(node, lo, split_at)
        attach_range!(node, split_at + 1, hi)
        return nothing
    end

    root_split = div(length(leaves), 2)
    attach_range!(root, firstindex(leaves), root_split)
    attach_range!(root, root_split + 1, lastindex(leaves))
    _check_tree_edges(root, tree_edges)
    return TreeTopology(root, tree_edges)
end

function _check_tree_edges(root::Symbol, tree_edges::Vector{Pair{Symbol,Symbol}})
    seen = Set([root])
    for edge in tree_edges
        parent = first(edge)
        child = last(edge)
        parent in seen || throw(ArgumentError("parent $parent is not top-down in $tree_edges"))
        child in seen && throw(ArgumentError("node $child appears twice in $tree_edges"))
        push!(seen, child)
    end
    return nothing
end

"""
    electron_ops()

Spin-orbital qubit operators for an explicit Jordan-Wigner encoding.
`C` annihilates and `Cd` creates in the local occupation basis
`|0>, |1>`. `F = (-1)^n` supplies the Jordan-Wigner string.
"""
function electron_ops()
    S = spin_ops()
    return (; C=S.Sp, Cd=S.Sm, N=S.N, I=S.I, F=S.Z, P=S.P)
end

orbital(site::Int, spin::Symbol) =
    Symbol(:e, site, spin === :up ? :u : :d)

electron_orbitals() = [orbital(site, spin) for site in LATTICE_SITES for spin in SPINS]
phonon_sites() = [Symbol(:ph, site) for site in LATTICE_SITES]
pp_ancilla_site(ph::Symbol) = Symbol(ph, :_B, 1)

"""
    open_boundary_bonds_2x2()

Nearest-neighbor bonds of a 2x2 square with open boundary conditions.
The site numbering is row-major:

    1 -- 2
    |    |
    3 -- 4
"""
open_boundary_bonds_2x2() = [(1, 2), (3, 4), (1, 3), (2, 4)]

function _jw_hop_terms(orbitals::Vector{Symbol}, a::Symbol, b::Symbol,
                       coeff::Number, E)
    ia = findfirst(==(a), orbitals)
    ib = findfirst(==(b), orbitals)
    ia === nothing && throw(ArgumentError("unknown orbital $a"))
    ib === nothing && throw(ArgumentError("unknown orbital $b"))
    ia == ib && throw(ArgumentError("hopping needs two distinct orbitals"))
    if ia > ib
        ia, ib = ib, ia
        a, b = b, a
    end

    string_ops = [SiteOp(orbitals[k], :F, E.F) for k in (ia + 1):(ib - 1)]

    left_to_right = SiteOp[SiteOp(a, :Cd, E.Cd)]
    append!(left_to_right, string_ops)
    push!(left_to_right, SiteOp(b, :C, E.C))

    right_to_left = SiteOp[SiteOp(a, :C, E.C)]
    append!(right_to_left, string_ops)
    push!(right_to_left, SiteOp(b, :Cd, E.Cd))

    return (Term(coeff, left_to_right), Term(coeff, right_to_left))
end

function hubbard_2x2_opsum(; t::Real=1.0, U::Real=4.0, shifted::Bool=true)
    E = electron_ops()
    orbitals = electron_orbitals()
    H = OpSum()

    for (i, j) in open_boundary_bonds_2x2(), spin in SPINS
        a = orbital(i, spin)
        b = orbital(j, spin)
        for term in _jw_hop_terms(orbitals, a, b, -t, E)
            H += term
        end
    end

    for site in LATTICE_SITES
        up = orbital(site, :up)
        dn = orbital(site, :dn)
        H += Term(U, SiteOp(up, :N, E.N), SiteOp(dn, :N, E.N))
        if shifted
            # Particle-hole symmetric onsite term:
            # U * (n_up - 1/2) * (n_dn - 1/2).
            H += Term(-U / 2, SiteOp(up, :N, E.N))
            H += Term(-U / 2, SiteOp(dn, :N, E.N))
            H += Term(U / 4, SiteOp(up, :I, E.I))
        end
    end

    return H
end

function hubbard_holstein_2x2_opsum(; t::Real=1.0, U::Real=4.0,
                                    omega::Real=0.8, g::Real=0.35,
                                    nmax::Int=1, shifted::Bool=true,
                                    centered_density::Bool=true)
    E = electron_ops()
    B = boson_ops(nmax)
    H = hubbard_2x2_opsum(; t, U, shifted)

    for site in LATTICE_SITES
        ph = Symbol(:ph, site)
        up = orbital(site, :up)
        dn = orbital(site, :dn)
        H += Term(omega, SiteOp(ph, :N, B.N))
        H += Term(g, SiteOp(up, :N, E.N), SiteOp(ph, :X, B.X))
        H += Term(g, SiteOp(dn, :N, E.N), SiteOp(ph, :X, B.X))
        if centered_density
            # Holstein coupling to n_up + n_dn - 1 at half filling.
            H += Term(-g, SiteOp(up, :I, E.I), SiteOp(ph, :X, B.X))
        end
    end

    return H
end

function hubbard_2x2_binary()
    E = electron_ops()
    orbitals = electron_orbitals()
    topo = balanced_binary_leaf_topology(orbitals; prefix=:hub)
    phys = Dict(site => E.P for site in orbitals)
    return topo, phys, orbitals
end

function hubbard_holstein_2x2_binary(; nmax::Int=1)
    E = electron_ops()
    B = boson_ops(nmax)
    orbitals = electron_orbitals()
    phonons = phonon_sites()
    leaves = vcat(orbitals, phonons)
    topo = balanced_binary_leaf_topology(leaves; prefix=:hh)
    phys = Dict(site => E.P for site in orbitals)
    for ph in phonons
        phys[ph] = B.P
    end
    return topo, phys, orbitals, phonons
end

function _occupied_orbitals()
    # Neel-like half-filled seed in the N_up = N_down = 2 sector.
    return Set([orbital(1, :up), orbital(2, :dn),
                orbital(3, :dn), orbital(4, :up)])
end

function half_filled_product_states(orbitals::Vector{Symbol})
    occupied = _occupied_orbitals()
    states = Dict{Symbol,Vector{ComplexF64}}()
    for site in orbitals
        states[site] = site in occupied ? ComplexF64[0, 1] : ComplexF64[1, 0]
    end
    return states
end

function half_filled_pp_basis(orbitals::Vector{Symbol}, phonons::Vector{Symbol})
    occupied = _occupied_orbitals()
    basis = Dict{Symbol,Any}()
    q0 = U1Irrep(0)
    for site in orbitals
        basis[site] = q0 => (site in occupied ? 2 : 1)
    end
    for ph in phonons
        basis[ph] = q0
        basis[pp_ancilla_site(ph)] = q0
    end
    return basis
end

function electron_number_opsum()
    E = electron_ops()
    H = OpSum()
    for site in electron_orbitals()
        H += Term(1.0, SiteOp(site, :N, E.N))
    end
    return H
end

function double_occupancy_opsum()
    E = electron_ops()
    H = OpSum()
    for site in LATTICE_SITES
        H += Term(1.0, SiteOp(orbital(site, :up), :N, E.N),
                  SiteOp(orbital(site, :dn), :N, E.N))
    end
    return H
end

function phonon_number_opsum(nmax::Int)
    B = boson_ops(nmax)
    H = OpSum()
    for ph in phonon_sites()
        H += Term(1.0, SiteOp(ph, :N, B.N))
    end
    return H
end

function dress_like_holstein(H::OpSum, topo::TreeTopology, phys, phonons; nmax::Int)
    return ppdress(H, topo, phys; nmax, boson_sites=phonons)
end

function max_bond_dim(psi::TTNS)
    t = topology(psi)
    dims = [dim(domain(psi.tensors[n])[1]) for n in 1:nnodes(t) if t.parent[n] != 0]
    return isempty(dims) ? 1 : maximum(dims)
end

end # module Hubbard2x2Common
