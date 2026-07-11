# Purification problem construction: doubled topology, lifted operators.
# Implements §05 plan §2.1 (purification_problem, physical_ttno).

"""
    purification_problem(K::OpSum, topo::TreeTopology, phys;
                        ancilla_prefix=:thermal,
                        pp_pairs=Dict{Symbol,Symbol}(),
                        hermitian=true,
                        elt=ComplexF64) -> PurificationProblem

Build the doubled-topology purification problem. Each logical physical degree
of freedom receives one thermal ancilla carrying `dual(L_i)`. For ordinary
sites, `L_i = P_i`; for PP pairs, `L_i` is the constrained `P + B_PP` subspace
(dim = `nmax+1`), and the ancilla is a trivial-PP-sector space.

Thermal ancilla names are `Symbol(site, :_, ancilla_prefix)`. PP pairs are
declared explicitly in `pp_pairs` (P site => B_PP leaf); each receives one
`B_thermal` attached to `P`.
"""
function purification_problem(K::OpSum, topo::TreeTopology, phys;
                             ancilla_prefix::Symbol=:thermal,
                             pp_pairs::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                             hermitian::Bool=true,
                             elt::Type{<:Number}=ComplexF64)
    S = _unify_spacetype(phys)
    phys_typed = Dict{Symbol,S}(k => v for (k, v) in phys)

    # Validate pp_pairs
    for (p, bpp) in pp_pairs
        haskey(phys_typed, p) ||
            throw(ArgumentError("PP pair P site $p not in phys"))
        haskey(phys_typed, bpp) ||
            throw(ArgumentError("PP B_PP site $bpp not in phys"))
        nodeindex(topo, p)
        nodeindex(topo, bpp)
    end
    pp_p_sites = Set(keys(pp_pairs))
    pp_b_sites = Set(values(pp_pairs))

    # Determine which sites get thermal ancillas
    thermal_sites = Symbol[]
    for site in sort!(collect(keys(phys_typed)); by=string)
        if site in pp_b_sites
            continue  # B_PP is not an independent thermal site
        end
        push!(thermal_sites, site)
    end

    # Generate ancilla names and check collisions
    ancilla_of = Dict{Symbol,Symbol}()
    existing_names = Set(topo.ids)
    for site in sort!(thermal_sites; by=string)
        anc = Symbol(site, :_, ancilla_prefix)
        anc in existing_names &&
            throw(ArgumentError("ancilla name $anc collides with existing node"))
        haskey(ancilla_of, anc) &&
            throw(ArgumentError("ancilla name $anc generated twice"))
        ancilla_of[site] = anc
        push!(existing_names, anc)
    end

    # Build doubled topology: original edges + ancilla edges
    orig_edges = Pair{Symbol,Symbol}[
        nodeid(topo, topo.parent[i]) => nodeid(topo, i)
        for i in 2:nnodes(topo)
    ]
    new_edges = copy(orig_edges)
    for site in sort!(thermal_sites; by=string)
        push!(new_edges, site => ancilla_of[site])
    end
    doubled_topo = TreeTopology(nodeid(topo, topo.root), new_edges)

    # Build doubled phys dict
    phys_doubled = Dict{Symbol,S}()
    for (site, P) in phys_typed
        phys_doubled[site] = P
    end
    for site in thermal_sites
        P = phys_typed[site]
        if haskey(pp_pairs, site)
            # PP pair: B_thermal is trivial-PP-sector with dim = nmax+1
            nmax_plus_1 = _pp_logical_dim(P)
            pp_S = _pp_spacetype(P)
            phys_doubled[ancilla_of[site]] = pp_S(0 => nmax_plus_1)
        else
            # Ordinary site: ancilla carries dual(P)
            phys_doubled[ancilla_of[site]] = dual(P)
        end
    end

    # Build inverse maps
    physical_of = Dict(v => k for (k, v) in ancilla_of)
    pp_ancilla_of = Dict{Symbol,Symbol}(pp_pairs)
    thermal_ancilla_of = Dict{Symbol,Symbol}()
    for site in thermal_sites
        thermal_ancilla_of[site] = ancilla_of[site]
    end

    # Build logical groups
    logical_groups = Vector{Vector{Symbol}}()
    for site in sort!(collect(keys(phys_typed)); by=string)
        if site in pp_b_sites
            continue
        end
        if haskey(pp_pairs, site)
            push!(logical_groups, [site, pp_pairs[site]])
        else
            push!(logical_groups, [site])
        end
    end

    # Compute log_hilbert_dim
    log_dim = 0.0
    for group in logical_groups
        if length(group) == 1
            log_dim += log(dim(phys_typed[group[1]]))
        else
            # PP pair: logical dim = nmax+1 (not dim(P)*dim(B_PP))
            P = phys_typed[group[1]]
            log_dim += log(_pp_logical_dim(P))
        end
    end

    # Lift K to doubled topology as TTNO
    K_ttno = ttno_from_opsum(K, doubled_topo, phys_doubled;
                             hermitian=hermitian, elt=elt)

    metadata = (;
        topology_hash = hash(topo),
        n_ancillas = length(ancilla_of),
        n_pp_pairs = length(pp_pairs),
    )

    return PurificationProblem{S}(
        topo, doubled_topo, phys_typed, phys_doubled,
        ancilla_of, physical_of, pp_ancilla_of, thermal_ancilla_of,
        logical_groups, K_ttno, log_dim, hermitian, elt, metadata,
    )
end

"""
    physical_ttno(problem::PurificationProblem, O::OpSum;
                  hermitian=false, elt=problem.elt) -> TTNO

Lift a physical operator `O` to the doubled topology. The operator acts only
on physical sites; ancilla sites carry identities via ordinary TTNO
construction.
"""
function physical_ttno(problem::PurificationProblem, O::OpSum;
                      hermitian::Bool=false, elt::Type{<:Number}=problem.elt)
    return ttno_from_opsum(O, problem.topo_doubled, problem.phys_doubled;
                          hermitian=hermitian, elt=elt)
end

function _unify_spacetype(phys::Dict{Symbol,<:ElementarySpace})
    isempty(phys) && return ComplexSpace
    spaces = collect(values(phys))
    S = spacetype(spaces[1])
    for P in spaces
        spacetype(P) === S ||
            throw(ArgumentError("all physical spaces must share one spacetype; got $S and $(spacetype(P))"))
    end
    return S
end

function _pp_logical_dim(P::ElementarySpace)
    if spacetype(P) === ComplexSpace
        return dim(P)
    elseif spacetype(P) === GradedSpace
        return dim(P)
    else
        return dim(P)
    end
end

function _pp_spacetype(P::ElementarySpace)
    S = spacetype(P)
    if S === ComplexSpace
        return U1Space  # PP lifts to U(1)
    else
        return S
    end
end
