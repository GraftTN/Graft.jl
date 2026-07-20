# Canonical infinite-temperature state |I⟩ and state types.
# Implements §05 plan §2.3 (infinite_temperature_state).

function _flat_basis_local(V::ElementarySpace)
    Q = sectortype(V)
    out = Tuple{Q,Int}[]
    for s in sectors(V), i in 1:dim(V, s)
        push!(out, (s, i))
    end
    return out
end

function _fuse_abelian_local(a, b)
    fused = a ⊗ b
    length(fused) == 1 ||
        throw(ArgumentError("only abelian one-channel fusion supported in Thermal"))
    return only(fused)
end

function _basis_coord_local(legs::Vector{S}) where {S<:ElementarySpace}
    isempty(legs) && return Dict{Tuple,Tuple{sectortype(S),Int}}(() => (one(sectortype(S)), 1))
    Q = sectortype(first(legs))
    unitq = one(Q)
    # TensorKit block-row layout: abelian fusion trees iterate first-leg-
    # fastest, each owning one contiguous row range with degeneracy indices
    # column-major inside it (identical to the Networks/TTNOBuild coords;
    # sweeping basis positions directly would interleave degenerate sectors).
    K = length(legs)
    legsectors = [collect(sectors(V)) for V in legs]
    legoffsets = map(legs) do V
        offsets = Dict{Q,Int}()
        offset = 0
        for s in sectors(V)
            offsets[s] = offset
            offset += dim(V, s)
        end
        offsets
    end
    rows = Dict{Q,Int}()
    coord = Dict{Tuple,Tuple{Q,Int}}()
    for T in CartesianIndices(Tuple(length.(legsectors)))
        secs = ntuple(j -> legsectors[j][T[j]], K)
        q = unitq
        for s in secs
            q = _fuse_abelian_local(q, s)
        end
        for D in CartesianIndices(ntuple(j -> dim(legs[j], secs[j]), K))
            idx = ntuple(j -> legoffsets[j][secs[j]] + D[j], K)
            row = get(rows, q, 0) + 1
            rows[q] = row
            coord[idx] = (q, row)
        end
    end
    return coord
end

function _identity_tensor(::Type{T}, P::S) where {T<:Number, S<:ElementarySpace}
    Q = sectortype(P)
    if Q === Trivial
        d = dim(P)
        m = zeros(T, d, d)
        for i in 1:d
            m[i, i] = one(T)
        end
        return TensorMap(m, P ← P)
    end
    t = zeros(T, P ← P)
    for (q, b) in blocks(t)
        for i in 1:size(b, 1)
            b[i, i] = one(T)
        end
    end
    return t
end

function _coevaluation_core(::Type{T}, P::S, V_parent::S) where {T<:Number, S<:ElementarySpace}
    d = dim(P)
    dualP = dual(P)
    t = zeros(T, dualP ⊗ P ← V_parent)
    Q = sectortype(P)

    if Q === Trivial
        for (_, b) in blocks(t)
            for n in 1:d
                b[(n - 1) * d + n, 1] = one(T) / sqrt(T(d))
            end
        end
    else
        unitq = one(Q)
        legs = S[dualP, P]
        coord = _basis_coord_local(legs)
        Pbasis = _flat_basis_local(P)
        dualPbasis = _flat_basis_local(dualP)
        for n in 1:d
            dp_sec = dualPbasis[n]
            p_sec = Pbasis[n]
            fused = _fuse_abelian_local(dp_sec[1], p_sec[1])
            fused == unitq || continue
            cq, row = coord[(n, n)]
            for (q, b) in blocks(t)
                q == cq && (b[row, 1] = one(T) / sqrt(T(d)))
            end
        end
    end
    return t
end

function _pp_coevaluation_core(::Type{T}, P::S, Bth_space::S, V_parent::S) where {T<:Number, S<:ElementarySpace}
    d = dim(P)
    dualP = dual(P)
    t = zeros(T, dualP ⊗ Bth_space ⊗ P ← V_parent)
    Q = sectortype(P)
    unitq = one(Q)

    legs = S[dualP, Bth_space, P]
    coord = _basis_coord_local(legs)

    Pbasis = _flat_basis_local(P)
    dualPbasis = _flat_basis_local(dualP)
    Bthbasis = _flat_basis_local(Bth_space)

    for n in 0:(d - 1)
        p_sec = Pbasis[n + 1]
        # TensorKit preserves the primal sector iteration order under `dual`:
        # P=(0,1,...,nmax) gives dual(P)=(0,-1,...,-nmax). Pair occupation n
        # with the same flat position; reversing the index retains only the
        # middle occupation in the neutral block for odd `d`.
        dp_sec = dualPbasis[n + 1]
        bth_sec = Bthbasis[n + 1]
        fused = _fuse_abelian_local(_fuse_abelian_local(dp_sec[1], bth_sec[1]), p_sec[1])
        fused == unitq || continue
        cq, row = coord[(n + 1, n + 1, n + 1)]
        for (q, b) in blocks(t)
            q == cq && (b[row, 1] = one(T) / sqrt(T(d)))
        end
    end
    return t
end

function _add_trivial_legs(core::AbstractTensorMap{T,S}, n_trivial::Int, ::Type{S}) where {T<:Number, S<:ElementarySpace}
    n_trivial == 0 && return core
    unit = oneunit(S)
    oldcod = codomain(core)
    newcod = reduce(⊗, ntuple(_ -> unit, n_trivial)) ⊗ oldcod
    dom = domain(core)
    t = zeros(T, newcod ← dom)
    for (q, sb) in blocks(core)
        for (q2, db) in blocks(t)
            q == q2 || continue
            r, c = size(sb)
            db[1:r, 1:c] .= sb
        end
    end
    return t
end

function infinite_temperature_state(problem::PurificationProblem; T=ComplexF64)
    topo = problem.topo_doubled
    S = _problem_spacetype(problem)
    phys = problem.phys_doubled
    pp_ancilla_of = problem.pp_ancilla_of
    unit = oneunit(S)

    pp_b_sites = Set(values(pp_ancilla_of))

    bond_space = Dict{Int,S}()
    for child in 1:nnodes(topo)
        topo.parent[child] == 0 && continue
        child_sym = nodeid(topo, child)
        if haskey(problem.physical_of, child_sym)
            psite = problem.physical_of[child_sym]
            P = phys[psite]
            if haskey(pp_ancilla_of, psite)
                nmax_plus_1 = _pp_logical_dim(P)
                bond_space[child] = S(0 => nmax_plus_1)
            else
                bond_space[child] = dual(P)
            end
        elseif child_sym in pp_b_sites
            bond_space[child] = phys[child_sym]
        else
            bond_space[child] = unit
        end
    end

    tensors = Vector{AbstractTensorMap{T,S}}(undef, nnodes(topo))
    for n in 1:nnodes(topo)
        sym = nodeid(topo, n)
        node_children = topo.children[n]

        if topo.parent[n] == 0
            V_parent = unit
        else
            V_parent = bond_space[n]
        end

        has_phys_n = haskey(phys, sym)

        if !has_phys_n
            cod_legs = S[bond_space[c] for c in node_children]
            cod = isempty(cod_legs) ? unit : reduce(⊗, cod_legs)
            A = zeros(T, cod ← V_parent)
            for (_, b) in blocks(A)
                b[1, 1] = one(T)
            end
            tensors[n] = A
            continue
        end

        P = phys[sym]

        n_trivial_children = 0
        ancilla_child_idx = 0
        bth_child_idx = 0

        for (k, c) in enumerate(node_children)
            csym = nodeid(topo, c)
            if haskey(problem.physical_of, csym)
                if haskey(pp_ancilla_of, sym)
                    bth_child_idx = k
                else
                    ancilla_child_idx = k
                end
            elseif csym in pp_b_sites
                continue
            else
                n_trivial_children += 1
            end
        end

        if haskey(pp_ancilla_of, sym)
            Bth_space = bond_space[node_children[bth_child_idx]]
            core = _pp_coevaluation_core(T, P, Bth_space, V_parent)
            full = _add_trivial_legs(core, n_trivial_children, S)
            tensors[n] = full
        elseif ancilla_child_idx > 0
            core = _coevaluation_core(T, P, V_parent)
            full = _add_trivial_legs(core, n_trivial_children, S)
            tensors[n] = full
        else
            cod_legs = S[bond_space[c] for c in node_children]
            push!(cod_legs, P)
            cod = isempty(cod_legs) ? P : reduce(⊗, cod_legs)
            A = zeros(T, cod ← V_parent)
            for (_, b) in blocks(A)
                b[1, 1] = one(T)
            end
            tensors[n] = A
        end
    end

    for n in 1:nnodes(topo)
        sym = nodeid(topo, n)
        if haskey(problem.physical_of, sym) || sym in pp_b_sites
            P = phys[sym]
            tensors[n] = _identity_tensor(T, P)
        end
    end

    ψ = TTNS(topo, tensors, topo.root)
    normalize!(ψ)
    logZ0 = problem.log_hilbert_dim
    return PurifiedState(ψ, 0.0, 0.0, logZ0,
                         (; problem_hash = hash(problem.topo_orig),))
end

function _problem_spacetype(problem::PurificationProblem{S}) where {S<:ElementarySpace}
    return S
end
