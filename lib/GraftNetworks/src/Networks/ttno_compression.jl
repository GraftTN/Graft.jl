# CP1 — three-stage whole-tree TTNO compression (ttno-compression plan v2):
#
#     Stage 1: exact-certified deparallelization over the whole tree
#     Stage 2: one complete tree QR canonicalization sweep
#     Stage 3: one complete tree sector-SVD sweep
#
# Stage 1 removes only channels with an exact certificate (bitwise identity,
# exact negation, exact zero, or an explicitly supplied provenance
# certificate) and never invokes numerical rank estimation. The
# numerical-rank reveal retained from the earlier substrate is named
# `_numerical_zero_svd_edge!` and executes only inside Stage 3 as the
# caller-certified numerical-zero cutoff (exact-rank mode) or the explicit
# approximate policy. Mutation is transactional at the compress! boundary:
# all three stages run on a working copy and the input TTNO is updated only
# after complete validation.

# ---------------------------------------------------------------------------
# Exact witnesses and optional build provenance
# ---------------------------------------------------------------------------

"""
    TTNOExactChannelRelation(child, sector, kept, removed, factor)

One caller-supplied exact channel relation for Stage 1: on the edge below
`child`, inside the coupled `sector` block of the child unfolding, channel
column `removed` equals `factor` times channel column `kept` *algebraically*
(for example because a compiler emitted both from one coefficient atom with
exact scalars). Column indices address the sector block columns at compress
time. Relations are validated against the data with a strict consistency
guard and rejected — retaining the original channels — when they disagree;
a numerical tolerance is never promoted into a witness.
"""
struct TTNOExactChannelRelation{Q}
    child::Symbol
    sector::Q
    kept::Int
    removed::Int
    factor::ComplexF64
end

"""
    TTNOExactProvenance(relations)

Optional exact build provenance consumed by Stage 1. It is deliberately
independent of the diagram-compiler IR: any producer that can certify exact
duplicate or proportional channels may construct it.
"""
struct TTNOExactProvenance{Q}
    relations::Vector{TTNOExactChannelRelation{Q}}
end

"""
    TTNOExactWitness(child, sector, kept, removed, factor, source)

The recorded witness and reconstruction datum of one Stage-1 elimination:
channel `removed` was removed because it equals `factor` times channel
`kept` with certificate `source` (`:provenance`, `:bitwise_duplicate`,
`:bitwise_negation`, or `:bitwise_zero`; a zero channel records
`factor == 0` and `kept == 0`). Replaying the witnesses reconstructs the
pre-deparallelized action exactly.
"""
struct TTNOExactWitness{Q}
    child::Symbol
    sector::Q
    kept::Int
    removed::Int
    factor::ComplexF64
    source::Symbol
end

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

"""
    TTNOCompressionSectorReport

Per-sector stage provenance for one edge: ranks after every stage, the
exact witness count, and the two distinct discarded quantities —
`numerical_zero_discarded_norm` from the exact-rank numerical-zero cutoff
and `physical_discarded_norm` from an explicit approximate truncation
policy. Both norms are edge-local factor tails.
"""
struct TTNOCompressionSectorReport{Q}
    sector::Q
    input_rank::Int
    exact_deparallelized_rank::Int
    post_qr_rank::Int
    retained_svd_rank::Int
    exact_witness_count::Int
    numerical_zero_discarded_norm::Float64
    physical_discarded_norm::Float64
end

"""
    TTNOCompressionEdgeReport

Stage provenance for one child-to-parent TTNO edge: aggregate and
per-sector ranks for every stage, the exact witnesses with their
reconstruction data, and any fallback reasons for rejected relations
(undercompression is valid; uncertified overcompression is not).
"""
struct TTNOCompressionEdgeReport{Q}
    child::Symbol
    parent::Symbol
    input_rank::Int
    exact_deparallelized_rank::Int
    post_qr_rank::Int
    retained_svd_rank::Int
    exact_witness_count::Int
    numerical_zero_discarded_norm::Float64
    physical_discarded_norm::Float64
    fallback_reasons::Vector{String}
    witnesses::Vector{TTNOExactWitness{Q}}
    sectors::Vector{TTNOCompressionSectorReport{Q}}
end

"""
    TTNOCompressionReport

Result of [`compress!`](@ref). `stage_trace` records the executed
`stage => edge` sequence and proves the global phase ordering (all Stage-1
work precedes any QR; all QR precedes any SVD). The
`aggregate_local_discarded_norm` is a root-sum-square of edge-local factor
tails; it is *not* a global operator-norm bound — no such bound is claimed
unless a separate theorem and implementation establish it.
"""
struct TTNOCompressionReport{Q}
    mode::Symbol
    sector_aware::Bool
    numerical_zero_atol::Float64
    scheme::TruncationScheme
    sweep_root::Symbol
    stage_trace::Vector{Pair{Symbol,Symbol}}
    edges::Vector{TTNOCompressionEdgeReport{Q}}
    total_before_dimension::Int
    total_after_dimension::Int
    compression_ratio::Float64
    aggregate_local_discarded_norm::Float64
end

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

function _compression_sector_dimensions(V::ElementarySpace)
    Q = sectortype(V)
    dimensions = Dict{Q,Int}()
    for q in sectors(V)
        n = dim(V, q)
        n == 0 || (dimensions[q] = n)
    end
    return dimensions
end

function _compression_singular_tail_norms(M::AbstractTensorMap,
                                           kept_space::ElementarySpace)
    Q = sectortype(kept_space)
    kept = _compression_sector_dimensions(kept_space)
    tails = Dict{Q,Float64}()
    for (q, values) in pairs(svd_vals(M))
        nkeep = get(kept, q, 0)
        nkeep <= length(values) ||
            throw(ArgumentError("compression kept more singular values than exist in sector $q"))
        tails[q] = Float64(sqrt(sum(abs2, @view values[(nkeep + 1):end])))
    end
    return tails
end

"""
Named numerical-zero truncation scheme: the Stage-3 exact-rank mode removes
only caller-certified numerical zeros through this explicitly numerical
threshold. This is numerical behavior — it is never part of the exact
Stage-1 contract.
"""
function _numerical_zero_scheme(numerical_zero_atol::Float64)
    effective_atol = numerical_zero_atol == 0.0 ? nextfloat(0.0) : numerical_zero_atol
    return TruncationScheme(atol=effective_atol)
end

function _compression_unfold(A::AbstractTensorMap)
    N = numind(A)
    N >= 1 || throw(ArgumentError("TTNO tensor has no parent-edge leg"))
    return permute(A, (ntuple(identity, N - 1), (N,)))
end

function _restore_compression_unfold(Q::AbstractTensorMap, original::AbstractTensorMap)
    N = numind(original)
    No = numout(original)
    return permute(Q, (ntuple(identity, No), ntuple(i -> No + i, N - No)))
end

function _absorb_compression_factor!(O::TTNO, child::Int, factor::AbstractTensorMap)
    t = topology(O)
    parent = t.parent[child]
    parent == 0 && throw(ArgumentError("root has no compression parent edge"))
    slot = childslot(t, parent, child)
    # `absorb_on_leg` performs the required transpose while reconnecting the
    # TTNO virtual edge.  Unlike a TTNS gauge move, a factorization link here
    # is already oriented as the operator-edge reconstruction factor; applying
    # `pivotal_link` a second time would insert a physical fZ2 parity phase.
    O.tensors[parent] = absorb_on_leg(O.tensors[parent], factor, slot)
    return O
end

# ---------------------------------------------------------------------------
# Stage 1 — exact-certified deparallelization
# ---------------------------------------------------------------------------

function _stage1_new_space(::Type{S}, kept_counts) where {S<:ElementarySpace}
    if S === ComplexSpace
        total = sum(values(kept_counts); init=0)
        return ComplexSpace(total)
    end
    Q = sectortype(S)
    pairs = [q => kept_counts[q]
             for q in sort!(collect(keys(kept_counts)); by=string)]
    return Vect[Q](pairs...)
end

"""
Exact Stage-1 elimination on the edge below `child`. Only exact
certificates remove channels: validated provenance relations first, then
bitwise duplicate, bitwise negation, and bitwise zero columns. Everything
else is retained. Returns the witnesses and any fallback reasons.
"""
function _stage1_exact_deparallelize_edge!(O::TTNO, child::Int,
                                           relations::Vector{TTNOExactChannelRelation{Q}}) where {Q}
    t = topology(O)
    original = O.tensors[child]
    M = _compression_unfold(original)
    elt = scalartype(M)
    child_id = nodeid(t, child)

    witnesses = TTNOExactWitness{Q}[]
    fallbacks = String[]

    blocklist = sort!([(string(q), q, Matrix(b)) for (q, b) in blocks(M)];
                      by=first)
    # removed[q][j] = (kept column, factor) rows of the exact selection map
    # G with M == M_kept * G; kept[q] lists retained columns in order.
    kept = Dict{Q,Vector{Int}}()
    removed = Dict{Q,Dict{Int,Tuple{Int,ComplexF64}}}()
    for (_, q, _) in blocklist
        removed[q] = Dict{Int,Tuple{Int,ComplexF64}}()
    end

    for rel in relations
        index = findfirst(x -> x[2] == rel.sector, blocklist)
        if index === nothing
            push!(fallbacks, "provenance relation on absent sector $(rel.sector)")
            continue
        end
        b = blocklist[index][3]
        ncols = size(b, 2)
        if !(1 <= rel.kept <= ncols && 1 <= rel.removed <= ncols) ||
           rel.kept == rel.removed
            push!(fallbacks,
                  "provenance relation indices ($(rel.kept), $(rel.removed)) invalid for sector $(rel.sector) with $ncols channels")
            continue
        end
        if haskey(removed[rel.sector], rel.removed) ||
           haskey(removed[rel.sector], rel.kept)
            push!(fallbacks,
                  "provenance relation chains through an already removed channel in sector $(rel.sector)")
            continue
        end
        guard = norm(b[:, rel.removed] .- elt(rel.factor) .* b[:, rel.kept])
        scale = norm(b[:, rel.removed]) + abs(rel.factor) * norm(b[:, rel.kept])
        if guard > 1e-8 * max(scale, 1e-12)
            push!(fallbacks,
                  "provenance relation for sector $(rel.sector) channel $(rel.removed) disagrees with the data (residual $guard)")
            continue
        end
        removed[rel.sector][rel.removed] = (rel.kept, ComplexF64(rel.factor))
        push!(witnesses, TTNOExactWitness{Q}(
            child_id, rel.sector, rel.kept, rel.removed,
            ComplexF64(rel.factor), :provenance))
    end

    for (_, q, b) in blocklist
        ncols = size(b, 2)
        keptq = Int[]
        for j in 1:ncols
            haskey(removed[q], j) && continue
            matched = false
            for i in keptq
                if b[:, j] == b[:, i]
                    removed[q][j] = (i, ComplexF64(1.0))
                    push!(witnesses, TTNOExactWitness{Q}(
                        child_id, q, i, j, ComplexF64(1.0),
                        :bitwise_duplicate))
                    matched = true
                    break
                elseif b[:, j] == -b[:, i]
                    removed[q][j] = (i, ComplexF64(-1.0))
                    push!(witnesses, TTNOExactWitness{Q}(
                        child_id, q, i, j, ComplexF64(-1.0),
                        :bitwise_negation))
                    matched = true
                    break
                end
            end
            matched && continue
            if !isempty(keptq) && iszero(b[:, j])
                removed[q][j] = (0, ComplexF64(0.0))
                push!(witnesses, TTNOExactWitness{Q}(
                    child_id, q, 0, j, ComplexF64(0.0), :bitwise_zero))
                continue
            end
            push!(keptq, j)
        end
        if isempty(keptq)
            # Never empty a sector: retain the lowest column instead.
            delete!(removed[q], 1)
            filter!(w -> !(w.sector == q && w.removed == 1), witnesses)
            push!(keptq, 1)
            push!(fallbacks,
                  "sector $q would be emptied; retained one channel instead")
        end
        kept[q] = keptq
    end

    isempty(witnesses) && return witnesses, fallbacks

    kept_counts = Dict(q => length(kept[q]) for (_, q, _) in blocklist)
    V_new = _stage1_new_space(spacetype(M), kept_counts)
    Mk = zeros(elt, codomain(M) ← V_new)
    G = zeros(elt, V_new ← domain(M))
    for (_, q, b) in blocklist
        bMk = block(Mk, q)
        bG = block(G, q)
        keptq = kept[q]
        for (a, col) in enumerate(keptq)
            bMk[:, a] .= b[:, col]
            bG[a, col] = one(elt)
        end
        for (j, (i, factor)) in sort!(collect(removed[q]); by=first)
            iszero(factor) && continue
            a = findfirst(==(i), keptq)
            a === nothing && continue
            bG[a, j] = elt(factor)
        end
    end
    O.tensors[child] = _restore_compression_unfold(Mk, original)
    _absorb_compression_factor!(O, child, G)
    return witnesses, fallbacks
end

# ---------------------------------------------------------------------------
# Stage 2 — complete tree QR canonicalization sweep
# ---------------------------------------------------------------------------

function _qr_canonicalize_edge!(O::TTNO, child::Int)
    original = O.tensors[child]
    Q, C = left_orth(_compression_unfold(original); alg=:qr)
    O.tensors[child] = _restore_compression_unfold(Q, original)
    _absorb_compression_factor!(O, child, C)
    return O
end

"Whole-tree post-QR validation: arrows plus per-edge canonicality."
function _validate_qr_sweep(O::TTNO)
    t = topology(O)
    check_arrows(O)
    for child in postorder(t)
        child == t.root && continue
        M = _compression_unfold(O.tensors[child])
        err = norm(M' * M - id(domain(M)))
        scale = max(sqrt(Float64(dim(domain(M)))), 1.0)
        err <= 1e-10 * scale || throw(ArgumentError(
            "QR sweep left edge $(nodeid(t, child)) non-canonical (error $err)"))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Stage 3 — complete tree sector-SVD sweep (center-carrying descent)
# ---------------------------------------------------------------------------

_cp_others(N::Int, k::Int) = Tuple(i for i in 1:N if i != k)

function _cp_restore_perm(N::Int, No::Int, k::Int)
    src(i) = i < k ? i : (i == k ? N : i - 1)
    return (ntuple(i -> src(i), No), ntuple(j -> src(No + j), N - No))
end

"""
Stage-3 truncating split of the edge between `parent` and `child`, with the
sweep center at `parent`: the parent tensor is SVD-split with respect to the
child leg (the environment below the edge is orthogonal from the Stage-2 QR
sweep, the environment above through the center), the truncated link is
absorbed into the child, and the center moves into the child. This helper —
the renamed numerical-rank reveal of the earlier substrate — is numerical
behavior and never executes in Stage 1.
"""
function _stage3_split_edge!(O::TTNO, parent::Int, child::Int,
                             scheme::TruncationScheme)
    t = topology(O)
    slot = childslot(t, parent, child)
    A = O.tensors[parent]
    N, No = numind(A), numout(A)
    M = permute(A, (_cp_others(N, slot), (slot,)))
    U, S, Vᴴ, discarded = split_svd_with_error(M, scheme)
    tails = _compression_singular_tail_norms(M, space(S, 1))
    O.tensors[parent] = permute(U, _cp_restore_perm(N, No, slot))
    # Downward link transport composes the transposed link on the child's
    # parent domain leg. Unlike a TTNS gauge down-move, a TTNO operator-edge
    # reconstruction link is already correctly oriented: adding the
    # `pivotal_link` twist here would insert a physical fZ2 parity phase on
    # odd blocks (verified on both dual and non-dual edges).
    link = transpose(S * Vᴴ)
    childA = O.tensors[child]
    if numin(childA) == 2
        pad = one(scalartype(childA)) * id(domain(childA)[1])
        O.tensors[child] = childA * (pad ⊗ link)
    else
        O.tensors[child] = childA * link
    end
    return Float64(discarded), tails
end

"""
One complete Stage-3 sector-SVD sweep over every edge. The sweep walks the
tree in preorder with an explicit orthogonality center: each edge is
truncated exactly once with both environments orthogonal, and the center
returns through exact QR moves (canonical factorization entry points, no
truncation) after each subtree.
"""
function _stage3_svd_sweep!(O::TTNO, scheme::TruncationScheme,
                            stage_trace::Vector{Pair{Symbol,Symbol}},
                            record!::Function)
    t = topology(O)
    function descend(p::Int)
        for c in t.children[p]
            push!(stage_trace, :svd => nodeid(t, c))
            discarded, tails = _stage3_split_edge!(O, p, c, scheme)
            record!(c, discarded, tails)
            descend(c)
            # Center return: exact canonical move, not a second QR stage.
            _qr_canonicalize_edge!(O, c)
        end
        return nothing
    end
    descend(t.root)
    return nothing
end

# ---------------------------------------------------------------------------
# Option validation
# ---------------------------------------------------------------------------

function _require_compression_options(mode::Symbol, compression_atol::Real,
                                      scheme::TruncationScheme)
    mode in (:exact_rank, :approximate) ||
        throw(ArgumentError("compress! mode must be :exact_rank or :approximate"))
    atol = Float64(compression_atol)
    isfinite(atol) && atol >= 0 ||
        throw(ArgumentError("compression_atol must be a finite nonnegative real number"))
    trivial = scheme.maxdim == typemax(Int) && scheme.atol == 0.0 &&
        scheme.rtol == 0.0 && scheme.discarded_weight == 0.0
    if mode === :exact_rank
        trivial || throw(ArgumentError(
            "mode=:exact_rank permits no physical truncation; certify numerical zeros only through compression_atol and use mode=:approximate with an explicit TruncationScheme for physical truncation"))
    else
        trivial && throw(ArgumentError(
            "mode=:approximate requires an explicit nontrivial TruncationScheme"))
        atol == 0.0 || throw(ArgumentError(
            "mode=:approximate takes its entire policy from `scheme`; compression_atol must be 0"))
    end
    return atol
end

# Category capability negotiation (CP1d contract for later CAT milestones):
# the generic pipeline never branches on a named symmetry. The supported
# profile factorizes block-locally over Abelian coupled sectors; a later
# category adapter must declare which route, fusion-multiplicity,
# channel-copy-degeneracy, and frame blocks may participate in QR or SVD,
# and unsupported profiles fail closed here — before any stage mutates —
# without a dense fallback. Unfreezing a non-Abelian backend therefore
# extends this predicate and the block enumeration, not the stage logic.
function _require_supported_compression_sectors(O::TTNO, sector_aware::Bool)
    for A in O.tensors
        S = spacetype(A)
        if S !== ComplexSpace && !sector_aware
            throw(ArgumentError("sector_aware=false is invalid for a symmetry-blocked TTNO"))
        end
        if S !== ComplexSpace && !sector_cost_supported(space(A))
            throw(ArgumentError("non-abelian TTNO compression is unsupported without SU2Reduce; no dense fallback is available"))
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# compress!
# ---------------------------------------------------------------------------

"""
    compress!(O::TTNO; sector_aware=true, mode=:exact_rank,
              compression_atol, scheme=TruncationScheme(),
              provenance=nothing) -> TTNOCompressionReport

Compress a TTNO in place through the fixed three-stage whole-tree pipeline:
(1) exact-certified deparallelization over the entire tree — bitwise
duplicate/negation/zero channels plus validated
[`TTNOExactProvenance`](@ref) relations, never numerical rank estimation;
(2) one complete deterministic QR canonicalization sweep with whole-tree
arrow and canonicality validation; (3) one complete sector-resolved SVD
sweep. `mode=:exact_rank` accepts only a trivial `scheme` and removes
numerical zeros through the explicitly numerical `compression_atol`
threshold; `mode=:approximate` requires an explicit nontrivial `scheme`
(and `compression_atol == 0`) and reports every discarded component under
that policy.

Mutation is transactional: the pipeline runs on a working copy and `O` is
updated only after all three stages and their validations succeed. The
report's aggregate discarded norm is a root-sum-square of edge-local factor
tails, not a global operator-norm bound. Non-abelian sectors are rejected
explicitly without a dense fallback.
"""
function compress!(O::TTNO;
                   sector_aware::Bool=true,
                   mode::Symbol=:exact_rank,
                   compression_atol::Real,
                   scheme::TruncationScheme=TruncationScheme(),
                   provenance::Union{Nothing,TTNOExactProvenance}=nothing)
    atol = _require_compression_options(mode, compression_atol, scheme)
    check_arrows(O)
    _require_supported_compression_sectors(O, sector_aware)

    t = topology(O)
    root_tensor = O.tensors[t.root]
    Q = sectortype(domain(root_tensor)[numin(root_tensor)])
    work = TTNO(t, copy.(O.tensors); ishermitian=O.ishermitian)

    sweep = [child for child in postorder(t) if child != t.root]
    stage_trace = Pair{Symbol,Symbol}[]

    relations_by_child = Dict{Int,Vector{TTNOExactChannelRelation{Q}}}()
    if provenance !== nothing
        for rel in provenance.relations
            rel isa TTNOExactChannelRelation{Q} || throw(ArgumentError(
                "provenance sector type does not match the TTNO sector type"))
            haskey(t.index, rel.child) || throw(ArgumentError(
                "provenance names unknown node $(rel.child)"))
            child = t.index[rel.child]
            child == t.root && throw(ArgumentError(
                "provenance names the root, which has no compression edge"))
            push!(get!(relations_by_child, child,
                       TTNOExactChannelRelation{Q}[]), rel)
        end
    end

    input_dims = Dict{Int,Dict{Q,Int}}()
    stage1_dims = Dict{Int,Dict{Q,Int}}()
    qr_dims = Dict{Int,Dict{Q,Int}}()
    svd_dims = Dict{Int,Dict{Q,Int}}()
    witnesses = Dict{Int,Vector{TTNOExactWitness{Q}}}()
    fallbacks = Dict{Int,Vector{String}}()
    zero_norms = Dict{Int,Float64}()
    phys_norms = Dict{Int,Float64}()
    zero_tails = Dict{Int,Dict{Q,Float64}}()
    phys_tails = Dict{Int,Dict{Q,Float64}}()

    # Stage 1 completes over the entire tree before any QR.
    for child in sweep
        push!(stage_trace, :exact_deparallelize => nodeid(t, child))
        input_dims[child] = _compression_sector_dimensions(virtualspace(work, child))
        rels = get(relations_by_child, child, TTNOExactChannelRelation{Q}[])
        w, f = _stage1_exact_deparallelize_edge!(work, child, rels)
        witnesses[child] = w
        fallbacks[child] = f
        stage1_dims[child] = _compression_sector_dimensions(virtualspace(work, child))
    end
    check_arrows(work)

    # Stage 2 completes (and validates) over the entire tree before any SVD.
    for child in sweep
        push!(stage_trace, :qr => nodeid(t, child))
        _qr_canonicalize_edge!(work, child)
        qr_dims[child] = _compression_sector_dimensions(virtualspace(work, child))
    end
    _validate_qr_sweep(work)

    # Stage 3: one center-carrying sector-SVD sweep under exactly one policy.
    stage3_scheme = mode === :exact_rank ? _numerical_zero_scheme(atol) : scheme
    record! = function (child, discarded, tails)
        if mode === :exact_rank
            zero_norms[child] = discarded
            zero_tails[child] = tails
            phys_norms[child] = 0.0
            phys_tails[child] = Dict{Q,Float64}()
        else
            phys_norms[child] = discarded
            phys_tails[child] = tails
            zero_norms[child] = 0.0
            zero_tails[child] = Dict{Q,Float64}()
        end
        return nothing
    end
    _stage3_svd_sweep!(work, stage3_scheme, stage_trace, record!)
    for child in sweep
        svd_dims[child] = _compression_sector_dimensions(virtualspace(work, child))
    end
    check_arrows(work)

    reports = TTNOCompressionEdgeReport{Q}[]
    total_before = 0
    total_after = 0
    discarded_squared = 0.0
    for child in sweep
        labels = Set{Q}()
        for values in (input_dims[child], stage1_dims[child], qr_dims[child],
                       svd_dims[child])
            union!(labels, keys(values))
        end
        sector_reports = TTNOCompressionSectorReport{Q}[]
        for q in sort!(collect(labels); by=string)
            push!(sector_reports, TTNOCompressionSectorReport(
                q,
                get(input_dims[child], q, 0),
                get(stage1_dims[child], q, 0),
                get(qr_dims[child], q, 0),
                get(svd_dims[child], q, 0),
                count(w -> w.sector == q, witnesses[child]),
                get(zero_tails[child], q, 0.0),
                get(phys_tails[child], q, 0.0),
            ))
        end
        before_dimension = sum(values(input_dims[child]); init=0)
        after_dimension = sum(values(svd_dims[child]); init=0)
        push!(reports, TTNOCompressionEdgeReport(
            nodeid(t, child),
            nodeid(t, t.parent[child]),
            before_dimension,
            sum(values(stage1_dims[child]); init=0),
            sum(values(qr_dims[child]); init=0),
            after_dimension,
            length(witnesses[child]),
            zero_norms[child],
            phys_norms[child],
            fallbacks[child],
            witnesses[child],
            sector_reports,
        ))
        total_before += before_dimension
        total_after += after_dimension
        discarded_squared += zero_norms[child]^2 + phys_norms[child]^2
    end

    for n in 1:nnodes(t)
        O.tensors[n] = work.tensors[n]
    end
    check_arrows(O)
    compression_ratio = total_before == 0 ? 1.0 : total_after / total_before
    return TTNOCompressionReport(
        mode,
        sector_aware,
        atol,
        scheme,
        nodeid(t, t.root),
        stage_trace,
        reports,
        total_before,
        total_after,
        compression_ratio,
        sqrt(discarded_squared),
    )
end
