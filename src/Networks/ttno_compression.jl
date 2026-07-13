# Sector-aware TTNO compression. This is intentionally local to an operator
# edge: TensorKit factorisations retain block/sector structure, so no dense
# operator or cross-sector matrix is materialised.

"""
    TTNOCompressionSectorReport

Exact per-sector dimensions and local numerical discarded norms for one TTNO
edge. `deparallelized_norm` is caused only by the caller-certified
`compression_atol` numerical-rank cutoff in `mode=:exact_rank`;
`svd_discarded_norm` is zero in that mode.
"""
struct TTNOCompressionSectorReport{Q}
    sector::Q
    before_dimension::Int
    after_deparallelization_dimension::Int
    after_qr_dimension::Int
    after_svd_dimension::Int
    deparallelized_norm::Float64
    svd_discarded_norm::Float64
end

"""
    TTNOCompressionEdgeReport

Concrete compression record for a child-to-parent TTNO edge. Dimensions are
reported both in aggregate and per native symmetry sector.
"""
struct TTNOCompressionEdgeReport{Q}
    child::Symbol
    parent::Symbol
    before_dimension::Int
    after_deparallelization_dimension::Int
    after_qr_dimension::Int
    after_svd_dimension::Int
    deparallelized_norm::Float64
    svd_discarded_norm::Float64
    sectors::Vector{TTNOCompressionSectorReport{Q}}
end

"""
    TTNOCompressionReport

Result of [`compress!`](@ref). The report is separate from the mutated TTNO so
callers can include it in a Hamiltonian/audit record without retaining a second
operator network.
"""
struct TTNOCompressionReport{Q}
    mode::Symbol
    sector_aware::Bool
    compression_atol::Float64
    scheme::TruncationScheme
    edges::Vector{TTNOCompressionEdgeReport{Q}}
    total_before_dimension::Int
    total_after_dimension::Int
    compression_ratio::Float64
    aggregate_local_discarded_norm::Float64
end

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

function _exact_rank_scheme(compression_atol::Float64)
    # `TruncationScheme(atol=0)` intentionally maps to `notrunc()`. A smallest
    # positive threshold represents the requested exact-zero-only case while
    # retaining the single TruncationScheme-controlled factorisation path.
    effective_atol = compression_atol == 0.0 ? nextfloat(0.0) : compression_atol
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
    # Preserve TensorKit's graded pivotal convention when a factorisation
    # changes the dual orientation of the virtual link.
    O.tensors[parent] = absorb_on_leg(O.tensors[parent], _pivotal_link(factor), slot)
    return O
end

function _deparallelize_edge!(O::TTNO, child::Int, compression_atol::Float64)
    original = O.tensors[child]
    M = _compression_unfold(original)
    U, S, Vᴴ, discarded = split_svd_with_error(M, _exact_rank_scheme(compression_atol))
    tails = _compression_singular_tail_norms(M, space(S, 1))
    O.tensors[child] = _restore_compression_unfold(U, original)
    _absorb_compression_factor!(O, child, S * Vᴴ)
    return Float64(discarded), tails
end

function _qr_canonicalize_edge!(O::TTNO, child::Int)
    original = O.tensors[child]
    Q, C = left_orth(_compression_unfold(original); alg=:qr)
    O.tensors[child] = _restore_compression_unfold(Q, original)
    _absorb_compression_factor!(O, child, C)
    return O
end

function _svd_compress_edge!(O::TTNO, child::Int, scheme::TruncationScheme)
    original = O.tensors[child]
    U, S, Vᴴ, discarded = split_svd_with_error(_compression_unfold(original), scheme)
    O.tensors[child] = _restore_compression_unfold(U, original)
    _absorb_compression_factor!(O, child, S * Vᴴ)
    return Float64(discarded)
end

function _compression_sector_reports(before::Dict{Q,Int},
                                      after_deparallelization::Dict{Q,Int},
                                      after_qr::Dict{Q,Int},
                                      after_svd::Dict{Q,Int},
                                      deparallelized::Dict{Q,Float64}) where {Q}
    labels = Set{Q}()
    for values in (before, after_deparallelization, after_qr, after_svd)
        union!(labels, keys(values))
    end
    reports = TTNOCompressionSectorReport{Q}[]
    for q in sort!(collect(labels); by=string)
        push!(reports, TTNOCompressionSectorReport(
            q,
            get(before, q, 0),
            get(after_deparallelization, q, 0),
            get(after_qr, q, 0),
            get(after_svd, q, 0),
            get(deparallelized, q, 0.0),
            0.0,
        ))
    end
    return reports
end

function _require_exact_rank_options(mode::Symbol, compression_atol::Real,
                                     scheme::TruncationScheme)
    mode === :exact_rank ||
        throw(ArgumentError("compress! currently supports only mode=:exact_rank; approximate physical truncation is TODO(M6+)"))
    atol = Float64(compression_atol)
    isfinite(atol) && atol >= 0 ||
        throw(ArgumentError("compression_atol must be a finite nonnegative real number"))
    scheme.maxdim == typemax(Int) && scheme.atol == 0.0 && scheme.rtol == 0.0 &&
        scheme.discarded_weight == 0.0 ||
        throw(ArgumentError("mode=:exact_rank permits no implicit physical truncation; certify numerical zeros only through compression_atol and use a future explicit approximate compression mode for a nontrivial TruncationScheme"))
    return atol
end

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

"""
    compress!(O::TTNO; sector_aware=true, mode=:exact_rank,
              compression_atol, scheme=TruncationScheme()) -> TTNOCompressionReport

Compress a TTNO in place without materialising a dense operator. The fixed
leaf-to-root pipeline is: (1) sector-keyed exact/numerical-zero dependence
elimination, (2) standard QR canonicalisation, and (3) sector-resolved SVD
through [`TruncationScheme`](@ref). `mode=:exact_rank` accepts only a trivial
`scheme`; the sole permitted numerical reduction is the caller-certified
`compression_atol` numerical-rank cutoff. The report's aggregate local
discarded norm is a root-sum-square of edge-local factor tails, not a global
operator-norm error after parent-side rescaling.

Non-abelian sectors are rejected explicitly until `SU2Reduce` exists. The
returned report records every edge and sector; it is not a replacement TTNO.
"""
function compress!(O::TTNO;
                   sector_aware::Bool=true,
                   mode::Symbol=:exact_rank,
                   compression_atol::Real,
                   scheme::TruncationScheme=TruncationScheme())
    atol = _require_exact_rank_options(mode, compression_atol, scheme)
    check_arrows(O)
    _require_supported_compression_sectors(O, sector_aware)

    t = topology(O)
    root_tensor = O.tensors[t.root]
    Q = sectortype(domain(root_tensor)[numin(root_tensor)])
    reports = TTNOCompressionEdgeReport{Q}[]
    total_before = 0
    total_after = 0
    discarded_squared = 0.0

    for child in postorder(t)
        child == t.root && continue
        before = _compression_sector_dimensions(virtualspace(O, child))
        before_dimension = sum(values(before))
        deparallelized_norm, sector_tails = _deparallelize_edge!(O, child, atol)
        after_deparallelization = _compression_sector_dimensions(virtualspace(O, child))
        _qr_canonicalize_edge!(O, child)
        after_qr = _compression_sector_dimensions(virtualspace(O, child))
        svd_discarded_norm = _svd_compress_edge!(O, child, scheme)
        after_svd = _compression_sector_dimensions(virtualspace(O, child))
        sector_reports = _compression_sector_reports(
            before,
            after_deparallelization,
            after_qr,
            after_svd,
            sector_tails,
        )
        push!(reports, TTNOCompressionEdgeReport(
            nodeid(t, child),
            nodeid(t, t.parent[child]),
            before_dimension,
            sum(values(after_deparallelization)),
            sum(values(after_qr)),
            sum(values(after_svd)),
            deparallelized_norm,
            svd_discarded_norm,
            sector_reports,
        ))
        total_before += before_dimension
        total_after += sum(values(after_svd))
        discarded_squared += deparallelized_norm^2 + svd_discarded_norm^2
    end
    check_arrows(O)
    compression_ratio = total_before == 0 ? 1.0 : total_after / total_before
    return TTNOCompressionReport(
        mode,
        sector_aware,
        atol,
        scheme,
        reports,
        total_before,
        total_after,
        compression_ratio,
        sqrt(discarded_squared),
    )
end
