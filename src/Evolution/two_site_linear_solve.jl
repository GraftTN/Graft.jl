"""
    TwoSiteLinearPolicy(; trunc=TruncationScheme(), sweeps=4,
                        krylovdim=30, maxiter=100,
                        local_tol=1e-10, residual_tol=1e-10)

Policy for the opt-in two-site variational linear solver. `local_tol` controls
each projected Krylov solve, while `residual_tol` is applied only to the
uncompressed physical residual after a complete sweep. All truncation of the
updated bond is governed by `trunc`.
"""
struct TwoSiteLinearPolicy
    trunc::TruncationScheme
    sweeps::Int
    krylovdim::Int
    maxiter::Int
    local_tol::Float64
    residual_tol::Float64

    function TwoSiteLinearPolicy(trunc::TruncationScheme, sweeps::Int,
                                 krylovdim::Int, maxiter::Int,
                                 local_tol::Float64, residual_tol::Float64)
        _validate_two_site_truncation(trunc)
        sweeps >= 1 ||
            throw(ArgumentError("TwoSiteLinearPolicy sweeps must be positive"))
        krylovdim >= 2 ||
            throw(ArgumentError("TwoSiteLinearPolicy krylovdim must be at least 2"))
        maxiter >= 1 ||
            throw(ArgumentError("TwoSiteLinearPolicy maxiter must be positive"))
        isfinite(local_tol) && local_tol > 0 ||
            throw(ArgumentError("TwoSiteLinearPolicy local_tol must be finite and positive"))
        isfinite(residual_tol) && residual_tol > 0 ||
            throw(ArgumentError("TwoSiteLinearPolicy residual_tol must be finite and positive"))
        return new(trunc, sweeps, krylovdim, maxiter, local_tol, residual_tol)
    end
end

function TwoSiteLinearPolicy(;
        trunc::TruncationScheme=TruncationScheme(),
        sweeps::Integer=4,
        krylovdim::Integer=30,
        maxiter::Integer=100,
        local_tol::Real=1e-10,
        residual_tol::Real=1e-10)
    return TwoSiteLinearPolicy(
        trunc, Int(sweeps), Int(krylovdim), Int(maxiter),
        Float64(local_tol), Float64(residual_tol))
end

function _validate_two_site_truncation(trunc::TruncationScheme)
    trunc.maxdim >= 1 ||
        throw(ArgumentError("TwoSiteLinearPolicy trunc.maxdim must be positive"))
    isfinite(trunc.atol) && trunc.atol >= 0 ||
        throw(ArgumentError("TwoSiteLinearPolicy trunc.atol must be finite and nonnegative"))
    isfinite(trunc.rtol) && 0 <= trunc.rtol <= 1 ||
        throw(ArgumentError("TwoSiteLinearPolicy trunc.rtol must lie in [0, 1]"))
    isfinite(trunc.discarded_weight) && 0 <= trunc.discarded_weight <= 1 ||
        throw(ArgumentError(
            "TwoSiteLinearPolicy trunc.discarded_weight must lie in [0, 1]"))
    return nothing
end

"""
Diagnostics for one child-parent two-site linear update.

The two local residuals use the same exact projected right-hand side and
effective equation. The first is evaluated on the Krylov solution before the
SVD split; the second is evaluated on the merged tensor reconstructed after
the policy-controlled split. `discarded_weight` is the actual squared relative
two-norm lost by that split.
"""
struct TwoSiteLinearEdgeReport
    sweep::Int
    child::Int
    parent::Int
    center_on::Symbol
    local_residual_before_truncation::Float64
    local_residual_after_truncation::Float64
    retained_rank::Int
    discarded_norm::Float64
    discarded_weight::Float64
    solver_iterations::Int
    solver_operations::Int
    solver_converged::Bool
end

"""
    TwoSiteLinearReport

Typed result of [`two_site_linsolve!`](@ref). `physical_residuals` contains the
initial residual followed by one uncompressed residual per completed sweep.
`transaction_committed` is false when a local failure, non-finite result, or
exception caused the scratch state to be discarded.
"""
struct TwoSiteLinearReport
    edge_reports::Vector{TwoSiteLinearEdgeReport}
    physical_residuals::Vector{Float64}
    converged::Bool
    stop_reason::Symbol
    transaction_committed::Bool
    exception_message::Union{Nothing,String}
end

"""
    PairedLinearClassification

Diagnostic for a Residual Driven Expansion/two-site pair evaluated
against one physical residual tolerance.
"""
struct PairedLinearClassification
    classification::Symbol
    residual_driven_converged::Bool
    two_site_converged::Bool
    residual_driven_residual::Float64
    two_site_residual::Float64
    tolerance::Float64
end

"""
    classify_linear_pair(residual_driven_report, two_site_report; tolerance)

Classify paired adaptive-one-site and two-site runs from their final
uncompressed physical residuals:

- `:inconclusive_failure`
- `:two_site_converged_residual_driven_stalled`
- `:both_stalled`
- `:matched`
- `:inconsistent`
"""
function classify_linear_pair(
        residual_driven_report, two_site_report; tolerance::Real)
    tol = Float64(tolerance)
    isfinite(tol) && tol > 0 ||
        throw(ArgumentError("classify_linear_pair tolerance must be finite and positive"))
    inconclusive = _paired_report_uncommitted(residual_driven_report) ||
        _paired_report_uncommitted(two_site_report)
    residual_driven_residual = _final_paired_physical_residual(
        residual_driven_report, "Residual Driven Expansion")
    two_site_residual =
        _final_paired_physical_residual(two_site_report, "two-site")
    residual_driven_converged =
        isfinite(residual_driven_residual) && residual_driven_residual <= tol
    two_site_converged =
        isfinite(two_site_residual) && two_site_residual <= tol
    classification = if inconclusive
        :inconclusive_failure
    elseif two_site_converged && !residual_driven_converged
        :two_site_converged_residual_driven_stalled
    elseif !two_site_converged && !residual_driven_converged
        :both_stalled
    elseif two_site_converged && residual_driven_converged
        :matched
    else
        :inconsistent
    end
    return PairedLinearClassification(
        classification, residual_driven_converged, two_site_converged,
        residual_driven_residual, two_site_residual, tol)
end

function _paired_report_uncommitted(report)
    hasproperty(report, :committed) &&
        !Bool(getproperty(report, :committed)) && return true
    hasproperty(report, :transaction_committed) &&
        !Bool(getproperty(report, :transaction_committed)) && return true
    return false
end

function _final_paired_physical_residual(report, label::String)
    hasproperty(report, :physical_residuals) ||
        throw(ArgumentError(
            "classify_linear_pair: $label report has no physical_residuals"))
    residuals = getproperty(report, :physical_residuals)
    residuals isa AbstractVector ||
        throw(ArgumentError(
            "classify_linear_pair: $label physical_residuals must be a vector"))
    isempty(residuals) && return Inf
    residual = last(residuals)
    residual isa Real ||
        throw(ArgumentError(
            "classify_linear_pair: $label final physical residual must be real"))
    return Float64(residual)
end

"""
    PairedEdgeSubspaceEvidence

Per-edge comparison of the new child-side bases retained by Residual Driven
Expansion and the two-site solve relative to the identical initial basis.
`principal_cosines` are the singular values of the overlap between the two
orthonormal novel bases. The two directional projection errors report how
completely either novel basis is contained in the other.

Evidence is unavailable when an update changed a child's external codomain,
because those local tensor subspaces then live in different spaces. Such an
edge reports `available == false` and
`stop_reason == :incompatible_external_space`.
"""
struct PairedEdgeSubspaceEvidence
    edge::Pair{Symbol,Symbol}
    available::Bool
    stop_reason::Symbol
    initial_rank::Int
    residual_driven_rank::Int
    two_site_rank::Int
    residual_driven_novel_rank::Int
    two_site_novel_rank::Int
    principal_cosines::Vector{Float64}
    residual_driven_to_two_site_projection_error::Float64
    two_site_to_residual_driven_projection_error::Float64
end

"""
    PairedLinearDiagnostic

Typed reports, per-edge novel-subspace evidence, and physical-residual
classification from [`paired_linear_diagnostic`](@ref). Both solvers start
from independent copies of the same input state.
"""
struct PairedLinearDiagnostic
    residual_driven_report::ResidualDrivenReport
    two_site_report::TwoSiteLinearReport
    classification::PairedLinearClassification
    edge_subspace_evidence::Vector{PairedEdgeSubspaceEvidence}
end

"""
    paired_linear_diagnostic(
        psi, H, rhs, residual_policy, two_site_policy;
        a0=1, a1=1, krylovdim=30, maxiter=100, tol=1e-10,
        fit_nsweeps=4, fit_tol=1e-10, fit_verbose=false)
        -> PairedLinearDiagnostic

Run Residual Driven Expansion and the two-site diagnostic from identical
copies of `psi`. The truncation policy, total sweep ceiling, Krylov dimension,
iteration limit, local tolerance, and authoritative physical-residual
tolerance must match. Residual Driven Expansion's rank-growth controls must
also be nonbinding relative to the shared hard rank cap, so both paths can
reach the same per-edge variational space. All budgets and both linear systems
are validated before either scratch solve starts. The caller's `psi` is never
mutated.
"""
function paired_linear_diagnostic(
        psi::TTNS,
        H::TTNO,
        rhs::TTNS,
        residual_policy::ResidualDrivenExpansion,
        two_site_policy::TwoSiteLinearPolicy;
        a0::Number=one(eltype(psi)),
        a1::Number=one(eltype(psi)),
        krylovdim::Int=30,
        maxiter::Int=100,
        tol::Float64=1e-10,
        fit_nsweeps::Int=4,
        fit_tol::Float64=1e-10,
        fit_verbose::Bool=false)
    _check_paired_linear_diagnostic(
        psi,
        H,
        rhs,
        residual_policy,
        two_site_policy,
        a0,
        a1,
        krylovdim,
        maxiter,
        tol,
        fit_nsweeps,
        fit_tol,
    )

    residual_driven_state = copy(psi)
    two_site_state = copy(psi)
    _, residual_driven_report = residual_driven_linsolve!(
        residual_driven_state,
        H,
        rhs,
        residual_policy;
        a0,
        a1,
        krylovdim,
        maxiter,
        tol,
        fit_nsweeps,
        fit_tol,
        fit_verbose,
    )
    _, two_site_report = two_site_linsolve!(
        two_site_state,
        H,
        rhs,
        two_site_policy;
        a0,
        a1,
        verbose=fit_verbose,
    )
    classification = classify_linear_pair(
        residual_driven_report,
        two_site_report;
        tolerance=tol,
    )
    edge_subspace_evidence = _paired_edge_subspace_evidence(
        psi,
        residual_driven_state,
        two_site_state,
    )
    return PairedLinearDiagnostic(
        residual_driven_report,
        two_site_report,
        classification,
        edge_subspace_evidence,
    )
end

function _paired_edge_subspace_evidence(
        initial::TTNS,
        residual_driven::TTNS,
        two_site::TTNS)
    initial_basis = copy(initial)
    residual_driven_basis = copy(residual_driven)
    two_site_basis = copy(two_site)
    topo = topology(initial_basis)
    root = topo.root
    move_center!(initial_basis, root; cache=EnvCache(topo))
    move_center!(residual_driven_basis, root; cache=EnvCache(topo))
    move_center!(two_site_basis, root; cache=EnvCache(topo))

    evidence = PairedEdgeSubspaceEvidence[]
    for (child, parent) in edges(topo)
        initial_tensor = initial_basis.tensors[child]
        residual_driven_tensor = residual_driven_basis.tensors[child]
        two_site_tensor_ = two_site_basis.tensors[child]
        initial_rank = dim(domain(initial_tensor))
        residual_driven_rank = dim(domain(residual_driven_tensor))
        two_site_rank = dim(domain(two_site_tensor_))
        compatible =
            codomain(initial_tensor) == codomain(residual_driven_tensor) &&
            codomain(initial_tensor) == codomain(two_site_tensor_)
        if !compatible
            push!(evidence, PairedEdgeSubspaceEvidence(
                nodeid(topo, child) => nodeid(topo, parent),
                false,
                :incompatible_external_space,
                initial_rank,
                residual_driven_rank,
                two_site_rank,
                0,
                0,
                Float64[],
                NaN,
                NaN,
            ))
            continue
        end

        residual_driven_novel, residual_driven_novel_rank =
            _paired_novel_child_basis(
                residual_driven_tensor, initial_tensor)
        two_site_novel, two_site_novel_rank =
            _paired_novel_child_basis(two_site_tensor_, initial_tensor)
        principal_cosines = _paired_principal_cosines(
            residual_driven_novel, two_site_novel)
        residual_driven_to_two_site_projection_error =
            _paired_directional_projection_error(
                residual_driven_novel, two_site_novel)
        two_site_to_residual_driven_projection_error =
            _paired_directional_projection_error(
                two_site_novel, residual_driven_novel)
        push!(evidence, PairedEdgeSubspaceEvidence(
            nodeid(topo, child) => nodeid(topo, parent),
            true,
            :available,
            initial_rank,
            residual_driven_rank,
            two_site_rank,
            residual_driven_novel_rank,
            two_site_novel_rank,
            principal_cosines,
            residual_driven_to_two_site_projection_error,
            two_site_to_residual_driven_projection_error,
        ))
    end
    return evidence
end

function _paired_novel_child_basis(
        candidate::AbstractTensorMap,
        initial::AbstractTensorMap)
    novel = candidate - initial * (initial' * candidate)
    threshold = sqrt(eps(Float64)) * max(Float64(norm(candidate)), 1.0)
    Float64(norm(novel)) <= threshold && return nothing, 0
    basis, _, _ = split_svd(
        novel,
        TruncationScheme(atol=threshold),
    )
    rank = dim(domain(basis))
    return rank == 0 ? (nothing, 0) : (basis, rank)
end

function _paired_principal_cosines(
        first_basis::Union{Nothing,AbstractTensorMap},
        second_basis::Union{Nothing,AbstractTensorMap})
    (first_basis === nothing || second_basis === nothing) &&
        return Float64[]
    overlap_values = Float64[]
    for values in values(svd_vals(first_basis' * second_basis))
        append!(
            overlap_values,
            clamp.(Float64.(real.(values)), 0.0, 1.0),
        )
    end
    sort!(overlap_values; rev=true)
    return overlap_values
end

function _paired_directional_projection_error(
        source::Union{Nothing,AbstractTensorMap},
        target::Union{Nothing,AbstractTensorMap})
    source === nothing && return 0.0
    target === nothing && return 1.0
    residual = source - target * (target' * source)
    source_norm = Float64(norm(source))
    iszero(source_norm) && return 0.0
    return Float64(norm(residual)) / source_norm
end

function _check_paired_linear_diagnostic(
        psi::TTNS,
        H::TTNO,
        rhs::TTNS,
        residual_policy::ResidualDrivenExpansion,
        two_site_policy::TwoSiteLinearPolicy,
        a0::Number,
        a1::Number,
        krylovdim::Int,
        maxiter::Int,
        tol::Float64,
        fit_nsweeps::Int,
        fit_tol::Float64)
    residual_policy.trunc == two_site_policy.trunc ||
        throw(ArgumentError(
            "paired_linear_diagnostic: Residual Driven Expansion and " *
            "two-site truncation policies must match exactly"))
    trunc = residual_policy.trunc
    iszero(trunc.atol) &&
        iszero(trunc.rtol) &&
        iszero(trunc.discarded_weight) ||
        throw(ArgumentError(
            "paired_linear_diagnostic: matched references require a " *
            "hard-cap-only truncation policy with zero atol, rtol, and " *
            "discarded_weight"))
    residual_sweep_limit = try
        Base.checked_mul(
            fit_nsweeps,
            Base.checked_add(residual_policy.max_rounds, 1),
        )
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError(
            "paired_linear_diagnostic: total sweep budget overflows Int"))
    end
    residual_sweep_limit == two_site_policy.sweeps ||
        throw(ArgumentError(
            "paired_linear_diagnostic: two-site sweeps must equal the " *
            "Residual Driven Expansion total sweep ceiling " *
            "fit_nsweeps * (max_rounds + 1) = $residual_sweep_limit"))
    krylovdim == two_site_policy.krylovdim ||
        throw(ArgumentError(
            "paired_linear_diagnostic: Krylov dimensions must match"))
    maxiter == two_site_policy.maxiter ||
        throw(ArgumentError(
            "paired_linear_diagnostic: maxiter budgets must match"))
    tol == two_site_policy.local_tol ||
        throw(ArgumentError(
            "paired_linear_diagnostic: local solver tolerances must match"))
    tol == two_site_policy.residual_tol ||
        throw(ArgumentError(
            "paired_linear_diagnostic: physical residual tolerances must match"))
    isfinite(fit_tol) && fit_tol >= 0 ||
        throw(ArgumentError(
            "paired_linear_diagnostic: fit_tol must be finite and nonnegative"))

    _check_linsolve_args(
        psi,
        H,
        rhs,
        a0,
        a1,
        krylovdim,
        maxiter,
        tol,
        fit_nsweeps,
        fit_tol,
    )
    _rde_check_linear_system(psi, H, rhs, a0, a1)
    _rde_check_state_rank_cap(psi, residual_policy)
    rank_requirements =
        _paired_rank_growth_requirements(psi, residual_policy.trunc.maxdim)
    residual_policy.max_rounds >= Int(!iszero(rank_requirements.edges)) ||
        throw(ArgumentError(
            "paired_linear_diagnostic: at least one expansion round is " *
            "required to reach the shared rank cap"))
    residual_policy.max_add >= rank_requirements.per_edge ||
        throw(ArgumentError(
            "paired_linear_diagnostic: Residual Driven Expansion max_add " *
            "must not bind before the shared rank cap " *
            "(required $(rank_requirements.per_edge))"))
    residual_policy.max_total_add >= rank_requirements.total ||
        throw(ArgumentError(
            "paired_linear_diagnostic: Residual Driven Expansion " *
            "max_total_add must not bind before the shared rank cap " *
            "(required $(rank_requirements.total))"))
    residual_policy.max_edges >= rank_requirements.edges ||
        throw(ArgumentError(
            "paired_linear_diagnostic: Residual Driven Expansion max_edges " *
            "must cover every growable edge under the shared rank cap " *
            "(required $(rank_requirements.edges))"))
    _check_two_site_linsolve_args(
        psi,
        H,
        rhs,
        two_site_policy,
        a0,
        a1,
    )
    return nothing
end

function _paired_rank_growth_requirements(
        psi::TTNS, maxdim::Int)
    capacities = Int[]
    topo = topology(psi)
    for child in 1:nnodes(topo)
        topo.parent[child] == 0 && continue
        rank = dim(virtualspace(psi, child))
        push!(capacities, maxdim - rank)
    end
    total = try
        foldl(Base.checked_add, capacities; init=0)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError(
            "paired_linear_diagnostic: rank-growth budget overflows Int"))
    end
    return (
        per_edge=maximum(capacities; init=0),
        total,
        edges=count(>(0), capacities),
    )
end

"""
    two_site_linsolve!(psi, H, rhs, policy; a0=1, a1=1, verbose=false)
        -> (psi, report)

Solve `(a0 * I + a1 * H)psi = rhs` by alternating child-parent two-site
updates. Every update projects `rhs` exactly into the current external
environments, solves the `eff_h2` equation on the merged tensor, and splits it
under `policy.trunc`. Convergence is decided only from the uncompressed
physical residual after a complete sweep.

The operation is transactional: all sweeps run on a copy. A local Krylov
failure, a non-finite residual, or an exception returns the original `psi`
unchanged with `report.transaction_committed == false`. Exhausting the sweep
or rank budget is a valid, committed stalled solve and is reported truthfully.
"""
function two_site_linsolve!(
        psi::TTNS, H::TTNO, rhs::TTNS, policy::TwoSiteLinearPolicy;
        a0::Number=one(eltype(psi)), a1::Number=one(eltype(psi)),
        verbose::Bool=false)
    _check_two_site_linsolve_args(psi, H, rhs, policy, a0, a1)
    T = eltype(psi)
    a0T, a1T = convert(T, a0), convert(T, a1)
    edge_reports = TwoSiteLinearEdgeReport[]
    physical_residuals = Float64[]
    working = copy(psi)

    try
        initial_residual =
            _linear_physical_residual(working, H, rhs, a0T, a1T)
        push!(physical_residuals, initial_residual)
        if !isfinite(initial_residual)
            return psi, TwoSiteLinearReport(
                edge_reports, physical_residuals, false,
                :nonfinite_physical_residual, false, nothing)
        end
        if initial_residual <= policy.residual_tol
            _commit_two_site_linear_state!(psi, working)
            return psi, TwoSiteLinearReport(
                edge_reports, physical_residuals, true, :converged, true,
                nothing)
        end

        topo = topology(working)
        cache = EnvCache(topo)
        bonds = [n for n in postorder(topo) if topo.parent[n] != 0]
        for sweep in 1:policy.sweeps
            root_targets = Contractions._physless_root_two_site_targets(
                working, policy.trunc)
            Contractions._bootstrap_physless_root!(
                working, cache, root_targets)
            updates = Iterators.flatten((
                ((child, :m) for child in bonds),
                ((child, :n) for child in Iterators.reverse(bonds))))
            for (child, center_on) in updates
                parent = topo.parent[child]
                move_center!(working, child; cache)
                merged = two_site_tensor(working, child, parent)
                effective = eff_h2(cache, working, H, child, parent)
                rhs_cache = Networks._FitCache(topo, nothing)
                local_rhs = Networks._fit_two_site_tensor(
                    rhs_cache, working, rhs, child, parent)
                local_solution, local_info = KrylovKit.linsolve(
                    effective, local_rhs, merged, a0T, a1T;
                    krylovdim=policy.krylovdim,
                    maxiter=policy.maxiter,
                    tol=policy.local_tol,
                    ishermitian=_shifted_ishermitian(H, a0T, a1T),
                    isposdef=false)
                solver_converged = local_info.converged > 0
                pre_residual = _two_site_local_residual(
                    effective, local_rhs, local_solution, a0T, a1T)
                if !solver_converged || !isfinite(pre_residual)
                    push!(edge_reports, TwoSiteLinearEdgeReport(
                        sweep, child, parent, center_on,
                        pre_residual, NaN, dim(virtualspace(working, child)),
                        0.0, 0.0, local_info.numiter, local_info.numops,
                        solver_converged))
                    reason = solver_converged ?
                        :nonfinite_local_residual : :local_solver_failed
                    return psi, TwoSiteLinearReport(
                        edge_reports, physical_residuals, false, reason,
                        false, nothing)
                end

                invalidate_edge!(cache, child, parent)
                split_two_site!(
                    working, local_solution, child, parent;
                    trunc=policy.trunc, center_on)
                reconstructed =
                    two_site_tensor(working, child, parent)
                discarded_norm =
                    Float64(norm(local_solution - reconstructed))
                solution_norm = Float64(norm(local_solution))
                discarded_weight =
                    iszero(solution_norm) ?
                    (iszero(discarded_norm) ? 0.0 : Inf) :
                    abs2(discarded_norm / solution_norm)
                post_residual = _two_site_local_residual(
                    effective, local_rhs, reconstructed, a0T, a1T)
                retained_rank = dim(virtualspace(working, child))
                push!(edge_reports, TwoSiteLinearEdgeReport(
                    sweep, child, parent, center_on,
                    pre_residual, post_residual, retained_rank,
                    discarded_norm, discarded_weight,
                    local_info.numiter, local_info.numops, true))
                if !(isfinite(post_residual) &&
                     isfinite(discarded_norm) &&
                     isfinite(discarded_weight))
                    return psi, TwoSiteLinearReport(
                        edge_reports, physical_residuals, false,
                        :nonfinite_truncation_result, false, nothing)
                end
            end

            residual =
                _linear_physical_residual(working, H, rhs, a0T, a1T)
            push!(physical_residuals, residual)
            verbose && @info "two_site_linsolve! sweep" sweep residual
            if !isfinite(residual)
                return psi, TwoSiteLinearReport(
                    edge_reports, physical_residuals, false,
                    :nonfinite_physical_residual, false, nothing)
            end
            residual <= policy.residual_tol && break
        end

        converged = physical_residuals[end] <= policy.residual_tol
        stop_reason = converged ? :converged :
                      _two_site_budget_stop_reason(edge_reports, policy)
        _commit_two_site_linear_state!(psi, working)
        return psi, TwoSiteLinearReport(
            edge_reports, physical_residuals, converged, stop_reason, true,
            nothing)
    catch err
        err isa InterruptException && rethrow()
        return psi, TwoSiteLinearReport(
            edge_reports, physical_residuals, false, :exception, false,
            sprint(showerror, err))
    end
end

function _check_two_site_linsolve_args(
        psi::TTNS, H::TTNO, rhs::TTNS, policy::TwoSiteLinearPolicy,
        a0::Number, a1::Number)
    _check_linsolve_args(
        psi, H, rhs, a0, a1, policy.krylovdim, policy.maxiter,
        policy.local_tol, policy.sweeps, 0.0)
    isfinite(a0) ||
        throw(ArgumentError("two_site_linsolve!: a0 must be finite"))
    isfinite(a1) ||
        throw(ArgumentError("two_site_linsolve!: a1 must be finite"))
    nnodes(topology(psi)) >= 2 ||
        throw(ArgumentError(
            "two_site_linsolve!: the topology must contain at least one edge"))
    return nothing
end

function _two_site_local_residual(
        effective, local_rhs, local_solution, a0::Number, a1::Number)
    residual = local_rhs - a0 * local_solution - a1 * effective(local_solution)
    return Float64(norm(residual))
end

function _two_site_budget_stop_reason(
        reports::Vector{TwoSiteLinearEdgeReport},
        policy::TwoSiteLinearPolicy)
    truncated = any(
        report -> report.discarded_weight > eps(Float64), reports)
    cap_exhausted =
        policy.trunc.maxdim < typemax(Int) &&
        any(report -> report.retained_rank >= policy.trunc.maxdim &&
                      report.discarded_weight > eps(Float64), reports)
    cap_exhausted && return :rank_cap_exhausted
    truncated && return :truncation_limited
    return :sweep_budget_exhausted
end

function _commit_two_site_linear_state!(destination::TTNS, source::TTNS)
    destination.tensors .= source.tensors
    destination.center = center(source)
    return destination
end
