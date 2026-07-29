# Variational full-state linear solves on the TTNS manifold.  The TTNS set at
# fixed bond dimensions is not a vector space, so projecting every vector
# operation through fit! does not define the linear map required by GMRES.
# Instead, solve the variational normal equations one orthogonality center at
# a time.  Each local effective problem is genuinely linear.

const _LinInfo = NamedTuple{(:converged, :normres, :numiter, :numops),
                            Tuple{Int,Float64,Int,Int}}

struct _LocalLinearSolveFailure <: Exception
    node::Symbol
    reason::Symbol
end

function Base.showerror(io::IO, failure::_LocalLinearSolveFailure)
    description = if failure.reason === :local_solver_failed
        "local Krylov solve failed"
    elseif failure.reason === :nonfinite_local_solution
        "local Krylov solve produced a non-finite solution"
    else
        "local linear solve failed ($(failure.reason))"
    end
    print(io, "linsolve!: ", description, " at node ", failure.node)
end

"""
Local projected linear operator in the pivotal coordinates of the fit
projection. `normal` is the self-fit overlap map and `effective` is the
one-site Hamiltonian map, so this callable represents `a0*N + a1*Heff`.
"""
struct _LocalNormalEquationMap{N,H,C<:Number}
    normal::N
    effective::H
    a0::C
    a1::C
end

function (map::_LocalNormalEquationMap)(x::AbstractTensorMap)
    iszero(map.a0) && return map.a1 * map.effective(x)
    iszero(map.a1) && return map.a0 * map.normal(x)
    return map.a0 * map.normal(x) + map.a1 * map.effective(x)
end

"""
    linsolve!(ψ, H, rhs; a0=1, a1=1, krylovdim=30, maxiter=100,
              tol=1e-10, fit_nsweeps=4, fit_tol=1e-10) -> (ψ, info)

Solve `(a0 * I + a1 * H)ψ = rhs` on the fixed TTNS manifold carried by `ψ`.
One-site alternating sweeps solve the effective linear equation at each
orthogonality center.  `fit_nsweeps` is the maximum number of full alternating
sweeps; `krylovdim` and `maxiter` configure each local Krylov solve.  The
reported residual is the full Hilbert-space norm of
`rhs - a0*ψ - a1*H*ψ`, evaluated without variational compression.  The state
`ψ` is both the initial guess and the destination.
"""
function linsolve!(ψ::TTNS, H::TTNO, rhs::TTNS;
                   a0::Number=one(eltype(ψ)), a1::Number=one(eltype(ψ)),
                   krylovdim::Int=30, maxiter::Int=100, tol::Float64=1e-10,
                   fit_nsweeps::Int=4, fit_tol::Float64=1e-10,
                   fit_verbose::Bool=false,
                   _fail_on_local::Bool=false,
                   _root_first::Bool=false)
    _check_linsolve_args(ψ, H, rhs, a0, a1, krylovdim, maxiter, tol,
                         fit_nsweeps, fit_tol)
    T = eltype(ψ)
    a0T, a1T = convert(T, a0), convert(T, a1)
    cache = EnvCache(topology(ψ))
    order = postorder(topology(ψ))
    total_iterations = 0
    total_operations = 0
    residual = Inf
    for sweep in 1:fit_nsweeps
        directions = _root_first ?
            (Iterators.reverse(order), order) :
            (order, Iterators.reverse(order))
        for n in Iterators.flatten(directions)
            move_center!(ψ, n; cache)
            effective = eff_h1(cache, ψ, H, n)
            rhs_cache = Networks._FitCache(topology(ψ), nothing)
            local_rhs = Networks._fit_local_tensor(
                [rhs_cache], ψ, (rhs,), T[one(T)], n)
            local_solution, local_info = if _root_first
                # A root-first sweep can encounter a nontrivial pivotal
                # self-overlap before any leaf update transports that
                # coordinate change. Build a fresh normal map from the current
                # state at every node and solve the full projected equation.
                normal = Networks._fit_local_normal_map(ψ, n)
                combined = _LocalNormalEquationMap(
                    normal, effective, a0T, a1T)
                KrylovKit.linsolve(
                    combined, local_rhs, ψ.tensors[n], zero(T), one(T);
                    krylovdim, maxiter, tol,
                    ishermitian=_shifted_ishermitian(H, a0T, a1T),
                    isposdef=false)
            else
                KrylovKit.linsolve(
                    effective, local_rhs, ψ.tensors[n], a0T, a1T;
                    krylovdim, maxiter, tol,
                    ishermitian=_shifted_ishermitian(H, a0T, a1T),
                    isposdef=false)
            end
            total_iterations += local_info.numiter
            total_operations += local_info.numops
            _fail_on_local && local_info.converged <= 0 &&
                throw(_LocalLinearSolveFailure(
                    nodeid(topology(ψ), n), :local_solver_failed))
            _fail_on_local && !isfinite(Float64(norm(local_solution))) &&
                throw(_LocalLinearSolveFailure(
                    nodeid(topology(ψ), n), :nonfinite_local_solution))
            update_tensor!(ψ, n, local_solution; caches=(cache,))
        end
        residual = _linear_physical_residual(ψ, H, rhs, a0T, a1T)
        fit_verbose && @info "linsolve! ALS sweep" sweep residual
        residual <= tol && break
    end
    infoout = (; converged=Int(residual <= tol),
               normres=Float64(residual),
               numiter=total_iterations,
               numops=total_operations)
    return ψ, infoout
end

function _linear_physical_residual(ψ::TTNS, H::TTNO, rhs::TTNS,
                                   a0::Number, a1::Number)
    acted = apply(H, ψ; center=center(ψ))
    residual = exact_linear_combination(
        [rhs, ψ, acted], [one(eltype(ψ)), -a0, -a1];
        max_bond=4096, max_payload=100_000_000)
    return Float64(norm(residual))
end

function _check_linsolve_args(ψ::TTNS, H::TTNO, rhs::TTNS, a0::Number,
                              a1::Number, krylovdim::Int, maxiter::Int,
                              tol::Float64, fit_nsweeps::Int,
                              fit_tol::Float64)
    topology(ψ) == topology(H) == topology(rhs) ||
        throw(ArgumentError("linsolve!: ψ, H, and rhs must share topology"))
    ψ.hasphys == H.hasphys == rhs.hasphys ||
        throw(ArgumentError("linsolve!: ψ, H, and rhs must share physical layout"))
    spacetype(ψ) == spacetype(H) == spacetype(rhs) ||
        throw(ArgumentError("linsolve!: ψ, H, and rhs must share spacetype"))
    eltype(ψ) == eltype(rhs) ||
        throw(ArgumentError("linsolve!: ψ and rhs must have the same eltype; convert explicitly"))
    if !(eltype(ψ) <: Complex)
        ((a0 isa Complex && !isreal(a0)) ||
         (a1 isa Complex && !isreal(a1)) ||
         (eltype(H) <: Complex)) &&
            throw(ArgumentError("linsolve!: real-eltype ψ cannot solve a complex operator/system without explicit conversion"))
    end
    krylovdim >= 2 || throw(ArgumentError("linsolve!: krylovdim must be at least 2"))
    maxiter >= 1 || throw(ArgumentError("linsolve!: maxiter must be positive"))
    isfinite(tol) && tol > 0 ||
        throw(ArgumentError("linsolve!: tol must be finite and positive"))
    fit_nsweeps >= 1 || throw(ArgumentError("linsolve!: fit_nsweeps must be positive"))
    isfinite(fit_tol) && fit_tol >= 0 ||
        throw(ArgumentError("linsolve!: fit_tol must be finite and nonnegative"))
    return nothing
end

_shifted_ishermitian(H::TTNO, a0::Number, a1::Number) =
    ishermitian(H) && isreal(a0) && isreal(a1)

include("adaptive_linear_solve.jl")
include("two_site_linear_solve.jl")

"""Time discretization used by [`ImplicitLogTime`](@ref)."""
abstract type ImplicitLogScheme end

"""First-order A/L-stable backward Euler; retained as an explicit fallback."""
struct LogBackwardEuler <: ImplicitLogScheme end

"""
Second-order A-stable implicit trapezoid rule from arXiv:2606.02930,
Eq. (3). This is the default [`ImplicitLogTime`](@ref) scheme.
"""
struct LogTrapezoid <: ImplicitLogScheme end

"""
    LogGaussLegendre(stages=4)

Gauss-Legendre collocation on one time panel. `stages` collocation nodes give
classical order `2stages`; increasing `stages` is the paper's spectrally
accurate route for analytic imaginary-time trajectories. The paper specifies
Gauss-Legendre collocation but defers its implementation details; this type
uses the standard collocation Runge-Kutta system and solves its diagonalized
stage equations with [`linsolve!`](@ref).
"""
struct LogGaussLegendre <: ImplicitLogScheme
    stages::Int
    function LogGaussLegendre(stages::Int)
        stages >= 1 || throw(ArgumentError("LogGaussLegendre stages must be positive"))
        return new(stages)
    end
end
LogGaussLegendre(; stages::Int=4) = LogGaussLegendre(stages)

"""
    logarithmic_time_grid(tau_first, tau_max; nsteps_per_panel=1)

Paper-style logarithmic imaginary-time grid. The first panel is
`[0, tau_first]`; subsequent panel widths double until `tau_max`. Each panel is
split into `nsteps_per_panel` uniform steps, as used for the trapezoid rule in
arXiv:2606.02930. If `tau_max` lies inside the next dyadic panel, that final
panel is shortened while retaining the requested number of steps.
"""
function logarithmic_time_grid(tau_first::Real, tau_max::Real;
                               nsteps_per_panel::Integer=1)
    τ0, τmax = Float64(tau_first), Float64(tau_max)
    isfinite(τ0) && τ0 > 0 ||
        throw(ArgumentError("tau_first must be finite and positive"))
    isfinite(τmax) && τmax >= τ0 ||
        throw(ArgumentError("tau_max must be finite and at least tau_first"))
    nsteps_per_panel >= 1 ||
        throw(ArgumentError("nsteps_per_panel must be positive"))
    grid = Float64[0.0]
    left = 0.0
    panel = 0
    while left < τmax
        dyadic_right = ldexp(τ0, panel)
        isfinite(dyadic_right) ||
            throw(ArgumentError("logarithmic grid panel endpoint overflowed"))
        right = min(dyadic_right, τmax)
        append!(grid, range(left, right; length=Int(nsteps_per_panel) + 1)[2:end])
        left = right
        panel += 1
    end
    grid[end] = τmax
    return grid
end

"""
    ImplicitLogTime(; scheme=LogTrapezoid(), krylovdim=30, maxiter=100,
                    tol=1e-10, fit_nsweeps=4, fit_tol=1e-10,
                    expansion=nothing, two_site=nothing,
                    normalize=false, energy_shift=false)

A-stable implicit imaginary-time evolution on caller-supplied, possibly
logarithmic steps `dz = -δτ <= 0`. The default is the paper's trapezoid rule,

`(I + δτ H/2) ψ_new = (I - δτ H/2) ψ_old`.

`LogGaussLegendre(s)` advances one whole panel with `s` Gauss-Legendre
collocation nodes. `LogBackwardEuler()` selects the former first-order
behavior explicitly. TTNS bond spaces define the fixed variational manifold;
operator actions and all linear solves reuse `fit!` and `linsolve!`.
`expansion=ResidualDrivenExpansion(...)` opts into the production
residual-driven adaptive one-site solve, while
`two_site=TwoSiteLinearPolicy(...)` independently opts into the diagnostic
two-site solve. The two policies are mutually exclusive, and leaving both
unset preserves the fixed-manifold solver. Adaptive and two-site policies are
currently supported only with `LogBackwardEuler` and `LogTrapezoid`;
`LogGaussLegendre` retains its fixed-manifold stage solves because projecting
rank-grown stages back through its endpoint fit would discard that growth.

As in the paper, callers should shift a Hermitian Hamiltonian to nonnegative
spectrum; the scalar shift must be tracked separately when absolute
normalization or `logZ` is required. With `energy_shift=true`, every step
replaces `H` by `H - <H>_psi I`, using the normalized Rayleigh quotient at the
beginning of the step. This is the norm-drift control used by the
logarithmic-time METTS calculation; `last_shift` records the applied scalar.
"""
mutable struct ImplicitLogTime{S<:ImplicitLogScheme} <: Evolver
    scheme::S
    krylovdim::Int
    maxiter::Int
    tol::Float64
    fit_nsweeps::Int
    fit_tol::Float64
    fit_verbose::Bool
    normalize::Bool
    energy_shift::Bool
    expansion::Union{Nothing,ResidualDrivenExpansion}
    two_site::Union{Nothing,TwoSiteLinearPolicy}
    last_info::Union{Nothing,_LinInfo}
    last_stage_infos::Vector{_LinInfo}
    last_residual_driven_reports::Vector{ResidualDrivenReport}
    last_two_site_reports::Vector{TwoSiteLinearReport}
    last_fit_error::Union{Nothing,Float64}
    last_shift::Float64
end

function ImplicitLogTime(; scheme::ImplicitLogScheme=LogTrapezoid(),
                         krylovdim::Int=30, maxiter::Int=100,
                         tol::Float64=1e-10, fit_nsweeps::Int=4,
                         fit_tol::Float64=1e-10,
                         fit_verbose::Bool=false, normalize::Bool=false,
                         energy_shift::Bool=false,
                         expansion::Union{Nothing,ResidualDrivenExpansion}=nothing,
                         two_site::Union{Nothing,TwoSiteLinearPolicy}=nothing)
    expansion === nothing || two_site === nothing ||
        throw(ArgumentError(
            "ImplicitLogTime expansion and two_site are mutually exclusive"))
    scheme isa LogGaussLegendre &&
        (expansion !== nothing || two_site !== nothing) &&
        throw(ArgumentError(
            "ImplicitLogTime adaptive and two-site policies are not supported " *
            "with LogGaussLegendre"))
    krylovdim >= 2 || throw(ArgumentError("ImplicitLogTime krylovdim must be at least 2"))
    maxiter >= 1 || throw(ArgumentError("ImplicitLogTime maxiter must be positive"))
    isfinite(tol) && tol > 0 ||
        throw(ArgumentError("ImplicitLogTime tol must be finite and positive"))
    fit_nsweeps >= 1 || throw(ArgumentError("ImplicitLogTime fit_nsweeps must be positive"))
    isfinite(fit_tol) && fit_tol >= 0 ||
        throw(ArgumentError(
            "ImplicitLogTime fit_tol must be finite and nonnegative"))
    if two_site !== nothing
        two_site.krylovdim == krylovdim ||
            throw(ArgumentError(
                "ImplicitLogTime two_site.krylovdim must equal krylovdim"))
        two_site.maxiter == maxiter ||
            throw(ArgumentError(
                "ImplicitLogTime two_site.maxiter must equal maxiter"))
        two_site.local_tol == tol ||
            throw(ArgumentError(
                "ImplicitLogTime two_site.local_tol must equal tol"))
        two_site.residual_tol == tol ||
            throw(ArgumentError(
                "ImplicitLogTime two_site.residual_tol must equal tol"))
        two_site.sweeps == fit_nsweeps ||
            throw(ArgumentError(
                "ImplicitLogTime two_site.sweeps must equal fit_nsweeps"))
    end
    return ImplicitLogTime{typeof(scheme)}(
        scheme, krylovdim, maxiter, tol, fit_nsweeps, fit_tol,
        fit_verbose, normalize, energy_shift, expansion, two_site,
        nothing, _LinInfo[], ResidualDrivenReport[], TwoSiteLinearReport[],
        nothing, 0.0,
    )
end

supports_complex_step(::Type{<:ImplicitLogTime}) = false

function step!(ev::ImplicitLogTime, ψ::TTNS, H::TTNO, dz::Number)
    isreal(dz) ||
        throw(ArgumentError("ImplicitLogTime accepts only real imaginary-time steps dz = -δτ"))
    δ = real(dz)
    isfinite(δ) || throw(ArgumentError("ImplicitLogTime expects finite dz"))
    δ <= 0 ||
        throw(ArgumentError("ImplicitLogTime expects dz <= 0"))
    topology(ψ) == topology(H) ||
        throw(ArgumentError("ImplicitLogTime: H and ψ have mismatched topologies"))
    ψ.hasphys == H.hasphys ||
        throw(ArgumentError("ImplicitLogTime: H and ψ have mismatched physical layout"))
    spacetype(ψ) == spacetype(H) ||
        throw(ArgumentError("ImplicitLogTime: H and ψ have mismatched spacetype"))
    h = -δ
    if iszero(h)
        ev.last_shift = 0.0
        info = _zero_lininfo()
        _record_implicit_info!(ev, _LinInfo[info], nothing)
        return ψ
    end
    shift = if ev.energy_shift
        ishermitian(H) ||
            throw(ArgumentError("ImplicitLogTime energy_shift requires a Hermitian TTNO"))
        quotient = expect(ψ, H) / inner(ψ, ψ)
        abs(imag(quotient)) <= sqrt(eps(Float64)) *
            max(abs(real(quotient)), 1.0) ||
            throw(ArgumentError(
                "ImplicitLogTime energy shift has a non-real Rayleigh quotient: $quotient"))
        Float64(real(quotient))
    else
        0.0
    end
    ev.last_shift = shift
    _implicit_step!(ev, ev.scheme, ψ, H, h, shift)
    ev.normalize && _implicit_result_committed(ev) && normalize!(ψ)
    return ψ
end

function _implicit_result_committed(ev::ImplicitLogTime)
    if ev.expansion !== nothing
        return all(report.committed
                   for report in ev.last_residual_driven_reports)
    elseif ev.two_site !== nothing
        return all(report.transaction_committed
                   for report in ev.last_two_site_reports)
    end
    return true
end

function _implicit_step!(ev::ImplicitLogTime, ::LogBackwardEuler,
                         ψ::TTNS, H::TTNO, h::Real, shift::Real)
    rhs = copy(ψ)
    info, residual_driven_reports, two_site_reports =
        _implicit_linsolve!(
            ev, ψ, H, rhs;
            a0=one(eltype(ψ)) - h * shift, a1=h)
    _record_implicit_info!(
        ev, _LinInfo[info], nothing;
        residual_driven_reports, two_site_reports)
    return ψ
end

function _implicit_step!(ev::ImplicitLogTime, ::LogTrapezoid,
                         ψ::TTNS, H::TTNO, h::Real, shift::Real)
    old = copy(ψ)
    acted = apply(H, old; center=center(old))
    rhs = exact_linear_combination(
        [old, acted],
        [one(eltype(ψ)) + h * shift / 2, -h / 2];
        max_bond=4096, max_payload=100_000_000)
    info, residual_driven_reports, two_site_reports =
        _implicit_linsolve!(
            ev, ψ, H, rhs;
            a0=one(eltype(ψ)) - h * shift / 2, a1=h / 2)
    _record_implicit_info!(
        ev, _LinInfo[info], nothing;
        residual_driven_reports, two_site_reports)
    return ψ
end

function _implicit_step!(ev::ImplicitLogTime, scheme::LogGaussLegendre,
                         ψ::TTNS, H::TTNO, h::Real, shift::Real)
    eltype(ψ) <: Complex ||
        throw(ArgumentError("LogGaussLegendre requires a complex-eltype TTNS for its conjugate-paired shifted solves"))
    A, b, _ = _gauss_legendre_tableau(scheme.stages)
    F = LinearAlgebra.eigen(A)
    V = F.vectors
    LinearAlgebra.cond(V) <= inv(sqrt(eps(Float64))) ||
        throw(ArgumentError("Gauss-Legendre stage diagonalization is ill-conditioned for $(scheme.stages) stages"))
    q = V \ ones(ComplexF64, scheme.stages)
    endpoint_weights = vec(transpose(b) * V)
    old = copy(ψ)
    stages = Vector{typeof(ψ)}(undef, scheme.stages)
    infos = Vector{_LinInfo}(undef, scheme.stages)
    for i in 1:scheme.stages
        rhs = _scaled_copy(old, q[i])
        stage = copy(rhs)
        _, info = linsolve!(
            stage, H, rhs;
            a0=one(eltype(ψ)) - h * F.values[i] * shift,
            a1=h * F.values[i],
            krylovdim=ev.krylovdim,
            maxiter=ev.maxiter,
            tol=ev.tol,
            fit_nsweeps=ev.fit_nsweeps,
            fit_tol=ev.fit_tol,
            fit_verbose=ev.fit_verbose)
        stages[i] = stage
        infos[i] = info
    end
    sources = (old, stages..., stages...)
    operators = (
        nothing,
        ntuple(_ -> nothing, scheme.stages)...,
        ntuple(_ -> H, scheme.stages)...)
    coefficients = (
        one(eltype(ψ)),
        (convert(eltype(ψ), h * shift * w) for w in endpoint_weights)...,
        (convert(eltype(ψ), -h * w) for w in endpoint_weights)...)
    _, fit_errors = fit!(ψ, sources; Hs=operators, coeffs=coefficients,
                         nsweeps=ev.fit_nsweeps, tol=ev.fit_tol,
                         normalize=false, verbose=ev.fit_verbose)
    _record_implicit_info!(ev, infos, _last_fit_error(fit_errors))
    return ψ
end

function _implicit_linsolve!(
        ev::ImplicitLogTime, ψ::TTNS, H::TTNO, rhs::TTNS;
        a0::Number, a1::Number)
    if ev.expansion !== nothing
        _, report = residual_driven_linsolve!(
            ψ, H, rhs, ev.expansion;
            a0, a1,
            krylovdim=ev.krylovdim,
            maxiter=ev.maxiter,
            tol=ev.tol,
            fit_nsweeps=ev.fit_nsweeps,
            fit_tol=ev.fit_tol,
            fit_verbose=ev.fit_verbose)
        return _residual_driven_lininfo(report),
            ResidualDrivenReport[report], TwoSiteLinearReport[]
    elseif ev.two_site !== nothing
        _, report = two_site_linsolve!(
            ψ, H, rhs, ev.two_site; a0, a1, verbose=ev.fit_verbose)
        return _two_site_lininfo(report),
            ResidualDrivenReport[], TwoSiteLinearReport[report]
    end

    _, info = linsolve!(
        ψ, H, rhs;
        a0, a1,
        krylovdim=ev.krylovdim,
        maxiter=ev.maxiter,
        tol=ev.tol,
        fit_nsweeps=ev.fit_nsweeps,
        fit_tol=ev.fit_tol,
        fit_verbose=ev.fit_verbose)
    return info, ResidualDrivenReport[], TwoSiteLinearReport[]
end

function _residual_driven_lininfo(report::ResidualDrivenReport)
    normres = isempty(report.physical_residuals) ?
        Inf : last(report.physical_residuals)
    return (; converged=Int(report.converged),
            normres=Float64(normres),
            numiter=sum(info.numiter for info in report.solves; init=0),
            numops=sum(info.numops for info in report.solves; init=0))
end

function _two_site_lininfo(report::TwoSiteLinearReport)
    normres = isempty(report.physical_residuals) ?
        Inf : last(report.physical_residuals)
    return (; converged=Int(report.converged),
            normres=Float64(normres),
            numiter=sum(edge.solver_iterations for edge in report.edge_reports;
                        init=0),
            numops=sum(edge.solver_operations for edge in report.edge_reports;
                       init=0))
end

function _gauss_legendre_tableau(stages::Int)
    offdiag = [k / sqrt(4k^2 - 1) for k in 1:(stages - 1)]
    J = LinearAlgebra.SymTridiagonal(zeros(stages), offdiag)
    F = LinearAlgebra.eigen(J)
    order = sortperm(F.values)
    nodes = F.values[order]
    vectors = F.vectors[:, order]
    c = (nodes .+ 1) ./ 2
    b = abs2.(vectors[1, :])
    vandermonde = [c[i]^(m - 1) for i in 1:stages, m in 1:stages]
    lagrange_coeffs = vandermonde \ Matrix{Float64}(LinearAlgebra.I, stages, stages)
    A = zeros(Float64, stages, stages)
    for i in 1:stages, j in 1:stages, m in 1:stages
        A[i, j] += lagrange_coeffs[m, j] * c[i]^m / m
    end
    return A, b, c
end

function _scaled_copy(ψ::TTNS{S,T}, α::Number) where {S,T}
    φ = copy(ψ)
    n = center(φ)
    update_tensor!(φ, n, convert(T, α) * φ.tensors[n])
    return φ
end

_zero_lininfo() = (; converged=1, normres=0.0, numiter=0, numops=0)
_last_fit_error(errors) = isempty(errors) ? 0.0 : Float64(errors[end])

function _record_implicit_info!(ev::ImplicitLogTime, infos::Vector{_LinInfo},
                                fit_error::Union{Nothing,Float64};
                                residual_driven_reports::Vector{ResidualDrivenReport}=
                                    ResidualDrivenReport[],
                                two_site_reports::Vector{TwoSiteLinearReport}=
                                    TwoSiteLinearReport[])
    ev.last_stage_infos = infos
    ev.last_info = (; converged=all(info.converged == 1 for info in infos) ? 1 : 0,
                    normres=maximum(info.normres for info in infos; init=0.0),
                    numiter=sum(info.numiter for info in infos; init=0),
                    numops=sum(info.numops for info in infos; init=0))
    ev.last_residual_driven_reports = residual_driven_reports
    ev.last_two_site_reports = two_site_reports
    ev.last_fit_error = fit_error
    return ev
end
