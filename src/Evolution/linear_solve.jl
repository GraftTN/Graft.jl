# Variational full-state linear solves on the TTNS manifold. This reuses the
# GlobalKrylov vector wrapper and direct operator-aware fit compression so all
# linear-combination and matvec policies stay in one place.

const _LinInfo = NamedTuple{(:converged, :normres, :numiter, :numops),
                            Tuple{Int,Float64,Int,Int}}

"""
    linsolve!(ψ, H, rhs; a0=1, a1=1, krylovdim=30, maxiter=100,
              tol=1e-10, fit_nsweeps=4, fit_tol=1e-10) -> (ψ, info)

Solve `(a0 * I + a1 * H)ψ = rhs` on the fixed TTNS manifold carried by `ψ`.
The matrix-vector product contracts `H` directly into the public
operator-aware `fit!` path; KrylovKit GMRES handles the shifted system. The
state `ψ` is both the initial guess and the destination.
"""
function linsolve!(ψ::TTNS, H::TTNO, rhs::TTNS;
                   a0::Number=one(eltype(ψ)), a1::Number=one(eltype(ψ)),
                   krylovdim::Int=30, maxiter::Int=100, tol::Float64=1e-10,
                   fit_nsweeps::Int=4, fit_tol::Float64=1e-10,
                   fit_verbose::Bool=false)
    _check_linsolve_args(ψ, H, rhs, a0, a1, krylovdim, maxiter, tol,
                         fit_nsweeps, fit_tol)
    T = eltype(ψ)
    a0T, a1T = convert(T, a0), convert(T, a1)
    template = copy(ψ)
    b = _GKState(copy(rhs), template, fit_nsweeps, fit_tol, fit_verbose)
    x0 = _GKState(copy(ψ), template, fit_nsweeps, fit_tol, fit_verbose)
    op = _GKOperator(H, template, fit_nsweeps, fit_tol, fit_verbose)
    y, info = KrylovKit.linsolve(op, b, x0, a0T, a1T;
                                 krylovdim, maxiter, tol,
                                 ishermitian=_shifted_ishermitian(H, a0T, a1T),
                                 isposdef=false)
    _replace_state!(ψ, y.ψ)
    infoout = (; converged=info.converged,
               normres=Float64(info.normres),
               numiter=info.numiter,
               numops=info.numops)
    return ψ, infoout
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
    tol > 0 || throw(ArgumentError("linsolve!: tol must be positive"))
    fit_nsweeps >= 1 || throw(ArgumentError("linsolve!: fit_nsweeps must be positive"))
    fit_tol >= 0 || throw(ArgumentError("linsolve!: fit_tol must be nonnegative"))
    return nothing
end

_shifted_ishermitian(H::TTNO, a0::Number, a1::Number) =
    ishermitian(H) && isreal(a0) && isreal(a1)

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
arXiv:2606.02930. `tau_max / tau_first` must be a power of two.
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
    ratio = τmax / τ0
    isfinite(ratio) ||
        throw(ArgumentError("tau_max / tau_first must be finite"))
    npanels_after_first = round(Int, log2(ratio))
    isapprox(ratio, exp2(npanels_after_first); rtol=64eps(Float64), atol=0.0) ||
        throw(ArgumentError("tau_max / tau_first must be a power of two"))
    grid = Float64[0.0]
    left = 0.0
    for panel in 0:npanels_after_first
        right = ldexp(τ0, panel)
        append!(grid, range(left, right; length=Int(nsteps_per_panel) + 1)[2:end])
        left = right
    end
    grid[end] = τmax
    return grid
end

"""
    ImplicitLogTime(; scheme=LogTrapezoid(), krylovdim=30, maxiter=100,
                    tol=1e-10, fit_nsweeps=4, fit_tol=1e-10,
                    normalize=false)

A-stable implicit imaginary-time evolution on caller-supplied, possibly
logarithmic steps `dz = -δτ <= 0`. The default is the paper's trapezoid rule,

`(I + δτ H/2) ψ_new = (I - δτ H/2) ψ_old`.

`LogGaussLegendre(s)` advances one whole panel with `s` Gauss-Legendre
collocation nodes. `LogBackwardEuler()` selects the former first-order
behavior explicitly. TTNS bond spaces define the fixed variational manifold;
operator actions and all linear solves reuse `fit!` and `linsolve!`. As in the
paper, callers should shift a Hermitian Hamiltonian to nonnegative spectrum;
the scalar shift must be tracked separately when absolute normalization or
`logZ` is required.
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
    last_info::Union{Nothing,_LinInfo}
    last_stage_infos::Vector{_LinInfo}
    last_fit_error::Union{Nothing,Float64}
end

function ImplicitLogTime(; scheme::ImplicitLogScheme=LogTrapezoid(),
                         krylovdim::Int=30, maxiter::Int=100,
                         tol::Float64=1e-10, fit_nsweeps::Int=4,
                         fit_tol::Float64=1e-10,
                         fit_verbose::Bool=false, normalize::Bool=false)
    krylovdim >= 2 || throw(ArgumentError("ImplicitLogTime krylovdim must be at least 2"))
    maxiter >= 1 || throw(ArgumentError("ImplicitLogTime maxiter must be positive"))
    tol > 0 || throw(ArgumentError("ImplicitLogTime tol must be positive"))
    fit_nsweeps >= 1 || throw(ArgumentError("ImplicitLogTime fit_nsweeps must be positive"))
    fit_tol >= 0 || throw(ArgumentError("ImplicitLogTime fit_tol must be nonnegative"))
    return ImplicitLogTime{typeof(scheme)}(
        scheme, krylovdim, maxiter, tol, fit_nsweeps, fit_tol,
        fit_verbose, normalize, nothing, _LinInfo[], nothing,
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
        info = _zero_lininfo()
        _record_implicit_info!(ev, _LinInfo[info], nothing)
        return ψ
    end
    _implicit_step!(ev, ev.scheme, ψ, H, h)
    ev.normalize && normalize!(ψ)
    return ψ
end

function _implicit_step!(ev::ImplicitLogTime, ::LogBackwardEuler,
                         ψ::TTNS, H::TTNO, h::Real)
    rhs = copy(ψ)
    _, info = linsolve!(ψ, H, rhs; a0=one(eltype(ψ)), a1=h,
                        krylovdim=ev.krylovdim, maxiter=ev.maxiter,
                        tol=ev.tol, fit_nsweeps=ev.fit_nsweeps,
                        fit_tol=ev.fit_tol, fit_verbose=ev.fit_verbose)
    _record_implicit_info!(ev, _LinInfo[info], nothing)
    return ψ
end

function _implicit_step!(ev::ImplicitLogTime, ::LogTrapezoid,
                         ψ::TTNS, H::TTNO, h::Real)
    old = copy(ψ)
    rhs = copy(old)
    _, fit_errors = fit!(rhs, (old, old); Hs=(nothing, H),
                         coeffs=(one(eltype(ψ)), -h / 2),
                         nsweeps=ev.fit_nsweeps, tol=ev.fit_tol,
                         normalize=false, verbose=ev.fit_verbose)
    _replace_state!(ψ, rhs) # paper: use the right-hand side as initial guess
    _, info = linsolve!(ψ, H, rhs; a0=one(eltype(ψ)), a1=h / 2,
                        krylovdim=ev.krylovdim, maxiter=ev.maxiter,
                        tol=ev.tol, fit_nsweeps=ev.fit_nsweeps,
                        fit_tol=ev.fit_tol, fit_verbose=ev.fit_verbose)
    _record_implicit_info!(ev, _LinInfo[info], _last_fit_error(fit_errors))
    return ψ
end

function _implicit_step!(ev::ImplicitLogTime, scheme::LogGaussLegendre,
                         ψ::TTNS, H::TTNO, h::Real)
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
        _, info = linsolve!(stage, H, rhs; a0=one(eltype(ψ)),
                            a1=h * F.values[i],
                            krylovdim=ev.krylovdim, maxiter=ev.maxiter,
                            tol=ev.tol, fit_nsweeps=ev.fit_nsweeps,
                            fit_tol=ev.fit_tol, fit_verbose=ev.fit_verbose)
        stages[i] = stage
        infos[i] = info
    end
    sources = (old, stages...)
    operators = (nothing, ntuple(_ -> H, scheme.stages)...)
    coefficients = (one(eltype(ψ)),
                    (convert(eltype(ψ), -h * w) for w in endpoint_weights)...)
    _, fit_errors = fit!(ψ, sources; Hs=operators, coeffs=coefficients,
                         nsweeps=ev.fit_nsweeps, tol=ev.fit_tol,
                         normalize=false, verbose=ev.fit_verbose)
    _record_implicit_info!(ev, infos, _last_fit_error(fit_errors))
    return ψ
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
                                fit_error::Union{Nothing,Float64})
    ev.last_stage_infos = infos
    ev.last_info = (; converged=all(info.converged == 1 for info in infos) ? 1 : 0,
                    normres=maximum(info.normres for info in infos; init=0.0),
                    numiter=sum(info.numiter for info in infos; init=0),
                    numops=sum(info.numops for info in infos; init=0))
    ev.last_fit_error = fit_error
    return ev
end
