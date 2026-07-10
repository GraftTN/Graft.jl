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

"""
    ImplicitLogTime(; krylovdim=30, maxiter=100, tol=1e-10,
                    fit_nsweeps=4, fit_tol=1e-10, normalize=false)

Backward-Euler imaginary-time stepper on a logarithmic-time-compatible surface:
for `dz = -δτ <= 0`, one step solves `(I - dz * H) ψ_new = ψ_old`. This is the
first production consumer of [`linsolve!`](@ref); logarithmic grid scheduling
is supplied by the caller through the sequence of `dz` values.
"""
Base.@kwdef mutable struct ImplicitLogTime <: Evolver
    krylovdim::Int = 30
    maxiter::Int = 100
    tol::Float64 = 1e-10
    fit_nsweeps::Int = 4
    fit_tol::Float64 = 1e-10
    fit_verbose::Bool = false
    normalize::Bool = false
    last_info::Union{Nothing,_LinInfo} = nothing
end
supports_complex_step(::Type{ImplicitLogTime}) = false

function step!(ev::ImplicitLogTime, ψ::TTNS, H::TTNO, dz::Number)
    isreal(dz) ||
        throw(ArgumentError("ImplicitLogTime accepts only real imaginary-time steps dz = -δτ"))
    δ = real(dz)
    δ <= 0 ||
        throw(ArgumentError("ImplicitLogTime expects dz <= 0 for ψ <- (I - dz*H)^-1 ψ"))
    rhs = copy(ψ)
    _, info = linsolve!(ψ, H, rhs; a0=one(eltype(ψ)), a1=-δ,
                        krylovdim=ev.krylovdim, maxiter=ev.maxiter,
                        tol=ev.tol, fit_nsweeps=ev.fit_nsweeps,
                        fit_tol=ev.fit_tol, fit_verbose=ev.fit_verbose)
    ev.last_info = info
    ev.normalize && normalize!(ψ)
    return ψ
end
