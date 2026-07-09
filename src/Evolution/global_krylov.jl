# Global full-state Krylov evolution. The Krylov vectors are TTNS objects on a
# fixed target manifold; linear combinations and H applications are projected
# back with Networks.fit!, making the compression policy explicit on the
# evolver instead of relying on hidden global state.

const _VI = KrylovKit.VectorInterface
const _GKInfo = NamedTuple{(:converged, :normres, :numiter, :numops),
                           Tuple{Int,Float64,Int,Int}}

"""
    GlobalKrylov(; krylovdim=30, maxiter=100, tol=1e-12,
                 fit_nsweeps=4, fit_tol=1e-10, fit_verbose=false,
                 eager=false)

Full-state Krylov evolver (§5b) using KrylovKit's Lanczos/Arnoldi exponential
and the public `apply(H, ψ)` + `fit!` compression primitives. The TTNS bond
spaces of the input state define the fixed compression manifold for the step;
start from the desired target bond dimensions before calling this evolver.
"""
Base.@kwdef mutable struct GlobalKrylov <: Evolver
    krylovdim::Int = 30
    maxiter::Int = 100
    tol::Float64 = 1e-12
    fit_nsweeps::Int = 4
    fit_tol::Float64 = 1e-10
    fit_verbose::Bool = false
    eager::Bool = false
    last_info::Union{Nothing,_GKInfo} = nothing
end

struct _GKState{S<:ElementarySpace,T<:Number}
    ψ::TTNS{S,T}
    template::TTNS{S,T}
    fit_nsweeps::Int
    fit_tol::Float64
    fit_verbose::Bool
end

struct _GKOperator{S<:ElementarySpace,T<:Number,O<:TTNO}
    H::O
    template::TTNS{S,T}
    fit_nsweeps::Int
    fit_tol::Float64
    fit_verbose::Bool
end

function step!(ev::GlobalKrylov, ψ::TTNS, H::TTNO, dz::Number)
    _check_global_krylov(ev, ψ, H, dz)
    template = copy(ψ)
    x0 = _GKState(copy(ψ), template, ev.fit_nsweeps, ev.fit_tol, ev.fit_verbose)
    op = _GKOperator(H, template, ev.fit_nsweeps, ev.fit_tol, ev.fit_verbose)
    y, info = exponentiate(op, dz, x0;
                           ishermitian=ishermitian(H),
                           krylovdim=ev.krylovdim,
                           maxiter=ev.maxiter,
                           tol=ev.tol,
                           eager=ev.eager)
    ev.last_info = (; converged=info.converged,
                    normres=Float64(info.normres),
                    numiter=info.numiter,
                    numops=info.numops)
    _replace_state!(ψ, y.ψ)
    return ψ
end

function _check_global_krylov(ev::GlobalKrylov, ψ::TTNS, H::TTNO, dz::Number)
    topology(ψ) == topology(H) ||
        throw(ArgumentError("GlobalKrylov: H and ψ have mismatched topologies"))
    ψ.hasphys == H.hasphys ||
        throw(ArgumentError("GlobalKrylov: H and ψ have mismatched physical layout"))
    spacetype(ψ) == spacetype(H) ||
        throw(ArgumentError("GlobalKrylov: H and ψ have mismatched spacetype"))
    ev.krylovdim >= 2 || throw(ArgumentError("GlobalKrylov: krylovdim must be at least 2"))
    ev.maxiter >= 1 || throw(ArgumentError("GlobalKrylov: maxiter must be positive"))
    ev.fit_nsweeps >= 1 || throw(ArgumentError("GlobalKrylov: fit_nsweeps must be positive"))
    ev.tol > 0 || throw(ArgumentError("GlobalKrylov: tol must be positive"))
    ev.fit_tol >= 0 || throw(ArgumentError("GlobalKrylov: fit_tol must be nonnegative"))
    if !(eltype(ψ) <: Complex) && dz isa Complex && !isreal(dz)
        throw(ArgumentError("GlobalKrylov complex-step evolution requires a complex-eltype TTNS; convert explicitly"))
    end
    return nothing
end

function (op::_GKOperator)(x::_GKState)
    φ = copy(op.template)
    Hx = apply(op.H, x.ψ; center=center(φ))
    fit!(φ, Hx; nsweeps=op.fit_nsweeps, tol=op.fit_tol,
         normalize=false, verbose=op.fit_verbose)
    return _GKState(φ, op.template, op.fit_nsweeps, op.fit_tol, op.fit_verbose)
end

function _replace_state!(dst::TTNS, src::TTNS)
    topology(dst) == topology(src) || throw(ArgumentError("state topologies differ"))
    dst.hasphys == src.hasphys || throw(ArgumentError("state physical layouts differ"))
    spacetype(dst) == spacetype(src) || throw(ArgumentError("state spacetypes differ"))
    for n in 1:nnodes(dst.topo)
        update_tensor!(dst, n, copy(src.tensors[n]); gauge=false)
    end
    dst.center = center(src)
    check_arrows(dst)
    return dst
end

_gk_wrap(ψ::TTNS, x::_GKState) =
    _GKState(ψ, x.template, x.fit_nsweeps, x.fit_tol, x.fit_verbose)

function _gk_zero_state(x::_GKState{S,T}) where {S,T}
    φ = copy(x.template)
    n = center(φ)
    A = φ.tensors[n]
    update_tensor!(φ, n, zeros(T, codomain(A) ← domain(A)))
    return _gk_wrap(φ, x)
end

_VI.scalartype(::Type{<:_GKState{S,T}}) where {S,T} = T

function _VI.zerovector(x::_GKState{S,T}, ::Type{R}) where {S,T,R<:Number}
    R == T || throw(ArgumentError("GlobalKrylov vector scalar promotion from $T to $R is not implicit; use an explicit TTNS eltype conversion"))
    return _gk_zero_state(x)
end
_VI.zerovector!!(x::_GKState) = _gk_zero_state(x)

function _VI.scale(x::_GKState{S,T}, α::Number) where {S,T}
    αT = convert(T, α)
    φ = copy(x.ψ)
    n = center(φ)
    update_tensor!(φ, n, αT * φ.tensors[n])
    return _gk_wrap(φ, x)
end
_VI.scale!!(x::_GKState, α::Number) = _VI.scale(x, α)
_VI.scale!!(y::_GKState, x::_GKState, α::Number) = _VI.scale(x, α)

function _VI.add(y::_GKState{S,T}, x::_GKState{S,T},
                 α::Number, β::Number) where {S,T}
    αT, βT = convert(T, α), convert(T, β)
    φ = copy(y.ψ)
    fit!(φ, (x.ψ, y.ψ); coeffs=(αT, βT),
         nsweeps=y.fit_nsweeps, tol=y.fit_tol,
         normalize=false, verbose=y.fit_verbose)
    return _gk_wrap(φ, y)
end
_VI.add!!(y::_GKState, x::_GKState, α::Number, β::Number) =
    _VI.add(y, x, α, β)

_VI.inner(x::_GKState, y::_GKState) = inner(x.ψ, y.ψ)
LinearAlgebra.norm(x::_GKState) = sqrt(real(_VI.inner(x, x)))
