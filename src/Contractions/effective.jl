# Effective Hamiltonians (PyTreeNet: contractions/effective_hamiltonians.py).
#
# All three return *linear maps* (closures) suitable for KrylovKit
# (eigsolve/exponentiate/linsolve); required environments are built/cached on
# construction. Leg bookkeeping matches the input tensor exactly: the closure
# maps a tensor to one with identical space structure.
#
# TODO(MPI extension, §8 level 2): operator-term-level MPI Allreduce lives
# exactly here. It is outside the forwarded B1-B5/M0-local threading surface;
# H_eff·x splits over TTNO virtual-bond blocks, and DMRG/TDVP should share it
# when the MPI extension milestone is opened.

# open-leg label bookkeeping: result leg i of the applied map gets label -i.

"""
    eff_h1(cache, ψ, H, n) -> x -> H_eff·x

One-site effective Hamiltonian at node `n`: environments from every neighbour
+ the local TTNO tensor. `x` must have the structure of `ψ[n]`.
"""
function eff_h1(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int)
    t = ψ.topo
    hp = hasphys(ψ, n)
    W = H.tensors[n]
    envlist = [(w, env!(cache, ψ, H, w, n)) for w in neighbors(t, n)]
    isroot_ = t.parent[n] == 0
    K = nchildren(t, n)

    function apply1(x::AbstractTensorMap)
        Nx = numind(x)
        xidx = zeros(Int, Nx)
        widx = zeros(Int, numind(W))
        tensors = Any[x, W]
        indices = Vector{Int}[xidx, widx]
        conjs = Bool[false, false]
        nxt = Ref(0)
        fresh() = (nxt[] += 1; nxt[])

        if hp
            pin = fresh()
            xidx[K + 1] = pin
            widx[K + 2] = pin
            widx[K + 1] = -(K + 1)          # open physical (out) leg
        end
        for (w, E) in envlist
            lx = _stateleg(t, hp, n, w)
            kk, oo = fresh(), fresh()
            xidx[lx] = kk
            widx[_opleg(t, hp, n, w)] = oo
            push!(tensors, E); push!(indices, [kk, oo, -lx]); push!(conjs, false)
        end
        if isroot_
            ka, ko = fresh(), fresh()
            xidx[end] = ka
            widx[end] = ko
            cap = ones_tensor(scalartype(x), domain(x)[1] ⊗ domain(W)[numin(W)] ⊗ dual(domain(x)[1]))
            push!(tensors, cap); push!(indices, [ka, ko, -Nx]); push!(conjs, false)
        end
        y = ncon(tensors, indices, conjs)
        return repartition(y, Nx - 1, 1)
    end
    return apply1
end

"""
    eff_h0(cache, ψ, H, n, m) -> C -> K_eff·C

Zero-site (link) effective Hamiltonian on the edge between adjacent `n` and
`m`, acting on a link tensor `C :: V_new ← V_e` produced by a gauge move from
`n` towards `m` (TDVP backward step). Consumes `env(n→m)` and `env(m→n)`.
"""
function eff_h0(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int, m::Int)
    En = env!(cache, ψ, H, n, m)
    Em = env!(cache, ψ, H, m, n)
    function apply0(C::AbstractTensorMap)
        y = ncon([C, En, Em], [[1, 2], [1, 3, -1], [2, 3, -2]], [false, false, false])
        return repartition(y, 1, 1)
    end
    return apply0
end

"""
    two_site_tensor(ψ, n, m) -> Θ

Contract `ψ[n]` and `ψ[m]` over their shared edge (`m` must be the parent of
`n`). Result is all-codomain, legs ordered: `A_n`'s codomain legs (slots, phys),
then `A_m`'s codomain legs except the `n`-slot (original order), then `m`'s
parent leg. `split_two_site!` is the exact inverse bookkeeping.
"""
function two_site_tensor(ψ::TTNS, n::Int, m::Int)
    t = ψ.topo
    t.parent[n] == m || throw(ArgumentError("two_site_tensor: m must be the parent of n"))
    A, B = ψ.tensors[n], ψ.tensors[m]
    k = childslot(t, m, n)
    pn = numout(A)
    aidx = [-(1:pn); 1]
    bidx = zeros(Int, numind(B))
    o = pn
    for j in 1:numind(B)
        bidx[j] = j == k ? 1 : -(o += 1)
    end
    return ncon([A, B], [aidx, bidx], [false, false])
end

"""
    split_two_site!(ψ, Θ, n, m; trunc, center_on=:n) -> (ψ, info)

Truncated SVD of a two-site tensor back into `ψ[n]`, `ψ[m]` (inverse of
`two_site_tensor`), moving the orthogonality center onto `center_on ∈ (:n, :m)`.
All truncation passes through `TruncationScheme` (§9.5).
"""
function split_two_site!(ψ::TTNS, Θ::AbstractTensorMap, n::Int, m::Int;
                         trunc::TruncationScheme=Backend.NO_TRUNCATION, center_on::Symbol=:n)
    t = ψ.topo
    k = childslot(t, m, n)
    pn = numout(ψ.tensors[n])
    # (n legs) ← (m legs); explicit permute — repartition would REVERSE the
    # multi-leg domain order (planar bending), scrambling the m-leg bookkeeping
    N = numind(Θ)
    Θs = permute(Θ, (ntuple(identity, pn), ntuple(j -> pn + j, N - pn)))
    U, S, Vh = split_svd(Θs, trunc)
    if center_on === :n
        An = U * S
        Rm = Vh
    else
        An = U
        Rm = S * Vh
    end
    Km = numind(ψ.tensors[m]) - 1                     # m's codomain legs
    p1 = ntuple(j -> j == k ? 1 : 1 + (j < k ? j : j - 1), Km)
    Am = permute(Rm, (p1, (numind(Rm),)))
    ψ.tensors[n] = An
    ψ.tensors[m] = Am
    ψ.center = center_on === :n ? n : m
    return ψ
end

"""
    eff_h2(cache, ψ, H, n, m) -> Θ -> H_eff·Θ

Two-site effective Hamiltonian on the bond `(n, m)` (`m` parent of `n`),
acting on `two_site_tensor(ψ, n, m)`-structured tensors.
"""
function eff_h2(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int, m::Int)
    t = ψ.topo
    t.parent[n] == m || throw(ArgumentError("eff_h2: m must be the parent of n"))
    hpn, hpm = hasphys(ψ, n), hasphys(ψ, m)
    Wn, Wm = H.tensors[n], H.tensors[m]
    k = childslot(t, m, n)
    Kn, Km = nchildren(t, n), nchildren(t, m)
    envn = [(w, env!(cache, ψ, H, w, n)) for w in neighbors(t, n) if w != m]
    envm = [(w, env!(cache, ψ, H, w, m)) for w in neighbors(t, m) if w != n]
    isroot_ = t.parent[m] == 0
    pn = Kn + (hpn ? 1 : 0)

    function apply2(x::AbstractTensorMap)
        Nx = numind(x)
        xidx = zeros(Int, Nx)
        wnidx = zeros(Int, numind(Wn))
        wmidx = zeros(Int, numind(Wm))
        tensors = Any[x, Wn, Wm]
        indices = Vector{Int}[xidx, wnidx, wmidx]
        conjs = Bool[false, false, false]
        nxt = Ref(0)
        fresh() = (nxt[] += 1; nxt[])

        # x leg positions: n-part 1..pn, then m's codomain legs except slot k,
        # then m's parent leg. Position of m's leg j (j ≠ k) in x:
        mpos(j) = pn + (j < k ? j : j - 1)

        # operator virtual bond between Wn and Wm
        ob = fresh()
        wnidx[end] = ob
        wmidx[k] = ob
        # physical legs
        if hpn
            pin = fresh()
            xidx[Kn + 1] = pin
            wnidx[Kn + 2] = pin
            wnidx[Kn + 1] = -(Kn + 1)
        end
        if hpm
            pin = fresh()
            xidx[mpos(Km + 1)] = pin
            wmidx[Km + 2] = pin
            wmidx[Km + 1] = -mpos(Km + 1)
        end
        # environments around n
        for (w, E) in envn
            lx = childslot(t, n, w)      # all non-parent neighbours of n are children
            kk, oo = fresh(), fresh()
            xidx[lx] = kk
            wnidx[lx] = oo
            push!(tensors, E); push!(indices, [kk, oo, -lx]); push!(conjs, false)
        end
        # environments around m
        for (w, E) in envm
            if t.parent[m] == w
                lw, lx = numind(Wm), Nx
            else
                lw = childslot(t, m, w)
                lx = mpos(lw)
            end
            kk, oo = fresh(), fresh()
            xidx[lx] = kk
            wmidx[lw] = oo
            push!(tensors, E); push!(indices, [kk, oo, -lx]); push!(conjs, false)
        end
        if isroot_
            ka, ko = fresh(), fresh()
            xidx[Nx] = ka
            wmidx[end] = ko
            cap = ones_tensor(scalartype(x), space(x, Nx)' ⊗ domain(Wm)[numin(Wm)] ⊗ space(x, Nx))
            push!(tensors, cap); push!(indices, [ka, ko, -Nx]); push!(conjs, false)
        end
        return ncon(tensors, indices, conjs)
    end
    return apply2
end
