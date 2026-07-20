# Effective Hamiltonians (PyTreeNet: contractions/effective_hamiltonians.py).
#
# All three return callable `EffectiveMap`s suitable for KrylovKit. Required
# environments, the root cap, and the labelled contraction specification are
# constructed once per local-map visit; the matvec itself executes a cached
# binary plan. The private `_ncon_effective_reference` remains deliberately
# available for A/B tests and benchmark validation.
#
# TODO(MPI extension, §8 level 2): operator-term-level MPI Allreduce lives
# exactly here. H_eff·x splits over TTNO virtual-bond blocks, and DMRG/TDVP
# should share it when the MPI extension milestone is opened.

# Open-leg label bookkeeping: result leg i gets label -i. Builders below retain
# the legacy ncon labels verbatim, but attach an explicit Phase-1 env-first
# static-slot order so Planning never mistakes the physical x–W leg for an
# environment edge.

function _ncon_effective_reference(spec::ContractionSpec, x::AbstractTensorMap,
                                   statics::Tuple)
    return Planning.ncon_reference(spec, x, statics)
end

function _h1_spec(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int)
    t = ψ.topo
    x = ψ.tensors[n]
    hp = hasphys(ψ, n)
    W = H.tensors[n]
    envlist = [(w, env!(cache, ψ, H, w, n)) for w in neighbors(t, n)]
    isroot_ = t.parent[n] == 0
    K = nchildren(t, n)
    Nx = numind(x)

    xidx = zeros(Int, Nx)
    widx = zeros(Int, numind(W))
    labels = Vector{Int}[xidx, widx]
    conjs = Bool[false, false]
    statics = (W,)
    protos = (x, W)
    children = Int[]
    parents = Int[]
    caps = Int[]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    if hp
        pin = fresh()
        xidx[K + 1] = pin
        widx[K + 2] = pin
        widx[K + 1] = -(K + 1)       # open physical (out) leg
    end
    for (w, E) in envlist
        lx = _stateleg(t, hp, n, w)
        kk, oo = fresh(), fresh()
        xidx[lx] = kk
        widx[_opleg(t, hp, n, w)] = oo
        push!(labels, [kk, oo, -lx]); push!(conjs, false)
        statics = (statics..., E)
        protos = (protos..., E)
        slot = length(labels)
        if t.parent[n] == w
            push!(parents, slot)
        else
            push!(children, slot)
        end
    end
    if isroot_
        ka, ko = fresh(), fresh()
        xidx[end] = ka
        widx[end] = ko
        cap = _root_cap!(cache, scalartype(x),
                         domain(x)[1] ⊗ domain(W)[numin(W)] ⊗ dual(domain(x)[1]))
        push!(labels, [ka, ko, -Nx]); push!(conjs, false)
        statics = (statics..., cap)
        protos = (protos..., cap)
        push!(caps, length(labels))
    end
    # x → child envs → W → parent env/root cap
    preferred = vcat(children, [2], parents, caps)
    spec = ContractionSpec(labels, conjs, Nx, (Nx - 1, 1), 1;
                           preferred_slots=preferred)
    return spec, statics, protos
end

"""
    eff_h1(cache, ψ, H, n) -> EffectiveMap

One-site effective Hamiltonian at node `n`. The returned callable maps a tensor
with the structure of `ψ[n]` to the same `(N-1, 1)` TensorMap partition. The
plan cache is shape-only and safely survives ordinary environment invalidation.
Set `optimize=false` to force the Phase-1 env-first plan; `memory_weight`
selects the dense FLOP-plus-live-byte objective and is part of cache identity.
`memory_cap_bytes` is a hard conservative live-memory cap and is likewise
part of cache identity.  `sector_aware=true` (the default) uses the Phase-3 exact
unique-fusion block-GEMM objective when the TensorKit spaces support it;
non-unique fusion spaces retain the dense model.  Planar/anyonic execution is
outside the current regular TensorOperations backend surface.
"""
function eff_h1(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int;
                optimize::Bool=true, memory_weight::Real=1,
                sector_aware::Bool=true,
                memory_cap_bytes::Union{Nothing,Real}=nothing)
    spec, statics, protos = _h1_spec(cache, ψ, H, n)
    return _effective_map!(cache, :h1, spec, protos, statics,
                           scalartype(ψ.tensors[n]);
                           optimize=optimize, memory_weight=memory_weight,
                           sector_aware=sector_aware,
                           memory_cap_bytes=memory_cap_bytes,
                           output_twists=_euclidean_output_legs(ψ, n))
end

function _h0_input_space(En::AbstractTensorMap, Em::AbstractTensorMap)
    # C's first flat leg contracts env(n→m)'s ket leg; its second (domain)
    # leg contracts env(m→n)'s ket leg. This is the link shape produced by the
    # QR/CBE split seam without allocating a data-valued C just for planning.
    return dual(space(En, 1)) ← space(Em, 1)
end

function _h0_spec(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int, m::Int)
    En = env!(cache, ψ, H, n, m)
    Em = env!(cache, ψ, H, m, n)
    Cspace = _h0_input_space(En, Em)
    spec = ContractionSpec(Vector{Int}[[1, 2], [1, 3, -1], [2, 3, -2]],
                           Bool[false, false, false], 2, (1, 1), 1;
                           preferred_slots=[2, 3])
    return spec, (En, Em), (Cspace, En, Em)
end

"""
    eff_h0(cache, ψ, H, n, m) -> EffectiveMap

Zero-site (link) effective Hamiltonian on adjacent nodes `n, m`, acting on the
gauge link tensor used by the TDVP backward step. The returned map has direct
`(1, 1)` output partitioning; no trailing `repartition` copy is made. Planner
keywords have the same semantics as `eff_h1`.
"""
function eff_h0(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int, m::Int;
                optimize::Bool=true, memory_weight::Real=1,
                sector_aware::Bool=true,
                memory_cap_bytes::Union{Nothing,Real}=nothing)
    spec, statics, protos = _h0_spec(cache, ψ, H, n, m)
    Cspace = first(protos)
    twists = isdual(codomain(Cspace)[1]) &&
        _component_has_dual_physical(ψ, n, m) ? (1,) : ()
    return _effective_map!(cache, :h0, spec, protos, statics,
                           scalartype(ψ.tensors[n]);
                           optimize=optimize, memory_weight=memory_weight,
                           sector_aware=sector_aware,
                           memory_cap_bytes=memory_cap_bytes,
                           output_twists=twists)
end

"""
    two_site_space(ψ, n, m) -> TensorMapSpace

No-data prototype for `two_site_tensor(ψ, n, m)`. `m` must be the parent of
`n`; the all-codomain leg order is exactly the data contraction's order:
child-node codomain legs, followed by every parent-node flat leg except the
crossed child slot. It lets h2 planning discover Θ's shape without allocating
or contracting Θ.
"""
function two_site_space(ψ::TTNS, n::Int, m::Int)
    t = ψ.topo
    t.parent[n] == m || throw(ArgumentError("two_site_space: m must be the parent of n"))
    A, B = ψ.tensors[n], ψ.tensors[m]
    k = childslot(t, m, n)
    legs = [space(A, i) for i in 1:numout(A)]
    append!(legs, (space(B, j) for j in 1:numind(B) if j != k))
    cod = reduce(⊗, legs)
    return cod ← one(cod)
end

"""
    two_site_tensor(ψ, n, m) -> Θ

Contract `ψ[n]` and `ψ[m]` over their shared edge (`m` must be the parent of
`n`). Result is all-codomain, legs ordered: `A_n`'s codomain legs (slots,
physical), then `A_m`'s flat legs except the `n` slot (original order).
`split_two_site!` is the exact inverse bookkeeping.
"""
function two_site_tensor(ψ::TTNS, n::Int, m::Int)
    t = ψ.topo
    t.parent[n] == m || throw(ArgumentError("two_site_tensor: m must be the parent of n"))
    A, B = ψ.tensors[n], ψ.tensors[m]
    k = childslot(t, m, n)
    pn = numout(A)
    # This is a fixed binary contraction, so bypass `ncon`'s dynamic label
    # parser and call the TensorOperations expert API through L0. Besides
    # removing repeated label allocations, this keeps the large generic ncon
    # method out of TDVP2's first-use JIT path.
    pA = (Tuple(1:pn), (pn + 1,))
    bopen = Tuple(j for j in 1:numind(B) if j != k)
    pB = ((k,), bopen)
    nopen = pn + length(bopen)
    pAB = (Tuple(1:nopen), ())
    return Backend.contract_pair(A, pA, false, B, pB, false, pAB)
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

function _h2_spec(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int, m::Int)
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
    xspace = two_site_space(ψ, n, m)
    Nx = numind(xspace)

    xidx = zeros(Int, Nx)
    wnidx = zeros(Int, numind(Wn))
    wmidx = zeros(Int, numind(Wm))
    labels = Vector{Int}[xidx, wnidx, wmidx]
    conjs = Bool[false, false, false]
    statics = (Wn, Wm)
    protos = (xspace, Wn, Wm)
    envnslots = Int[]
    envmchildren = Int[]
    envmparents = Int[]
    caps = Int[]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    # x leg positions: n part 1:pn, then m's flat legs except slot k.
    mpos(j) = pn + (j < k ? j : j - 1)

    # Operator virtual bond between Wn and Wm.
    ob = fresh()
    wnidx[end] = ob
    wmidx[k] = ob
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
    # Environments around n: all are children because m is n's parent.
    for (w, E) in envn
        lx = childslot(t, n, w)
        kk, oo = fresh(), fresh()
        xidx[lx] = kk
        wnidx[lx] = oo
        push!(labels, [kk, oo, -lx]); push!(conjs, false)
        statics = (statics..., E)
        protos = (protos..., E)
        push!(envnslots, length(labels))
    end
    # Environments around m: defer its parent until after Wm joins.
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
        push!(labels, [kk, oo, -lx]); push!(conjs, false)
        statics = (statics..., E)
        protos = (protos..., E)
        if t.parent[m] == w
            push!(envmparents, length(labels))
        else
            push!(envmchildren, length(labels))
        end
    end
    if isroot_
        ka, ko = fresh(), fresh()
        xidx[Nx] = ka
        wmidx[end] = ko
        # `xspace` is a no-data TensorMapSpace rather than an AbstractTensorMap;
        # TensorKit exposes its flat leg spaces through `getindex`, not `space`.
        # This is exactly the final leg of the Θ prototype used by the former
        # data-valued implementation.
        cap = _root_cap!(cache, scalartype(ψ.tensors[n]),
                         dual(xspace[Nx]) ⊗ domain(Wm)[numin(Wm)] ⊗ xspace[Nx])
        push!(labels, [ka, ko, -Nx]); push!(conjs, false)
        statics = (statics..., cap)
        protos = (protos..., cap)
        push!(caps, length(labels))
    end
    # Θ → n child envs → Wn → m child envs → Wm → m parent/root cap.
    preferred = vcat(envnslots, [2], envmchildren, [3], envmparents, caps)
    spec = ContractionSpec(labels, conjs, Nx, (Nx, 0), 1;
                           preferred_slots=preferred)
    return spec, statics, protos
end

"""
    eff_h2(cache, ψ, H, n, m) -> EffectiveMap

Two-site effective Hamiltonian on child-parent bond `(n, m)`, acting on the
all-codomain structure returned by `two_site_tensor(ψ, n, m)`. Planner
keywords have the same semantics as `eff_h1`.
"""
function eff_h2(cache::EnvCache, ψ::TTNS, H::TTNO, n::Int, m::Int;
                optimize::Bool=true, memory_weight::Real=1,
                sector_aware::Bool=true,
                memory_cap_bytes::Union{Nothing,Real}=nothing)
    spec, statics, protos = _h2_spec(cache, ψ, H, n, m)
    t = ψ.topo
    Kn, Km = nchildren(t, n), nchildren(t, m)
    k = childslot(t, m, n)
    pn = Kn + (hasphys(ψ, n) ? 1 : 0)
    mpos(j) = pn + (j < k ? j : j - 1)
    xspace = first(protos)
    twists = Int[]
    for (j, child) in enumerate(t.children[n])
        isdual(xspace[j]) && _component_has_dual_physical(ψ, child, n) &&
            push!(twists, j)
    end
    hasphys(ψ, n) && isdual(xspace[Kn + 1]) && push!(twists, Kn + 1)
    for (j, child) in enumerate(t.children[m])
        child == n && continue
        pos = mpos(j)
        isdual(xspace[pos]) && _component_has_dual_physical(ψ, child, m) &&
            push!(twists, pos)
    end
    hasphys(ψ, m) && isdual(xspace[mpos(Km + 1)]) &&
        push!(twists, mpos(Km + 1))
    if t.parent[m] != 0
        pos = mpos(parentleg(ψ, m))
        isdual(xspace[pos]) &&
            _component_has_dual_physical(ψ, t.parent[m], m) && push!(twists, pos)
    end
    return _effective_map!(cache, :h2, spec, protos, statics,
                           scalartype(ψ.tensors[n]);
                           optimize=optimize, memory_weight=memory_weight,
                           sector_aware=sector_aware,
                           memory_cap_bytes=memory_cap_bytes,
                           output_twists=Tuple(twists))
end
