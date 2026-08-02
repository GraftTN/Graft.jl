"""
    OrientedTwoSiteFactorFrame

Planned, factorized Hamiltonian frame for the directed edge `source -> target`.
`source_action` and `target_action` are the two local ket--operator--environment
halves.  In each half the site-external bra legs are the codomain and the open
state and TTNO edge legs form the two-factor domain.  The type is deliberately
algorithm independent: it contains no expansion or truncation policy.
"""
struct OrientedTwoSiteFactorFrame{A<:AbstractTensorMap,B<:AbstractTensorMap}
    source::Int
    target::Int
    source_action::A
    target_action::B
end

"""
    contract_oriented_two_site(cache, frame)

Close the state and TTNO edge channels of an oriented factor frame.  This is
primarily the exact C1 identity for consumers and tests; it still contracts
the already factorized local halves and never constructs a two-site ket.
"""
function contract_oriented_two_site(cache::EnvCache,
                                    frame::OrientedTwoSiteFactorFrame;
                                    optimize::Bool=true)
    A, B = frame.source_action, frame.target_action
    ea, eb = numind(A) - 2, numind(B) - 2
    labels = Vector{Int}[
        [ntuple(i -> -i, ea)..., 1, 2],
        [ntuple(i -> -(ea + i), eb)..., 1, 2],
    ]
    spec = ContractionSpec(labels, Bool[false, false], ea + eb,
                           (ea + eb, 0), nothing;
                           preferred_slots=[1, 2])
    return _planned_execute!(cache, :oriented_two_site_close, spec, (A, B),
                             scalartype(A); optimize)
end

"""Contract a rank-two edge link into the source half, leaving state/TTNO channels open."""
function contract_source_factor(cache::EnvCache,
                                frame::OrientedTwoSiteFactorFrame,
                                link::AbstractTensorMap;
                                optimize::Bool=true)
    A = frame.source_action
    ea = numind(A) - 2
    numind(link) == 2 || throw(ArgumentError("edge link must have rank two"))
    labels = Vector{Int}[
        [ntuple(i -> -i, ea)..., 1, -(ea + 2)],
        [1, -(ea + 1)],
    ]
    spec = ContractionSpec(labels, Bool[false, false], ea + 2, (ea, 2),
                           nothing; preferred_slots=[1, 2])
    return _planned_execute!(cache, :oriented_source_factor, spec, (A, link),
                             scalartype(A); optimize)
end

"""Contract a rank-two bridge into the target half, leaving bridge/TTNO channels open."""
function contract_target_factor(cache::EnvCache,
                                frame::OrientedTwoSiteFactorFrame,
                                bridge::AbstractTensorMap;
                                optimize::Bool=true)
    B = frame.target_action
    eb = numind(B) - 2
    numind(bridge) == 2 || throw(ArgumentError("factor bridge must have rank two"))
    labels = Vector{Int}[
        [ntuple(i -> -i, eb)..., 1, -(eb + 2)],
        [-(eb + 1), 1],
    ]
    spec = ContractionSpec(labels, Bool[false, false], eb + 2, (eb, 2),
                           nothing; preferred_slots=[2, 1])
    return _planned_execute!(cache, :oriented_target_factor, spec, (B, bridge),
                             scalartype(B); optimize)
end

"""
Close a factorized two-site action after projecting the target external frame.
The result maps the projected target channel to the source external frame.
"""
function contract_projected_two_site(cache::EnvCache,
                                     frame::OrientedTwoSiteFactorFrame,
                                     link::AbstractTensorMap,
                                     target_basis::AbstractTensorMap;
                                     optimize::Bool=true)
    A, B = frame.source_action, frame.target_action
    ea = numind(A) - 2
    numind(link) == 2 || throw(ArgumentError("edge link must have rank two"))
    codomain(target_basis) == codomain(B) || throw(SpaceMismatch(
        "target basis does not span the target external frame"))
    projected = target_basis' * B
    labels = Vector{Int}[
        [ntuple(i -> -i, ea)..., 1, 3],
        [1, 2],
        [-(ea + 1), 2, 3],
    ]
    spec = ContractionSpec(labels, Bool[false, false, false], ea + 1,
                           (ea, 1), nothing; preferred_slots=[2, 3, 1])
    return _planned_execute!(cache, :oriented_projected_close, spec,
                             (A, link, projected), scalartype(A); optimize)
end

function _oriented_site_tensor(psi::TTNS, site::Int, peer::Int)
    t = topology(psi)
    A = psi.tensors[site]
    N = numind(A)
    if t.parent[site] == peer
        return A, ntuple(identity, N)
    elseif t.parent[peer] == site
        k = childslot(t, site, peer)
        order = (Backend._others(N, k)..., k)
        return permute(A, (Tuple(order[1:(N - 1)]), (order[N],))), order
    end
    throw(ArgumentError("oriented factor-frame endpoints must be adjacent"))
end

function _site_factor_action(cache::EnvCache, psi::TTNS, H::TTNO,
                             site::Int, peer::Int,
                             X::AbstractTensorMap, order::Tuple;
                             optimize::Bool=true)
    t = topology(psi)
    hp = hasphys(psi, site)
    W = H.tensors[site]
    N = numind(X)
    N == length(order) || throw(ArgumentError(
        "oriented site factor has the wrong rank"))
    raw_to_oriented = invperm(collect(order))
    active_state = N
    active_state_raw = t.parent[site] == peer ? parentleg(psi, site) :
        childslot(t, site, peer)
    active_op_raw = _opleg(t, hp, site, peer)
    envlist = [(w, env!(cache, psi, H, w, site))
               for w in neighbors(t, site) if w != peer]

    xidx = zeros(Int, N)
    widx = zeros(Int, numind(W))
    labels = Vector{Int}[xidx, widx]
    conjs = Bool[false, false]
    operands = (X, W)
    children = Int[]
    parents = Int[]
    caps = Int[]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    # Keep the directed state and TTNO edge legs open as the two domain
    # factors, after all site-external output legs.
    xidx[active_state] = -N
    widx[active_op_raw] = -(N + 1)
    if hp
        raw_phys = physleg(psi, site)
        phys = raw_to_oriented[raw_phys]
        pin = fresh()
        xidx[phys] = pin
        widx[physleg(H, site) + 1] = pin
        widx[physleg(H, site)] = -phys
    end
    for (w, E) in envlist
        raw_state = _stateleg(t, hp, site, w)
        state = raw_to_oriented[raw_state]
        kk, oo = fresh(), fresh()
        xidx[state] = kk
        widx[_opleg(t, hp, site, w)] = oo
        push!(labels, [kk, oo, -state])
        push!(conjs, false)
        operands = (operands..., E)
        slot = length(labels)
        if t.parent[site] == w
            push!(parents, slot)
        else
            push!(children, slot)
        end
    end
    if t.parent[site] == 0
        raw_parent = numind(psi.tensors[site])
        state = raw_to_oriented[raw_parent]
        ka, ko = fresh(), fresh()
        xidx[state] = ka
        widx[end] = ko
        cap = _root_cap!(cache, scalartype(X),
                         domain(psi.tensors[site])[1] ⊗
                         domain(W)[numin(W)] ⊗
                         dual(domain(psi.tensors[site])[1]))
        push!(labels, [ka, ko, -state])
        push!(conjs, false)
        operands = (operands..., cap)
        push!(caps, length(labels))
    end

    preferred = vcat(children, [2], parents, caps)
    spec = ContractionSpec(labels, conjs, N + 1, (N - 1, 2), 1;
                           preferred_slots=preferred)
    result = _planned_execute!(cache, :oriented_site_factor, spec, operands,
                               scalartype(X); optimize)

    # The external output legs inherit exactly the same pivotal convention as
    # an h1 result.  The two central factor legs are ket/operator channels and
    # are never Euclidean-bra outputs.
    for raw_leg in _euclidean_output_legs(psi, site)
        raw_leg == active_state_raw && continue
        oriented_leg = raw_to_oriented[raw_leg]
        oriented_leg < N && (result = Backend.twist(result, oriented_leg))
    end
    return result
end

"""
    oriented_two_site_factor_frame(cache, psi, H, source, target)

Build the two planned local halves of the Hamiltonian sandwich on a directed
adjacent edge without materializing either a two-site ket or `eff_h2(Theta)`.
The returned halves are expressed in source/target edge-in-domain frames.
"""
function oriented_two_site_factor_frame(cache::EnvCache, psi::TTNS, H::TTNO,
                                        source::Int, target::Int;
                                        source_tensor::Union{Nothing,AbstractTensorMap}=nothing,
                                        target_tensor::Union{Nothing,AbstractTensorMap}=nothing,
                                        optimize::Bool=true)
    topology(psi) == topology(H) || throw(ArgumentError(
        "oriented factor frame requires matching TTNS and TTNO topologies"))
    source_default, source_order = _oriented_site_tensor(psi, source, target)
    target_default, target_order = _oriented_site_tensor(psi, target, source)
    Xs = source_tensor === nothing ? source_default : source_tensor
    Xt = target_tensor === nothing ? target_default : target_tensor
    codomain(Xs) == codomain(source_default) || throw(SpaceMismatch(
        "source tensor does not match the directed site frame"))
    codomain(Xt) == codomain(target_default) || throw(SpaceMismatch(
        "target tensor does not match the directed site frame"))
    source_action = _site_factor_action(
        cache, psi, H, source, target, Xs, source_order; optimize)
    target_action = _site_factor_action(
        cache, psi, H, target, source, Xt, target_order; optimize)
    return OrientedTwoSiteFactorFrame(source, target,
                                      source_action, target_action)
end
