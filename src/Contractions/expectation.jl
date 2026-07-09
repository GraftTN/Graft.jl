# Scalar contractions: overlaps and expectation values
# (PyTreeNet: contractions/state_state_contraction.py + state_operator_contraction.py).

"""
    inner(φ, ψ) -> ⟨φ|ψ⟩

Full overlap of two TTNS on the same topology (bra `φ` is conjugated).
"""
function inner(φ::TTNS, ψ::TTNS)
    φ.topo == ψ.topo || throw(ArgumentError("inner: mismatched topologies"))
    c = EnvCache(ψ.topo)
    r = ψ.topo.root
    for w in neighbors(ψ.topo, r)
        env!(c, ψ, nothing, φ, w, r)
    end
    return build_env(ψ, nothing, φ, r, 0, c.envs)
end

"""
    expect(ψ, H::TTNO; cache=EnvCache(ψ.topo)) -> ⟨ψ|H|ψ⟩

Raw (unnormalized) sandwich expectation value. Pass a warm `EnvCache` to reuse
sweep environments; the contraction closes at the orthogonality center so a
cache maintained by a sweep kernel is maximally reused.
"""
function expect(ψ::TTNS, H::TTNO; cache::EnvCache=EnvCache(ψ.topo))
    n = ψ.center
    for w in neighbors(ψ.topo, n)
        env!(cache, ψ, H, w, n)
    end
    return build_env(ψ, H, ψ, n, 0, cache.envs)
end

"""
    expect(ψ, op, site::Symbol) -> ⟨ψ|op_site|ψ⟩ / ⟨ψ|ψ⟩

Local single-site expectation value (`op :: P ← P`). Works on a gauge-moved
copy; normalized by construction.
"""
function expect(ψ::TTNS, op::AbstractTensorMap, site::Symbol)
    n = nodeindex(ψ.topo, site)
    ϕ = ψ.center == n ? ψ : move_center!(copy(ψ), n)
    A = ϕ.tensors[n]
    p = physleg(ϕ, n)
    N = numind(A)
    aidx = collect(1:N); aidx[p] = N + 1          # ket phys ↔ op P_in
    cidx = collect(1:N); cidx[p] = N + 2          # bra phys ↔ op P_out
    val = ncon([A, op, A], [aidx, [N + 2, N + 1], cidx], [false, false, true])
    return val / dot(A, A)
end
