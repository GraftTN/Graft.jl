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
    return build_env(c, ψ, nothing, φ, r, 0)
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
    return build_env(cache, ψ, H, ψ, n, 0)
end

function _local_expect_spec(ψ::TTNS, op::AbstractTensorMap, n::Int)
    A = ψ.tensors[n]
    p = physleg(ψ, n)
    N = numind(A)
    aidx = collect(1:N); aidx[p] = N + 1          # ket phys ↔ op P_in
    bidx = collect(1:N); bidx[p] = N + 2          # bra phys ↔ op P_out
    spec = ContractionSpec(Vector{Int}[aidx, [N + 2, N + 1], bidx],
                           Bool[false, false, true], 0, (0, 0), nothing;
                           preferred_slots=[1, 3, 2])
    return spec, (A, op, A)
end

"""Retained private ncon reference for local-expectation A/B tests."""
function _local_expect_ncon_reference(ψ::TTNS, op::AbstractTensorMap, n::Int)
    A = ψ.tensors[n]
    p = physleg(ψ, n)
    N = numind(A)
    aidx = collect(1:N); aidx[p] = N + 1
    bidx = collect(1:N); bidx[p] = N + 2
    val = ncon([A, op, A], [aidx, [N + 2, N + 1], bidx],
               [false, false, true])
    return val / dot(A, A)
end

"""
    expect(ψ, op, site::Symbol; cache=EnvCache(ψ.topo)) -> ⟨ψ|op_site|ψ⟩ / ⟨ψ|ψ⟩

Local single-site expectation value (`op :: P ← P`). Works on a gauge-moved
copy; normalized by construction.
"""
function expect(ψ::TTNS, op::AbstractTensorMap, site::Symbol;
                cache::EnvCache=EnvCache(ψ.topo))
    n = nodeindex(ψ.topo, site)
    ϕ = ψ.center == n ? ψ : move_center!(copy(ψ), n)
    spec, operands = _local_expect_spec(ϕ, op, n)
    val = _planned_execute!(cache, :local_ket_op_bra, spec, operands,
                            scalartype(operands[1]))
    return val / dot(operands[1], operands[1])
end
