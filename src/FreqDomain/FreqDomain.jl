"""
L5d — Frequency-domain kernel (architecture §5d). **All TODO (M6).**

TaSK — tangent-space Krylov resolvent solver (Kovalska–von Delft–Gleis,
arXiv:2510.07279): projects (ω − H + E_g)⁻¹ into the tangent space of a
converged ground-state TTNS and outputs discrete spectral poles {ω_α, S_α}
directly on the real axis. No time evolution involved.

Planned reuse: tangent-space projection = orthogonality-center machinery
(Networks), H·(tangent basis) = Contractions.EnvCache, orthogonalization =
KrylovKit. New work: tangent-basis bookkeeping at branching nodes.
Known limits to surface honestly: T = 0 only; single-site tangent-space
expressivity (ship the 2-site variance Δ²⊥ as a diagnostic output).
Division of labor with the companion `GraftImpurity.jl` spectral layer (§6.5):
TaSK sweeps the full spectrum cheaply; high-variance windows get complex-time
Krylov refinement — both emit the same discrete-pole format.
"""
module FreqDomain

export task_resolvent

# TODO(M6): implement the TaSK kernel; each step costs about one single-site
# sweep — an order of magnitude cheaper than complex-time evolution.
"""
    task_resolvent(ψgs, H, excitation_op, ωgrid; kwargs...) -> poles

Tangent-space Krylov resolvent kernel. TODO(M6) — no methods yet.
"""
function task_resolvent end

end # module FreqDomain
