# SD6 — public typed opt-in compiler facade (state-diagram-compiler plan
# v2, SD6 deliverables).
#
# The facade pairs exactly one lowering kernel with exactly one merge
# kernel. The legacy `ttno_from_opsum` compiler remains the package
# default and the explicit migration oracle; the typed compiler is opt-in
# until the SD7 promotion gate is passed and explicitly approved. There is
# no symbol-based mode API: strategies are the typed kernel values.

"""
    compile_ttno(H::OpSum, topo::TreeTopology, phys;
                 lowering=AbelianScalarLowering(),
                 merge=StateDiagramMerge(SGEOptimizer()),
                 hermitian=false, elt=ComplexF64) ->
        (TTNO, TTNOBuildReport, Union{TTNOExactProvenance,Nothing})

Opt-in typed StateDiagram compiler: normalize the input, lower every term
through `lowering`, combine channels through `merge`, and realize the TTNO
producer-independently, returning the operator together with its complete
build and capability report. A [`DirectSumMerge`](@ref) additionally returns
compiler-certified [`TTNOExactProvenance`](@ref) for exact Stage 1
compression; optimizing StateDiagram merge plans return `nothing` in the
third tuple slot.

Supported merge kernels: [`DirectSumMerge`](@ref) (uncompressed correctness
oracle), and [`StateDiagramMerge`](@ref) with a [`StructuralOptimizer`](@ref)
(exact structural sharing), [`GammaCoverOptimizer`](@ref) (raw-Gamma minimum
covers), or [`SGEOptimizer`](@ref) (raw-versus-eliminated cover selection).
Unsupported category profiles fail closed with a typed
[`MissingCategoryCapability`](@ref) before any densification.

The legacy [`ttno_from_opsum`](@ref) remains the default compiler and the
explicit oracle; this facade changes no default (plan v2 SD7 gates the
promotion, which additionally requires explicit approval).
"""
function compile_ttno(H::OpSum, topo::TreeTopology,
                      phys::Dict{Symbol,<:ElementarySpace};
                      lowering::AbstractOperatorLoweringKernel=AbelianScalarLowering(),
                      merge::AbstractTTNOMergeKernel=StateDiagramMerge(SGEOptimizer()),
                      hermitian::Bool=false,
                      elt::Type{<:Number}=ComplexF64)
    input = TTNOBuildInput(
        H, topo, phys;
        hermiticity=hermitian ? AssertedHermitian() : NoHermiticityAssertion())
    expansions = lower_terms(input, lowering)
    plan = merge_channels(input, expansions, merge)
    operator, report = realize_ttno(input, plan; elt)
    provenance = if plan isa DirectSumPlan
        compiler_exact_provenance(input, plan)
    else
        nothing
    end
    return operator, report, provenance
end
