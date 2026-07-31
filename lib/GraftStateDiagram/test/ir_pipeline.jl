function duplicate_term_input()
    spin = Symbolic.spin_ops()
    topo = Trees.mps_topology(2)
    phys = Dict(:site1 => spin.P, :site2 => spin.P)
    term(coefficient) = Symbolic.Term(
        coefficient,
        Symbolic.SiteOp(:site1, :X, spin.X),
        Symbolic.SiteOp(:site2, :Z, spin.Z),
    )
    hamiltonian = Symbolic.OpSum() + term(0.5) + term(0.25)
    return StateDiagram.TTNOBuildInput(hamiltonian, topo, phys)
end

@testset "typed input and lowering" begin
    input = duplicate_term_input()
    input_again = duplicate_term_input()
    @test input isa StateDiagram.TTNOBuildInput
    @test input == input_again
    @test hash(input) == hash(input_again)
    @test length(input.terms) == 2
    @test length(input.operator_table) == 2

    expansions = StateDiagram.lower_terms(
        input, GraftStateDiagram.AbelianScalarLowering())
    @test length(expansions) == length(input.terms)
    @test all(expansion -> StateDiagram.validate_expansion(input, expansion),
              expansions)
    @test StateDiagram.serialize_expansions(input, expansions) ==
        StateDiagram.serialize_expansions(input_again,
            StateDiagram.lower_terms(
                input_again, GraftStateDiagram.AbelianScalarLowering()))

    spin = Symbolic.spin_ops()
    topo = Trees.mps_topology(2)
    phys = Dict(:site1 => spin.P, :site2 => spin.P)
    collision = Symbolic.OpSum() +
        Symbolic.Term(1.0, Symbolic.SiteOp(:site1, :K, spin.X)) +
        Symbolic.Term(1.0, Symbolic.SiteOp(:site2, :K, spin.Z))
    @test_throws ArgumentError StateDiagram.TTNOBuildInput(
        collision, topo, phys)
end

@testset "direct-sum and structural merge" begin
    input = duplicate_term_input()
    expansions = StateDiagram.lower_terms(
        input, GraftStateDiagram.AbelianScalarLowering())

    direct = StateDiagram.merge_channels(
        input, expansions, GraftStateDiagram.DirectSumMerge())
    @test direct isa StateDiagram.DirectSumPlan
    @test direct.expansions == expansions

    merged = StateDiagram.merge_channels(
        input,
        expansions,
        GraftStateDiagram.StateDiagramMerge(
            GraftStateDiagram.StructuralOptimizer()),
    )
    @test merged isa StateDiagram.StateDiagram
    @test !isempty(merged.proofs)
    @test !isempty(merged.log.entries)
    @test StateDiagram.validate_merge_plan(input, merged)
    @test StateDiagram.unmerge_expansions(merged) == expansions

    second_pass = StateDiagram.merge_channels(
        input,
        merged.expansions,
        GraftStateDiagram.StateDiagramMerge(
            GraftStateDiagram.StructuralOptimizer()),
    )
    @test isempty(second_pass.proofs)
    @test second_pass.expansions == merged.expansions
end
