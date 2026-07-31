using Test
using GraftFoundation: U1Irrep, mps_topology, nnodes, nodeindex
using GraftSymbolic: OpSum, SiteOp, Term, boson_ops, charge, coefficient,
    nterms, ppdress, sites, spin_ops_u1

@testset "GraftSymbolic operator and OpSum contracts" begin
    spin = spin_ops_u1()
    raising = SiteOp(:left, :Sp, spin.Sp)
    lowering = SiteOp(:right, :Sm, spin.Sm)

    @test charge(raising) == U1Irrep(1)
    @test charge(lowering) == U1Irrep(-1)
    @test_throws ArgumentError Term(1.0, raising)

    hopping = Term(0.75, raising, lowering)
    field = Term(-0.2, SiteOp(:left, :Z, spin.Z))
    hamiltonian = OpSum() + hopping + field
    scaled = (2 - im) * hamiltonian

    @test nterms(hamiltonian) == length(hamiltonian) == 2
    @test sites(first(hamiltonian)) == [:left, :right]
    @test coefficient(first(scaled)) == (2 - im) * coefficient(hopping)
    @test collect(hamiltonian) == [hopping, field]
    @test_throws ArgumentError Term(
        1.0, raising, SiteOp(:left, :Z, spin.Z))
end

@testset "GraftSymbolic projected-purification rewrite" begin
    boson = boson_ops(2)
    topology = mps_topology(1)
    physical = Dict(:site1 => boson.P)
    original = OpSum() + Term(
        0.4, SiteOp(:site1, :X, boson.X))

    dressed, dressed_topology, dressed_physical = ppdress(
        original, topology, physical; nmax=2, boson_sites=[:site1])
    ancilla = :site1_B1

    @test nterms(original) == 1
    @test nnodes(topology) == 1
    @test nterms(dressed) == 2
    @test nnodes(dressed_topology) == 2
    @test nodeindex(dressed_topology, ancilla) > 0
    @test Set(keys(dressed_physical)) == Set((:site1, ancilla))
    @test all(coefficient(term) == 0.4 for term in dressed)
    @test all(Set(sites(term)) == Set((:site1, ancilla)) for term in dressed)
    @test Set(
        Tuple(operator.name for operator in term.ops) for term in dressed
    ) == Set(((:Bpd, :Bbd), (:Bp, :Bb)))
end
