using GraftTestUtils
using Test

@testset "GraftTestUtils smoke" begin
    @test exact_groundstate([2.0 0.0; 0.0 -1.0])[1] == -1.0
    @test exact_thermal_Z(zeros(2, 2), 3.0) == 2.0
end
