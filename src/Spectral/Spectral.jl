"""
M1 spectral post-processing.

This module contains only small dense numerical kernels. Time evolution stays
in `Evolution`; bath realization stays in companion packages.
"""
module Spectral

using LinearAlgebra
using ..Networks
using ..Contractions
using ..Evolution: CorrelatorSeries
using ..Parallel: AbstractDistributedContext, distributed_rank,
    distributed_size, distributed_allreduce_sum!

export ExponentialSum, evaluate, rank_from_svals, esprit,
    LinearPredictionResult, linear_prediction, predict,
    exponential_sum,
    ComplexTimeKrylovResult, complex_time_krylov

include("exponential_sum.jl")
include("linear_prediction.jl")
include("complex_time_krylov.jl")

end # module Spectral
