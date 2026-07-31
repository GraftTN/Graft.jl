module GraftTTNOBuild

import GraftFoundation
import GraftNetworks
import GraftSymbolic

const Backend = GraftFoundation.Backend
const Trees = GraftFoundation.Trees
const Networks = GraftNetworks.Networks
const Symbolic = GraftSymbolic.Symbolic

include("TTNOBuild/TTNOBuild.jl")

using .LegacyTTNOBuild: ttno_from_opsum, THCFactorization, THCReport, isdf_thc,
    fit_thc, reconstruct_thc

export LegacyTTNOBuild, ttno_from_opsum, THCFactorization, THCReport, isdf_thc,
    fit_thc, reconstruct_thc

include("precompile.jl")

end # module GraftTTNOBuild
