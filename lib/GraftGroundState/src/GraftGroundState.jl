module GraftGroundState

import GraftContractions
import GraftFoundation
import GraftNetworks
import GraftParallel

const Backend = GraftFoundation.Backend
const Trees = GraftFoundation.Trees
const Networks = GraftNetworks.Networks
const Contractions = GraftContractions.Contractions
const Parallel = GraftParallel.Parallel

include("GroundState/GroundState.jl")

using .GroundState

export GroundState
export dmrg1!, dmrg2!, dmrg1_3s!, expand!, groundstate_eigsolve_backend

include("precompile.jl")

end # module GraftGroundState
