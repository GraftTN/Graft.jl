module GraftGroundStateMPIExt

using MPI
using GraftContractions: with_workspace_map
import GraftGroundState: groundstate_eigsolve_backend
import GraftParallel: AbstractRootDrivenContext, distributed_eigsolve,
    root_driven_solver

function distributed_eigsolve(
        context::AbstractRootDrivenContext, effective, x,
        howmany::Integer, which; kwargs...)
    return with_workspace_map(effective) do workspace
        root_driven_solver(context, workspace) do driven
            groundstate_eigsolve_backend(
                driven, x, howmany, which; kwargs...)
        end
    end
end

end # module GraftGroundStateMPIExt
