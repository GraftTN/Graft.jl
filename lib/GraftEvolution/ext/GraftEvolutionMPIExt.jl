module GraftEvolutionMPIExt

using MPI
using GraftContractions: with_workspace_map
import GraftEvolution: evolution_exponentiate_backend
import GraftParallel: AbstractRootDrivenContext, distributed_exponentiate,
    root_driven_solver

function distributed_exponentiate(
        context::AbstractRootDrivenContext, effective, x, time; kwargs...)
    return with_workspace_map(effective) do workspace
        root_driven_solver(context, workspace) do driven
            evolution_exponentiate_backend(driven, time, x; kwargs...)
        end
    end
end

end # module GraftEvolutionMPIExt
