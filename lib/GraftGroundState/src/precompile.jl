import PrecompileTools

if get(ENV, "GRAFT_FULL_PRECOMPILE", "false") == "true"
    include("precompile/dense_solvers.jl")
end
