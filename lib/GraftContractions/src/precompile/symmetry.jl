# Minimal one-site charged states and neutral operators are sufficient to
# specialize the sector-aware environment and effective-map paths. Building
# them here avoids a runtime edge to Symbolic, TTNOBuild, or test support.
function _precompile_sector_state(physical, sector)
    topo = Trees.mps_topology(1)
    rootspace = Backend.Vect[typeof(sector)](sector => 1)
    tensor = zeros(
        ComplexF64, Backend.:←(physical, rootspace))
    for (block_sector, block) in Backend.blocks(tensor)
        block_sector == sector && (block[1, 1] = one(ComplexF64))
    end
    return Networks.TTNS(topo, [tensor], topo.root)
end

function _precompile_sector_identity(physical)
    topo = Trees.mps_topology(1)
    unit = Backend.oneunit(physical)
    domain = Backend.:⊗(physical, unit)
    tensor = zeros(ComplexF64, Backend.:←(physical, domain))
    for (_, block) in Backend.blocks(tensor)
        for index in 1:min(size(block)...)
            block[index, index] = one(ComplexF64)
        end
    end
    return Networks.TTNO(topo, [tensor]; ishermitian=true)
end

function _precompile_sector_contractions(physical, sector)
    state = _precompile_sector_state(physical, sector)
    operator = _precompile_sector_identity(physical)
    topo = Networks.topology(state)
    Contractions.inner(state, state)
    cache = Contractions.EnvCache(topo)
    Contractions.expect(state, operator; cache)
    root = topo.root
    Contractions.eff_h1(cache, state, operator, root)(state.tensors[root])
    applied = Networks.apply(operator, state)
    Contractions.inner(applied, applied)
    return nothing
end

PrecompileTools.@compile_workload begin
    let
        physical = Backend.U1Space(0 => 1, 1 => 1)
        _precompile_sector_contractions(physical, Backend.U1Irrep(1))
    end
end

PrecompileTools.@compile_workload begin
    let
        even = Backend.FermionParity(0)
        odd = Backend.FermionParity(1)
        physical = Backend.Vect[Backend.FermionParity](even => 1, odd => 1)
        _precompile_sector_contractions(physical, odd)
    end
end
