module GraftFoundation

include("Backend/Backend.jl")
include("Trees/Trees.jl")

using .Backend
using .Trees

export Backend, Trees

export ℂ, ComplexSpace, GradedSpace, ElementarySpace, ProductSpace, Vect,
    U1Space, Z2Space, U1Irrep, ZNIrrep, SU2Irrep, FermionParity, Trivial,
    ⊠, ⊕, ⊗, ←, dual, oneunit, fuse, dim, space, sectortype, spacetype, sectors,
    isdual
export AbstractTensorMap, TensorMap, DiagonalTensorMap, id, isometry, unitary,
    permute, repartition, flip, twist, twist!, catdomain, catcodomain, numind,
    numout, numin, codomain, domain, blocks, block, norm, dot, tr, scalartype,
    Rsymbol
export left_orth, right_orth, left_null, qr_compact, svd_compact, svd_trunc,
    svd_vals, truncrank, trunctol, truncerror, notrunc
export @tensor, ncon, contract_pair, pair_cost, pair_workload_profile,
    space_signature, sector_cost_supported, sector_cost_nontrivial,
    sector_block_peak, tensor_scalar, contract_pair!, allocate_contract_pair,
    contract_pair_compatible
export FermionSector, AbelianSector, TruncationScheme, truncspec, split_svd,
    split_svd_with_error, svd_factor_leg_with_error, absorb_on_leg,
    transform_leg, transform_leg_space, orth_factor_leg, trivialspace,
    ones_tensor, sector_fusion_symbol, sector_braiding_symbol

export TreeTopology, nnodes, nodeid, nodeindex, isroot, isleaf, leaves,
    nchildren, neighbors, childslot, edges, postorder, preorder, path_to_root,
    path_between, subtree_nodes, tdvp_update_path, orth_path_segments,
    sweep_order, mps_topology, star_topology, binary_topology, fork_topology,
    mount_chain, is_t3ns

end # module GraftFoundation
