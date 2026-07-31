"""
Narrow package boundary consumed by the typed StateDiagram compiler.

These bindings expose the already validated braided lowering and TensorKit
materialization substrate without making the legacy builder depend on the
typed compiler.
"""
module LegacyLoweringInterface

using ..LegacyTTNOBuild: _Euler, _insub, _subframe,
    _build_braided_term_plan, _local_morphism_signature, _fuse_charge,
    _OMITTED_IDENTITY, _ExplicitLocal, _siteop_matrix,
    _graded_siteop_matrix, _sector_tuple_groups, _add_block_entry!

export _Euler, _insub, _subframe, _build_braided_term_plan,
    _local_morphism_signature, _fuse_charge, _OMITTED_IDENTITY,
    _ExplicitLocal, _siteop_matrix, _graded_siteop_matrix,
    _sector_tuple_groups, _add_block_entry!

end # module LegacyLoweringInterface
