"""
Cross-cutting — checkpoint/restart (architecture §7). Checkpointing is M0
infrastructure, not a patch: "job may be killed anytime" (Snellius/GPFS) is the
default assumption.

Format: JLD2 (Julia-native, zero glue). A mirrored HDF5 layout for external
tools (h5py / TRIQS-side inspection) is TODO. Contents contract (§7): a
checkpoint holds the complete state to resume one solve — solver-state structs
carry TTNS tensors + center, truncation parameters, sweep/step counters and
grid positions, RNG states (JLD2 serializes `Xoshiro` fine, §9.6), finished
sample sets. `EnvCache` is big-but-rebuildable: dropped by default.

Write atomicity: temp file + `mv -f` rename, keeping the last `keep` rotations —
a sudden stop costs at most one checkpoint interval. IO stays off the sweep
critical path (§9.12): call from sweep-gap callbacks.
"""
module Checkpoints

using JLD2

export checkpoint!, resume

"""
    checkpoint!(state, path; keep=3, metadata=NamedTuple())

Atomically write `state` (any serializable solver-state struct) to `path`,
rotating up to `keep` previous checkpoints as `path.1`, `path.2`, ….

Self-consistency contract (§7): callers on the DMFT side should put a bath
parameter hash into `metadata` so a restart against a silently-changed bath is
detectable.
"""
function checkpoint!(state, path::AbstractString; keep::Int=3, metadata=NamedTuple())
    tmp = path * ".tmp"
    jldsave(tmp; state, metadata, format_version=1)
    for i in (keep - 1):-1:1
        older = string(path, ".", i)
        newer = i == 1 ? path : string(path, ".", i - 1)
        isfile(newer) && mv(newer, older; force=true)
    end
    mv(tmp, path; force=true)
    return path
end

"""
    resume(path) -> (; state, metadata)

Load a checkpoint written by [`checkpoint!`](@ref). Returns the stored solver
state ready to continue.
"""
function resume(path::AbstractString)
    data = load(path)
    return (; state=data["state"], metadata=get(data, "metadata", NamedTuple()))
end

# TODO(§10.8): `with_checkpoint(itr; every, path)` iterator combinator so sweep
# kernels only implement "one step" and checkpointing wraps around them.
# TODO(§7): TRIQS interop — (a) pure-Julia HDF5 reader for TRIQS archive
# layouts (BlockGf/Gf/U-matrix, production default), (b) PythonCall bridge as a
# package extension `GRAFTTriqsExt`; both converge on one intermediate
# representation before mapping to TensorKit sectors.

end # module Checkpoints
