start_ns = time_ns()
using Graft
load_seconds = (time_ns() - start_ns) / 1e9
runtime = parallel_runtime_config()
println(
    "GRAFT_COLD_START ",
    "load_seconds=$(repr(load_seconds)) ",
    "custom_sysimage=$(Base.JLOptions().image_file_specified != 0) ",
    "julia_version=$(runtime.julia_version) ",
    "machine=$(runtime.machine) ",
    "julia_threads=$(runtime.julia_threads) ",
    "blas_vendor=$(runtime.blas_vendor) ",
    "blas_threads=$(runtime.blas_threads) ",
    "strided_threads=$(runtime.strided_threads) ",
    "peak_rss_bytes=$(Int(Sys.maxrss()))",
)
