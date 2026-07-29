#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import time


def rss_bytes(pid: int) -> int:
    with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is unavailable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=("thermal", "rsvd"), required=True)
    parser.add_argument("--mode", choices=("serial", "threaded"), required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--memory-cap-bytes", type=int, default=2_147_483_648)
    args = parser.parse_args()

    environment = os.environ.copy()
    environment.update(
        GRAFT_P1_WORKLOAD=args.workload,
        GRAFT_P1_MODE=args.mode,
        GRAFT_P1_SAMPLES=str(args.samples),
        GRAFT_P1_MEMORY_CAP_BYTES=str(args.memory_cap_bytes),
    )
    command = [
        "julia",
        "--startup-file=no",
        "--project=.",
        f"-t{args.threads}",
        "benchmark/p1_fanout_workloads.jl",
    ]
    process = subprocess.Popen(
        command,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    assert process.stdin is not None
    ready = process.stdout.readline().strip()
    if not ready.startswith("GRAFT_P1_READY "):
        stderr = process.stderr.read() if process.stderr is not None else ""
        process.wait()
        print(ready)
        print(stderr, file=sys.stderr)
        return process.returncode or 1
    baseline = int(
        next(field.split("=", 1)[1] for field in ready.split()
             if field.startswith("baseline_rss_bytes="))
    )
    peak = rss_bytes(process.pid)
    process.stdin.write("\n")
    process.stdin.flush()

    while process.poll() is None:
        try:
            peak = max(peak, rss_bytes(process.pid))
        except (FileNotFoundError, ProcessLookupError, RuntimeError):
            break
        time.sleep(0.002)

    stdout = process.stdout.read()
    stderr = process.stderr.read() if process.stderr is not None else ""
    returncode = process.wait()
    if stderr:
        print(stderr, file=sys.stderr, end="")
    print(ready)
    print(stdout, end="")
    print(
        "GRAFT_P1_RSS "
        f"workload={args.workload} "
        f"mode={args.mode} "
        f"julia_threads={args.threads} "
        f"baseline_rss_bytes={baseline} "
        f"monitored_peak_rss_bytes={peak} "
        f"monitored_growth_bytes={max(0, peak - baseline)}"
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
