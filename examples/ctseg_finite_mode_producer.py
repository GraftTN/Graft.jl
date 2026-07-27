#!/usr/bin/env python3
"""Produce a strict Graft CTSEG result artifact in an external TRIQS env."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class Action:
    action_hash: str
    beta: float
    mu: float
    static_interaction: float
    n0: float
    orbital_convention: str
    density_convention: str
    bath_energies: tuple[float, ...]
    bath_couplings: tuple[complex, ...]
    boson_frequencies: tuple[float, ...]
    boson_couplings: tuple[float, ...]


def _float_list(text: str) -> tuple[float, ...]:
    if not text:
        return ()
    return tuple(float(value) for value in text.split(";"))


def _complex_list(text: str) -> tuple[complex, ...]:
    if not text:
        return ()
    result = []
    for value in text.split(";"):
        real_part, imag_part = value.split(":", maxsplit=1)
        result.append(complex(float(real_part), float(imag_part)))
    return tuple(result)


def read_action(path: Path) -> Action:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("CTSEG input CSV contains no kernel rows")
    required = {
        "action_hash",
        "beta",
        "mu",
        "static_interaction",
        "n0",
        "orbital_convention",
        "density_convention",
        "bath_energies",
        "bath_couplings",
        "boson_frequencies",
        "boson_couplings",
        "kernel",
        "index",
        "frequency",
        "value_re",
        "value_im",
    }
    if set(rows[0]) != required:
        raise ValueError("unexpected CTSEG input CSV schema")
    metadata_fields = [
        "action_hash",
        "beta",
        "mu",
        "static_interaction",
        "n0",
        "orbital_convention",
        "density_convention",
        "bath_energies",
        "bath_couplings",
        "boson_frequencies",
        "boson_couplings",
    ]
    first = rows[0]
    for row in rows[1:]:
        if any(row[field] != first[field] for field in metadata_fields):
            raise ValueError("inconsistent action metadata across input rows")
    action = Action(
        action_hash=first["action_hash"],
        beta=float(first["beta"]),
        mu=float(first["mu"]),
        static_interaction=float(first["static_interaction"]),
        n0=float(first["n0"]),
        orbital_convention=first["orbital_convention"],
        density_convention=first["density_convention"],
        bath_energies=_float_list(first["bath_energies"]),
        bath_couplings=_complex_list(first["bath_couplings"]),
        boson_frequencies=_float_list(first["boson_frequencies"]),
        boson_couplings=_float_list(first["boson_couplings"]),
    )
    if action.beta <= 0:
        raise ValueError("beta must be positive")
    if len(action.bath_energies) != len(action.bath_couplings):
        raise ValueError("bath arrays have unequal lengths")
    if len(action.boson_frequencies) != len(action.boson_couplings):
        raise ValueError("boson arrays have unequal lengths")
    if any(frequency <= 0 for frequency in action.boson_frequencies):
        raise ValueError("boson frequencies must be positive")
    if action.orbital_convention not in {"spinless", "spinful"}:
        raise ValueError("unsupported orbital convention")
    if action.density_convention not in {"shifted", "unshifted"}:
        raise ValueError("unsupported density convention")
    return action


def _triqs_imports():
    from triqs_ctseg import Solver
    from triqs.operators import Operator, n

    try:
        from triqs.gfs import Fourier, make_gf_from_fourier
    except ImportError:
        from triqs.gf import Fourier, make_gf_from_fourier
    return Solver, Operator, n, Fourier, make_gf_from_fourier


def _set_scalar_gf_values(gf, value_function) -> None:
    import numpy as np

    for index, point in enumerate(gf.mesh):
        gf.data[index, 0, 0] = value_function(complex(point.value))
    if not np.all(np.isfinite(gf.data)):
        raise ValueError("nonfinite Green-function input")


def _block_values(block) -> tuple["object", "object"]:
    import numpy as np

    mesh = np.asarray([float(point.value) for point in block.mesh])
    values = np.asarray(block.data).reshape(len(mesh), -1)[:, 0].real
    return mesh, values


def run_replica(action: Action, args, replica: int):
    import numpy as np

    Solver, Operator, n, Fourier, make_gf_from_fourier = _triqs_imports()
    active = ["up", "down"] if action.orbital_convention == "spinful" else ["0"]
    flavors = active if len(active) > 1 else ["0", "inactive"]
    solver = Solver(
        beta=action.beta,
        n_tau=args.n_tau,
        n_tau_bosonic=args.n_tau,
        gf_struct=[[flavor, 1] for flavor in flavors],
    )

    for flavor in active:
        delta_iw = make_gf_from_fourier(solver.Delta_tau[flavor])
        _set_scalar_gf_values(
            delta_iw,
            lambda z: sum(
                abs(coupling) ** 2 / (z - energy)
                for energy, coupling in zip(
                    action.bath_energies, action.bath_couplings)
            ),
        )
        solver.Delta_tau[flavor] << Fourier(delta_iw)

    for left in active:
        for right in active:
            interaction_iw = make_gf_from_fourier(solver.D0_tau[left, right])
            _set_scalar_gf_values(
                interaction_iw,
                lambda z: -sum(
                    2.0 * coupling**2 * frequency
                    / (frequency**2 - z**2)
                    for frequency, coupling in zip(
                        action.boson_frequencies,
                        action.boson_couplings,
                    )
                ),
            )
            solver.D0_tau[left, right] << Fourier(interaction_iw)

    # D0 represents coupling to n. Expanding g*(n-n0) gives an additional
    # one-body polaron shift +2*n0*g^2/omega for every active flavor.
    shift = 2.0 * action.n0 * sum(
        coupling**2 / frequency
        for frequency, coupling in zip(
            action.boson_frequencies, action.boson_couplings)
    )
    h_loc0 = Operator()
    for flavor in active:
        h_loc0 += (-action.mu + shift) * n(flavor, 0)
    h_int = Operator()
    if len(active) == 2:
        h_int += (
            action.static_interaction
            * n(active[0], 0)
            * n(active[1], 0)
        )

    solver.solve(
        h_loc0=h_loc0,
        h_int=h_int,
        measure_nn_tau=True,
        measure_densities=True,
        length_cycle=args.length_cycle,
        n_warmup_cycles=args.warmup_cycles,
        n_cycles=args.cycles,
        random_seed=args.seed + 104729 * replica,
    )

    target_tau = np.linspace(0.0, action.beta, args.output_grid)
    green_by_flavor = {}
    for flavor in active:
        mesh, values = _block_values(solver.results.G_tau[flavor])
        green_by_flavor[flavor] = np.interp(target_tau, mesh, values)
    green = green_by_flavor[active[0]]
    densities = {
        flavor: 1.0 + values[0]
        for flavor, values in green_by_flavor.items()
    }
    density = sum(densities.values())

    nn_total = np.zeros_like(target_tau)
    for left in active:
        for right in active:
            mesh, values = _block_values(solver.results.nn_tau[left, right])
            nn_total += np.interp(target_tau, mesh, values)
    chi = nn_total - density**2
    double_occupancy = None
    if len(active) == 2:
        mesh, values = _block_values(
            solver.results.nn_tau[active[0], active[1]])
        double_occupancy = float(np.interp(0.0, mesh, values))

    return {
        "tau": target_tau,
        "density": density,
        "double_occupancy": double_occupancy,
        "Gtau": green,
        "chi_nn": chi,
    }


def _jackknife(values):
    import numpy as np

    array = np.asarray(values)
    count = array.shape[0]
    if count < 2:
        raise ValueError("at least two replicas are required for jackknife errors")
    total = np.sum(array, axis=0)
    leave_one_out = (total - array) / (count - 1)
    estimate = np.mean(array, axis=0)
    center = np.mean(leave_one_out, axis=0)
    stderr = np.sqrt(
        (count - 1) / count
        * np.sum(np.abs(leave_one_out - center) ** 2, axis=0)
    )
    return estimate, stderr


def _fourier(values, times, beta: float, fermionic: bool, count: int = 16):
    import numpy as np

    transformed = []
    frequencies = []
    for index in range(count):
        frequency = (
            (2 * index + 1) * math.pi / beta
            if fermionic
            else 2 * index * math.pi / beta
        )
        frequencies.append(frequency)
        phase = np.exp(1j * frequency * times)
        transformed.append(np.trapezoid(values * phase, times, axis=-1))
    return np.asarray(frequencies), np.stack(transformed, axis=-1)


def _datum_rows(action: Action, replicas):
    import numpy as np

    rows = []
    density, density_error = _jackknife(
        [replica["density"] for replica in replicas])
    rows.append(("density", "scalar", 0.0, density, density_error))
    if replicas[0]["double_occupancy"] is not None:
        value, error = _jackknife(
            [replica["double_occupancy"] for replica in replicas])
        rows.append(("double_occupancy", "scalar", 0.0, value, error))

    times = replicas[0]["tau"]
    for observable in ("Gtau", "chi_nn"):
        mean, error = _jackknife(
            [replica[observable] for replica in replicas])
        rows.extend(
            (observable, "tau", coordinate, value, stderr)
            for coordinate, value, stderr in zip(times, mean, error)
        )

    green_replicas = np.asarray(
        [replica["Gtau"] for replica in replicas])
    frequencies, transformed = _fourier(
        green_replicas, times, action.beta, True)
    mean, error = _jackknife(transformed)
    rows.extend(
        ("Giw", "fermionic_iw", frequency, value, stderr)
        for frequency, value, stderr in zip(frequencies, mean, error)
    )

    chi_replicas = np.asarray(
        [replica["chi_nn"] for replica in replicas])
    frequencies, transformed = _fourier(
        chi_replicas, times, action.beta, False)
    mean, error = _jackknife(transformed)
    rows.extend(
        ("chi_nn_iv", "bosonic_iv", frequency, value, stderr)
        for frequency, value, stderr in zip(frequencies, mean, error)
    )
    return rows


def write_results(path: Path, action: Action, args, rows) -> None:
    header = [
        "action_hash",
        "beta",
        "mu",
        "static_interaction",
        "n0",
        "orbital_convention",
        "density_convention",
        "endpoint_convention",
        "density_connected",
        "fourier_convention",
        "equilibrated",
        "error_stable",
        "nsamples",
        "jackknife_bins",
        "cycles_per_replica",
        "warmup_cycles",
        "length_cycle",
        "seed",
        "observable",
        "axis",
        "coordinate",
        "mean_re",
        "mean_im",
        "stderr",
    ]
    prefix = [
        action.action_hash,
        action.beta,
        action.mu,
        action.static_interaction,
        action.n0,
        action.orbital_convention,
        action.density_convention,
        "zero_plus",
        "true",
        "positive_exponential",
        str(args.equilibrated).lower(),
        str(args.error_stable).lower(),
        args.cycles * args.replicas,
        args.replicas,
        args.cycles,
        args.warmup_cycles,
        args.length_cycle,
        args.seed,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for observable, axis, coordinate, mean, stderr in rows:
            value = complex(mean)
            writer.writerow(
                prefix
                + [
                    observable,
                    axis,
                    float(coordinate),
                    value.real,
                    value.imag,
                    float(stderr),
                ]
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=1_000_000)
    parser.add_argument("--warmup-cycles", type=int, default=100_000)
    parser.add_argument("--length-cycle", type=int, default=16)
    parser.add_argument("--n-tau", type=int, default=1001)
    parser.add_argument("--output-grid", type=int, default=65)
    parser.add_argument("--seed", type=int, default=26060727)
    parser.add_argument("--equilibrated", action="store_true")
    parser.add_argument("--error-stable", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    action = read_action(args.input)
    if args.replicas < 2:
        raise ValueError("replicas must be at least two")
    if args.output_grid < 2:
        raise ValueError("output-grid must be at least two")
    if args.dry_run:
        print(
            f"action={action.action_hash} beta={action.beta} "
            f"orbitals={action.orbital_convention} "
            f"bath_modes={len(action.bath_energies)} "
            f"boson_modes={len(action.boson_frequencies)}"
        )
        return
    replicas = [
        run_replica(action, args, replica)
        for replica in range(args.replicas)
    ]
    rows = _datum_rows(action, replicas)
    write_results(args.output, action, args, rows)
    print(
        f"wrote {len(rows)} rows to {args.output}; "
        f"equilibrated={args.equilibrated} "
        f"error_stable={args.error_stable}"
    )


if __name__ == "__main__":
    main()
