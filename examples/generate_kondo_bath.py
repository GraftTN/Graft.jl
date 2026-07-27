#!/usr/bin/env python3
"""Generate the paper-scale semicircular Kondo bath with adapol."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from adapol import hybfit


def semicircular_hybridization(z: np.ndarray) -> np.ndarray:
    root = np.sqrt(z * z - 1.0)
    root = np.where(np.signbit(root.imag) == np.signbit(z.imag), root, -root)
    return 2.0 * (z - root)


def discrete_hybridization(
    z: np.ndarray, energies: np.ndarray, couplings: np.ndarray
) -> np.ndarray:
    return np.sum(
        np.abs(couplings)[None, :] ** 2 / (z[:, None] - energies[None, :]),
        axis=1,
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "kondo_semicircular_bath_29.csv",
    )
    args = parser.parse_args()

    beta = 10_000.0
    half_grid_size = 4_000
    indices = np.arange(-half_grid_size, half_grid_size, dtype=float)
    frequencies = 1j * (2.0 * indices + 1.0) * np.pi / beta
    target = semicircular_hybridization(frequencies)
    energies, bath_hybridizations, training_error, _ = hybfit(
        target,
        frequencies,
        tol=1e-6,
        mmax=50,
        maxiter=500,
    )
    couplings = bath_hybridizations[:, 0]

    validation_omega = np.geomspace(np.pi / 1024.0, 100.0, 4096)
    validation_z = 1j * validation_omega
    validation_error = np.max(
        np.abs(
            discrete_hybridization(validation_z, energies, couplings)
            - semicircular_hybridization(validation_z)
        )
    )
    if len(energies) != 29 or validation_error > 1e-6:
        raise RuntimeError(
            "adapol fit failed the paper gate: "
            f"sites={len(energies)}, validation_error={validation_error:.6e}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((energies, couplings.real, couplings.imag))
    np.savetxt(
        args.output,
        data,
        delimiter=",",
        header="energy,coupling_re,coupling_im",
        comments="",
        fmt="%.17g",
    )
    print(
        f"wrote {len(energies)} sites to {args.output}; "
        f"training_error={training_error:.6e}, "
        f"validation_error={validation_error:.6e}"
    )


if __name__ == "__main__":
    main()
