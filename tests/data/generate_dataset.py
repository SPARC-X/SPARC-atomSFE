"""Generate one dataset for atom tests.

This script is a single-case dataset generator:
1) it creates exactly one dataset per run (no FE/radius sweep, no reference set),
2) it writes data under a short folder name: ``<xc_lower>/`` (default preset) or
   ``hf/charged/`` (HF charged preset); FE count and domain radius stay in the manifest
   only, not in the path,
3) it writes a one-entry manifest for reproducibility.

Presets (``--dataset``):
- ``default``: neutral atoms Z=1..92 under ``lda_svwn/`` (or ``<xc-functional>/``).
- ``hf_charged``: Hartree-Fock for the Lehtola (2019) Table 5 species set; ``Z`` and
  ``N_e`` are the fixed lists ``HF_CHARGED_ATOMIC_NUMBER_LIST`` and
  ``HF_CHARGED_N_ELECTRONS_LIST`` (same order as ``reference/hf/charged_atoms_hf.json``
  ``records``). Output under ``hf/charged/``. For this preset,
  ``--xc-functional`` / ``--use-oep`` are ignored.

How to run:
- Dry run (default LDA preset):
  python delta/atom/tests/data/generate_dataset.py --dry-run

- Plan HF charged dataset:
  python delta/atom/tests/data/generate_dataset.py --dataset hf_charged --dry-run

- Regenerate HF charged dataset:
  python delta/atom/tests/data/generate_dataset.py --dataset hf_charged --regenerate-data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atom.src.data.data_manager import AtomicDataManager

# --- Solver settings for one dataset (default preset) ---
XC_FUNCTIONAL            = "LDA_SVWN"
USE_OEP_DEFAULT          = False
DOMAIN_SIZE              = 40.0
FINITE_ELEMENTS_NUMBER   = 12
POLYNOMIAL_ORDER         = 20
QUADRATURE_POINT_NUMBER  = 60
MESH_TYPE                = "exponential"
MESH_CONCENTRATION       = 101.0
SCF_TOLERANCE            = 1e-11
USE_PRECONDITIONER       = True
ATOMIC_NUMBER_LIST       = list(range(1, 93))

# --- HF charged (Lehtola 2019 Table 5): same species order as charged_atoms_hf.json ---
# Species: H-, He, Li+, Li-, Be, B+, C-, N, O+, F-, Ne, Na+, Na-, Mg, Al+, Si-, P, S+, Cl-, Ar
HF_CHARGED_ATOMIC_NUMBER_LIST = \
    [1, 2, 3, 3, 4, 5, 6, 7, 8,  9, 10, 11, 11, 12, 13, 14, 15, 16, 17, 18]
HF_CHARGED_N_ELECTRONS_LIST = \
    [2, 2, 2, 4, 4, 4, 7, 7, 7, 10, 10, 10, 12, 12, 12, 15, 15, 15, 18, 18]

_HF_CHARGED_DATA_SUBDIR_PREFIX = "hf/charged"
_HF_CHARGED_SCF_TOLERANCE = 1e-10


def _data_subdir_slug(xc_functional: str, use_oep: bool) -> str:
    """One folder per XC label, e.g. ``lda_svwn``, ``gga_pbe``, ``hf``."""
    suffix = "_oep" if use_oep else ""
    return f"{xc_functional.lower()}{suffix}"


def build_manifest(
    base          : Path,
    xc_functional : str,
    use_oep       : bool,
    fe_number     : int,
    radius        : float,
) -> list[dict]:
    """Build a one-entry manifest for single-case dataset generation (default layout)."""
    arm = {
        "xc_functional"          : str(xc_functional),
        "use_oep"                : bool(use_oep),
        "study"                  : "flat_single",
        "finite_elements_number" : int(fe_number),
        "domain_size"            : float(radius),
        "data_subdir"            : _data_subdir_slug(xc_functional, use_oep),
    }
    arm["data_root"] = str(base / arm["data_subdir"])
    return [arm]


def build_manifest_hf_charged(
    base      : Path,
    fe_number : int,
    radius    : float,
) -> list[dict]:
    """One-entry manifest: HF, Lehtola charged table, under ``hf/charged/``."""
    arm = {
        "preset"                   : "hf_charged",
        "xc_functional"            : "HF",
        "use_oep"                  : False,
        "study"                    : "flat_single",
        "finite_elements_number"   : int(fe_number),
        "domain_size"              : float(radius),
        "data_subdir"              : _HF_CHARGED_DATA_SUBDIR_PREFIX,
        "atomic_number_list"       : list(HF_CHARGED_ATOMIC_NUMBER_LIST),
        "n_electrons_list"         : [float(x) for x in HF_CHARGED_N_ELECTRONS_LIST],
        "scf_tolerance"            : _HF_CHARGED_SCF_TOLERANCE,
        "hybrid_mixing_parameter"  : 1.0,
        "max_scf_iterations_outer" : 50,
    }
    arm["data_root"] = str(base / arm["data_subdir"])
    return [arm]


def _generate_job(base_dir: Path, job: dict) -> None:
    sub = job["data_subdir"]
    print("\n" + "=" * 75)
    print(
        "Generating: "
        f"{job['xc_functional']} :: {sub}  "
        f"(FE={job['finite_elements_number']}, domain_size={job['domain_size']}, "
        f"use_oep={job['use_oep']})"
    )
    if job.get("preset") == "hf_charged":
        n = len(job["atomic_number_list"])
        print(f"  preset=hf_charged, {n} configurations (Lehtola Table 5 order)")
    print("=" * 75)

    atomic_data_manager = AtomicDataManager(
        data_root=str(base_dir / sub),
        scf_xc_functional=str(job["xc_functional"]),
        forward_pass_xc_functionals=None,
        auto_confirm=True,
    )

    z_list = job.get("atomic_number_list", ATOMIC_NUMBER_LIST)
    ne_list = job.get("n_electrons_list")

    atomic_data_manager.generate_data(
        atomic_number_list        = z_list,
        n_electrons_list          = ne_list,
        start_configuration_index = 1,
        domain_size               = float(job["domain_size"]),
        finite_elements_number    = int(job["finite_elements_number"]),
        polynomial_order          = POLYNOMIAL_ORDER,
        quadrature_point_number   = QUADRATURE_POINT_NUMBER,
        mesh_type                 = MESH_TYPE,
        mesh_concentration        = MESH_CONCENTRATION,
        scf_tolerance             = float(job.get("scf_tolerance", SCF_TOLERANCE)),
        max_scf_iterations        = 200,
        max_scf_iterations_outer  = job.get("max_scf_iterations_outer"),
        use_preconditioner          = USE_PRECONDITIONER,
        use_oep                   = bool(job["use_oep"]),
        hybrid_mixing_parameter   = job.get("hybrid_mixing_parameter"),
        save_energy_density       = True,
        save_intermediate         = False,
        save_full_spectrum        = True,
        save_full_orbitals        = False,
        save_derivative_matrix    = False,
        verbose                   = True,
        overwrite                 = True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one convergence-oriented dataset with a single FE/radius parameter set."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("default", "hf_charged"),
        default="default",
        help="default=LDA_SVWN Z=1..92; hf_charged=HF Lehtola charged species (see module lists).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset root (default: current folder of this script).",
    )
    parser.add_argument(
        "--xc-functional",
        type=str,
        default=XC_FUNCTIONAL,
        help="SCF XC functional (ignored for --dataset hf_charged).",
    )
    parser.add_argument(
        "--use-oep",
        action="store_true",
        default=USE_OEP_DEFAULT,
        help="Enable OEP (ignored for --dataset hf_charged).",
    )
    parser.add_argument(
        "--domain-size",
        type=float,
        default=DOMAIN_SIZE,
        help="Domain radius in Bohr for the single dataset.",
    )
    parser.add_argument(
        "--finite-elements-number",
        type=int,
        default=FINITE_ELEMENTS_NUMBER,
        help="Finite-element count for the single dataset.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write manifest and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned job only; do not call generate_data.",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="If set, run dataset generation for the single planned job.",
    )
    args = parser.parse_args()

    base_dir: Path = args.base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "hf_charged":
        if len(HF_CHARGED_ATOMIC_NUMBER_LIST) != len(HF_CHARGED_N_ELECTRONS_LIST):
            print("HF charged list length mismatch", file=sys.stderr)
            sys.exit(1)
        manifest = build_manifest_hf_charged(
            base=base_dir,
            fe_number=int(args.finite_elements_number),
            radius=float(args.domain_size),
        )
    else:
        manifest = build_manifest(
            base          = base_dir,
            xc_functional = args.xc_functional,
            use_oep       = bool(args.use_oep),
            fe_number     = int(args.finite_elements_number),
            radius        = float(args.domain_size),
        )

    manifest_path = base_dir / "convergence_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest ({len(manifest)} arm) -> {manifest_path}")

    if args.manifest_only:
        return

    job = manifest[0]
    if args.dry_run:
        print(
            f"  [PLAN] {job['xc_functional']} :: {job['data_subdir']}: "
            f"FE={job['finite_elements_number']}, R={job['domain_size']}, use_oep={job['use_oep']}"
        )
        if job.get("preset") == "hf_charged":
            print(f"  hf_charged: {len(HF_CHARGED_ATOMIC_NUMBER_LIST)} (Z, N_e) pairs")
        return

    if args.regenerate_data:
        _generate_job(base_dir, job)
        print("\nData regeneration completed.")
    else:
        print("\nSkip data regeneration (default behavior). Use --regenerate-data to regenerate.")

    print("\nDone.")


if __name__ == "__main__":
    main()
