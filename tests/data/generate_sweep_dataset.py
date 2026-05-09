"""Dataset generation helper for FE / domain-radius sweeps under atom tests.

This script is the multi-case companion to ``generate_dataset.py``: both drive
``AtomicDataManager`` with the same SCF settings; this entrypoint only plans many
mesh/domain combinations (sweeps) for discretization studies. It does not imply
that other generators produce SCF runs that fail to converge.

This script provides a reproducible entrypoint for:
1) optional regeneration of large sweep batch datasets,
2) optional regeneration of summary figures used in manuscripts,
3) consistent manifest output for FE/domain sweep case tracking.

Script intent:
- Keep defaults safe for daily use (reuse existing data by default).
- Make expensive regeneration explicit via command-line flags.
- Keep folder structure stable across XC/use_oep variants.

How to run:
- Dry run (plan only):
  python delta/atom/tests/data/generate_sweep_dataset.py --dry-run

- Regenerate dataset only:
  python delta/atom/tests/data/generate_sweep_dataset.py --regenerate-data

- Enable both use_oep=False/True job sets:
  python delta/atom/tests/data/generate_sweep_dataset.py --use-oep-options both --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atom.src.data.data_manager import AtomicDataManager

# --- Solver settings (kept aligned with datagen_lda.py defaults) ---
XC_FUNCTIONAL            = "LDA_SVWN"
USE_OEP_OPTIONS_DEFAULT  = [False]
DOMAIN_SIZE_REF          = 40.0
FINITE_ELEMENTS_REF      = 12
POLYNOMIAL_ORDER         = 20
QUADRATURE_POINT_NUMBER  = 60
MESH_TYPE                = "exponential"
MESH_CONCENTRATION       = 101.0
SCF_TOLERANCE            = 1e-11
USE_PRECONDITIONER       = True
ATOMIC_NUMBER_LIST       = list(range(1, 93))
FE_SWEEP_RANGE           = range(2, 15)
DOMAIN_SWEEP_VALUES      = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]


def _parse_use_oep_options(mode: str) -> list[bool]:
    if mode == "false":
        return [False]
    if mode == "true":
        return [True]
    return [False, True]


def _functional_root(base: Path, xc_functional: str, use_oep: bool) -> Path:
    suffix = "_oep" if use_oep else ""
    return base / f"{xc_functional.lower()}{suffix}"


def _subdir_fe_sweep(xc_functional: str, use_oep: bool, fe: int) -> str:
    return str(
        _functional_root(Path("."), xc_functional, use_oep)
        / "finite_element_sweep"
        / f"fe{fe:02d}_R{int(DOMAIN_SIZE_REF):03d}"
    )


def _subdir_domain_sweep(xc_functional: str, use_oep: bool, r: float) -> str:
    return str(
        _functional_root(Path("."), xc_functional, use_oep)
        / "domain_radius_sweep"
        / f"fe{FINITE_ELEMENTS_REF:02d}_R{int(r):03d}"
    )


def build_manifest(base: Path, xc_functional: str, use_oep_options: list[bool]) -> list[dict]:
    """Build all jobs for selected sweeps and OEP modes."""
    arms: list[dict] = []

    for use_oep in use_oep_options:
        for fe in FE_SWEEP_RANGE:
            arms.append(
                {
                    "xc_functional"         : xc_functional,
                    "use_oep"               : bool(use_oep),
                    "study"                 : "finite_element_sweep",
                    "finite_elements_number": int(fe),
                    "domain_size"           : float(DOMAIN_SIZE_REF),
                    "data_subdir"           : _subdir_fe_sweep(xc_functional, bool(use_oep), fe),
                }
            )

        for r in DOMAIN_SWEEP_VALUES:
            arms.append(
                {
                    "xc_functional"         : xc_functional,
                    "use_oep"               : bool(use_oep),
                    "study"                 : "domain_radius_sweep",
                    "finite_elements_number": int(FINITE_ELEMENTS_REF),
                    "domain_size"           : float(r),
                    "data_subdir"           : _subdir_domain_sweep(xc_functional, bool(use_oep), r),
                }
            )

    for arm in arms:
        arm["data_root"] = str(base / arm["data_subdir"])
    return arms


def _generate_jobs(base_dir: Path, jobs: list[dict]) -> None:
    for job in jobs:
        sub = job["data_subdir"]
        print("\n" + "=" * 75)
        print(
            "Generating: "
            f"{job['xc_functional']} :: {sub}  "
            f"(FE={job['finite_elements_number']}, domain_size={job['domain_size']}, "
            f"use_oep={job['use_oep']})"
        )
        print("=" * 75)

        atomic_data_manager = AtomicDataManager(
            data_root                   = str(base_dir / sub),
            scf_xc_functional           = str(job["xc_functional"]),
            forward_pass_xc_functionals = None,
            auto_confirm                = True,
        )

        atomic_data_manager.generate_data(
            atomic_number_list        = ATOMIC_NUMBER_LIST,
            n_electrons_list          = None,
            start_configuration_index = 1,
            domain_size               = float(job["domain_size"]),
            finite_elements_number    = int(job["finite_elements_number"]),
            polynomial_order          = POLYNOMIAL_ORDER,
            quadrature_point_number   = QUADRATURE_POINT_NUMBER,
            mesh_type                 = MESH_TYPE,
            mesh_concentration        = MESH_CONCENTRATION,
            scf_tolerance             = SCF_TOLERANCE,
            max_scf_iterations        = 200,
            use_preconditioner        = USE_PRECONDITIONER,
            use_oep                   = bool(job["use_oep"]),
            save_energy_density       = True,
            save_intermediate         = False,
            save_full_spectrum        = True,
            save_full_orbitals        = False,
            save_derivative_matrix    = False,
            verbose                   = True,
            overwrite                 = True,
        )


def _regenerate_summary(base_dir: Path) -> None:
    summary_script = Path(__file__).resolve().parents[3] / "data_functional_sweeps" / "summary.py"
    if not summary_script.is_file():
        raise FileNotFoundError(f"Cannot find summary script: {summary_script}")

    print(f"\nRegenerating summary figures with: {summary_script}")
    command = [
        sys.executable,
        str(summary_script),
        "--base-dir",
        str(base_dir),
        "--out-dir",
        str(base_dir),
        "--refresh",
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate datasets over finite-element and domain-radius sweeps (stable folder layout). "
            "By default, existing dataset/summary are reused."
        )
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
        help="SCF XC functional label (default: LDA_SVWN).",
    )
    parser.add_argument(
        "--use-oep-options",
        type=str,
        choices=("false", "true", "both"),
        default="false",
        help="Job set selection for use_oep: false, true, or both (default: false).",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write manifest and exit.",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="If set, run only one job by index in manifest.",
    )
    parser.add_argument(
        "--only-subdir",
        type=str,
        default=None,
        help="If set, only run one exact data_subdir path from manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs only (no generation, no summary).",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="If set, regenerate selected dataset cases.",
    )
    parser.add_argument(
        "--regenerate-summary",
        action="store_true",
        help="If set, regenerate summary figures from dataset folders.",
    )
    args = parser.parse_args()

    base_dir: Path = args.base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    use_oep_options = _parse_use_oep_options(args.use_oep_options)
    manifest = build_manifest(base_dir, args.xc_functional, use_oep_options)
    manifest_path = base_dir / "convergence_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest ({len(manifest)} arms) -> {manifest_path}")

    if args.manifest_only:
        return

    to_run = list(manifest)
    if args.only_subdir:
        to_run = [j for j in to_run if j["data_subdir"] == args.only_subdir]
        if not to_run:
            raise SystemExit(f"No matching job for subdir {args.only_subdir!r}")

    if args.job_index is not None:
        if args.job_index < 0 or args.job_index >= len(manifest):
            raise SystemExit(f"--job-index must be in [0, {len(manifest) - 1}]")
        selected = manifest[args.job_index]
        to_run = [selected]
        print(
            f"Selected job index {args.job_index}: "
            f"{selected['data_subdir']} (FE={selected['finite_elements_number']}, "
            f"R={selected['domain_size']}, use_oep={selected['use_oep']})"
        )

    if args.dry_run:
        for j in to_run:
            print(
                f"  [PLAN] {j['xc_functional']} :: {j['data_subdir']}: "
                f"FE={j['finite_elements_number']}, R={j['domain_size']}, use_oep={j['use_oep']}"
            )
        return

    if args.regenerate_data:
        _generate_jobs(base_dir, to_run)
        print("\nData regeneration completed.")
    else:
        print("\nSkip data regeneration (default behavior). Use --regenerate-data to regenerate.")

    if args.regenerate_summary:
        _regenerate_summary(base_dir)
        print("Summary regeneration completed.")
    else:
        print("Skip summary regeneration (default behavior). Use --regenerate-summary to regenerate.")

    print("\nDone.")


if __name__ == "__main__":
    main()
