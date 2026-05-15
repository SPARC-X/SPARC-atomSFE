"""Build per-dataset summary JSON files from configuration out.txt logs.

Output layout mirrors dataset directories:
  <base-dir>/summary/<functional>/<sweep>/<dataset>/fe12_R040__z1_92.json

Filenames are ``{fe##}_R###__z1_92.json`` (or ``__charged`` for charged subsets), derived from
``input_parameters`` unless ``--output-name`` is set.

Two-part dataset roots (no sweep/case folder) are also supported, e.g. ``hf/charged``.

For each configuration, this script extracts:
1) energy components from the final "Total Energy (...)" block in out.txt
   (including TOTAL ENERGY),
2) occupied eigenvalues,
3) the lowest 5 unoccupied eigenvalues.

Important behavior:
- No error metrics are computed or stored.
- Dry-run mode performs scanning/parsing but does not create folders or files.
- Progress is shown with a live-refresh status line (similar to cleanup_dataset.py).

How to run:
- Write summaries under ``summary/`` (default ``base-dir`` is this script's folder):
  python delta/atom/tests/data/build_summary_from_out.py

- Scan only (no files written):
  python delta/atom/tests/data/build_summary_from_out.py --dry-run

- Custom dataset root or summary folder name:
  python delta/atom/tests/data/build_summary_from_out.py --base-dir path/to/tests/data --summary-dir-name summary
"""

from __future__ import annotations

import argparse
import json
import re

from summary_naming import summary_basename_from_payload
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atom.utils.occupation_states import OccupationInfo


# ``(functional, leaf)`` paths that are dataset roots with only two segments under base-dir.
_FLAT_TWO_PART_DATASET_ROOTS = frozenset(
    {
        ("hf", "charged"),
    }
)


def _dataset_rel_path_accepted(rel_dataset: Path) -> bool:
    """True if ``rel_dataset`` matches sweep layout (>=3 parts) or a registered flat root."""
    parts = rel_dataset.parts
    if len(parts) >= 3:
        return True
    if len(parts) == 2 and parts in _FLAT_TWO_PART_DATASET_ROOTS:
        return True
    return False


ENERGY_LINE_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9\-\s\(\)\/]+?)\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+Ha\s*$"
)
INPUT_PARAMETER_LINE_PATTERN = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*?)\s*$")
TOTAL_ENERGY_HEADER_PATTERN = re.compile(r"^\s*Total\s+Energy\s*\(", re.IGNORECASE)
NOT_CONVERGED_PATTERN = re.compile(r"did\s+not\s+converge", re.IGNORECASE)
INPUT_PARAMETER_EXCLUDE_KEYS = {"atomic_number", "n_electrons"}
ENERGY_COMPONENT_KEY_MAP = {
    "Kinetic (radial)"    : "kinetic_radial",
    "Kinetic (angular)"   : "kinetic_angular",
    "Total Kinetic"       : "total_kinetic",
    "External potential"  : "external_potential",
    "Hartree"             : "hartree",
    "Exchange"            : "exchange",
    "Correlation"         : "correlation",
    "Nonlocal PSP"        : "nonlocal_psp",
    "HF Exchange"         : "hf_exchange",
    "Exact exchange (HF)" : "exact_exchange_hf",
    "RPA Correlation"     : "rpa_correlation",
    "ML Energy Correction": "ml_energy_correction",
    "Total Potential"     : "total_potential",
    "TOTAL ENERGY"        : "total_energy",
}


def _truncate_path(path_text: str, max_len: int = 120) -> str:
    if len(path_text) <= max_len:
        return path_text
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return f"{path_text[:left]}...{path_text[-right:]}"


def _print_live_status(prefix: str, rel_path: str, index: int, total: int) -> None:
    status = (
        f"[{prefix}] {index}/{total} checking={_truncate_path(rel_path, max_len=140)}"
    )
    print(f"\r{status}", end="", flush=True)


def _print_live_scan_status(
    prefix: str,
    rel_path: str,
    visited_dirs: int,
    matched_cfg_dirs: int,
    collected_cases: int,
) -> None:
    status = (
        f"[{prefix}] visited_dirs={visited_dirs:,} matched_cfg_dirs={matched_cfg_dirs:,} "
        f"cases={collected_cases:,} checking={_truncate_path(rel_path, max_len=110)}"
    )
    print(f"\r{status}", end="", flush=True)


def _clear_live_status_line() -> None:
    print("\r" + (" " * 220) + "\r", end="", flush=True)


def _normalize_key(label: str) -> str:
    key = label.strip().lower()
    key = re.sub(r"[^\w]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    return key


def _extract_final_energy_block(text: str) -> dict[str, float]:
    lines = text.splitlines()
    header_idx = -1
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("total energy ("):
            header_idx = i
    if header_idx < 0:
        return {}

    energies: dict[str, float] = {}
    for line in lines[header_idx:]:
        stripped = line.strip()
        if ":" not in stripped or "Ha" not in stripped:
            continue
        label_raw, value_raw = stripped.split(":", 1)
        label = label_raw.strip()

        key = ENERGY_COMPONENT_KEY_MAP.get(label, _normalize_key(label))
        value_token = value_raw.replace("Ha", "").strip().split()[0]
        try:
            energies[key] = float(value_token)
        except (TypeError, ValueError):
            continue
    return energies


def _coerce_input_parameter_value(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return value

    lower = value.lower()
    if lower == "none":
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False

    # Handle values like "1    (H)" by taking the leading numeric token.
    numeric_prefix = value.split("(")[0].strip()
    if re.fullmatch(r"[-+]?\d+", numeric_prefix):
        try:
            return int(numeric_prefix)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", numeric_prefix):
        try:
            return float(numeric_prefix)
        except ValueError:
            pass

    return value


def _extract_input_parameters_block(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    start_idx = -1
    for i, line in enumerate(lines):
        if "INPUT PARAMETERS" in line.upper():
            start_idx = i
            break
    if start_idx < 0:
        return {}

    params: dict[str, Any] = {}
    for line in lines[start_idx + 1 :]:
        upper = line.upper()
        if (
            "PSEUDOPOTENTIAL INFORMATION" in upper
            or "OCCUPATION INFORMATION" in upper
            or "SELF-CONSISTENT FIELD" in upper
        ):
            break

        m = INPUT_PARAMETER_LINE_PATTERN.match(line)
        if not m:
            continue
        key = m.group(1).strip()
        if key in INPUT_PARAMETER_EXCLUDE_KEYS:
            continue
        raw_value = m.group(2).strip()
        params[key] = _coerce_input_parameter_value(raw_value)
    return params


def _find_out_txt(configuration_dir: Path) -> Path | None:
    matches = sorted(configuration_dir.rglob("out.txt"))
    if not matches:
        return None
    return matches[0]


def _read_atomic_number(configuration_dir: Path) -> int | None:
    meta_path = configuration_dir / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if "atomic_number" not in meta:
        return None
    try:
        return int(round(float(meta["atomic_number"])))
    except (TypeError, ValueError):
        return None


def _read_meta(configuration_dir: Path) -> dict[str, Any] | None:
    meta_path = configuration_dir / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_n_electrons(configuration_dir: Path) -> float | None:
    meta = _read_meta(configuration_dir)
    if not meta or "n_electrons" not in meta:
        return None
    try:
        return float(meta["n_electrons"])
    except (TypeError, ValueError):
        return None


def _read_occupied_state_count(configuration_dir: Path) -> int | None:
    meta = _read_meta(configuration_dir)
    if not meta or "atomic_number" not in meta:
        return None
    try:
        z = int(round(float(meta["atomic_number"])))
        n_electrons = float(meta.get("n_electrons", z))
    except (TypeError, ValueError):
        return None
    occ = OccupationInfo(
        z_nuclear         = z,
        z_valence         = z,
        all_electron_flag = True,
        n_electrons       = n_electrons,
    )
    return int(occ.n_states)


def _read_full_eigen_energies(configuration_dir: Path) -> np.ndarray | None:
    candidates = sorted(configuration_dir.rglob("full_eigen_energies.txt"))
    if not candidates:
        return None
    try:
        return np.atleast_1d(np.loadtxt(candidates[0], dtype=float))
    except (OSError, ValueError):
        return None


def _iter_dataset_dirs(base_dir: Path):
    """Yield dataset directories incrementally from configuration_* folders."""
    seen: set[Path] = set()
    matched_cfg_dirs = 0
    for cfg_dir in base_dir.rglob("configuration_*"):
        if not cfg_dir.is_dir():
            continue
        matched_cfg_dirs += 1
        dataset_dir = cfg_dir.parent
        try:
            rel_dataset = dataset_dir.relative_to(base_dir).as_posix()
        except ValueError:
            continue
        if rel_dataset.startswith("summary/") or rel_dataset == "summary":
            continue

        _print_live_scan_status(
            prefix           = "DATASET-SCAN",
            rel_path         = rel_dataset,
            visited_dirs     = matched_cfg_dirs,
            matched_cfg_dirs = matched_cfg_dirs,
            collected_cases  = len(seen),
        )

        if dataset_dir in seen:
            continue
        seen.add(dataset_dir)
        _clear_live_status_line()
        yield dataset_dir

    _clear_live_status_line()


def _build_dataset_summary(dataset_dir: Path) -> dict[str, Any]:
    configs = sorted(
        p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("configuration_")
    )
    rows: list[dict[str, Any]] = []
    dataset_input_parameters: dict[str, Any] | None = None

    for idx, cfg in enumerate(configs, start=1):
        _print_live_status(
            prefix = "CFG",
            rel_path = cfg.as_posix(),
            index = idx,
            total = len(configs),
        )
        out_path = _find_out_txt(cfg)
        if out_path is None:
            continue
        try:
            out_text = out_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        energies = _extract_final_energy_block(out_text)
        if not energies:
            continue
        input_parameters = _extract_input_parameters_block(out_text)
        if dataset_input_parameters is None and input_parameters:
            dataset_input_parameters = input_parameters

        full_eigs = _read_full_eigen_energies(cfg)
        n_occ = _read_occupied_state_count(cfg)
        occupied_eigs: list[float] = []
        lowest_unoccupied_5: list[float] = []
        if full_eigs is not None and n_occ is not None:
            n_occ_clamped = max(0, min(int(n_occ), int(full_eigs.shape[0])))
            occupied_eigs = [float(v) for v in full_eigs[:n_occ_clamped].tolist()]
            lowest_unoccupied_5 = [
                float(v) for v in full_eigs[n_occ_clamped : n_occ_clamped + 5].tolist()
            ]

        row: dict[str, Any] = {
            "configuration"                : cfg.name,
            "atomic_number"                : _read_atomic_number(cfg),
            "n_electrons"                  : _read_n_electrons(cfg),
            "converged"                    : not bool(NOT_CONVERGED_PATTERN.search(out_text)),
            "energies_ha"                  : energies,
            "total_energy_ha"              : energies.get("total_energy"),
            "occupied_state_count"         : n_occ,
            "occupied_eigenvalues_ha"      : occupied_eigs,
            "lowest_5_unoccupied_eigs_ha"  : lowest_unoccupied_5,
        }
        rows.append(row)

    _clear_live_status_line()
    return {
        "dataset_dir"       : dataset_dir.as_posix(),
        "input_parameters"  : dataset_input_parameters if dataset_input_parameters is not None else {},
        "n_configurations"  : len(configs),
        "n_summarized"      : len(rows),
        "config_summaries"  : rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-configuration energy summaries from out.txt into summary/ tree."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset base directory containing functional folders (default: script directory).",
    )
    parser.add_argument(
        "--summary-dir-name",
        type=str,
        default="summary",
        help="Name of summary folder to create under base-dir (default: summary).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Override summary JSON basename (default: auto, e.g. fe12_R040__z1_92.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report planned outputs without writing files.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    summary_root = base_dir / args.summary_dir_name
    mode = "DRY-RUN" if args.dry_run else "WRITE"
    print(f"[{mode}] base_dir={base_dir}")
    print(f"[{mode}] summary_root={summary_root}")
    print(f"[{mode}] output_name={args.output_name or '(auto)'}")
    written = 0
    print(f"[{mode}] Streaming dataset discovery and summary writing...")
    for i, dataset_dir in enumerate(_iter_dataset_dirs(base_dir), start=1):
        rel_dataset = dataset_dir.relative_to(base_dir)
        if not _dataset_rel_path_accepted(rel_dataset):
            _clear_live_status_line()
            print(
                "[WARN] Skip unexpected dataset path "
                "(need functional/sweep/case, or a registered two-part root such as hf/charged): "
                f"{rel_dataset.as_posix()}",
                flush=True,
            )
            continue

        dataset_summary = _build_dataset_summary(dataset_dir)
        out_name = args.output_name or summary_basename_from_payload(dataset_summary, rel_dataset)
        out_path = summary_root / rel_dataset / out_name
        _print_live_status(
            prefix   = "DATASET",
            rel_path = rel_dataset.as_posix(),
            index    = i,
            total    = i,
        )
        _clear_live_status_line()
        print(
            f"[DATASET {i}] summarized "
            f"{dataset_summary['n_summarized']}/{dataset_summary['n_configurations']}",
            flush=True,
        )

        if args.dry_run:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(dataset_summary, indent=2) + "\n", encoding="utf-8")
        print(
            f"[DATASET {i}] wrote {_truncate_path(out_path.as_posix(), max_len=150)}",
            flush=True,
        )
        written += 1

    if args.dry_run:
        print("[DRY-RUN] Completed. No files written.")
    else:
        print(f"[WRITE] Done. Wrote {written} summary JSON files under: {summary_root}")


if __name__ == "__main__":
    main()

