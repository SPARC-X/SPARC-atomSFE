"""Shared summary JSON filenames: {mesh_tag}__{z_suffix}.json (e.g. fe12_R040__z1_92.json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUMMARY_GLOB = "fe*_R*__*.json"


def mesh_tag(finite_element_number: int, domain_size: float) -> str:
    return f"fe{int(finite_element_number):02d}_R{int(round(float(domain_size))):03d}"


def z_suffix(rel_path: Path | str, n_configurations: int) -> str:
    parts = Path(rel_path).parts
    if any(p == "charged" for p in parts):
        return "charged"
    if int(n_configurations) == 92:
        return "z1_92"
    return f"z{int(n_configurations)}c"


def summary_basename(mesh: str, z_part: str) -> str:
    return f"{mesh}__{z_part}.json"


def summary_basename_from_payload(payload: dict[str, Any], rel_path: Path | str) -> str:
    params = payload.get("input_parameters") or {}
    fe = int(params.get("finite_element_number", 12))
    r = float(params.get("domain_size", 40.0))
    mesh = mesh_tag(fe, r)
    n_cfg = int(payload.get("n_configurations", 0))
    return summary_basename(mesh, z_suffix(rel_path, n_cfg))


def summary_path_for_case_dir(case_dir: Path) -> Path | None:
    """Return summary JSON inside a sweep case directory (e.g. .../fe12_R040/)."""
    preferred = case_dir / summary_basename(case_dir.name, "z1_92")
    if preferred.is_file():
        return preferred
    matches = sorted(case_dir.glob(SUMMARY_GLOB))
    if matches:
        return matches[0]
    legacy = case_dir / "configuration_energy_summary.json"
    return legacy if legacy.is_file() else None


def glob_sweep_summaries(sweep_dir: Path) -> list[Path]:
    paths = sorted(sweep_dir.glob(f"*/{SUMMARY_GLOB}"))
    if paths:
        return paths
    return sorted(sweep_dir.glob("*/configuration_energy_summary.json"))


def resolve_summary_path(directory: Path) -> Path | None:
    """First ``fe*_R*__*.json`` in *directory*, else legacy ``configuration_energy_summary.json``."""
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(SUMMARY_GLOB))
    if matches:
        return matches[0]
    legacy = directory / "configuration_energy_summary.json"
    return legacy if legacy.is_file() else None


def resolve_summary_under(root: Path, case: str = "") -> Path | None:
    """Summary under flat *root*, or *root/case* (sweep case dir or ``.json`` basename)."""
    if case:
        sub = root / case
        if sub.is_file() and sub.suffix.lower() == ".json":
            return sub
        if sub.is_dir():
            found = resolve_summary_path(sub)
            if found is not None:
                return found
    return resolve_summary_path(root)


def default_all_electron_summary(data_dir: Path, xc: str) -> Path:
    d = data_dir / "summary" / "all_electron" / xc
    found = resolve_summary_path(d)
    return found if found is not None else d / "fe12_R040__z1_92.json"


def default_pseudo_summary(data_dir: Path, xc: str) -> Path:
    d = data_dir / "summary" / "pseudo_potential" / xc
    found = resolve_summary_path(d)
    return found if found is not None else d / "fe10_R040__z7c.json"
