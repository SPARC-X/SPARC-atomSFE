"""
Compare GGA-PBE finite-element summary against HelFEM reference JSON (even Z subset).

Default comparison:
- Reference: tests/data/reference/gga_pbe/lehtola_even_z_2_to_20_gga_pbe.json
- Ours:      tests/data/summary/all_electron/gga_pbe/fe12_R040__z1_92.json

Outputs a text report in this folder:
    gga_pbe_accuracy_test_lehtola_summary.txt

Run from repo root::

    python atom/tests/data/compare/gga_pbe_accuracy_test_lehtola.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import sys

_DATA_DIR = Path(__file__).resolve().parent.parent
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
from summary_naming import default_all_electron_summary

_DEFAULT_REFERENCE = _DATA_DIR / "reference" / "gga_pbe" / "lehtola_even_z_2_to_20_gga_pbe.json"
_DEFAULT_SUMMARY = default_all_electron_summary(_DATA_DIR, "gga_pbe")
_DEFAULT_OUT_TXT = Path(__file__).resolve().parent / "gga_pbe_accuracy_test_lehtola_summary.txt"

# Aufbau-like occupied subshell order (by l) for low-Z atoms:
# 1s, 2s, 2p, 3s, 3p, 4s, ...
_OCCUPIED_L_SEQUENCE = [0, 0, 1, 0, 1, 0, 2, 1, 0]


def _fmt_fixed_total_digits(value: float, total_digits: int = 12) -> str:
    """Format in fixed-point (no scientific) with roughly `total_digits` significant width."""
    if not np.isfinite(value):
        return str(value)

    abs_v = abs(value)
    if abs_v < 1.0:
        int_digits = 1
    else:
        int_digits = int(math.floor(math.log10(abs_v))) + 1

    frac_digits = max(0, total_digits - int_digits)
    return f"{value:.{frac_digits}f}"


def _load_reference_even_z(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_z: dict[int, dict] = {}
    for atom in data.get("atoms", []):
        z = int(atom["z"])
        ev = atom.get("eigenvalues_ha") or {}
        alpha = ev.get("occupied_eigenvalues_alpha_ha") or []
        by_z[z] = {
            "symbol": str(atom.get("symbol", "")),
            "total_energy_ha": float(atom["total_energy_ha"]),
            "occupied_eigenvalues_ha": np.asarray(alpha, dtype=float),
            "n_occupied": int(len(alpha)),
        }
    return by_z


def _load_summary(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_z: dict[int, dict] = {}
    for row in data.get("config_summaries", []):
        z = int(row.get("atomic_number"))
        by_z[z] = {
            "configuration": str(row.get("configuration", f"configuration_{z:03d}")),
            "converged": bool(row.get("converged", True)),
            "total_energy_ha": None
            if row.get("total_energy_ha") is None
            else float(row["total_energy_ha"]),
            "occupied_eigenvalues_ha": np.asarray(
                row.get("occupied_eigenvalues_ha") or [], dtype=float
            ),
            "occupied_state_count": int(row.get("occupied_state_count", 0)),
        }
    return by_z


def _expand_l1_triply(occupied_eigs: np.ndarray) -> np.ndarray:
    """
    Expand occupied eigenvalues by magnetic degeneracy for l=1 states:
    each l=1 entry is repeated 3 times before comparison.
    """
    if occupied_eigs.size == 0:
        return occupied_eigs
    if occupied_eigs.size > len(_OCCUPIED_L_SEQUENCE):
        raise ValueError(
            f"Unsupported occupied count={occupied_eigs.size}; "
            f"extend _OCCUPIED_L_SEQUENCE for this case."
        )
    expanded: list[float] = []
    for i, eps in enumerate(occupied_eigs):
        l_val = _OCCUPIED_L_SEQUENCE[i]
        mult = 3 if l_val == 1 else 1
        expanded.extend([float(eps)] * mult)
    return np.asarray(expanded, dtype=float)


def _format_report(
    *,
    ref_path: Path,
    summary_path: Path,
    rows: list[dict],
    skipped_notes: list[str],
) -> str:
    lines: list[str] = []
    lines.append("GGA-PBE accuracy vs HelFEM reference (selected Z subset)")
    lines.append("")
    lines.append(f"reference JSON: {ref_path}")
    lines.append(f"summary JSON:   {summary_path}")
    lines.append(f"compared Z:     {len(rows)}")
    if skipped_notes:
        lines.append(f"comparison notes: {len(skipped_notes)}")
    lines.append("")
    if skipped_notes:
        lines.append("Comparison notes")
        for s in skipped_notes:
            lines.append(f"  - {s}")
        lines.append("")

    hdr = (
        f"{'Z':>3}  {'sym':^4}  {'config':<18}  {'|E_ref| (Ha)':>18}  {'|E_ours| (Ha)':>18}  "
        f"{'dE (ours-ref)':>14}  {'|dE|':>12}  {'n_ref':>6}  {'n_ours':>6}  "
        f"{'max |d eps|':>12}  {'mean |d eps|':>13}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    abs_de: list[float] = []
    max_eps: list[float] = []
    mean_eps: list[float] = []
    for r in rows:
        abs_de.append(r["abs_de"])
        max_eps.append(r["max_abs_eps"])
        mean_eps.append(r["mean_abs_eps"])
        e_ref_s = _fmt_fixed_total_digits(abs(r["e_ref"]))
        e_ours_s = _fmt_fixed_total_digits(abs(r["e_ours"]))
        lines.append(
            f"{r['Z']:3d}  {r['symbol']:^4}  {r['configuration']:<18}  {e_ref_s:>18}  {e_ours_s:>18}  "
            f"{r['de']:14.4e}  {r['abs_de']:12.4e}  {r['n_ref']:6d}  {r['n_ours']:6d}  "
            f"{r['max_abs_eps']:12.4e}  {r['mean_abs_eps']:13.4e}"
        )

    lines.append("-" * len(hdr))
    lines.append("")
    lines.append("Aggregate")
    lines.append(f"  max  |dE|                 = {max(abs_de):.6e} Ha")
    lines.append(f"  mean |dE|                 = {float(np.mean(abs_de)):.6e} Ha")
    lines.append(f"  max  max|d epsilon_i|     = {max(max_eps):.6e} Ha")
    lines.append(f"  mean max|d epsilon_i|     = {float(np.mean(max_eps)):.6e} Ha")
    lines.append(f"  mean mean|d epsilon_i|    = {float(np.mean(mean_eps)):.6e} Ha")
    lines.append("")
    worst_e = max(rows, key=lambda x: x["abs_de"])
    worst_eps = max(rows, key=lambda x: x["max_abs_eps"])
    lines.append("Worst cases")
    lines.append(
        f"  total energy: Z={worst_e['Z']} ({worst_e['symbol']}, {worst_e['configuration']}), "
        f"dE={worst_e['de']:+.6e} Ha"
    )
    lines.append(
        f"  eigenvalues:  Z={worst_eps['Z']} ({worst_eps['symbol']}, {worst_eps['configuration']}), "
        f"max|d eps|={worst_eps['max_abs_eps']:.6e} Ha"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare GGA-PBE summary to HelFEM even-Z reference and write text report."
    )
    ap.add_argument("--reference-json", type=Path, default=_DEFAULT_REFERENCE)
    ap.add_argument("--summary-json", type=Path, default=_DEFAULT_SUMMARY)
    ap.add_argument("--out-txt", type=Path, default=_DEFAULT_OUT_TXT)
    args = ap.parse_args()

    ref_path = args.reference_json.resolve()
    summary_path = args.summary_json.resolve()
    if not ref_path.is_file():
        print(f"Missing reference JSON: {ref_path}", file=sys.stderr)
        sys.exit(1)
    if not summary_path.is_file():
        print(f"Missing summary JSON: {summary_path}", file=sys.stderr)
        sys.exit(1)

    ref_by_z = _load_reference_even_z(ref_path)
    ours_by_z = _load_summary(summary_path)

    rows: list[dict] = []
    skipped_notes: list[str] = []
    for z in sorted(ref_by_z):
        r = ref_by_z[z]
        if z not in ours_by_z:
            skipped_notes.append(f"Z={z}: missing in summary JSON.")
            continue
        o = ours_by_z[z]
        if not o["converged"]:
            skipped_notes.append(f"Z={z}: summary not converged ({o['configuration']}).")
            continue
        if o["total_energy_ha"] is None or o["occupied_eigenvalues_ha"].size == 0:
            skipped_notes.append(f"Z={z}: missing total energy or occupied eigenvalues.")
            continue

        e_ref = float(r["total_energy_ha"])
        e_ours = float(o["total_energy_ha"])
        de = e_ours - e_ref
        ref_eps = r["occupied_eigenvalues_ha"]
        our_eps = _expand_l1_triply(o["occupied_eigenvalues_ha"])
        n_ref = int(ref_eps.size)
        n_ours = int(our_eps.size)
        n_cmp = min(n_ref, n_ours)
        if n_cmp == 0:
            skipped_notes.append(f"Z={z}: no occupied eigenvalues to compare.")
            continue
        diff_eps = np.abs(our_eps[:n_cmp] - ref_eps[:n_cmp])
        rows.append(
            {
                "Z": z,
                "symbol": r["symbol"],
                "configuration": o["configuration"],
                "e_ref": e_ref,
                "e_ours": e_ours,
                "de": de,
                "abs_de": abs(de),
                "n_ref": n_ref,
                "n_ours": n_ours,
                "max_abs_eps": float(np.max(diff_eps)),
                "mean_abs_eps": float(np.mean(diff_eps)),
            }
        )
        if n_ref != n_ours:
            skipped_notes.append(
                f"Z={z}: occupied state count mismatch (ref={n_ref}, ours={n_ours}); compared first {n_cmp}."
            )

    if not rows:
        print("No comparable entries found.", file=sys.stderr)
        sys.exit(1)

    txt = _format_report(
        ref_path=ref_path,
        summary_path=summary_path,
        rows=rows,
        skipped_notes=skipped_notes,
    )
    out_txt = args.out_txt.resolve()
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(txt, encoding="utf-8")
    sys.stdout.write(txt)
    print(f"(also wrote {out_txt})", flush=True)


if __name__ == "__main__":
    main()
