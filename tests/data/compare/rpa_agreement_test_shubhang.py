"""
Compare RPA-OEP finite-element summary against Shubhang RPA-OEP reference JSON.

Default comparison:
- Reference: tests/data/reference/rpa_oep/shubhang_z_1_to_18_rpa_oep.json
- Ours:      tests/data/summary/all_electron/rpa_oep/fe12_R040__z1_92.json (when present)

Outputs a text report in this folder:
    rpa_agreement_test_shubhang_summary.txt
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

_DEFAULT_REFERENCE = (
    _DATA_DIR / "reference" / "rpa_oep" / "shubhang_z_1_to_18_rpa_oep.json"
)
_DEFAULT_SUMMARY = default_all_electron_summary(_DATA_DIR, "rpa_oep")
_DEFAULT_OUT_TXT = (
    Path(__file__).resolve().parent / "rpa_agreement_test_shubhang_summary.txt"
)


def _fmt_fixed_total_digits(value: float, total_digits: int = 12) -> str:
    """Format fixed-point with a target count of total digits."""
    if not np.isfinite(value):
        return str(value)

    abs_v = abs(value)
    if abs_v < 1.0:
        int_digits = 1
    else:
        int_digits = int(math.floor(math.log10(abs_v))) + 1

    frac_digits = max(0, total_digits - int_digits)
    return f"{value:.{frac_digits}f}"


def _load_summary_like(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_z: dict[int, dict] = {}
    for row in data.get("config_summaries", []):
        z_raw = row.get("atomic_number")
        if z_raw is None:
            continue
        z = int(z_raw)
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


def _format_report(
    *,
    ref_path: Path,
    summary_path: Path,
    rows: list[dict],
    skipped_notes: list[str],
    ignored_summary_z: list[int],
) -> str:
    lines: list[str] = []
    lines.append("RPA-OEP agreement vs Shubhang reference")
    lines.append("")
    lines.append(f"reference JSON: {ref_path}")
    lines.append(f"summary JSON:   {summary_path}")
    lines.append(f"compared Z:     {len(rows)}")
    if skipped_notes:
        lines.append(f"skipped entries: {len(skipped_notes)}")
    if ignored_summary_z:
        lines.append(f"ignored summary-only Z: {len(ignored_summary_z)}")
    lines.append("")
    if ignored_summary_z:
        lo = ignored_summary_z[:10]
        lo_text = ", ".join(str(z) for z in lo)
        tail = " ..." if len(ignored_summary_z) > 10 else ""
        lines.append(
            f"Ignored summary-only configurations (not present in reference): {lo_text}{tail}"
        )
        lines.append("")
    if skipped_notes:
        lines.append("Skipped details")
        for s in skipped_notes:
            lines.append(f"  - {s}")
        lines.append("")

    hdr = (
        f"{'Z':>3}  {'config':<18}  {'|E_ref| (Ha)':>18}  {'|E_ours| (Ha)':>18}  "
        f"{'dE (ours-ref)':>14}  {'|dE|':>12}  {'n_ref':>6}  {'n_ours':>6}  "
        f"{'max |d eps|':>12}  {'mean |d eps|':>13}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    energy_rows = [r for r in rows if r["valid_energy"]]
    eig_rows = [r for r in rows if r["valid_eigenvalues"]]
    for r in rows:
        e_ref_s = _fmt_fixed_total_digits(abs(r["e_ref"])) if r["has_e_ref"] else "-"
        e_ours_s = _fmt_fixed_total_digits(abs(r["e_ours"])) if r["has_e_ours"] else "-"
        de_s = f"{r['de']:14.4e}" if r["valid_energy"] else f"{'-':>14}"
        abs_de_s = f"{r['abs_de']:12.4e}" if r["valid_energy"] else f"{'-':>12}"
        n_ref_s = f"{r['n_ref']:6d}" if r["n_ref"] is not None else f"{'-':>6}"
        n_ours_s = f"{r['n_ours']:6d}" if r["n_ours"] is not None else f"{'-':>6}"
        max_eps_s = f"{r['max_abs_eps']:12.4e}" if r["valid_eigenvalues"] else f"{'-':>12}"
        mean_eps_s = (
            f"{r['mean_abs_eps']:13.4e}" if r["valid_eigenvalues"] else f"{'-':>13}"
        )
        lines.append(
            f"{r['Z']:3d}  {r['configuration']:<18}  {e_ref_s:>18}  {e_ours_s:>18}  "
            f"{de_s}  {abs_de_s}  {n_ref_s}  {n_ours_s}  {max_eps_s}  {mean_eps_s}"
        )

    lines.append("-" * len(hdr))
    lines.append("")
    lines.append("Aggregate")
    lines.append(f"  valid rows (energy)       = {len(energy_rows)}")
    lines.append(f"  valid rows (eigenvalues)  = {len(eig_rows)}")
    if energy_rows:
        abs_de = [r["abs_de"] for r in energy_rows]
        lines.append(f"  max  |dE|                 = {max(abs_de):.6e} Ha")
        lines.append(f"  mean |dE|                 = {float(np.mean(abs_de)):.6e} Ha")
    else:
        lines.append("  max  |dE|                 = -")
        lines.append("  mean |dE|                 = -")
    if eig_rows:
        max_eps = [r["max_abs_eps"] for r in eig_rows]
        mean_eps = [r["mean_abs_eps"] for r in eig_rows]
        lines.append(f"  max  max|d epsilon_i|     = {max(max_eps):.6e} Ha")
        lines.append(f"  mean max|d epsilon_i|     = {float(np.mean(max_eps)):.6e} Ha")
        lines.append(f"  mean mean|d epsilon_i|    = {float(np.mean(mean_eps)):.6e} Ha")
    else:
        lines.append("  max  max|d epsilon_i|     = -")
        lines.append("  mean max|d epsilon_i|     = -")
        lines.append("  mean mean|d epsilon_i|    = -")
    lines.append("")
    lines.append("Worst cases")
    if energy_rows:
        worst_e = max(energy_rows, key=lambda x: x["abs_de"])
        lines.append(
            f"  total energy: Z={worst_e['Z']} ({worst_e['configuration']}), dE={worst_e['de']:+.6e} Ha"
        )
    else:
        lines.append("  total energy: -")
    if eig_rows:
        worst_eps = max(eig_rows, key=lambda x: x["max_abs_eps"])
        lines.append(
            f"  eigenvalues:  Z={worst_eps['Z']} ({worst_eps['configuration']}), max|d eps|={worst_eps['max_abs_eps']:.6e} Ha"
        )
    else:
        lines.append("  eigenvalues:  -")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare RPA-OEP summary to Shubhang reference and write text report."
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

    ref_by_z = _load_summary_like(ref_path)
    ours_by_z = _load_summary_like(summary_path)

    rows: list[dict] = []
    skipped_notes: list[str] = []
    compared_z = sorted(ref_by_z)
    ignored_summary_z = sorted(set(ours_by_z) - set(ref_by_z))
    for z in compared_z:
        r = ref_by_z.get(z)
        o = ours_by_z.get(z)
        configuration = o["configuration"] if o is not None else f"configuration_{z:03d}"
        ref_not_converged = (r is not None) and (not r["converged"])

        e_ref = None if (r is None or ref_not_converged) else r["total_energy_ha"]
        e_ours = None if o is None else o["total_energy_ha"]
        has_e_ref = e_ref is not None
        has_e_ours = e_ours is not None
        valid_energy = has_e_ref and has_e_ours
        de = (float(e_ours) - float(e_ref)) if valid_energy else None
        abs_de = abs(de) if de is not None else None

        ref_eps = np.asarray(
            [] if (r is None or ref_not_converged) else r["occupied_eigenvalues_ha"],
            dtype=float,
        )
        our_eps = np.asarray(
            [] if o is None else o["occupied_eigenvalues_ha"], dtype=float
        )
        n_ref = int(ref_eps.size) if (r is not None and not ref_not_converged) else None
        n_ours = int(our_eps.size) if o is not None else None
        n_cmp = min(int(ref_eps.size), int(our_eps.size))
        valid_eigenvalues = n_cmp > 0
        max_abs_eps = (
            float(np.max(np.abs(our_eps[:n_cmp] - ref_eps[:n_cmp])))
            if valid_eigenvalues
            else None
        )
        mean_abs_eps = (
            float(np.mean(np.abs(our_eps[:n_cmp] - ref_eps[:n_cmp])))
            if valid_eigenvalues
            else None
        )

        if ref_not_converged:
            skipped_notes.append(f"Z={z}: reference not converged ({r['configuration']}).")
        if o is None:
            skipped_notes.append(f"Z={z}: missing in summary JSON.")
        elif not o["converged"]:
            skipped_notes.append(f"Z={z}: summary not converged ({o['configuration']}).")
        if not valid_eigenvalues and (r is not None or o is not None):
            skipped_notes.append(f"Z={z}: no occupied eigenvalues to compare.")
        if (
            valid_eigenvalues
            and n_ref is not None
            and n_ours is not None
            and n_ref != n_ours
        ):
            skipped_notes.append(
                f"Z={z}: occupied state count mismatch (ref={n_ref}, ours={n_ours}); compared first {n_cmp}."
            )

        rows.append(
            {
                "Z": z,
                "configuration": configuration,
                "has_e_ref": has_e_ref,
                "has_e_ours": has_e_ours,
                "valid_energy": valid_energy,
                "valid_eigenvalues": valid_eigenvalues,
                "e_ref": None if e_ref is None else float(e_ref),
                "e_ours": None if e_ours is None else float(e_ours),
                "de": de,
                "abs_de": abs_de,
                "n_ref": n_ref,
                "n_ours": n_ours,
                "max_abs_eps": max_abs_eps,
                "mean_abs_eps": mean_abs_eps,
            }
        )

    txt = _format_report(
        ref_path=ref_path,
        summary_path=summary_path,
        rows=rows,
        skipped_notes=skipped_notes,
        ignored_summary_z=ignored_summary_z,
    )
    out_txt = args.out_txt.resolve()
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(txt, encoding="utf-8")
    sys.stdout.write(txt)
    print(f"(also wrote {out_txt})", flush=True)


if __name__ == "__main__":
    main()
