"""
Compare ATOM LDA_SVWN summary against FEATOM reference JSON.

Default comparison:
- Reference: tests/data/reference/lda_svwn/featom_atoms_lda.json
- Ours:      tests/data/summary/lda_svwn/finite_element_sweep/fe12_R040/configuration_energy_summary.json

Outputs a text report in this folder:
    lda_svwn_accuracy_test_featom_summary.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_REFERENCE = _DATA_DIR / "reference" / "lda_svwn" / "featom_atoms_lda.json"
_DEFAULT_SUMMARY = (
    _DATA_DIR
    / "summary"
    / "lda_svwn"
    / "finite_element_sweep"
    / "fe12_R040"
    / "configuration_energy_summary.json"
)
_DEFAULT_OUT_TXT = Path(__file__).resolve().parent / "lda_svwn_accuracy_test_featom_summary.txt"


def _fmt_fixed_total_digits(value: float, total_digits: int = 12) -> str:
    """
    Format number in fixed-point (no scientific notation) such that
    digits before and after decimal point sum to `total_digits`.
    """
    if not np.isfinite(value):
        return str(value)

    abs_v = abs(value)
    if abs_v < 1.0:
        int_digits = 1  # keep leading zero as integer part
    else:
        int_digits = int(math.floor(math.log10(abs_v))) + 1

    frac_digits = max(0, total_digits - int_digits)
    return f"{value:.{frac_digits}f}"


def _load_reference(path: Path) -> tuple[dict[int, dict], set[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    failures = data.get("failures", [])
    by_z: dict[int, dict] = {}
    for rec in records:
        z = int(rec["Z"])
        by_z[z] = {
            "total_energy_au": float(rec["total_energy_au"]),
            "occupied_eigenvalues_au": np.asarray(rec["occupied_eigenvalues_au"], dtype=float),
            "n_occupied_states": int(rec["n_occupied_states"]),
        }
    fail_z = {int(row["Z"]) for row in failures if "Z" in row}
    return by_z, fail_z


def _load_summary(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_z: dict[int, dict] = {}
    for row in data.get("config_summaries", []):
        z = int(row.get("atomic_number"))
        by_z[z] = {
            "configuration": str(row.get("configuration", f"configuration_Z{z:03d}")),
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
) -> str:
    lines: list[str] = []
    lines.append("LDA_SVWN accuracy vs FEATOM reference")
    lines.append("")
    lines.append(f"reference JSON: {ref_path}")
    lines.append(f"summary JSON:   {summary_path}")
    lines.append(f"compared Z:     {len(rows)}")
    if skipped_notes:
        lines.append(f"skipped entries: {len(skipped_notes)}")
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
        mean_eps_s = f"{r['mean_abs_eps']:13.4e}" if r["valid_eigenvalues"] else f"{'-':>13}"
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
    ap = argparse.ArgumentParser(description="Compare LDA_SVWN summary to FEATOM reference and write text report.")
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

    ref_by_z, ref_failed_z = _load_reference(ref_path)
    ours_by_z = _load_summary(summary_path)

    rows: list[dict] = []
    skipped_notes: list[str] = []
    for z in sorted(set(ref_by_z) | set(ref_failed_z) | set(ours_by_z)):
        r = ref_by_z.get(z)
        o = ours_by_z.get(z)
        ref_failed = z in ref_failed_z
        configuration = o["configuration"] if o is not None else f"configuration_Z{z:03d}"

        e_ref = None if r is None or ref_failed else r["total_energy_au"]
        e_ours = None if o is None else o["total_energy_ha"]
        has_e_ref = e_ref is not None
        has_e_ours = e_ours is not None
        valid_energy = has_e_ref and has_e_ours
        de = (float(e_ours) - float(e_ref)) if valid_energy else None
        abs_de = abs(de) if de is not None else None

        ref_eps = np.asarray([] if r is None or ref_failed else r["occupied_eigenvalues_au"], dtype=float)
        our_eps = np.asarray([] if o is None else o["occupied_eigenvalues_ha"], dtype=float)
        n_ref_full = int(ref_eps.size)
        n_ours = int(our_eps.size) if o is not None else None
        n_ref = n_ref_full if not ref_failed and r is not None else None
        n_cmp = min(n_ref_full, int(our_eps.size))
        valid_eigenvalues = n_cmp > 0
        max_abs_eps = float(np.max(np.abs(our_eps[:n_cmp] - ref_eps[:n_cmp]))) if valid_eigenvalues else None
        mean_abs_eps = float(np.mean(np.abs(our_eps[:n_cmp] - ref_eps[:n_cmp]))) if valid_eigenvalues else None

        if ref_failed:
            skipped_notes.append(f"Z={z}: marked as failed in reference JSON.")
        if o is None:
            skipped_notes.append(f"Z={z}: missing in summary JSON.")
        elif not o["converged"]:
            skipped_notes.append(f"Z={z}: summary not converged ({o['configuration']}).")
        if not valid_eigenvalues and (r is not None or o is not None):
            skipped_notes.append(f"Z={z}: no occupied eigenvalues to compare.")
        if valid_eigenvalues and n_ref is not None and n_ours is not None and n_ref != n_ours:
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
    )
    out_txt = args.out_txt.resolve()
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(txt, encoding="utf-8")
    sys.stdout.write(txt)
    print(f"(also wrote {out_txt})", flush=True)


if __name__ == "__main__":
    main()
