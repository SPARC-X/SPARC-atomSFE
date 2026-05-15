"""
Compare GGA-PBE finite-element summary against ONCVPSP GGA-PBE reference JSON.

Outputs a text report in this folder:
    gga_pbe_accuracy_test_oncvpsp_summary.txt
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

_DEFAULT_REFERENCE = _DATA_DIR / "reference" / "gga_pbe" / "oncvpsp_atoms_gga_pbe.json"
_DEFAULT_SUMMARY = default_all_electron_summary(_DATA_DIR, "gga_pbe")
_DEFAULT_OUT_TXT = Path(__file__).resolve().parent / "gga_pbe_accuracy_test_oncvpsp_summary.txt"


def _fmt_fixed_total_digits(value: float, total_digits: int = 12) -> str:
    if not np.isfinite(value):
        return "-"
    abs_v = abs(value)
    if abs_v < 1.0:
        int_digits = 1
    else:
        int_digits = int(math.floor(math.log10(abs_v))) + 1
    frac_digits = max(0, total_digits - int_digits)
    return f"{value:.{frac_digits}f}"


def _load_reference(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_z: dict[int, dict] = {}
    for rec in data.get("atoms", []):
        z = int(rec["atomic_number"])
        eigs = np.asarray(
            [float(e["energy_ha"]) for e in rec.get("occupied_eigenvalues_ha", [])],
            dtype=float,
        )
        by_z[z] = {
            "symbol": str(rec.get("symbol", "")),
            "converged": bool(rec.get("converged", True)),
            "total_energy_ha": None
            if rec.get("total_energy_ha") is None
            else float(rec["total_energy_ha"]),
            "occupied_eigenvalues_ha": eigs,
            "occupied_state_count": int(eigs.size),
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


def _build_rows(z_list: list[int], ref_by_z: dict[int, dict], ours_by_z: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    for z in z_list:
        ref = ref_by_z.get(z)
        ours = ours_by_z.get(z)
        configuration = ours["configuration"] if ours is not None else f"configuration_{z:03d}"
        symbol = ref["symbol"] if ref is not None else ""
        row = {
            "Z": z,
            "symbol": symbol,
            "configuration": configuration,
            "valid": False,
            "e_ref": None,
            "e_ours": None,
            "de": None,
            "abs_de": None,
            "n_ref": None,
            "n_ours": None,
            "max_abs_eps": None,
            "mean_abs_eps": None,
        }

        if ref is not None and ours is not None and ref["converged"] and ours["converged"]:
            e_ref = ref["total_energy_ha"]
            e_ours = ours["total_energy_ha"]
            ref_eps = np.sort(ref["occupied_eigenvalues_ha"])
            our_eps = np.sort(ours["occupied_eigenvalues_ha"])
            n_ref = int(ref_eps.size)
            n_ours = int(our_eps.size)
            n_cmp = min(n_ref, n_ours)
            if e_ref is not None and e_ours is not None and n_cmp > 0:
                diff_eps = np.abs(our_eps[:n_cmp] - ref_eps[:n_cmp])
                de = float(e_ours) - float(e_ref)
                row.update(
                    {
                        "valid": True,
                        "e_ref": float(e_ref),
                        "e_ours": float(e_ours),
                        "de": de,
                        "abs_de": abs(de),
                        "n_ref": n_ref,
                        "n_ours": n_ours,
                        "max_abs_eps": float(np.max(diff_eps)),
                        "mean_abs_eps": float(np.mean(diff_eps)),
                    }
                )
        rows.append(row)
    return rows


def _format_section(title: str, rows: list[dict]) -> list[str]:
    lines: list[str] = []
    valid_rows = [r for r in rows if r["valid"]]
    lines.append(title)
    lines.append(f"compared Z:     {len(rows)}")
    lines.append("")

    hdr = (
        f"{'Z':>3}  {'sym':^4}  {'config':<18}  {'|E_ref| (Ha)':>18}  {'|E_ours| (Ha)':>18}  "
        f"{'dE (ours-ref)':>14}  {'|dE|':>12}  {'n_ref':>6}  {'n_ours':>6}  "
        f"{'max |d eps|':>12}  {'mean |d eps|':>13}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for r in rows:
        if r["valid"]:
            lines.append(
                f"{r['Z']:3d}  {r['symbol']:^4}  {r['configuration']:<18}  "
                f"{_fmt_fixed_total_digits(abs(r['e_ref'])):>18}  "
                f"{_fmt_fixed_total_digits(abs(r['e_ours'])):>18}  "
                f"{r['de']:14.4e}  {r['abs_de']:12.4e}  {r['n_ref']:6d}  {r['n_ours']:6d}  "
                f"{r['max_abs_eps']:12.4e}  {r['mean_abs_eps']:13.4e}"
            )
        else:
            lines.append(
                f"{r['Z']:3d}  {r['symbol']:^4}  {r['configuration']:<18}  "
                f"{'-':>18}  {'-':>18}  {'-':>14}  {'-':>12}  {'-':>6}  {'-':>6}  "
                f"{'-':>12}  {'-':>13}"
            )

    lines.append("-" * len(hdr))
    lines.append("")

    if valid_rows:
        abs_de = [r["abs_de"] for r in valid_rows]
        max_eps = [r["max_abs_eps"] for r in valid_rows]
        mean_eps = [r["mean_abs_eps"] for r in valid_rows]
        lines.append("Aggregate")
        lines.append(f"  valid rows                = {len(valid_rows)}")
        lines.append(f"  max  |dE|                 = {max(abs_de):.6e} Ha")
        lines.append(f"  mean |dE|                 = {float(np.mean(abs_de)):.6e} Ha")
        lines.append(f"  max  max|d epsilon_i|     = {max(max_eps):.6e} Ha")
        lines.append(f"  mean max|d epsilon_i|     = {float(np.mean(max_eps)):.6e} Ha")
        lines.append(f"  mean mean|d epsilon_i|    = {float(np.mean(mean_eps)):.6e} Ha")
        lines.append("")
        worst_e = max(valid_rows, key=lambda x: x["abs_de"])
        worst_eps = max(valid_rows, key=lambda x: x["max_abs_eps"])
        lines.append("Worst cases")
        lines.append(
            f"  total energy: Z={worst_e['Z']} ({worst_e['symbol']}, {worst_e['configuration']}), dE={worst_e['de']:+.6e} Ha"
        )
        lines.append(
            f"  eigenvalues:  Z={worst_eps['Z']} ({worst_eps['symbol']}, {worst_eps['configuration']}), max|d eps|={worst_eps['max_abs_eps']:.6e} Ha"
        )
    else:
        lines.append("Aggregate")
        lines.append("  valid rows                = 0")
    lines.append("")
    return lines


def _format_report(*, ref_path: Path, summary_path: Path, rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("GGA-PBE accuracy vs ONCVPSP reference")
    lines.append("")
    lines.append(f"reference JSON: {ref_path}")
    lines.append(f"summary JSON:   {summary_path}")
    lines.append("")
    lines.extend(_format_section("Section A: ONCVPSP selected subset", rows))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare GGA-PBE summary to ONCVPSP reference and write text report.")
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

    ref_by_z = _load_reference(ref_path)
    ours_by_z = _load_summary(summary_path)
    selected_z = sorted(ref_by_z.keys())
    rows = _build_rows(selected_z, ref_by_z, ours_by_z)

    txt = _format_report(ref_path=ref_path, summary_path=summary_path, rows=rows)
    out_txt = args.out_txt.resolve()
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(txt, encoding="utf-8")
    sys.stdout.write(txt)
    print(f"(also wrote {out_txt})", flush=True)


if __name__ == "__main__":
    main()
