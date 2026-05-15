r"""
HF charged-set accuracy: Lehtola (2019) Table 5 reference vs run summary.

Loads ``reference/hf/lehtola_charged_atoms_hf.json`` (``E_fem_au``, ``E_erkale_au``,
``E_gaussian_au``; see ``reference/hf/README.md``) and
``fe12_R040__charged.json`` under ``summary/all_electron/hf/charged/`` (or sweep case dir).

Configurations are paired **by index** with the reference ``records`` list, in the
same order as ``generate_dataset.py`` (``HF_CHARGED_*_LIST``). A mismatch in
``(atomic_number, n_electrons)`` vs that order raises an error.

Compared quantities:
- Total energy vs literature FEM (``E_fem_au``), Erkale, and Gaussian column.
- HOMO and HF exchange are **ours only** (no literature columns in the JSON).

Non-converged configurations are skipped in aggregates (same spirit as
``hf_accuracy_test_neural_lehtola.py``).

Run from ``delta/``::

    python atomSFE/tests/data/compare/hf_accuracy_test_charged_lehtola.py
    python atomSFE/tests/data/compare/hf_accuracy_test_charged_lehtola.py --case charged --out-txt path/to/report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atom.utils.occupation_states import OccupationInfo

_DATA_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_REFERENCE = _DATA_DIR / "reference" / "hf" / "lehtola_charged_atoms_hf.json"
_DEFAULT_SUMMARY_HF = _DATA_DIR / "summary" / "hf"
_DEFAULT_CASE = "charged"

# Same order as ``reference/hf/lehtola_charged_atoms_hf.json`` / ``generate_dataset.HF_CHARGED_*``.
_HF_CHARGED_Z = [1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 17, 18]
_HF_CHARGED_N = [2, 2, 2, 4, 4, 4, 7, 7, 7, 10, 10, 10, 12, 12, 12, 15, 15, 15, 18, 18]


@dataclass(frozen=True)
class RefChargedRow:
    species: str
    e_fem: float
    e_erkale: float
    e_gaussian: float
    delta_fem_minus_erkale_microeh: float


@dataclass(frozen=True)
class OursChargedRow:
    z: int
    n_electrons: float
    e_tot: float
    eps_homo: float
    e_x: float
    converged: bool


@dataclass
class CompareChargedRow:
    species: str
    z: int
    n_e: float
    e_fem: float
    e_ours: float
    d_ours_minus_fem: float
    e_erkale: float
    d_ours_minus_erkale: float
    e_gaussian: float
    d_ours_minus_gaussian: float
    homo_ours: float
    ex_ours: float


def _load_reference_charged(path: Path) -> list[RefChargedRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[RefChargedRow] = []
    for rec in data.get("records", []):
        out.append(
            RefChargedRow(
                species=str(rec["species"]),
                e_fem=float(rec["E_fem_au"]),
                e_erkale=float(rec["E_erkale_au"]),
                e_gaussian=float(rec["E_gaussian_au"]),
                delta_fem_minus_erkale_microeh=float(rec["delta_fem_minus_erkale_microEh"]),
            )
        )
    return out


def _hf_exchange_from_row(row: dict) -> float | None:
    eh = row.get("energies_ha") or {}
    if not isinstance(eh, dict):
        return None
    if "hf_exchange" in eh and eh["hf_exchange"] is not None:
        return float(eh["hf_exchange"])
    if "exchange" in eh and eh["exchange"] is not None:
        return float(eh["exchange"])
    return None


def _load_summary_charged_ordered(summary_json: Path) -> list[OursChargedRow | None]:
    """One entry per ``config_summaries`` row (None if energy/HOMO/exchange missing)."""
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    out: list[OursChargedRow | None] = []
    for row in data.get("config_summaries", []):
        z = row.get("atomic_number")
        n_e = row.get("n_electrons")
        if z is None or n_e is None:
            out.append(None)
            continue
        z = int(z)
        n_e = float(n_e)
        conv = bool(row.get("converged", True))
        e_tot = row.get("total_energy_ha")
        occ = row.get("occupied_eigenvalues_ha") or []
        ex = _hf_exchange_from_row(row)
        if e_tot is None or len(occ) == 0 or ex is None:
            out.append(None)
            continue
        occ_arr = np.asarray(occ, dtype=float)
        homo = float(np.max(occ_arr))
        out.append(
            OursChargedRow(
                z=z,
                n_electrons=n_e,
                e_tot=float(e_tot),
                eps_homo=homo,
                e_x=ex,
                converged=conv,
            )
        )
    return out


def _build_comparison(
    ref_rows: list[RefChargedRow],
    ours_rows: list[OursChargedRow | None],
) -> tuple[list[CompareChargedRow], int]:
    if len(ref_rows) != len(_HF_CHARGED_Z):
        raise ValueError(f"reference records length {len(ref_rows)} != {_HF_CHARGED_Z}")
    if len(ours_rows) != len(ref_rows):
        raise ValueError(
            f"summary config count {len(ours_rows)} != reference count {len(ref_rows)}"
        )
    skipped = 0
    rows: list[CompareChargedRow] = []
    for i, r in enumerate(ref_rows):
        z_exp, n_exp = _HF_CHARGED_Z[i], float(_HF_CHARGED_N[i])
        o = ours_rows[i]
        if o is None:
            raise ValueError(f"configuration index {i + 1}: missing energy/HOMO/exchange in summary")
        if o.z != z_exp or abs(o.n_electrons - n_exp) > 1e-6:
            raise ValueError(
                f"configuration index {i + 1} ({r.species}): expected Z={z_exp}, N={n_exp}, "
                f"got Z={o.z}, N={o.n_electrons} (summary order must match Lehtola table order)."
            )
        if not o.converged:
            skipped += 1
            continue
        rows.append(
            CompareChargedRow(
                species=r.species,
                z=o.z,
                n_e=o.n_electrons,
                e_fem=r.e_fem,
                e_ours=o.e_tot,
                d_ours_minus_fem=o.e_tot - r.e_fem,
                e_erkale=r.e_erkale,
                d_ours_minus_erkale=o.e_tot - r.e_erkale,
                e_gaussian=r.e_gaussian,
                d_ours_minus_gaussian=o.e_tot - r.e_gaussian,
                homo_ours=o.eps_homo,
                ex_ours=o.e_x,
            )
        )
    return rows, skipped


def _format_report(
    *,
    ref_path: Path,
    summary_json: Path,
    case: str,
    rows: list[CompareChargedRow],
    skipped_nonconverged: int,
) -> str:
    def _is_spin_polarized(z: int, n_e: float) -> bool:
        occ = OccupationInfo(
            z_nuclear=z,
            z_valence=z,
            all_electron_flag=True,
            n_electrons=n_e,
        )
        return not np.isclose(occ.n_free_electrons_up, occ.n_free_electrons_dn, atol=1e-12)

    def _append_group_report(
        lines_out: list[str],
        *,
        group_title: str,
        group_rows: list[CompareChargedRow],
    ) -> None:
        lines_out.append(group_title)
        lines_out.append("")
        if not group_rows:
            lines_out.append("  (no converged configurations in this group)")
            lines_out.append("")
            return
        hdr = (
            f"{'species':^8}  {'Z':>3}  {'N':>4}  "
            f"{'E_FEM ref':>15}  {'E_tot ours':>15}  {'d-FEM':>12}  {'|dF|':>12}  "
            f"{'d-Erk':>12}  {'|dE|':>12}  {'d-Gauss':>12}  {'|dG|':>12}  "
            f"{'HOMO ours':>12}  {'E_x ours':>12}"
        )
        lines_out.append(hdr)
        lines_out.append("-" * len(hdr))
        abs_df: list[float] = []
        abs_de: list[float] = []
        abs_dg: list[float] = []
        for w in group_rows:
            abs_df.append(abs(w.d_ours_minus_fem))
            abs_de.append(abs(w.d_ours_minus_erkale))
            abs_dg.append(abs(w.d_ours_minus_gaussian))
            lines_out.append(
                f"{w.species:^8}  {w.z:3d}  {w.n_e:4.0f}  "
                f"{w.e_fem:15.9f}  {w.e_ours:15.9f}  {w.d_ours_minus_fem:12.4e}  {abs(w.d_ours_minus_fem):12.4e}  "
                f"{w.d_ours_minus_erkale:12.4e}  {abs(w.d_ours_minus_erkale):12.4e}  "
                f"{w.d_ours_minus_gaussian:12.4e}  {abs(w.d_ours_minus_gaussian):12.4e}  "
                f"{w.homo_ours:12.8f}  {w.ex_ours:12.8f}"
            )
        lines_out.append("-" * len(hdr))
        lines_out.append("")
        lines_out.append(f"Aggregate for {group_title} (absolute errors vs reference columns, Ha)")
        lines_out.append(f"  max  |d vs FEM|      = {max(abs_df):.6e}")
        lines_out.append(f"  mean |d vs FEM|      = {float(np.mean(abs_df)):.6e}")
        lines_out.append(f"  max  |d vs Erkale|   = {max(abs_de):.6e}")
        lines_out.append(f"  mean |d vs Erkale|   = {float(np.mean(abs_de)):.6e}")
        lines_out.append(f"  max  |d vs Gaussian| = {max(abs_dg):.6e}")
        lines_out.append(f"  mean |d vs Gaussian| = {float(np.mean(abs_dg)):.6e}")
        lines_out.append("")
        worst_f = max(group_rows, key=lambda r: abs(r.d_ours_minus_fem))
        worst_e = max(group_rows, key=lambda r: abs(r.d_ours_minus_erkale))
        worst_g = max(group_rows, key=lambda r: abs(r.d_ours_minus_gaussian))
        lines_out.append("Largest |d vs FEM| (diagnosis)")
        lines_out.append(
            f"  {worst_f.species}  Z={worst_f.z}  d(ours - E_FEM) = {worst_f.d_ours_minus_fem:+.6e} Ha"
        )
        lines_out.append("Largest |d vs Erkale|")
        lines_out.append(
            f"  {worst_e.species}  Z={worst_e.z}  d(ours - Erkale) = {worst_e.d_ours_minus_erkale:+.6e} Ha"
        )
        lines_out.append("Largest |d vs lit. Gaussian|")
        lines_out.append(
            f"  {worst_g.species}  Z={worst_g.z}  d(ours - Gauss) = {worst_g.d_ours_minus_gaussian:+.6e} Ha"
        )
        lines_out.append("")

    lines: list[str] = []
    lines.append("HF Lehtola Table 5 charged set (reference vs summary)")
    lines.append("")
    lines.append(f"reference:     {ref_path}")
    lines.append(f"summary JSON:  {summary_json}")
    lines.append(f"case:          {case}")
    lines.append(
        f"species listed: {len(rows)}  (skipped not converged: {skipped_nonconverged})"
    )
    lines.append("")
    spin_unpolarized_rows = [r for r in rows if not _is_spin_polarized(r.z, r.n_e)]
    spin_polarized_rows = [r for r in rows if _is_spin_polarized(r.z, r.n_e)]
    _append_group_report(
        lines,
        group_title=f"Spin-unpolarized atoms (count: {len(spin_unpolarized_rows)})",
        group_rows=spin_unpolarized_rows,
    )
    _append_group_report(
        lines,
        group_title=f"Spin-polarized atoms (count: {len(spin_polarized_rows)})",
        group_rows=spin_polarized_rows,
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare HF charged-set summary to Lehtola Table 5 reference (text report)."
    )
    ap.add_argument(
        "--reference-json",
        type=Path,
        default=_DEFAULT_REFERENCE,
        help="Reference dataset (default: reference/hf/lehtola_charged_atoms_hf.json).",
    )
    ap.add_argument(
        "--summary-hf-root",
        type=Path,
        default=_DEFAULT_SUMMARY_HF,
        help="Root containing HF summaries (default: tests/data/summary/all_electron/hf).",
    )
    ap.add_argument(
        "--case",
        type=str,
        default=_DEFAULT_CASE,
        help=(
            "Subpath under --summary-hf-root (e.g. charged/ for flat fe12_R040__charged.json). "
            f"(default: {_DEFAULT_CASE})."
        ),
    )
    ap.add_argument(
        "--out-txt",
        type=Path,
        default=Path(__file__).resolve().parent / "hf_accuracy_test_charged_lehtola_summary.txt",
        help="Output text report path.",
    )
    args = ap.parse_args()

    ref_path = args.reference_json.resolve()
    if not ref_path.is_file():
        print(f"Missing reference file: {ref_path}", file=sys.stderr)
        sys.exit(1)

    summary_json = resolve_summary_under(args.summary_hf_root.resolve(), args.case)
    if summary_json is None or not summary_json.is_file():
        print(f"Missing summary file: {summary_json}", file=sys.stderr)
        sys.exit(1)

    try:
        ref_rows = _load_reference_charged(ref_path)
        ours_rows = _load_summary_charged_ordered(summary_json)
        rows, skipped = _build_comparison(ref_rows, ours_rows)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    if not rows:
        print(
            "No converged configurations to compare (check SCF convergence in summary).",
            file=sys.stderr,
        )
        sys.exit(1)

    text = _format_report(
        ref_path=ref_path,
        summary_json=summary_json,
        case=args.case,
        rows=rows,
        skipped_nonconverged=skipped,
    )
    out_txt = args.out_txt.resolve()
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    print(f"(also wrote {out_txt})", flush=True)


if __name__ == "__main__":
    main()
