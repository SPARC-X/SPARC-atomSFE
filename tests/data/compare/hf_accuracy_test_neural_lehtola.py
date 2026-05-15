r"""
HF accuracy check: literature reference (``reference/hf``) vs run summaries (``summary/all_electron/hf``).

Loads Cinal (2020) closed-(sub)shell reference from ``lehtola_closed_subshell_atoms_hf.json``:
``minus_E_tot_au``, ``minus_epsilon_homo_au``, ``minus_E_x_au`` follow
``reference/hf/README.md`` (stored positive magnitudes for $-E_{\mathrm{tot}}$,
$-\varepsilon_{\mathrm{HOMO}}$, $-E_{\mathrm{x}}$).

Loads ``fe12_R040__z1_92.json`` (or sweep-case summary) for one HF case and compares per $Z$:
total energy, HOMO (= ``max(occupied_eigenvalues_ha)``), and HF exchange
(``energies_ha.hf_exchange``, falling back to ``energies_ha.exchange``).
Non-converged configurations are skipped.

Writes a detailed text report to ``--out-txt`` and prints the same summary to stdout.

Run from ``delta/``::

    python atomSFE/tests/data/compare/hf_accuracy_test_neural_lehtola.py
    python atomSFE/tests/data/compare/hf_accuracy_test_neural_lehtola.py --case finite_element_sweep/fe12_R040 --out-txt path/to/report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import sys

_DATA_DIR = Path(__file__).resolve().parent.parent
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
from summary_naming import resolve_summary_under

_DEFAULT_REFERENCE = _DATA_DIR / "reference" / "hf" / "lehtola_closed_subshell_atoms_hf.json"
_DEFAULT_SUMMARY_HF = _DATA_DIR / "summary" / "all_electron" / "hf"
_DEFAULT_CASE = ""

# Text report: ``g`` with alternate form ``#`` keeps trailing zeros while using
# significant-digit rules (see format mini-language: ``#`` + ``g``).
_REPORT_SIG_DIGITS = 13
_W_COL_E = 26  # E_tot ref / ours
_W_COL_H = 24  # HOMO ref / ours
_W_COL_D = 22  # deltas and |delta|


@dataclass(frozen=True)
class RefRow:
    z: int
    symbol: str
    e_tot: float
    eps_homo: float
    e_x: float


@dataclass(frozen=True)
class OursRow:
    z: int
    e_tot: float
    eps_homo: float
    e_x: float
    converged: bool


def _load_reference_closed_subshell(path: Path) -> dict[int, RefRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, RefRow] = {}
    for rec in data.get("records", []):
        z = int(rec["Z"])
        minus_e = float(rec["minus_E_tot_au"])
        minus_homo = float(rec["minus_epsilon_homo_au"])
        minus_ex = float(rec["minus_E_x_au"])
        sym = str(rec.get("atom", ""))
        out[z] = RefRow(
            z=z,
            symbol=sym,
            e_tot=-minus_e,
            eps_homo=-minus_homo,
            e_x=-minus_ex,
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


def _load_summary_rows(summary_json: Path) -> dict[int, OursRow]:
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    out: dict[int, OursRow] = {}
    for row in data.get("config_summaries", []):
        z = row.get("atomic_number")
        if z is None:
            continue
        z = int(z)
        conv = bool(row.get("converged", True))
        e_tot = row.get("total_energy_ha")
        occ = row.get("occupied_eigenvalues_ha") or []
        ex = _hf_exchange_from_row(row)
        if e_tot is None or len(occ) == 0 or ex is None:
            continue
        occ_arr = np.asarray(occ, dtype=float)
        homo = float(np.max(occ_arr))
        out[z] = OursRow(
            z=z,
            e_tot=float(e_tot),
            eps_homo=homo,
            e_x=ex,
            converged=conv,
        )
    return out


@dataclass
class CompareRow:
    z: int
    symbol: str
    e_tot_ref: float
    e_tot_ours: float
    d_e_tot: float
    homo_ref: float
    homo_ours: float
    d_homo: float
    ex_ref: float
    ex_ours: float
    d_ex: float


def _build_comparison(
    ref_by_z: dict[int, RefRow],
    ours_by_z: dict[int, OursRow],
) -> tuple[list[CompareRow], int]:
    """Return (rows for converged overlap, count skipped not converged)."""
    shared = sorted(set(ref_by_z.keys()) & set(ours_by_z.keys()))
    skipped = 0
    rows: list[CompareRow] = []
    for z in shared:
        r = ref_by_z[z]
        o = ours_by_z[z]
        if not o.converged:
            skipped += 1
            continue
        rows.append(
            CompareRow(
                z          = z,
                symbol     = r.symbol,
                e_tot_ref  = r.e_tot,
                e_tot_ours = o.e_tot,
                d_e_tot    = o.e_tot - r.e_tot,
                homo_ref   = r.eps_homo,
                homo_ours  = o.eps_homo,
                d_homo     = o.eps_homo - r.eps_homo,
                ex_ref     = r.e_x,
                ex_ours    = o.e_x,
                d_ex       = o.e_x - r.e_x,
            )
        )
    return rows, skipped


def _format_report(
    *,
    ref_path: Path,
    summary_json: Path,
    case: str,
    rows: list[CompareRow],
    skipped_nonconverged: int,
) -> str:
    lines: list[str] = []
    lines.append("HF closed-subshell accuracy (reference vs summary)")
    lines.append("")
    lines.append(f"reference:     {ref_path}")
    lines.append(f"summary JSON:  {summary_json}")
    lines.append(f"case:          {case}")
    lines.append(f"atoms listed:  {len(rows)}  (skipped not converged in overlap: {skipped_nonconverged})")
    lines.append("")
    sig = _REPORT_SIG_DIGITS
    we, wh, wd = _W_COL_E, _W_COL_H, _W_COL_D
    hdr = (
        f"{'Z':>4}  {'sym':^4}  "
        f"{'E_tot ref':^{we}}  {'E_tot ours':^{we}}  {'dE_tot':^{wd}}  {'|dE|':^{wd}}  "
        f"{'HOMO ref':^{wh}}  {'HOMO ours':^{wh}}  {'dHOMO':^{wd}}  {'|dH|':^{wd}}  "
        f"{'E_x ref':^{wh}}  {'E_x ours':^{wh}}  {'dE_x':^{wd}}  {'|dx|':^{wd}}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    abs_de = []
    abs_dh = []
    abs_dx = []
    for w in rows:
        abs_de.append(abs(w.d_e_tot))
        abs_dh.append(abs(w.d_homo))
        abs_dx.append(abs(w.d_ex))
        lines.append(
            f"{w.z:4d}  {w.symbol:^4}  "
            f"{w.e_tot_ref:#{we}.{sig}g}  {w.e_tot_ours:#{we}.{sig}g}  "
            f"{w.d_e_tot:#{wd}.{sig}g}  {abs(w.d_e_tot):#{wd}.{sig}g}  "
            f"{w.homo_ref:#{wh}.{sig}g}  {w.homo_ours:#{wh}.{sig}g}  "
            f"{w.d_homo:#{wd}.{sig}g}  {abs(w.d_homo):#{wd}.{sig}g}  "
            f"{w.ex_ref:#{wh}.{sig}g}  {w.ex_ours:#{wh}.{sig}g}  "
            f"{w.d_ex:#{wd}.{sig}g}  {abs(w.d_ex):#{wd}.{sig}g}"
        )
    lines.append("-" * len(hdr))
    lines.append("")
    lines.append("Aggregate (absolute errors, Ha)")
    lines.append(f"  max  |d E_tot|   = {max(abs_de):#.{sig}g}")
    lines.append(f"  mean |d E_tot|   = {float(np.mean(abs_de)):#.{sig}g}")
    lines.append(f"  max  |d HOMO|   = {max(abs_dh):#.{sig}g}")
    lines.append(f"  mean |d HOMO|   = {float(np.mean(abs_dh)):#.{sig}g}")
    lines.append(f"  max  |d E_x|    = {max(abs_dx):#.{sig}g}")
    lines.append(f"  mean |d E_x|    = {float(np.mean(abs_dx)):#.{sig}g}")
    lines.append("")
    worst_e = max(rows, key=lambda r: abs(r.d_e_tot))
    worst_h = max(rows, key=lambda r: abs(r.d_homo))
    worst_x = max(rows, key=lambda r: abs(r.d_ex))
    lines.append("Largest signed deltas (for diagnosis)")
    lines.append(
        f"  E_tot:  Z={worst_e.z} ({worst_e.symbol})  dE_tot = {worst_e.d_e_tot:+#.{sig}g} Ha"
    )
    lines.append(
        f"  HOMO:   Z={worst_h.z} ({worst_h.symbol})  dHOMO  = {worst_h.d_homo:+#.{sig}g} Ha"
    )
    lines.append(
        f"  E_x:    Z={worst_x.z} ({worst_x.symbol})  dE_x   = {worst_x.d_ex:+#.{sig}g} Ha"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare HF summary to closed-subshell reference (text report only)."
    )
    ap.add_argument(
        "--reference-json",
        type=Path,
        default=_DEFAULT_REFERENCE,
        help="Reference dataset (default: reference/hf/lehtola_closed_subshell_atoms_hf.json).",
    )
    ap.add_argument(
        "--summary-hf-root",
        type=Path,
        default=_DEFAULT_SUMMARY_HF,
        help="Root containing HF summary JSON (default: tests/data/summary/all_electron/hf).",
    )
    ap.add_argument(
        "--case",
        type=str,
        default=_DEFAULT_CASE,
        help=(
            "Optional subpath under --summary-hf-root (sweep case dir or empty for flat JSON). "
            f"(default: {repr(_DEFAULT_CASE)})."
        ),
    )
    ap.add_argument(
        "--out-txt",
        type=Path,
        default=Path(__file__).resolve().parent / "hf_accuracy_test_neural_lehtola_summary.txt",
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

    ref_by_z = _load_reference_closed_subshell(ref_path)
    ours_by_z = _load_summary_rows(summary_json)
    rows, skipped = _build_comparison(ref_by_z, ours_by_z)

    if not rows:
        print(
            "No overlapping converged Z between reference and summary "
            f"(reference Z={sorted(ref_by_z.keys())}, check convergence / exchange field).",
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
