"""
Compare ``tests/data/summary/pseudo/<xc>/configuration_energy_summary.json``
against ``tests/data/reference/pseudo/msparc_atoms_*.json`` for four functionals
(LDA_SVWN, GGA_PBE, RSCAN, PBE0).

Per element:
- total energy ``Etot`` vs ``total_energy_ha`` from the summary row
- occupied KS eigenvalues matched on ``(n, l)`` using ``date_pseudo/<XC>/<element>/data/atom_dataset.json``
  when present under the inferred repo root (``Path(__file__).resolve().parents[4]``); if that file is
  missing, falls back to comparing **sorted** occupied eigenvalue lists (same length only).

Writes:
    pseudo_accuracy_test_msparc_summary.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_DATA_DIR = Path(__file__).resolve().parent.parent
_COMPARE_DIR = Path(__file__).resolve().parent
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OUT_TXT = _COMPARE_DIR / "pseudo_accuracy_test_msparc_summary.txt"

# (summary subdir under tests/data/summary/pseudo, M-SPARC reference filename, date_pseudo folder name)
_CASES: tuple[tuple[str, str, str], ...] = (
    ("lda_svwn", "msparc_atoms_lda_svwn.json", "LDA_SVWN"),
    ("gga_pbe", "msparc_atoms_gga_pbe.json", "GGA_PBE"),
    ("rscan", "msparc_atoms_rscan.json", "rSCAN"),
    ("pbe0", "msparc_atoms_pbe0.json", "PBE0"),
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ref_element_states(ref_el: dict[str, Any]) -> tuple[float, list[tuple[int, int, float]]]:
    # Backward compatible:
    # - old reference schema: {"energies_Ha": {"Etot": ...}, "occupied_states": [...]}
    # - new reference schema: {"Etot": ..., "occupied_states": [...]}
    eh = ref_el.get("energies_Ha") or {}
    etot = float(ref_el.get("Etot", eh.get("Etot", float("nan"))))
    states: list[tuple[int, int, float]] = []
    for st in ref_el.get("occupied_states") or []:
        n = int(st["n"])
        l = int(st["l"])
        eu = float(st.get("eigenvalue_up_Ha", 0.0))
        ed = float(st.get("eigenvalue_down_Ha", eu))
        eps = 0.5 * (eu + ed)
        states.append((n, l, eps))
    states.sort(key=lambda t: (t[0], t[1]))
    return etot, states


def _normalize_reference_doc(ref_doc: Any) -> tuple[dict[str, Any], str]:
    """
    Return a per-element mapping and an XC label from either schema:
    - old: {"xc": "...", "elements": {"He": {...}, ...}}
    - new: [{"element": "He", "xc": "...", ...}, ...]
    """
    if isinstance(ref_doc, dict):
        ref_elements = ref_doc.get("elements") or {}
        xc_label = str(ref_doc.get("xc", ""))
        return ref_elements, xc_label

    if isinstance(ref_doc, list):
        ref_elements: dict[str, Any] = {}
        xc_label = ""
        for item in ref_doc:
            if not isinstance(item, dict):
                continue
            el = str(item.get("element", "")).strip()
            if not el:
                continue
            ref_elements[el] = item
            if not xc_label and item.get("xc"):
                xc_label = str(item["xc"])
        return ref_elements, xc_label

    return {}, ""


def _ours_nl_eps_from_atom_dataset(path: Path) -> list[tuple[int, int, float]] | None:
    if not path.is_file():
        return None
    try:
        d = _load_json(path)
    except (json.JSONDecodeError, OSError):
        return None
    occ = d.get("occupied_states") or []
    out: list[tuple[int, int, float]] = []
    for st in occ:
        out.append((int(st["n"]), int(st["l"]), float(st["eigenvalue_Ha"])))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _format_case_report(
    *,
    xc_label: str,
    summary_path: Path,
    ref_path: Path,
    repo_root: Path,
    date_pseudo_xc: str,
    rows: list[dict[str, Any]],
    notes: list[str],
) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"XC = {xc_label}")
    lines.append("=" * 78)
    lines.append(f"summary: {summary_path}")
    lines.append(f"reference: {ref_path}")
    lines.append(f"repo root (for atom_dataset): {repo_root}")
    lines.append(f"date_pseudo folder: {date_pseudo_xc}")
    lines.append("")
    if notes:
        lines.append("Notes")
        for n in notes:
            lines.append(f"  - {n}")
        lines.append("")

    hdr = (
        f"{'El':<4}  {'Z':>3}  {'dE_tot (Ha)':>14}  {'max|d eps|':>14}  "
        f"{'mean|d eps|':>14}  {'n_orb':>5}  {'match':<8}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        lines.append(
            f"{r['element']:<4}  {r['Z']:3d}  {r['dE']:14.6e}  {r['max_abs_deps']:14.6e}  "
            f"{r['mean_abs_deps']:14.6e}  {r['n_orb']:5d}  {r['match_mode']:<8}"
        )
    lines.append("-" * len(hdr))
    if rows:
        des = [abs(r["dE"]) for r in rows]
        mx = [r["max_abs_deps"] for r in rows]
        mn = [r["mean_abs_deps"] for r in rows]
        lines.append("")
        lines.append("Aggregate (this XC)")
        lines.append(f"  max  |dE_tot|   = {max(des):.6e} Ha")
        lines.append(f"  mean |dE_tot|   = {float(np.mean(des)):.6e} Ha")
        lines.append(f"  max  max|d eps| = {max(mx):.6e} Ha")
        lines.append(f"  mean max|d eps| = {float(np.mean(mx)):.6e} Ha")
        lines.append(f"  mean mean|d eps| = {float(np.mean(mn)):.6e} Ha")
    lines.append("")
    return "\n".join(lines) + "\n"


def _run_one_case(
    *,
    summary_subdir: str,
    ref_name: str,
    date_pseudo_xc: str,
    repo_root: Path,
) -> tuple[str, list[dict[str, Any]]]:
    summary_path = _DATA_DIR / "summary" / "pseudo" / summary_subdir / "configuration_energy_summary.json"
    ref_path = _DATA_DIR / "reference" / "pseudo" / ref_name
    notes: list[str] = []
    if not summary_path.is_file():
        return f"SKIP {summary_subdir}: missing {summary_path}\n", []
    if not ref_path.is_file():
        return f"SKIP {summary_subdir}: missing {ref_path}\n", []

    ref_doc = _load_json(ref_path)
    ref_elements, ref_xc = _normalize_reference_doc(ref_doc)
    summ = _load_json(summary_path)
    xc_label = ref_xc or summary_subdir

    rows: list[dict[str, Any]] = []
    for row in summ.get("config_summaries") or []:
        el = str(row.get("element", ""))
        if el not in ref_elements:
            notes.append(f"{el}: not in reference elements; skipped.")
            continue
        z = int(row.get("atomic_number", 0))
        e_ours = float(row.get("total_energy_ha", float("nan")))
        etot_ref, ref_states = _ref_element_states(ref_elements[el])

        dE = e_ours - etot_ref
        ad_path = repo_root / "date_pseudo" / date_pseudo_xc / el / "data" / "atom_dataset.json"
        ours_nl = _ours_nl_eps_from_atom_dataset(ad_path)
        if ours_nl is None:
            notes.append(f"{el}: atom_dataset missing at {ad_path}; using sorted-eigenvalue fallback.")
            ref_eps = np.asarray([t[2] for t in ref_states], dtype=float)
            ours_eps = np.asarray(row.get("occupied_eigenvalues_ha") or [], dtype=float)
            ref_eps.sort()
            ours_eps.sort()
            n = min(ref_eps.size, ours_eps.size)
            if ref_eps.size != ours_eps.size:
                notes.append(
                    f"{el}: occupied count ref={ref_eps.size} ours={ours_eps.size}; comparing first {n}."
                )
            if n == 0:
                notes.append(f"{el}: no eigenvalues to compare; skipped.")
                continue
            diff = np.abs(ours_eps[:n] - ref_eps[:n])
            rows.append(
                {
                    "element": el,
                    "Z": z,
                    "dE": dE,
                    "max_abs_deps": float(np.max(diff)),
                    "mean_abs_deps": float(np.mean(diff)),
                    "n_orb": n,
                    "match_mode": "sorted",
                }
            )
            continue

        if len(ours_nl) != len(ref_states):
            notes.append(
                f"{el}: orbital count mismatch ref={len(ref_states)} ours={len(ours_nl)}; "
                "pairing min length by sorted (n,l)."
            )
        n_orb = min(len(ours_nl), len(ref_states))
        diffs: list[float] = []
        for i in range(n_orb):
            nr, lr, er = ref_states[i]
            no, lo, eo = ours_nl[i]
            if (nr, lr) != (no, lo):
                notes.append(
                    f"{el}: (n,l) mismatch at index {i}: ref ({nr},{lr}) vs ours ({no},{lo}); using |Δε| anyway."
                )
            diffs.append(abs(eo - er))
        arr = np.asarray(diffs, dtype=float)
        rows.append(
            {
                "element": el,
                "Z": z,
                "dE": dE,
                "max_abs_deps": float(np.max(arr)) if arr.size else 0.0,
                "mean_abs_deps": float(np.mean(arr)) if arr.size else 0.0,
                "n_orb": n_orb,
                "match_mode": "n,l",
            }
        )

    txt = _format_case_report(
        xc_label=xc_label,
        summary_path=summary_path,
        ref_path=ref_path,
        repo_root=repo_root,
        date_pseudo_xc=date_pseudo_xc,
        rows=rows,
        notes=notes,
    )
    return txt, rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare summary/pseudo configuration_energy_summary vs reference/pseudo M-SPARC JSON."
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=_DEFAULT_REPO_ROOT,
        help="Repository root containing date_pseudo/ (default: four parents above this file).",
    )
    ap.add_argument("--out-txt", type=Path, default=_DEFAULT_OUT_TXT)
    args = ap.parse_args()
    repo_root = args.repo_root.resolve()

    chunks: list[str] = []
    chunks.append("Pseudo valence dataset vs M-SPARC (reference/pseudo) - four functionals")
    chunks.append("")
    chunks.append(f"repo_root: {repo_root}")
    chunks.append("")

    all_des: list[float] = []
    all_maxeps: list[float] = []

    for summary_subdir, ref_name, date_pseudo_xc in _CASES:
        block, rows = _run_one_case(
            summary_subdir=summary_subdir,
            ref_name=ref_name,
            date_pseudo_xc=date_pseudo_xc,
            repo_root=repo_root,
        )
        chunks.append(block)
        for r in rows:
            all_des.append(abs(float(r["dE"])))
            all_maxeps.append(float(r["max_abs_deps"]))

    if all_des:
        chunks.append("=" * 78)
        chunks.append("Global (all four XCs, all compared elements)")
        chunks.append("=" * 78)
        chunks.append(f"  max  |dE_tot|   = {max(all_des):.6e} Ha")
        chunks.append(f"  mean |dE_tot|   = {float(np.mean(all_des)):.6e} Ha")
        if all_maxeps:
            chunks.append(f"  max  max|d eps| = {max(all_maxeps):.6e} Ha")
            chunks.append(f"  mean max|d eps| = {float(np.mean(all_maxeps)):.6e} Ha")
        else:
            chunks.append("  (no eigenvalue rows aggregated)")
        chunks.append("")

    out_txt = args.out_txt.resolve()
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    final = "\n".join(chunks)
    out_txt.write_text(final, encoding="utf-8")
    sys.stdout.write(final)
    print(f"(also wrote {out_txt})", flush=True)


if __name__ == "__main__":
    main()
