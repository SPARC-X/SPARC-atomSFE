"""
Generate LaTeX table bodies for pseudopotential_accuracy.tex from summary JSON
and M-SPARC reference (same (n,l) pairing as pseudo_accuracy_test_msparc.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_DATA_DIR = Path(__file__).resolve().parent.parent
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
from summary_naming import default_pseudo_summary

_REFERENCE_DIR = _DATA_DIR / "reference" / "pseudo_potential"
_REPO_ROOT = Path(__file__).resolve().parents[4]

_CASES: tuple[tuple[str, str, str], ...] = (
    ("lda_svwn", "msparc_atoms_lda_svwn.json", "LDA_SVWN"),
    ("gga_pbe", "msparc_atoms_gga_pbe.json", "GGA_PBE"),
    ("rscan", "msparc_atoms_rscan.json", "rSCAN"),
    ("pbe0", "msparc_atoms_pbe0.json", "PBE0"),
)

_ELEMENT_ORDER = ("He", "N", "O", "Mn", "Fe", "Mo", "Cs")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_reference_doc(ref_doc: Any) -> dict[str, Any]:
    if isinstance(ref_doc, dict):
        return ref_doc.get("elements") or {}
    if isinstance(ref_doc, list):
        return {str(item["element"]): item for item in ref_doc if isinstance(item, dict) and item.get("element")}
    return {}


def _ref_states(ref_el: dict[str, Any]) -> list[dict[str, Any]]:
    states = list(ref_el.get("occupied_states") or [])
    states.sort(key=lambda st: (int(st["n"]), int(st["l"])))
    return states


def _ref_eps(st: dict[str, Any]) -> float:
    eu = float(st.get("eigenvalue_up_Ha", 0.0))
    ed = float(st.get("eigenvalue_down_Ha", eu))
    return 0.5 * (eu + ed)


def _g_nl(st: dict[str, Any]) -> int:
    ou = float(st.get("occupation_up", 0.0))
    od = float(st.get("occupation_down", 0.0))
    return int(round(ou + od))


def _ours_nl(path: Path) -> list[tuple[int, int, float]]:
    d = _load_json(path)
    out: list[tuple[int, int, float]] = []
    for st in d.get("occupied_states") or []:
        out.append((int(st["n"]), int(st["l"]), float(st["eigenvalue_Ha"])))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _fmt_sci(x: float) -> str:
    if x == 0.0:
        return "$0$"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10.0**exp)
    mant = float(f"{mant:.3g}")
    if abs(mant) >= 10.0:
        mant /= 10.0
        exp += 1
    mant_str = f"{mant:.2f}".rstrip("0").rstrip(".")
    return rf"${mant_str} \times 10^{{{exp}}}$"


def _fmt_pos_ha(x: float, *, eig: bool = False) -> str:
    ax = abs(x)
    if eig:
        s = f"{x:.6f}"
    elif ax >= 100.0:
        s = f"{x:.7f}"
    elif ax >= 10.0:
        s = f"{x:.8f}"
    else:
        s = f"{x:.9f}"
    s = s.rstrip("0").rstrip(".")
    return f"${s}$"


def _collect_element(
    row: dict[str, Any],
    ref_el: dict[str, Any],
    repo_root: Path,
    date_pseudo_xc: str,
) -> dict[str, Any] | None:
    el = str(row.get("element", ""))
    e_ours = float(row.get("total_energy_ha", float("nan")))
    e_ref = float(ref_el.get("Etot", (ref_el.get("energies_Ha") or {}).get("Etot", float("nan"))))
    ad = repo_root / "date_pseudo" / date_pseudo_xc / el / "data" / "atom_dataset.json"
    ours_nl = _ours_nl(ad) if ad.is_file() else None
    ref_states = _ref_states(ref_el)
    if ours_nl is None:
        return None
    orbitals: list[dict[str, Any]] = []
    n_orb = min(len(ours_nl), len(ref_states))
    for i in range(n_orb):
        nr, lr = int(ref_states[i]["n"]), int(ref_states[i]["l"])
        no, lo, eo = ours_nl[i]
        er = _ref_eps(ref_states[i])
        orbitals.append(
            {
                "n": nr,
                "l": lr,
                "g": _g_nl(ref_states[i]),
                "eps_ours": -eo,
                "eps_ref": -er,
                "diff_eps": abs(eo - er),
            }
        )
    return {
        "element": el,
        "Z": int(row.get("atomic_number", 0)),
        "E_ours": -e_ours,
        "E_ref": -e_ref,
        "diff_E": abs(e_ours - e_ref),
        "orbitals": orbitals,
    }


def _emit_element_block(el_data: dict[str, Any]) -> list[str]:
    el = el_data["element"]
    z = el_data["Z"]
    nrows = len(el_data["orbitals"])
    lines: list[str] = []
    e_o = _fmt_pos_ha(el_data["E_ours"])
    e_r = _fmt_pos_ha(el_data["E_ref"])
    d_e = _fmt_sci(el_data["diff_E"])
    for i, orb in enumerate(el_data["orbitals"]):
        eps_o = _fmt_pos_ha(orb["eps_ours"], eig=True)
        eps_r = _fmt_pos_ha(orb["eps_ref"], eig=True)
        d_eps = _fmt_sci(orb["diff_eps"])
        if i == 0:
            lines.append(
                f"        \\multirow[t]{{{nrows}}}{{*}}{{{el}}} & "
                f"\\multirow[t]{{{nrows}}}{{*}}{{{z}}} & "
                f"\\multirow[t]{{{nrows}}}{{*}}{{{e_o}}} & "
                f"\\multirow[t]{{{nrows}}}{{*}}{{{e_r}}} & "
                f"\\multirow[t]{{{nrows}}}{{*}}{{{d_e}}} & "
                f"{orb['n']} & {orb['l']} & {orb['g']} & {eps_o} & {eps_r} & {d_eps} \\\\"
            )
        else:
            lines.append(
                f"         &  &  &  &  & {orb['n']} & {orb['l']} & {orb['g']} & "
                f"{eps_o} & {eps_r} & {d_eps} \\\\"
            )
    return lines


def generate_table_body(summary_subdir: str, ref_name: str, date_pseudo_xc: str) -> str:
    summary_path = default_pseudo_summary(_DATA_DIR, summary_subdir)
    ref_path = _REFERENCE_DIR / ref_name
    summ = _load_json(summary_path)
    ref_elements = _normalize_reference_doc(_load_json(ref_path))
    by_el = {str(r["element"]): r for r in summ.get("config_summaries") or []}
    lines: list[str] = []
    for el in _ELEMENT_ORDER:
        if el not in by_el or el not in ref_elements:
            continue
        block = _collect_element(by_el[el], ref_elements[el], _REPO_ROOT, date_pseudo_xc)
        if block is None:
            raise RuntimeError(f"{el}: missing atom_dataset for {date_pseudo_xc}")
        lines.extend(_emit_element_block(block))
    return "\n".join(lines)


def main() -> None:
    labels = ("lda_svwn", "gga_pbe", "rscan", "pbe0")
    for (summary_subdir, ref_name, date_pseudo_xc), label in zip(_CASES, labels):
        print(f"%% --- {label} ---")
        print(generate_table_body(summary_subdir, ref_name, date_pseudo_xc))
        print()


if __name__ == "__main__":
    main()
