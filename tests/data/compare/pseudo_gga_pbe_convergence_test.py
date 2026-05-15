"""Pseudo GGA-PBE sweep convergence test: max error vs finest reference.

Reads ``fe*_R*__*.json`` under ``summary/pseudo_potential/gga_pbe/<sweep>/<case>/`` and plots domain-radius vs
finite-element panels.

Run::

    python atomSFE/tests/data/compare/pseudo_gga_pbe_convergence_test.py
    python atomSFE/tests/data/compare/pseudo_gga_pbe_convergence_test.py --out path/to/figure.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MultipleLocator

_DATA_DIR = Path(__file__).resolve().parent.parent
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
from summary_naming import glob_sweep_summaries

_SUMMARY_DIR = _DATA_DIR / "summary"
_DEFAULT_GGA_PBE_ROOT = _SUMMARY_DIR / "pseudo_potential" / "gga_pbe"

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 15,
    }
)

AXIS_LABEL_FONTSIZE = 18
X_AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 15
LEGEND_FONTSIZE = 18


def _load_dataset_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_x_from_dataset_name(name: str, mode: str) -> float:
    fe_txt, r_txt = name.split("_")
    fe = float(fe_txt.replace("fe", ""))
    r = float(r_txt.replace("R", ""))
    return r if mode == "domain_radius_sweep" else fe


def _per_atom_metrics(payload: dict) -> dict[str, tuple[float, np.ndarray]]:
    out: dict[str, tuple[float, np.ndarray]] = {}
    for row in payload.get("config_summaries", []):
        z = row.get("atomic_number")
        if z is None:
            continue
        e_tot = row.get("total_energy_ha")
        occ = row.get("occupied_eigenvalues_ha") or []
        if e_tot is None or len(occ) == 0:
            continue
        out[str(int(z))] = (float(e_tot), np.asarray(occ, dtype=float))
    return out


def _build_curve(mode: str, gga_pbe_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sweep_dir = gga_pbe_root / mode
    files = glob_sweep_summaries(sweep_dir)
    if not files:
        raise RuntimeError(f"No summary files found in {sweep_dir}")

    x_and_payload = []
    for p in files:
        x = _parse_x_from_dataset_name(p.parent.name, mode)
        x_and_payload.append((x, _load_dataset_summary(p)))
    x_and_payload.sort(key=lambda t: t[0])

    x_ref, payload_ref = x_and_payload[-1]
    ref_map = _per_atom_metrics(payload_ref)

    xs = []
    y_energy = []
    y_eigen = []

    for x, payload in x_and_payload:
        if np.isclose(x, x_ref):
            continue
        if mode == "domain_radius_sweep" and np.isclose(x, 35.0):
            continue
        cur_map = _per_atom_metrics(payload)
        shared = sorted(set(ref_map.keys()) & set(cur_map.keys()))
        if not shared:
            continue

        e_errs = []
        eig_errs = []
        for z in shared:
            e_ref, eig_ref_all = ref_map[z]
            e_cur, eig_cur_all = cur_map[z]
            e_errs.append(abs(e_cur - e_ref))
            n = min(eig_ref_all.shape[0], eig_cur_all.shape[0])
            if n <= 0:
                continue
            per_atom_mean_eig_err = float(np.mean(np.abs(eig_cur_all[:n] - eig_ref_all[:n])))
            eig_errs.append(per_atom_mean_eig_err)

        if not e_errs or not eig_errs:
            continue
        xs.append(float(x))
        y_energy.append(float(np.max(np.asarray(e_errs, dtype=float))))
        y_eigen.append(float(np.max(np.asarray(eig_errs, dtype=float))))

    return np.asarray(xs), np.asarray(y_energy), np.asarray(y_eigen)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pseudo GGA-PBE sweep convergence-test figure from summary JSON (max error vs finest case).",
    )
    ap.add_argument(
        "--gga-pbe-root",
        type=Path,
        default=_DEFAULT_GGA_PBE_ROOT,
        help="Path to pseudo/gga_pbe summary root (contains domain_radius_sweep/, finite_element_sweep/).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "pseudo_gga_pbe_convergence_test_summary.png",
        help="Output PNG path.",
    )
    args = ap.parse_args()
    root = args.gga_pbe_root.resolve()
    out_png = args.out.resolve()

    x_r, y_r_energy, y_r_eigen = _build_curve("domain_radius_sweep", root)
    x_fe, y_fe_energy, y_fe_eigen = _build_curve("finite_element_sweep", root)
    fe_mask = (x_fe > 1.0) & (~np.isclose(x_fe, 13.0))
    x_fe = x_fe[fe_mask]
    y_fe_energy = y_fe_energy[fe_mask]
    y_fe_eigen = y_fe_eigen[fe_mask]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    ax_l.semilogy(x_r, np.maximum(y_r_energy, 1e-20), marker="o", lw=1.8, label="Energy")
    ax_l.semilogy(x_r, np.maximum(y_r_eigen, 1e-20), marker="s", lw=1.8, label="Eigenvalues")
    ax_l.set_xlabel(r"$R_{max}$ (Bohr)", fontsize=X_AXIS_LABEL_FONTSIZE)
    ax_l.set_ylabel(r"Error (Ha)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_l.yaxis.set_major_locator(LogLocator(base=10))
    ax_l.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax_l.yaxis.set_major_formatter(LogFormatterMathtext())
    ax_l.minorticks_on()
    ax_l.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax_l.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_FONTSIZE - 1)
    ax_l.grid(True, which="major", axis="both", alpha=0.32, linestyle="-", linewidth=0.75)
    ax_l.grid(True, which="minor", axis="y", alpha=0.40, linestyle=":", linewidth=0.70)
    ax_l.legend(fontsize=LEGEND_FONTSIZE)

    ax_r.semilogy(x_fe, np.maximum(y_fe_energy, 1e-20), marker="o", lw=1.8, label="Energy")
    ax_r.semilogy(x_fe, np.maximum(y_fe_eigen, 1e-20), marker="s", lw=1.8, label="Eigenvalues")
    ax_r.set_xlabel(r"$N_{fe}$", fontsize=X_AXIS_LABEL_FONTSIZE)
    ax_r.set_ylabel(r"Error (Ha)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_r.yaxis.set_major_locator(LogLocator(base=10, numticks=100))
    ax_r.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(np.arange(2, 10)), numticks=100))
    ax_r.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax_r.yaxis.set_minor_formatter(
        LogFormatterMathtext(base=10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
    )
    ax_r.minorticks_on()
    ax_r.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax_r.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_FONTSIZE - 1)
    ax_r.tick_params(axis="y", which="minor", labelleft=False)
    if x_fe.size > 0:
        x_min = int(np.floor(float(np.min(x_fe))))
        x_max = int(np.ceil(float(np.max(x_fe))))
        ax_r.set_xlim(x_min - 0.5, x_max + 0.5)
        ax_r.xaxis.set_major_locator(MultipleLocator(1))
    ax_r.grid(True, which="major", axis="both", alpha=0.32, linestyle="-", linewidth=0.75)
    ax_r.grid(True, which="minor", axis="y", alpha=0.45, linestyle=":", linewidth=0.75)
    ax_r.legend(fontsize=LEGEND_FONTSIZE)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig("pseudo_gga_pbe_convergence_test_summary.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure: {out_png}")


if __name__ == "__main__":
    main()
