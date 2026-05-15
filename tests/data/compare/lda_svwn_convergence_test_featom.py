r"""
FEATOM reference vs LDA_SVWN sweep: per-state absolute eigenvalue error.

Left: fixed $N_{\mathrm{fe}}$, vary domain radius $R$ (one curve per $R$).
Right: fixed $R$, vary $N_{\mathrm{fe}}$ (one curve per FE count).

Reference eigenvalues (Hartree) are listed in ``ref_eigenvalues`` (18 states, same
ordering as the FEATOM benchmark). Data are read from committed summaries under
``tests/data/summary/all_electron/lda_svwn/``: each sweep case folder
``domain_radius_sweep/fe{fe}_R{rr}/`` or ``finite_element_sweep/...`` contains
``fe*_R*__*.json``; the chosen configuration's
``occupied_eigenvalues_ha`` are compared to the reference.

Run from anywhere (defaults read ``atomSFE/tests/data/summary/all_electron/lda_svwn/``, write
``atomSFE/tests/data/compare/lda_svwn_convergence_test_featom_summary.png``; no CLI flags)::

    python atomSFE/tests/data/compare/lda_svwn_convergence_test_featom.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (
    LogFormatterMathtext,
    LogLocator,
    NullLocator,
)

_TESTS_DATA = Path(__file__).resolve().parent.parent
if str(_TESTS_DATA) not in sys.path:
    sys.path.insert(0, str(_TESTS_DATA))
from summary_naming import summary_path_for_case_dir

_DEFAULT_LDA_SVWN_ROOT = _TESTS_DATA / "summary" / "all_electron" / "lda_svwn"
_DEFAULT_CONFIGURATION = "configuration_092"
_DEFAULT_FIXED_FE = 12
_DEFAULT_FIXED_R = 40
_DEFAULT_OUT_PNG = Path(__file__).resolve().parent / "lda_svwn_convergence_test_featom_summary.png"

# Distinct curve colors (matplotlib tab10); self-contained so this script needs no repo paths outside ``atom/``.
_COLORS: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def _finalize_paper_semilogy_axes(ax: plt.Axes, *, x_mode: str = "fe") -> None:
    """Log-y grid/ticks in a publication-friendly style; no minor ticks on x (``x_mode='fe'``)."""
    _ = x_mode  # reserved for parity with older callers
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.25)
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(NullLocator())


# Reference eigenvalues from FEATOM (Hartree); 18 states, fixed ordering for comparison.
ref_eigenvalues = np.array(
    [
        -3689.35513984,
        -639.77872809,
        -619.10855018,
        -161.11807321,
        -150.97898016,
        -131.97735828,
        -40.52808425,
        -35.85332083,
        -27.12321230,
        -15.02746007,
        -8.82408940,
        -7.01809220,
        -3.86617513,
        -0.36654335,
        -1.32597632,
        -0.82253797,
        -0.14319018,
        -0.13094786,
    ],
    dtype=float,
)

CASE_NAME_PATTERN = re.compile(r"^fe(?P<fe>\d+)_R(?P<radius>\d+)$")
N_REF = int(ref_eigenvalues.size)


def parse_case_name(case_dir_name: str) -> tuple[int, int]:
    m = CASE_NAME_PATTERN.match(case_dir_name)
    if m is None:
        raise ValueError(f"Unrecognized case directory name: {case_dir_name!r}")
    return int(m.group("fe")), int(m.group("radius"))


def read_occupied_eigenvalues_from_case_summary(
    case_dir: Path, configuration: str
) -> np.ndarray | None:
    p = summary_path_for_case_dir(case_dir)
    if p is None or not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    for row in data.get("config_summaries", []):
        if row.get("configuration") != configuration:
            continue
        if not row.get("converged", True):
            return None
        occ = row.get("occupied_eigenvalues_ha")
        if occ is None:
            return None
        return np.asarray(occ, dtype=float)
    return None


def iter_case_dirs(sweep_root: Path) -> Iterator[Path]:
    if not sweep_root.is_dir():
        return
    for p in sorted(sweep_root.iterdir(), key=lambda x: x.name):
        if p.is_dir():
            yield p


def collect_radius_curves(
    sweep_dir     : Path,
    configuration : str,
    fixed_fe      : int,
    ref           : np.ndarray,
) -> list[tuple[float, np.ndarray]]:
    """Return list of (R_bohr, abs_error[0:n_ref]) sorted by R."""
    rows: list[tuple[float, np.ndarray]] = []
    for case in iter_case_dirs(sweep_dir):
        try:
            fe, r_bohr = parse_case_name(case.name)
        except ValueError:
            continue
        if fe != fixed_fe:
            continue
        ev = read_occupied_eigenvalues_from_case_summary(case, configuration)
        if ev is None or ev.size < ref.size:
            continue
        err = np.abs(ev[: ref.size] - ref)
        rows.append((float(r_bohr), err))
    rows.sort(key=lambda t: t[0])
    return rows


def collect_fe_curves(
    sweep_dir     : Path,
    configuration : str,
    fixed_r       : int,
    ref           : np.ndarray,
) -> list[tuple[float, np.ndarray]]:
    """Return list of (N_fe, abs_error[0:n_ref]) sorted by N_fe; skips N_fe=2."""
    rows: list[tuple[float, np.ndarray]] = []
    for case in iter_case_dirs(sweep_dir):
        try:
            fe, r_bohr = parse_case_name(case.name)
        except ValueError:
            continue
        if int(fe) == 2:
            continue
        if int(r_bohr) != int(fixed_r):
            continue
        ev = read_occupied_eigenvalues_from_case_summary(case, configuration)
        if ev is None or ev.size < ref.size:
            continue
        err = np.abs(ev[: ref.size] - ref)
        rows.append((float(fe), err))
    rows.sort(key=lambda t: t[0])
    return rows


def plot_featom_panels(
    radius_rows : list[tuple[float, np.ndarray]],
    fe_rows     : list[tuple[float, np.ndarray]],
    out_path    : Path,
    *,
    dpi         : int = 300,
) -> None:
    x = np.arange(1, N_REF + 1, dtype=int)
    colors = _COLORS
    fig, (ax_r, ax_fe) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.9),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.09},
    )

    for j, (r_bohr, err) in enumerate(radius_rows):
        c = colors[j % len(colors)]
        ax_r.semilogy(
            x,
            np.maximum(err, 1e-20),
            marker="o",
            ms=3.0,
            lw=1.5,
            label=rf"$R={int(r_bohr)}$",
            color=c,
        )

    for j, (n_fe, err) in enumerate(fe_rows):
        c = colors[j % len(colors)]
        ax_fe.semilogy(
            x,
            np.maximum(err, 1e-20),
            marker="o",
            ms=3.0,
            lw=1.5,
            label=rf"$N_{{\mathrm{{fe}}}}={int(n_fe)}$",
            color=c,
        )

    y_label = r"$|\varepsilon_i-\varepsilon_i^{\mathrm{ref}}|$ (Ha)"
    ax_r.set_xlabel(r"Eigenvalue index $i$")
    ax_r.set_ylabel(y_label)
    ax_r.set_xticks(x)
    ax_fe.set_xlabel(r"Eigenvalue index $i$")
    ax_fe.set_ylabel(y_label)
    ax_fe.set_xticks(x)

    _finalize_paper_semilogy_axes(ax_r, x_mode="fe")
    _finalize_paper_semilogy_axes(ax_fe, x_mode="fe")
    ax_r.tick_params(axis="y", which="major", labelsize=9)
    ax_fe.tick_params(axis="y", which="major", labelsize=9)
    ax_r.legend(fontsize=9, ncol=2, loc="upper right")
    ax_fe.legend(fontsize=9, ncol=2, loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = _DEFAULT_LDA_SVWN_ROOT.resolve()
    out_path = _DEFAULT_OUT_PNG.resolve()

    if ref_eigenvalues.size != N_REF:
        print("ref_eigenvalues length mismatch", file=sys.stderr)
        sys.exit(1)

    r_sweep = root / "domain_radius_sweep"
    fe_sweep = root / "finite_element_sweep"

    radius_rows = collect_radius_curves(
        r_sweep,
        _DEFAULT_CONFIGURATION,
        _DEFAULT_FIXED_FE,
        ref_eigenvalues,
    )
    fe_rows = collect_fe_curves(
        fe_sweep,
        _DEFAULT_CONFIGURATION,
        _DEFAULT_FIXED_R,
        ref_eigenvalues,
    )

    if not radius_rows:
        print(
            f"No radius curves (check {r_sweep} and fe={_DEFAULT_FIXED_FE}).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not fe_rows:
        print(
            f"No FE curves (check {fe_sweep} and R={_DEFAULT_FIXED_R}).",
            file=sys.stderr,
        )
        sys.exit(1)

    plot_featom_panels(radius_rows, fe_rows, out_path)
    print(f"Wrote {out_path} ({len(radius_rows)} R-curves, {len(fe_rows)} FE-curves).")


if __name__ == "__main__":
    main()
