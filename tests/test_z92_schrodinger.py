"""
Test case for Z92 Schrodinger mode against analytic hydrogenic levels.

This test validates the `xc_functional='Schrodinger'` mode where both XC and
Hartree contributions are disabled in the Hamiltonian build path.
"""

# If True, write paper/figures/test_z92_schrodinger_summary.pdf after the test suite (tests always run).
REGENERATE_SUMMARY_PDF = False

from pathlib import Path
import sys
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atom.solver import AtomicDFTSolver

_SCHRODINGER_RUN_CACHE = None


def print_test_passed(test_name: str):
    print("\t {:<50} : test passed".format(test_name))


def print_test_failed(test_name: str, error_msg: str = ""):
    print("\t {:<50} : test FAILED".format(test_name))
    if error_msg:
        print("\t\t Error: {}".format(error_msg))

def _run_as_bool(test_func):
    try:
        test_func()
        return True
    except Exception:
        return False


def hydrogenic_analytic_energy(n: int, z: int) -> float:
    return -(z**2) / (2.0 * (n**2))


def z92_schrodinger_occupied_abs_errors(results, z: int = 92) -> np.ndarray:
    """Per-occupied-state |E_numeric - E_analytic| (Ha) for hydrogenic comparison."""
    occ_info = results["occupation_info"]
    n_occ = occ_info.n_states
    full_eigs = np.asarray(results["full_eigen_energies"], dtype=float)
    if full_eigs.shape[0] < n_occ:
        raise ValueError(
            f"full_eigen_energies length {full_eigs.shape[0]} < n occupied states {n_occ}"
        )

    occ_n = np.asarray(occ_info.occ_n, dtype=int)
    numeric = full_eigs[:n_occ]
    analytic = np.array(
        [hydrogenic_analytic_energy(n=int(occ_n[i]), z=z) for i in range(n_occ)],
        dtype=float,
    )
    return np.abs(numeric - analytic)


def run_z92_schrodinger_once():
    global _SCHRODINGER_RUN_CACHE
    if _SCHRODINGER_RUN_CACHE is not None:
        _, elapsed = _SCHRODINGER_RUN_CACHE
        print(f"Using cached Z92 Schrodinger results (computed in {elapsed:.2f} seconds)")
        return _SCHRODINGER_RUN_CACHE

    print("Running Z92 Schrodinger calculation...")
    start_time = time.time()
    solver = AtomicDFTSolver(
        atomic_number           = 92,
        n_electrons             = 92,
        xc_functional           = "Schrodinger",
        domain_size             = 40.0,
        finite_element_number   = 12,
        polynomial_order        = 20,
        quadrature_point_number = 60,
        mesh_type               = "exponential",
        mesh_concentration      = 101.0,
        scf_tolerance           = 1e-20,
        verbose                 = True,
        all_electron_flag       = True,
        use_oep                 = False,
        use_preconditioner      = True,
    )
    results = solver.solve(save_full_spectrum=True, use_warm_start=False)
    elapsed = time.time() - start_time
    print(f"Computation time: {elapsed:.2f} seconds")

    _SCHRODINGER_RUN_CACHE = (results, elapsed)
    return _SCHRODINGER_RUN_CACHE


def test_z92_schrodinger_basic():
    print("\n" + "=" * 60)
    print("Test: Basic Z92 Schrodinger calculation")
    print("=" * 60)

    try:
        results, _ = run_z92_schrodinger_once()

        assert results is not None, "Results should not be None"
        assert "full_eigen_energies" in results, "Results should contain full_eigen_energies"
        assert "full_l_terms" in results, "Results should contain full_l_terms"
        assert bool(results["converged"]), "SCF should converge"

        print_test_passed("Basic Z92 Schrodinger calculation")
        return

    except Exception as e:
        print_test_failed("Basic Z92 Schrodinger calculation", str(e))
        import traceback
        traceback.print_exc()
        raise


def test_z92_schrodinger_hydrogenic_eigenvalues():
    print("\n" + "=" * 60)
    print("Test: Z92 Schrodinger eigenvalues vs hydrogenic analytic")
    print("=" * 60)

    z = 92

    try:
        results, _ = run_z92_schrodinger_once()

        occ_info = results["occupation_info"]
        n_occ    = occ_info.n_states
        full_eigs = np.asarray(results["full_eigen_energies"], dtype=float)
        if full_eigs.shape[0] < n_occ:
            raise ValueError(
                f"full_eigen_energies length {full_eigs.shape[0]} < n occupied states {n_occ}"
            )

        occ_n = np.asarray(occ_info.occ_n, dtype=int)
        occ_l = np.asarray(occ_info.occ_l, dtype=int)
        f_occ = np.asarray(occ_info.occ_spin_up_plus_spin_down, dtype=float)
        degen = 2 * (2 * occ_l + 1)

        abs_err = z92_schrodinger_occupied_abs_errors(results, z=z)
        numeric = full_eigs[:n_occ]
        analytic = np.array(
            [hydrogenic_analytic_energy(n=int(occ_n[i]), z=z) for i in range(n_occ)],
            dtype=float,
        )
        max_err = float(np.max(abs_err))

        print("\nPer-state comparison (all occupied KS states, hydrogenic E_n = -Z^2/(2n^2)):")
        print(
            f"{'#':<4} {'n':<4} {'l':<4} {'occ':<6} {'g':<4} "
            f"{'numeric (Ha)':<22} {'analytic (Ha)':<22} {'|err|':<12}"
        )
        print("-" * 92)
        for i in range(n_occ):
            print(
                f"{i:<4} {occ_n[i]:<4} {occ_l[i]:<4} {f_occ[i]:<6.4g} {degen[i]:<4} "
                f"{numeric[i]:<22.13f} {analytic[i]:<22.13f} {abs_err[i]:<12.3e}"
            )
        print(f"\nMax abs error (over {n_occ} occupied states): {max_err:.6e} Ha")

        tolerance = 1e-5
        if max_err < tolerance:
            print_test_passed(f"Z92 Schrodinger eigenvalues (max err < {tolerance:.1e} Ha)")
            return

        error_msg = f"Max error {max_err:.3e} exceeds tolerance {tolerance:.1e}"
        print_test_failed(
            "Z92 Schrodinger eigenvalues",
            error_msg,
        )
        raise AssertionError(error_msg)

    except Exception as e:
        print_test_failed("Z92 Schrodinger eigenvalue comparison", str(e))
        import traceback
        traceback.print_exc()
        raise


def write_z92_schrodinger_summary_pdf(schrodinger_errors: np.ndarray) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    schrodinger_errors = np.asarray(schrodinger_errors, dtype=float)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
        }
    )

    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    violin = ax.violinplot(
        [schrodinger_errors],
        positions=[0],
        widths=0.72,
        points=300,
        bw_method=0.18,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    body = violin["bodies"][0]
    body.set_facecolor("#f2c7b0")
    body.set_edgecolor("#e95d1a")
    body.set_alpha(0.85)
    body.set_linewidth(0.9)

    mean_v = float(np.mean(schrodinger_errors))
    median_v = float(np.median(schrodinger_errors))

    ax.scatter(
        [0],
        [mean_v],
        marker="s",
        s=38,
        color="red",
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
        label="Mean",
    )
    ax.scatter(
        [0],
        [median_v],
        marker="o",
        s=52,
        color="lime",
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
        label="Median",
    )

    ax.set_yscale("log")
    ax.set_xlim(-0.55, 0.55)
    ax.set_xticks([])
    ax.set_ylabel(r"Error (Ha)")
    ax.set_facecolor("white")
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, length=5.5, width=1.0)

    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="red", markeredgecolor="black", markersize=8, label="Mean"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lime", markeredgecolor="black", markersize=10, label="Median"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        borderpad=0.3,
        handletextpad=0.5,
        labelspacing=0.3,
    )

    figures_dir = Path(__file__).resolve().parent / "paper" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / f"{Path(__file__).stem}_summary.pdf"
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Z92 Schrodinger Test Suite")
    print("=" * 60)

    test_results = []
    test_results.append(("Basic calculation", _run_as_bool(test_z92_schrodinger_basic)))
    test_results.append(("Hydrogenic eigenvalue comparison", _run_as_bool(test_z92_schrodinger_hydrogenic_eigenvalues)))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in test_results if result)
    total  = len(test_results)
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"  {test_name:<35} : {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    exit_code = 0 if passed == total else 1

    if REGENERATE_SUMMARY_PDF:
        if _SCHRODINGER_RUN_CACHE is None:
            results, _ = run_z92_schrodinger_once()
        else:
            results, _ = _SCHRODINGER_RUN_CACHE
        errors = z92_schrodinger_occupied_abs_errors(results)
        out_pdf = write_z92_schrodinger_summary_pdf(errors)
        print(f"Saved plot to: {out_pdf}")

    sys.exit(exit_code)
