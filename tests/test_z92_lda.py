"""
Test case for Z92 LDA_SVWN calculation against reference values.

This module tests the accuracy of the AtomicDFTSolver for Z=92
using LDA_SVWN functional by comparing computed eigenvalues with reference
values from the featom paper.

Reference:
----------
Ondrej Certik, John E. Pask, Isuru Fernando, Rohit Goswami, N. Sukumar,
Lee. A. Collins, Gianmarco Manzini, Jiri Vackar,
High-order finite element method for atomic structure calculations,
Computer Physics Communications,
Volume 297,
2024,
109051,
ISSN 0010-4655,
https://doi.org/10.1016/j.cpc.2023.109051.
(https://www.sciencedirect.com/science/article/pii/S001046552300396X)
"""

from pathlib import Path
import sys
import numpy as np
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atom.solver import AtomicDFTSolver

_Z92_LDA_RUN_CACHE = None


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


def run_z92_lda_once():
    """Run the expensive Z92 LDA solve once and reuse across tests."""
    global _Z92_LDA_RUN_CACHE
    if _Z92_LDA_RUN_CACHE is not None:
        results, elapsed_time = _Z92_LDA_RUN_CACHE
        print(f"Using cached Z92 LDA_SVWN results (computed in {elapsed_time:.2f} seconds)")
        return _Z92_LDA_RUN_CACHE

    start_time = time.time()
    atomic_dft_solver = AtomicDFTSolver(
        atomic_number           = 92,
        xc_functional           = "LDA_SVWN",
        domain_size             = 40.0,
        finite_element_number   = 8,
        polynomial_order        = 31,
        quadrature_point_number = 70,
        mesh_type               = "exponential",
        mesh_concentration      = 101.0,
        scf_tolerance           = 1e-11,
        verbose                 = True,
        all_electron_flag       = True,
        use_oep                 = False,
        use_preconditioner      = True,
    )
    results = atomic_dft_solver.solve(save_energy_density=True)
    elapsed_time = time.time() - start_time
    _Z92_LDA_RUN_CACHE = (results, elapsed_time)
    return _Z92_LDA_RUN_CACHE


def test_z92_lda_svwn_eigenvalues():
    """Test Z92 LDA_SVWN calculation against reference eigenvalues."""
    print("\n" + "=" * 60)
    print("Test: Z92 LDA_SVWN eigenvalues vs. featom reference")
    print("=" * 60)
    print("Reference: Certik et al., Comput. Phys. Commun. 297, 109051 (2024)")
    print("=" * 60)

    ref_eigenvalues = np.array([
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
        -0.13094786
    ])

    try:
        results, elapsed_time = run_z92_lda_once()
        print(f"\nComputation time: {elapsed_time:.2f} seconds")

        rho                  = results["rho"]
        orbitals             = results["orbitals"]
        computed_eigenvalues = results["eigen_energies"]

        print(f"\nrho.shape      = {rho.shape}")
        print(f"orbitals.shape = {orbitals.shape}")
        print(f"Number of eigenvalues computed: {len(computed_eigenvalues)}")
        print(f"Number of reference eigenvalues: {len(ref_eigenvalues)}")

        n_compare       = min(len(computed_eigenvalues), len(ref_eigenvalues))
        computed_subset = computed_eigenvalues[:n_compare]
        ref_subset      = ref_eigenvalues[:n_compare]

        differences = computed_subset - ref_subset
        max_diff    = np.max(np.abs(differences))
        mean_diff   = np.mean(np.abs(differences))
        rms_diff    = np.sqrt(np.mean(differences**2))

        print(f"\nEigenvalue comparison (first {n_compare} eigenvalues):")
        print(f"  Max absolute difference:  {max_diff:.2e} Hartree")
        print(f"  Mean absolute difference: {mean_diff:.2e} Hartree")
        print(f"  RMS difference:           {rms_diff:.2e} Hartree")

        print("\nDetailed comparison:")
        print(f"{'Index':<6} {'Computed':<15} {'Reference':<15} {'Difference':<15}")
        print("-" * 60)
        for i in range(n_compare):
            diff = computed_subset[i] - ref_subset[i]
            print(f"{i:<6} {computed_subset[i]:<15.8f} {ref_subset[i]:<15.8f} {diff:<15.8e}")

        if "e_x_local" in results and "e_c_local" in results:
            e_x_local = results["e_x_local"]
            e_c_local = results["e_c_local"]
            print(f"\ne_x_local.shape = {e_x_local.shape}")
            print(f"e_c_local.shape = {e_c_local.shape}")

        tolerance = 1e-5
        if max_diff < tolerance:
            print_test_passed(f"Z92 LDA_SVWN eigenvalues (max diff < {tolerance:.1e} Hartree)")
            return

        error_msg = f"Max difference {max_diff:.2e} exceeds tolerance {tolerance:.1e}"
        print_test_failed("Z92 LDA_SVWN eigenvalues", error_msg)
        print("\nWarning: Differences exceed tolerance but may still be acceptable")
        print("         depending on computational parameters and mesh settings.")
        raise AssertionError(error_msg)

    except Exception as e:
        print_test_failed("Z92 LDA_SVWN eigenvalues", str(e))
        import traceback
        traceback.print_exc()
        raise


def test_z92_lda_svwn_basic():
    """Test basic Z92 LDA_SVWN calculation without detailed comparison."""
    print("\n" + "=" * 60)
    print("Test: Basic Z92 LDA_SVWN calculation")
    print("=" * 60)

    try:
        results, elapsed_time = run_z92_lda_once()
        print(f"\nComputation time: {elapsed_time:.2f} seconds")

        assert results is not None, "Results should not be None"
        assert "rho" in results, "Results should contain 'rho'"
        assert "orbitals" in results, "Results should contain 'orbitals'"
        assert "eigen_energies" in results, "Results should contain 'eigen_energies'"

        rho            = results["rho"]
        orbitals       = results["orbitals"]
        eigen_energies = results["eigen_energies"]

        assert len(rho) > 0, "Density should have non-zero length"
        assert orbitals.shape[0] > 0, "Orbitals should have non-zero shape"
        assert len(eigen_energies) > 0, "Eigenvalues should have non-zero length"
        assert np.all(eigen_energies < 0), "All eigenvalues should be negative (bound states)"

        print_test_passed("Basic Z92 LDA_SVWN calculation")
        return

    except Exception as e:
        print_test_failed("Basic Z92 LDA_SVWN calculation", str(e))
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Z92 LDA_SVWN Test Suite")
    print("=" * 60)
    print("Reference: Certik et al., Comput. Phys. Commun. 297, 109051 (2024)")
    print("=" * 60)

    test_results = []
    test_results.append(("Basic calculation", _run_as_bool(test_z92_lda_svwn_basic)))
    test_results.append(("Eigenvalue comparison", _run_as_bool(test_z92_lda_svwn_eigenvalues)))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in test_results if result)
    total  = len(test_results)

    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"  {test_name:<30} : {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)
