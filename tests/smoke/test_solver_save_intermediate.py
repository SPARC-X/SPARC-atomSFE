"""Smoke tests for `save_intermediate` behavior in AtomicDFTSolver.

This module validates that enabling intermediate-data recording in SCF:
1) leaves core solver outputs available and well-formed,
2) populates `intermediate_info` with expected structure,
3) preserves numerical results compared with `save_intermediate=False`.

Test intent:
- Keep checks lightweight and robust as smoke tests.
- Validate behavior/consistency, not strict performance tuning.

How to run:
- Run only this file with pytest:
  python -m pytest tests/smoke/test_solver_save_intermediate.py -q

- Run all smoke tests:
  python -m pytest tests/smoke -q
"""

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atom.solver import AtomicDFTSolver


pytestmark = pytest.mark.smoke

# Shared smoke parameters (kept at file top for centralized edits)
ATOMIC_NUMBER           = 3
DOMAIN_SIZE             = 25.0
FINITE_ELEMENT_NUMBER   = 4
POLYNOMIAL_ORDER        = 20
QUADRATURE_POINT_NUMBER = 50
VERBOSE                 = False
XC_FUNCTIONAL           = "GGA_PBE"
ALL_ELECTRON_FLAG       = True
MESH_TYPE               = "exponential"
MESH_CONCENTRATION      = 101.0

SCF_SOLVER_KWARGS = {
    "atomic_number"           : ATOMIC_NUMBER,
    "domain_size"             : DOMAIN_SIZE,
    "finite_element_number"   : FINITE_ELEMENT_NUMBER,
    "polynomial_order"        : POLYNOMIAL_ORDER,
    "quadrature_point_number" : QUADRATURE_POINT_NUMBER,
    "verbose"                 : VERBOSE,
    "xc_functional"           : XC_FUNCTIONAL,
    "all_electron_flag"       : ALL_ELECTRON_FLAG,
    "mesh_type"               : MESH_TYPE,
    "mesh_concentration"      : MESH_CONCENTRATION,
}


def _build_solver() -> AtomicDFTSolver:
    return AtomicDFTSolver(**SCF_SOLVER_KWARGS)


def test_solve_without_intermediate() -> None:
    solver = _build_solver()
    results = solver.solve(save_intermediate=False)
    assert results["intermediate_info"] is None
    assert "rho" in results
    assert "orbitals" in results
    assert "energy" in results


def test_solve_with_intermediate() -> None:
    solver = _build_solver()
    results = solver.solve(save_intermediate=True)
    info = results["intermediate_info"]
    assert info is not None
    assert hasattr(info, "inner_iterations")
    assert hasattr(info, "outer_iterations")
    assert hasattr(info, "current_outer_iteration")
    assert isinstance(info.inner_iterations, list)
    assert isinstance(info.outer_iterations, list)
    assert isinstance(info.current_outer_iteration, (int, np.integer))
    assert len(info.inner_iterations) > 0


def test_intermediate_data_consistency() -> None:
    solver = _build_solver()
    results = solver.solve(save_intermediate=True)
    info = results["intermediate_info"]

    for inner_iter in info.inner_iterations:
        assert inner_iter.outer_iteration >= 0
        assert inner_iter.inner_iteration > 0
        assert inner_iter.rho_residual >= 0
        if hasattr(inner_iter, "rho"):
            assert len(inner_iter.rho) > 0
            assert np.all(np.isfinite(inner_iter.rho))
            rho_norm = getattr(inner_iter, "rho_norm", None)
            if rho_norm is not None:
                assert rho_norm > 0
            else:
                assert np.linalg.norm(inner_iter.rho) > 0


def test_save_intermediate_does_not_affect_results() -> None:
    solver = _build_solver()
    results_without = solver.solve(save_intermediate=False)
    results_with = solver.solve(save_intermediate=True)

    assert results_without["intermediate_info"] is None
    assert results_with["intermediate_info"] is not None
    assert np.isclose(results_without["energy"], results_with["energy"], rtol=1e-10)
    assert np.allclose(results_without["rho"], results_with["rho"], rtol=1e-10)
    assert np.allclose(results_without["orbitals"], results_with["orbitals"], rtol=1e-10)
    assert np.allclose(
        results_without["eigen_energies"], results_with["eigen_energies"], rtol=1e-10
    )
    assert results_without["converged"] == results_with["converged"]
    assert results_without["iterations"] == results_with["iterations"]
    assert np.isclose(
        results_without["rho_residual"], results_with["rho_residual"], rtol=1e-10
    )

