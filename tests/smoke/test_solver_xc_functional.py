"""Smoke tests for core AtomicDFTSolver workflows.

This module validates that representative solver configurations run end-to-end
without runtime errors and return core output fields (`rho`, `orbitals`).

Functional families covered in this file:
1) LDA      : `LDA_SVWN`
2) GGA      : `GGA_PBE`
3) meta-GGA : `RSCAN`
4) HF       : `HF`
5) Hybrid   : `PBE0`
6) RPA      : `RPA`

Test intent:
- Keep these tests lightweight and robust as smoke checks.
- Verify solver execution and basic output presence, not tight accuracy bounds.

How to run:
- Run only this file with pytest:
  python -m pytest tests/smoke/test_solver_xc_functional.py -q

- Run all smoke tests:
  python -m pytest tests/smoke -q
"""

from pathlib import Path
import sys

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
OEP_BASIS_NUMBER        = 5
VERBOSE                 = False
MESH_TYPE               = "exponential"
MESH_CONCENTRATION      = 101.0
ENABLE_PARALLELIZATION  = True
ALL_ELECTRON_FLAG       = True

LDA_SOLVER_KWARGS = {
    "atomic_number"           : ATOMIC_NUMBER,
    "verbose"                 : VERBOSE,
    "xc_functional"           : "LDA_SVWN",
    "all_electron_flag"       : ALL_ELECTRON_FLAG,
    "domain_size"             : DOMAIN_SIZE,
    "finite_element_number"   : FINITE_ELEMENT_NUMBER,
    "polynomial_order"        : POLYNOMIAL_ORDER,
    "quadrature_point_number" : QUADRATURE_POINT_NUMBER,
    "mesh_type"               : MESH_TYPE,
    "mesh_concentration"      : MESH_CONCENTRATION,
}

GGA_SOLVER_KWARGS = {
    "atomic_number"           : ATOMIC_NUMBER,
    "verbose"                 : VERBOSE,
    "xc_functional"           : "GGA_PBE",
    "all_electron_flag"       : ALL_ELECTRON_FLAG,
    "domain_size"             : DOMAIN_SIZE,
    "finite_element_number"   : FINITE_ELEMENT_NUMBER,
    "polynomial_order"        : POLYNOMIAL_ORDER,
    "quadrature_point_number" : QUADRATURE_POINT_NUMBER,
    "mesh_type"               : MESH_TYPE,
    "mesh_concentration"      : MESH_CONCENTRATION,
}

META_GGA_SOLVER_KWARGS = {
    "atomic_number"           : ATOMIC_NUMBER,
    "verbose"                 : VERBOSE,
    "xc_functional"           : "RSCAN",
    "all_electron_flag"       : ALL_ELECTRON_FLAG,
    "domain_size"             : DOMAIN_SIZE,
    "finite_element_number"   : FINITE_ELEMENT_NUMBER,
    "polynomial_order"        : POLYNOMIAL_ORDER,
    "quadrature_point_number" : QUADRATURE_POINT_NUMBER,
    "mesh_type"               : MESH_TYPE,
    "mesh_concentration"      : MESH_CONCENTRATION,
}

HF_SOLVER_KWARGS = {
    "atomic_number"           : ATOMIC_NUMBER,
    "domain_size"             : DOMAIN_SIZE,
    "finite_element_number"   : FINITE_ELEMENT_NUMBER,
    "polynomial_order"        : POLYNOMIAL_ORDER,
    "quadrature_point_number" : QUADRATURE_POINT_NUMBER,
    "oep_basis_number"        : OEP_BASIS_NUMBER,
    "verbose"                 : VERBOSE,
    "xc_functional"           : "HF",
    "all_electron_flag"       : ALL_ELECTRON_FLAG,
    "mesh_type"               : MESH_TYPE,
    "mesh_concentration"      : MESH_CONCENTRATION,
}

HYBRID_SOLVER_KWARGS = {
    "atomic_number"           : ATOMIC_NUMBER,
    "domain_size"             : DOMAIN_SIZE,
    "finite_element_number"   : FINITE_ELEMENT_NUMBER,
    "polynomial_order"        : POLYNOMIAL_ORDER,
    "quadrature_point_number" : QUADRATURE_POINT_NUMBER,
    "oep_basis_number"        : OEP_BASIS_NUMBER,
    "verbose"                 : VERBOSE,
    "xc_functional"           : "PBE0",
    "all_electron_flag"       : ALL_ELECTRON_FLAG,
    "use_oep"                 : False,
    "mesh_type"               : MESH_TYPE,
    "mesh_concentration"      : MESH_CONCENTRATION,
}

RPA_SOLVER_KWARGS = {
    "atomic_number"           : ATOMIC_NUMBER,
    "domain_size"             : DOMAIN_SIZE,
    "finite_element_number"   : FINITE_ELEMENT_NUMBER,
    "polynomial_order"        : POLYNOMIAL_ORDER,
    "quadrature_point_number" : QUADRATURE_POINT_NUMBER,
    "oep_basis_number"        : OEP_BASIS_NUMBER,
    "verbose"                 : VERBOSE,
    "xc_functional"           : "RPA",
    "all_electron_flag"       : ALL_ELECTRON_FLAG,
    "use_oep"                 : True,
    "mesh_type"               : MESH_TYPE,
    "mesh_concentration"      : MESH_CONCENTRATION,
    "enable_parallelization"  : ENABLE_PARALLELIZATION,
}


def _assert_basic_outputs(results: dict) -> None:
    assert results is not None
    assert "rho" in results
    assert "orbitals" in results
    assert results["rho"] is not None
    assert results["orbitals"] is not None


def test_lda() -> None:
    solver = AtomicDFTSolver(**LDA_SOLVER_KWARGS)
    results = solver.solve()
    _assert_basic_outputs(results)


def test_gga() -> None:
    solver = AtomicDFTSolver(**GGA_SOLVER_KWARGS)
    results = solver.solve()
    _assert_basic_outputs(results)


def test_meta_gga() -> None:
    solver = AtomicDFTSolver(**META_GGA_SOLVER_KWARGS)
    results = solver.solve()
    _assert_basic_outputs(results)


def test_hf() -> None:
    solver = AtomicDFTSolver(**HF_SOLVER_KWARGS)
    results = solver.solve()
    _assert_basic_outputs(results)


def test_hybrid() -> None:
    solver = AtomicDFTSolver(**HYBRID_SOLVER_KWARGS)
    results = solver.solve()
    _assert_basic_outputs(results)
    assert "v_x_local" in results
    assert "v_c_local" in results
    assert len(results["v_x_local"]) > 0
    assert len(results["v_c_local"]) > 0


def test_rpa() -> None:
    solver = AtomicDFTSolver(**RPA_SOLVER_KWARGS)
    results = solver.solve()
    _assert_basic_outputs(results)

