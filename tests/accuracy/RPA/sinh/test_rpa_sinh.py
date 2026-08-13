"""
RPA accuracy, timing and memory checks on the SINH omega mapping, against
tests/accuracy/RPA/rpa_oep_mapping/reference.json.

    python generate_reference_data.py            # once, populates reference.json
    pytest tests/accuracy/RPA/rpa_oep_mapping -m rpa_accuracy

Each of the 15 cases (5 elements x self-consistent RPA {AE, psp}, plus 5 elements x
non-self-consistent RPA@DFT {AE only}) is re-run fresh in its own subprocess -- the same
way generate_reference_data.py measured it, via cases.run_subprocess_case -- and checked:

    HARD  (fails, 1e-5 absolute): total_energy, rpa_correlation_energy,
          exact_exchange_energy, and -- self-consistent RPA only --
          ionization_potential (= -HOMO).  These four are the only quantities in
          reference.json and the only ones compared.

    HARD  (fails): the frequency grid actually in force.  See below.

    SOFT  (warns, 20% relative): wall time and peak RSS, from the case's own
          result.json.  Both scale with core count; see conftest.py.

A case with no reference.json entry is skipped, not failed -- it means the data has not
been generated, not that something is broken.

WHY THE GRID IS A HARD CHECK
----------------------------
frequency_grid_type is not a solver argument; it is a default on
RPACorrelation.__init__.  So these reference numbers depend on it while nothing in the
case definition pins it, and if the default moves the whole file silently becomes
numbers for a different quadrature -- which is exactly how ../algebraic_omega_mapping/
came to hold inverse_linear results that read as if they were current.  Comparing it
explicitly turns that into a failed assertion naming the cause.
"""

from __future__ import annotations

import json
import os
import warnings

import pytest

# Loaded by path under a unique module name.  Both this suite and ../algebraic/ have a
# cases.py, and a plain `import cases` binds sys.modules['cases'] to whichever is
# collected first -- silently giving one suite the other's Lmax, omega and expected
# grid.
import importlib.util as _importlib_util

_spec = _importlib_util.spec_from_file_location(
    "rpa_sinh_cases",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.py"))
C = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(C)

pytestmark = [pytest.mark.rpa_accuracy, pytest.mark.rpa_sinh]

ENERGY_ABS_TOL = 1e-5
PERF_REL_TOL   = 0.20
DEFAULT_TIMEOUT_S = 3600


def _load_reference():
    path = os.path.join(C.HERE, "reference.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


REFERENCE = _load_reference()


def _baseline_perf(case):
    """wall_s / peak_rss_mb from the case's own stored result.json, or (None, None)."""
    path = os.path.join(case["outdir"], "result.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as fh:
        record = json.load(fh)
    if not record.get("ok"):
        return None, None
    return record.get("wall_s"), record.get("peak_rss_mb")


@pytest.mark.parametrize("case", C.ALL_CASES, ids=C.case_key)
def test_rpa_case(case):
    key = C.case_key(case)
    ref = REFERENCE.get(key)
    if ref is None:
        pytest.skip("no reference.json entry for %s yet -- run "
                    "generate_reference_data.py" % key)

    baseline_wall, baseline_mem = _baseline_perf(case)
    timeout = max(DEFAULT_TIMEOUT_S, (baseline_wall or 0) * 5)

    result = C.run_subprocess_case(case, timeout=timeout)
    if not result.get("ok"):
        pytest.fail("%s did not complete: %s" % (key, result.get("error")))

    # ---- hard: the grid these numbers were measured on ----
    assert result.get("frequency_grid_type") == C.EXPECTED_GRID_TYPE, \
        ("%s: rpa.py's default frequency grid is %r, but reference.json was generated "
         "with %r -- these numbers are for a different quadrature, not a regression"
         % (key, result.get("frequency_grid_type"), C.EXPECTED_GRID_TYPE))
    assert result.get("frequency_base_rule") == C.EXPECTED_BASE_RULE, \
        ("%s: rpa.py's default base rule is %r, but reference.json was generated with "
         "%r -- a different rule on the same map is still a different quadrature"
         % (key, result.get("frequency_base_rule"), C.EXPECTED_BASE_RULE))
    if result.get("omega_ceiling") is not None:
        assert result["omega_ceiling"] == pytest.approx(C.EXPECTED_OMEGA_CEILING,
                                                        rel=1e-12), \
            ("%s: omega_ceiling is %r, reference generated with %r"
             % (key, result["omega_ceiling"], C.EXPECTED_OMEGA_CEILING))

    # ---- hard: the four physics quantities, and only these ----
    assert result["total_energy"] == pytest.approx(ref["total_energy"],
                                                   abs=ENERGY_ABS_TOL), \
        "%s: total energy mismatch" % key
    if ref.get("rpa_correlation_energy") is not None:
        assert result.get("rpa_correlation") == pytest.approx(
            ref["rpa_correlation_energy"], abs=ENERGY_ABS_TOL), \
            "%s: RPA correlation energy mismatch" % key
    if ref.get("exact_exchange_energy") is not None:
        assert result.get("exact_exchange") == pytest.approx(
            ref["exact_exchange_energy"], abs=ENERGY_ABS_TOL), \
            "%s: exact exchange energy mismatch" % key
    if case["mode"] == "RPA-OEP" and ref.get("ionization_potential") is not None:
        ip = -result["homo"] if result.get("homo") is not None else None
        assert ip == pytest.approx(ref["ionization_potential"], abs=ENERGY_ABS_TOL), \
            "%s: ionization potential (-HOMO) mismatch" % key

    # ---- soft: timing / memory, warning-only ----
    live_wall = result.get("wall_s")
    if baseline_wall and live_wall is not None:
        rel = abs(live_wall - baseline_wall) / baseline_wall
        if rel > PERF_REL_TOL:
            warnings.warn(
                "%s: wall time %.1fs vs reference %.1fs (%.0f%% off, >%.0f%% "
                "tolerance) -- see this folder's conftest.py core-count note"
                % (key, live_wall, baseline_wall, 100 * rel, 100 * PERF_REL_TOL))

    live_mem = result.get("peak_rss_mb")
    if baseline_mem and live_mem is not None:
        rel = abs(live_mem - baseline_mem) / baseline_mem
        if rel > PERF_REL_TOL:
            warnings.warn(
                "%s: peak RSS %.0f MiB vs reference %.0f MiB (%.0f%% off, >%.0f%% "
                "tolerance)" % (key, live_mem, baseline_mem, 100 * rel,
                                100 * PERF_REL_TOL))
