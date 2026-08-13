"""
RPA accuracy, timing, and memory checks against tests/accuracy/RPA/reference.json.

    python generate_reference_data.py         # once, on a 24-core allocation --
                                               # populates reference.json and each
                                               # case's own run.log/result.json
    pytest tests/accuracy/RPA -m rpa_accuracy  # run these tests

Each of the 15 cases (5 elements x self-consistent RPA {AE, psp}, plus 5 elements x
non-self-consistent RPA@DFT {AE only}) is re-run fresh, in its own subprocess (same
method generate_reference_data.py used, via cases.run_subprocess_case -- this keeps
peak-RSS measurement meaningful, see that function's docstring), and checked two ways:

    HARD  (fails the test, 1e-5 absolute tolerance): total_energy,
          rpa_correlation_energy, exact_exchange_energy, and -- self-consistent
          RPA only -- ionization_potential (= -HOMO eigenvalue).  These four are
          the only quantities in reference.json, and the only ones compared.

    SOFT  (warns, never fails, 20% relative tolerance): wall time and peak RSS,
          read from the case's own result.json (not from reference.json -- that
          file deliberately holds only the four physics quantities above).  These
          depend heavily on core count; see conftest.py's session warning.

A case with no reference.json entry yet (not generated) or no baseline result.json
is skipped, not failed -- it means the data hasn't been produced, not that
something is broken.
"""

from __future__ import annotations

import json
import os
import warnings

import pytest

# Loaded by path under a unique module name.  Both this suite and ../sinh/ have a
# cases.py, and a plain `import cases` binds sys.modules['cases'] to whichever is
# collected first -- silently giving one suite the other's Lmax, omega and expected
# grid.
import importlib.util as _importlib_util

_spec = _importlib_util.spec_from_file_location(
    "rpa_algebraic_cases",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.py"))
C = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(C)

pytestmark = [pytest.mark.rpa_accuracy, pytest.mark.rpa_algebraic]

ENERGY_ABS_TOL = 1e-5
PERF_REL_TOL   = 0.20
DEFAULT_TIMEOUT_S = 1800


def _load_reference():
    path = os.path.join(C.HERE, "reference.json")
    if not os.path.exists(path):
        return {}
    return json.load(open(path))


REFERENCE = _load_reference()


def _baseline_perf(case):
    """wall_s / peak_rss_mb from the case's own stored result.json, or (None, None)."""
    p = os.path.join(case["outdir"], "result.json")
    if not os.path.exists(p):
        return None, None
    r = json.load(open(p))
    if not r.get("ok"):
        return None, None
    return r.get("wall_s"), r.get("peak_rss_mb")


@pytest.mark.parametrize("case", C.ALL_CASES, ids=C.case_key)
def test_rpa_case(case):
    key = C.case_key(case)
    ref = REFERENCE.get(key)
    if ref is None:
        pytest.skip("no reference.json entry for %s yet -- run generate_reference_data.py" % key)

    baseline_wall, baseline_mem = _baseline_perf(case)
    timeout = max(DEFAULT_TIMEOUT_S, (baseline_wall or 0) * 5)

    result = C.run_subprocess_case(case, timeout=timeout)
    if not result.get("ok"):
        pytest.fail("%s did not complete: %s" % (key, result.get("error")))

    # ---- hard checks: the four physics quantities, and only these ----
    assert result["total_energy"] == pytest.approx(ref["total_energy"], abs=ENERGY_ABS_TOL), \
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

    # ---- soft checks: timing / memory, warning-only, never fail the test ----
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
                "tolerance)" % (key, live_mem, baseline_mem, 100 * rel, 100 * PERF_REL_TOL))
